import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.teachers.base import ARTeacher


class HierarchicalTeacher(ARTeacher):
    """Wraps a base ARTeacher and exposes a surface-space interface.

    The base teacher operates over a hidden vocabulary of size `base_teacher.dim`.
    Each hidden token id maps to `num_tuples` (M) distinct length-`chunk_size`
    surface tuples — one-hot sequences of dimension `chunk_dim` — via a chunk
    table with globally disjoint supports across (hidden_id, tuple_idx). Because
    supports are disjoint, any complete chunk uniquely identifies its hidden id,
    so surface→hidden decoding remains deterministic even with M > 1.

    The forward direction (hidden→surface) is stochastic when M > 1: each hidden
    token uniformly draws one of its M tuples. Next-slot prediction therefore
    marginalizes over both the hidden posterior *and* the within-hidden tuple
    posterior conditioned on any observed slots of the current chunk.

    Distribution sharpness is encoded in the base teacher's weight scale (see
    `LinearARTeacher.from_parameters(scale=...)`) — there is no temperature knob
    here. `next_token_log_probs`, `predict_next`, and `unroll` all return
    **log surface probabilities**.
    """

    def __init__(
        self,
        base_teacher: ARTeacher,
        chunk_dim: int,
        chunk_size: int,
        num_tuples: int = 1,
        chunk_seed: Optional[int] = None,
        chunk_table: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_size > chunk_dim:
            raise ValueError("chunk_size cannot exceed chunk_dim.")
        if num_tuples <= 0:
            raise ValueError("num_tuples must be a positive integer.")

        self.base_teacher = base_teacher
        self.chunk_dim = chunk_dim
        self.chunk_size = chunk_size
        self.num_tuples = num_tuples

        if chunk_table is None:
            generator = None
            if chunk_seed is not None:
                generator = torch.Generator().manual_seed(int(chunk_seed))
            chunk_table = self._generate_unique_chunks(generator=generator)
        else:
            expected = (self.hidden_dim, num_tuples, chunk_size, chunk_dim)
            if tuple(chunk_table.shape) != expected:
                raise ValueError(
                    f"chunk_table shape {tuple(chunk_table.shape)} != expected {expected}"
                )

        self.register_buffer("_chunk_table", chunk_table)
        self.register_buffer("_chunk_slot_indices", chunk_table.argmax(dim=-1))

    # --- ARTeacher interface ---
    @property
    def dim(self) -> int:
        return self.chunk_dim

    @property
    def context_length(self) -> int:
        return self.base_teacher.context_length * self.chunk_size

    @property
    def hidden_dim(self) -> int:
        return self.base_teacher.dim

    @property
    def window(self) -> int:
        return self.base_teacher.window

    @property
    def span_lengths(self) -> list:
        return [s * self.chunk_size for s in self.base_teacher.span_lengths]

    @property
    def stride(self) -> Optional[int]:
        base_stride = getattr(self.base_teacher, "stride", None)
        return base_stride * self.chunk_size if base_stride is not None else None

    def _get_weights(self) -> torch.Tensor:
        return self.base_teacher._get_weights()

    def next_token_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        """Predict slot 0 of the *next* chunk, given a chunk-aligned context.

        context: (B, context_length, chunk_dim). Returns (B, chunk_dim) log-probs.
        """
        if context.shape[-2] != self.context_length:
            raise ValueError(
                f"context has {context.shape[-2]} tokens; expected {self.context_length}"
            )
        hidden = self._decode_chunk_aligned(context)  # (B, base_ctx_h, hidden_dim)
        hidden_log_probs = self.base_teacher.next_token_log_probs(hidden)  # (B, hidden_dim)

        B = hidden_log_probs.shape[0]
        dummy_observed = torch.zeros(
            B, self.chunk_size, dtype=torch.long, device=hidden_log_probs.device
        )
        return self._hidden_probs_to_surface_logprobs(
            hidden_log_probs=hidden_log_probs,
            observed_slots=dummy_observed,
            num_observed=0,
            slot_to_predict=0,
        )

    def predict_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Autoregressive single-step prediction.

        Handles mid-chunk positions: if `T % chunk_size == s > 0`, uses the last
        s surface tokens as observed slots of the currently-emitting chunk and
        returns the conditional distribution over slot s.
        """
        T = prefix.shape[-2]
        if T < self.context_length:
            raise ValueError(
                f"prefix length {T} < context_length {self.context_length}"
            )

        s = T % self.chunk_size
        needed = self.context_length + s
        if T > needed:
            prefix = prefix[..., -needed:, :]

        # First context_length tokens are chunk-aligned.
        context_surface = prefix[..., : self.context_length, :]
        context_hidden = self._decode_chunk_aligned(context_surface)
        hidden_log_probs = self.base_teacher.next_token_log_probs(context_hidden)

        B = hidden_log_probs.shape[0]
        observed = torch.zeros(
            B, self.chunk_size, dtype=torch.long, device=prefix.device
        )
        if s > 0:
            partial = prefix[..., self.context_length :, :]  # (B, s, chunk_dim)
            observed[:, :s] = partial.argmax(dim=-1)

        return self._hidden_probs_to_surface_logprobs(
            hidden_log_probs=hidden_log_probs,
            observed_slots=observed,
            num_observed=s,
            slot_to_predict=s,
        )

    def unroll(
        self,
        sequence: torch.Tensor,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-position surface predictions over a chunk-aligned batch.

        sequence: (B, L_surf, chunk_dim) with L_surf a multiple of chunk_size and
        L_surf > context_length. Returns log-probs (B, L_surf - context_length,
        chunk_dim).
        """
        B, L_surf, D = sequence.shape
        if L_surf % self.chunk_size != 0:
            raise ValueError(
                f"sequence length {L_surf} must be a multiple of chunk_size {self.chunk_size}"
            )
        if L_surf <= self.context_length:
            raise ValueError(
                f"sequence length {L_surf} must exceed context_length {self.context_length}"
            )
        L_h = L_surf // self.chunk_size

        hidden = self._decode_chunk_aligned(sequence)  # (B, L_h, hidden_dim)
        hidden_log_probs = self.base_teacher.unroll(hidden)  # (B, L_h - base_ctx_h, hidden_dim)
        L_out_h = hidden_log_probs.shape[1]
        ctx_surf = self.context_length

        # Slot indices of the *predicted* chunks (for mid-chunk conditioning).
        pred_slot_idx = (
            sequence[:, ctx_surf:, :]
            .reshape(B, L_out_h, self.chunk_size, self.chunk_dim)
            .argmax(dim=-1)
        )  # (B, L_out_h, chunk_size)

        slot_logprobs = []
        for s in range(self.chunk_size):
            lp = self._hidden_probs_to_surface_logprobs(
                hidden_log_probs=hidden_log_probs,
                observed_slots=pred_slot_idx,
                num_observed=s,
                slot_to_predict=s,
            )
            slot_logprobs.append(lp)
        stacked = torch.stack(slot_logprobs, dim=2)  # (B, L_out_h, chunk_size, chunk_dim)
        surface_logprobs = stacked.reshape(B, L_out_h * self.chunk_size, D)

        if return_targets:
            targets = sequence[:, ctx_surf:, :]
            return surface_logprobs, targets
        return surface_logprobs

    def with_lag_restriction(self, k: int) -> "HierarchicalTeacher":
        """Restrict the underlying base teacher's lags; keep the same chunk table."""
        restricted_base = self.base_teacher.with_lag_restriction(k)
        return HierarchicalTeacher(
            base_teacher=restricted_base,
            chunk_dim=self.chunk_dim,
            chunk_size=self.chunk_size,
            num_tuples=self.num_tuples,
            chunk_table=self._chunk_table,
        )

    # --- Internals ---
    def _generate_unique_chunks(
        self, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        needed = self.hidden_dim * self.num_tuples
        total_permutations = math.perm(self.chunk_dim, self.chunk_size)
        if total_permutations < needed:
            raise ValueError(
                "Not enough unique chunk permutations to cover all hidden tokens "
                f"with {self.num_tuples} tuples each. "
                f"Need {needed}, but only {total_permutations} available."
            )

        chunks = torch.zeros(
            self.hidden_dim, self.num_tuples, self.chunk_size, self.chunk_dim
        )
        used: set = set()
        for hid in range(self.hidden_dim):
            for m in range(self.num_tuples):
                for _ in range(1000):
                    indices = torch.randperm(self.chunk_dim, generator=generator)[
                        : self.chunk_size
                    ]
                    signature = tuple(indices.tolist())
                    if signature not in used:
                        used.add(signature)
                        chunk = torch.zeros(self.chunk_size, self.chunk_dim)
                        chunk[torch.arange(self.chunk_size), indices] = 1.0
                        chunks[hid, m] = chunk
                        break
                else:
                    raise RuntimeError(
                        "Failed to sample a unique chunk sequence after many attempts."
                    )
        return chunks

    def decode_chunk_aligned(self, surface: torch.Tensor) -> torch.Tensor:
        """Public wrapper for `_decode_chunk_aligned` — surface -> hidden one-hots.

        Surface tokens must be chunk-aligned (length divisible by `chunk_size`)
        and each chunk must be a valid slot-tuple in the chunk table; otherwise
        the argmax falls back to hidden id 0.
        """
        return self._decode_chunk_aligned(surface)

    def _decode_chunk_aligned(self, surface: torch.Tensor) -> torch.Tensor:
        """(..., L_h * chunk_size, chunk_dim) -> (..., L_h, hidden_dim) one-hot.

        With num_tuples > 1, matches the observed slot sequence against any of
        the M tuples per hidden id. Supports are globally disjoint, so exactly
        one (hidden_id, tuple_idx) can match a valid input.
        """
        *lead, l_surf, cd = surface.shape
        if cd != self.chunk_dim:
            raise ValueError(f"Trailing dim {cd} != chunk_dim {self.chunk_dim}")
        if l_surf % self.chunk_size != 0:
            raise ValueError(
                f"Surface length {l_surf} is not chunk-aligned (chunk_size={self.chunk_size})"
            )
        l_h = l_surf // self.chunk_size
        chunks = surface.reshape(*lead, l_h, self.chunk_size, self.chunk_dim)
        slot_idx = chunks.argmax(dim=-1)  # (..., L_h, chunk_size)
        # table: (hidden_dim, M, chunk_size); expand slot_idx to (..., L_h, 1, 1, chunk_size)
        matches = (
            slot_idx.unsqueeze(-2).unsqueeze(-2) == self._chunk_slot_indices
        ).all(dim=-1)  # (..., L_h, hidden_dim, M)
        flat = matches.reshape(*matches.shape[:-2], self.hidden_dim * self.num_tuples)
        hidden_ids = flat.float().argmax(dim=-1) // self.num_tuples
        return F.one_hot(hidden_ids, num_classes=self.hidden_dim).to(surface.dtype)

    def _compat_mask(
        self, observed_slots: torch.Tensor, num_observed: int
    ) -> torch.Tensor:
        """Compatibility mask over (hidden_id, tuple_idx) given leading slots.

        Returns shape (..., hidden_dim, num_tuples). Entry (h, m) is 1 iff
        `_chunk_slot_indices[h, m, :num_observed] == observed_slots[..., :num_observed]`.
        For num_observed == 0, all entries are 1.
        """
        lead_shape = observed_slots.shape[:-1]
        if num_observed == 0:
            return torch.ones(
                (*lead_shape, self.hidden_dim, self.num_tuples),
                device=observed_slots.device,
                dtype=torch.float,
            )
        obs = observed_slots[..., :num_observed]  # (..., num_observed)
        table = self._chunk_slot_indices[:, :, :num_observed]  # (hidden_dim, M, num_observed)
        matches = (obs.unsqueeze(-2).unsqueeze(-2) == table).all(dim=-1)  # (..., hidden_dim, M)
        return matches.to(torch.float)

    def _hidden_probs_to_surface_logprobs(
        self,
        hidden_log_probs: torch.Tensor,
        observed_slots: torch.Tensor,
        num_observed: int,
        slot_to_predict: int,
    ) -> torch.Tensor:
        """Marginalize over (hidden_id, tuple_idx) to get surface log-probs.

        Uses a uniform 1/M prior over tuples per hidden id. Compatibility
        filtering conditions the posterior on the observed slot prefix; the
        result is mixed through the chunk table at `slot_to_predict`.
        """
        p_h = hidden_log_probs.exp()  # (..., hidden_dim)
        # Uniform tuple prior 1/M; broadcast into (..., hidden_dim, M)
        p_hm = p_h.unsqueeze(-1) / self.num_tuples
        compat = self._compat_mask(observed_slots, num_observed)  # (..., hidden_dim, M)
        p_posterior = p_hm * compat
        norm = p_posterior.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-30)
        p_posterior = p_posterior / norm
        # chunk_table[:, :, slot_to_predict, :] -> (hidden_dim, M, chunk_dim)
        slot_table = self._chunk_table[:, :, slot_to_predict, :]
        surface_probs = torch.einsum("...hm,hmd->...d", p_posterior, slot_table)
        return surface_probs.clamp(min=1e-30).log()

    def sample_surface_prefix(
        self,
        num_surface_tokens: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample a valid chunk-composed surface prefix.

        Uniform-random hidden ids -> chunk-table lookup. Guarantees every sampled
        surface token is a legitimate slot of some hidden id's chunk.
        """
        if num_surface_tokens % self.chunk_size != 0:
            raise ValueError(
                f"num_surface_tokens ({num_surface_tokens}) must be a multiple of "
                f"chunk_size ({self.chunk_size})"
            )
        n_hidden = num_surface_tokens // self.chunk_size
        table = self._chunk_table
        if device is not None:
            table = table.to(device)
        hidden_ids = torch.randint(
            0, self.hidden_dim, (n_hidden,), device=table.device
        )
        tuple_ids = torch.randint(
            0, self.num_tuples, (n_hidden,), device=table.device
        )
        chunks = table[hidden_ids, tuple_ids]  # (n_hidden, chunk_size, chunk_dim)
        return chunks.reshape(num_surface_tokens, self.chunk_dim)
