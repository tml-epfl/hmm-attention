from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.utils import split_into_windows

# Sentinel `context_length` for teachers that condition on the *entire* prefix
# (e.g. attention) rather than a fixed Markov window. Kept an int so downstream
# `getattr(teacher, "context_length")` reads stay typed; logic that must
# distinguish the two regimes checks `is_adaptive` / this sentinel directly.
ADAPTIVE = -1


class ARTeacher(nn.Module, ABC):
    """Abstract autoregressive teacher.

    Subclasses only need to implement `dim`, `context_length`, and
    `next_token_log_probs(context)`. Sequence-level operations (per-position
    unrolling and autoregressive-generation prefix slicing) are shared.

    Output convention: every teacher returns **log-probabilities** — a (B, dim)
    tensor that sums to 1 when exponentiated along the last dim. This is the
    only representation shared across teacher families (HierarchicalTeacher has
    no coherent raw-logit form because its output mixes softmaxed hidden
    distributions through a chunk table).
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vocab dimension."""

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Number of tokens the teacher conditions on to predict the next one.

        Returns `ADAPTIVE` for teachers that condition on the *entire* prefix
        (unbounded memory) rather than a fixed Markov window.
        """

    @property
    def is_adaptive(self) -> bool:
        """Whether the teacher conditions on the whole prefix (vs a fixed window)."""
        return self.context_length == ADAPTIVE

    @property
    def burn_in(self) -> int:
        """Minimum prefix length before the teacher can predict — i.e. how many
        tokens the dataset seeds before autoregression starts.

        **Invariant:** for non-adaptive teachers `burn_in == context_length`
        (this default). Downstream alignment — the bounded `unroll` offset,
        `TeacherEvaluator._align_data`, `NgramEvaluator._slice_data` — relies on
        this, so bounded teachers must not diverge.

        Adaptive teachers (`context_length == ADAPTIVE`) have no window to fall
        back on, so `burn_in` isn't derivable here — they **must** override this
        with their own positive minimum.
        """
        if self.is_adaptive:
            raise NotImplementedError(
                f"{type(self).__name__} is adaptive and must define `burn_in` "
                "(context_length is ADAPTIVE, so there is no window to default to)."
            )
        return self.context_length

    @abstractmethod
    def next_token_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        """Predict next-token log-probabilities from a fixed-size context.

        Args:
            context: shape (B, context_length, dim). Callers must hand exactly
                context_length tokens — see `predict_next` for automatic
                slicing from a longer prefix.

        Returns:
            (B, dim) log-probabilities. `.exp()` sums to 1 along the last dim.
        """

    def predict_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Autoregressive single-step prediction (log-probs).

        Requires at least `burn_in` tokens. Bounded teachers then condition on
        the trailing `context_length` tokens; adaptive teachers
        (`context_length == ADAPTIVE`) condition on the entire prefix.
        """
        T = prefix.shape[-2]
        if T < self.burn_in:
            raise ValueError(f"prefix length {T} < burn_in {self.burn_in}")
        if self.context_length != ADAPTIVE and T > self.context_length:
            prefix = prefix[..., -self.context_length :, :]
        return self.next_token_log_probs(prefix)

    def sequence_log_probs(self, sequence: torch.Tensor) -> torch.Tensor:
        """Causal per-position next-token log-probs in one shot.

        Returns (B, L, dim) where position `t` is the log-prob distribution over
        token `t+1` conditioned on `sequence[:, : t+1, :]`. Must be **causal**:
        position `t` sees only tokens `0..t`.

        Optional. Teachers with a natural full-sequence forward (attention,
        transformers) implement this once and get `unroll` — hence per-position
        evaluation — for free. Bounded teachers can instead rely on the default
        windowed `unroll` and leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `sequence_log_probs`."
        )

    def unroll(
        self,
        sequence: torch.Tensor,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-position predictions over a full sequence.

        Output `j` predicts `sequence[:, burn_in + j, :]`. Bounded teachers
        vectorize a fixed-window sweep via `split_into_windows`. Adaptive
        teachers get this for free from their causal `sequence_log_probs`: a
        single forward whose position `t` predicts token `t+1`, so predictions
        for tokens `burn_in..L-1` are positions `burn_in-1..L-2`.

        Args:
            sequence: shape (B, L, dim). Must satisfy L >= burn_in + 1.
            return_targets: if True, also return the ground-truth next tokens
                shape (B, L - burn_in, dim).

        Returns:
            Either log-probs (B, L - burn_in, dim) or a (log_probs, targets) tuple.
        """
        B, L, D = sequence.shape
        if L <= self.burn_in:
            raise ValueError(
                f"sequence length {L} must exceed burn_in {self.burn_in}"
            )
        if self.context_length == ADAPTIVE:
            all_log_probs = self.sequence_log_probs(sequence)  # (B, L, dim)
            log_probs = all_log_probs[:, self.burn_in - 1 : L - 1, :]
            if return_targets:
                return log_probs, sequence[:, self.burn_in :, :]
            return log_probs
        contexts, targets = split_into_windows(
            seq=sequence, window=self.context_length, pad=0
        )
        # contexts: (B * (L - ctx), ctx, D); targets: (B * (L - ctx), D)
        log_probs_flat = self.next_token_log_probs(contexts)  # (B * (L - ctx), D)
        log_probs = log_probs_flat.view(B, L - self.context_length, D)
        if return_targets:
            targets = targets.view(B, L - self.context_length, D)
            return log_probs, targets
        return log_probs

    def with_lag_restriction(self, k: int) -> "ARTeacher":
        """Return a shallow-copy teacher with AR context restricted to k lags.

        Only meaningful for teachers with an explicit multi-lag structure
        (e.g. LinearARTeacher). Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support lag restriction"
        )
