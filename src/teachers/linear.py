import warnings
from typing import List, Optional

import torch
import torch.nn as nn

from src.teachers.base import ARTeacher
from src.utils import random_orthogonal_matrices, random_unit_norm_matrix


class LinearARTeacher(ARTeacher):
    """Linear autoregressive teacher over one-hot / dense token spaces.

    The teacher looks at a fixed context of `context_length` tokens, splits it
    into `window` variable-length spans (`span_lengths[i]` tokens per span),
    aggregates each span (sum, or weighted sum via `span_position_weights`),
    and computes a linear combination of the per-span aggregates using
    per-lag weight matrices `_params[i]` shape `(dim, dim)`. Optionally, a
    `stride` schedule places span start positions with overlap.

    `next_token_logits` returns raw un-normalized outputs (call softmax with a
    temperature to obtain a probability distribution).
    """

    def __init__(
        self,
        params: torch.Tensor,
        span_lengths: List[int],
        stride: Optional[int] = None,
        span_position_weights: Optional[List[float]] = None,
        rank: Optional[int] = None,
        scale: float = 1.0,
        multiplicative_constant: float = 1.0,
        reverse_constants: bool = True,
        shared_matrix_across_lags: bool = False,
        orthogonal_matrices: bool = False,
    ) -> None:
        super().__init__()
        window, dim, dim2 = params.shape
        if dim != dim2:
            raise ValueError(f"params must be (window, dim, dim); got {tuple(params.shape)}")
        if len(span_lengths) != window:
            raise ValueError(
                f"span_lengths length {len(span_lengths)} must equal window {window}"
            )

        self._params = nn.Parameter(data=params.detach().clone())
        self._dim = dim
        self._window = window
        self._span_lengths = list(span_lengths)
        self._stride = stride
        self._span_position_weights = (
            list(span_position_weights) if span_position_weights is not None else None
        )

        # Metadata (preserved for verbose logging / analysis; not used in forward).
        self.rank = rank if rank is not None else dim
        self.scale = scale
        self.multiplicative_constant = multiplicative_constant
        self.reverse_constants = reverse_constants
        self.shared_matrix_across_lags = shared_matrix_across_lags
        self.orthogonal_matrices = orthogonal_matrices

    # --- ARTeacher interface ---
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def context_length(self) -> int:
        if self._stride is not None:
            return (self._window - 1) * self._stride + self._span_lengths[-1]
        return sum(self._span_lengths)

    @property
    def window(self) -> int:
        return self._window

    @property
    def span_lengths(self) -> List[int]:
        return list(self._span_lengths)

    @property
    def stride(self) -> Optional[int]:
        return self._stride

    @property
    def span_position_weights(self) -> Optional[List[float]]:
        return list(self._span_position_weights) if self._span_position_weights else None

    def _get_weights(self) -> torch.Tensor:
        return self._params

    def next_token_logits(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, context_length, dim). Returns (B, dim) raw logits."""
        if context.shape[-2] != self.context_length:
            raise ValueError(
                f"context has {context.shape[-2]} tokens; expected exactly {self.context_length}"
            )

        # Aggregate each span into a single (dim,)-vector per batch.
        aggregates = []
        for lag_idx, span_len in enumerate(self._span_lengths):
            if self._stride is not None:
                start_idx = lag_idx * self._stride
            else:
                start_idx = sum(self._span_lengths[:lag_idx])
            end_idx = start_idx + span_len
            span_tokens = context[..., start_idx:end_idx, :]  # (B, span_len, dim)

            if self._span_position_weights is not None:
                w = torch.tensor(
                    self._span_position_weights,
                    device=span_tokens.device,
                    dtype=span_tokens.dtype,
                ).view(1, span_len, 1)
                aggregates.append((span_tokens * w).sum(dim=-2))
            else:
                aggregates.append(span_tokens.sum(dim=-2))
        # (B, window, dim)
        x_agg = torch.stack(aggregates, dim=-2)

        # (window, dim, dim) x (B, window, dim, 1) -> (B, window, dim, 1) -> sum -> (B, dim)
        weights = self._params.unsqueeze(0)  # (1, window, dim, dim)
        out = torch.matmul(weights, x_agg.unsqueeze(-1)).squeeze(-1).sum(dim=-2)
        return out

    # --- Lag restriction ---
    def with_lag_restriction(self, k: int) -> "LinearARTeacher":
        """Return a shallow-copy teacher restricted to k of the `window` lags."""
        if k <= 0 or k > self._window:
            raise ValueError(f"k={k} must be in [1, window={self._window}]")

        if self.reverse_constants:
            params = self._params[:k]
            span_lengths = self._span_lengths[:k]
        else:
            params = self._params[-k:]
            span_lengths = self._span_lengths[-k:]

        return LinearARTeacher(
            params=params,
            span_lengths=span_lengths,
            stride=self._stride,
            span_position_weights=self._span_position_weights,
            rank=self.rank,
            scale=self.scale,
            multiplicative_constant=self.multiplicative_constant,
            reverse_constants=self.reverse_constants,
            shared_matrix_across_lags=self.shared_matrix_across_lags,
            orthogonal_matrices=self.orthogonal_matrices,
        )

    # --- Constructor helpers ---
    @classmethod
    def from_parameters(
        cls,
        dim: int,
        span_lengths: List[int],
        rank: int = 1,
        window: int = 1,
        scale: float = 1.0,
        multiplicative_constant: float = 1.0,
        reverse_constants: bool = True,
        shared_matrix_across_lags: bool = False,
        orthogonal_matrices: bool = False,
        stride: Optional[int] = None,
        span_position_weights: Optional[List[float]] = None,
    ) -> "LinearARTeacher":
        assert dim > 0, f"Dimension {dim} must be positive"
        assert rank > 0, f"Rank {rank} must be positive"
        assert window > 0, f"Window {window} must be positive"
        assert scale > 0, f"Scale {scale} must be positive"
        assert rank <= dim, f"Rank {rank} must be less than or equal to dim {dim}"

        if stride is not None:
            assert stride > 0, f"Stride {stride} must be positive"
            min_span = min(span_lengths)
            if stride > min_span:
                warnings.warn(
                    f"Stride ({stride}) > min span_length ({min_span}) creates gaps between intervals"
                )

        if span_position_weights is not None:
            if len(set(span_lengths)) != 1:
                raise ValueError(
                    f"All span_lengths must be equal when using span_position_weights. "
                    f"Got: {span_lengths}"
                )
            span_len = span_lengths[0]
            if len(span_position_weights) != span_len:
                raise ValueError(
                    f"span_position_weights length ({len(span_position_weights)}) "
                    f"must match span_length ({span_len})"
                )
            weight_sum = sum(span_position_weights)
            if weight_sum <= 0:
                raise ValueError(
                    f"span_position_weights must sum to a positive value. Got sum: {weight_sum}"
                )
            span_position_weights = [w / weight_sum for w in span_position_weights]

        if shared_matrix_across_lags:
            base_matrix = random_unit_norm_matrix(dim, rank)
            matrices = [base_matrix] * window
        elif orthogonal_matrices:
            matrices = random_orthogonal_matrices(window, dim, rank)
        else:
            matrices = [random_unit_norm_matrix(dim, rank) for _ in range(window)]

        constants = [
            multiplicative_constant**i
            for i in (reversed(range(window)) if reverse_constants else range(window))
        ]
        A = torch.stack(
            [matrix * const for matrix, const in zip(matrices, constants)], dim=0
        )
        A *= scale

        return cls(
            params=A,
            span_lengths=span_lengths,
            stride=stride,
            span_position_weights=span_position_weights,
            rank=rank,
            scale=scale,
            multiplicative_constant=multiplicative_constant,
            reverse_constants=reverse_constants,
            shared_matrix_across_lags=shared_matrix_across_lags,
            orthogonal_matrices=orthogonal_matrices,
        )
