from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.utils import split_into_windows


class ARTeacher(nn.Module, ABC):
    """Abstract autoregressive teacher.

    Subclasses only need to implement `dim`, `context_length`, and
    `next_token_logits(context)`. Sequence-level operations (per-position
    unrolling and autoregressive-generation prefix slicing) are shared.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vocab dimension."""

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Number of tokens the teacher looks at to predict the next one."""

    @abstractmethod
    def next_token_logits(self, context: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits (or log-probs) from a fixed-size context.

        Args:
            context: shape (B, context_length, dim). Callers must hand exactly
                context_length tokens — see `predict_next` for automatic
                slicing from a longer prefix.

        Returns:
            (B, dim) tensor. The convention (logits vs log-probs) is documented
            per subclass; downstream code that needs a normalized distribution
            should apply softmax when appropriate.
        """

    def predict_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Autoregressive single-step prediction.

        Auto-slices to the trailing `context_length` tokens so callers can hand
        any prefix `>= context_length` in length.
        """
        if prefix.shape[-2] < self.context_length:
            raise ValueError(
                f"prefix length {prefix.shape[-2]} < context_length {self.context_length}"
            )
        if prefix.shape[-2] > self.context_length:
            prefix = prefix[..., -self.context_length :, :]
        return self.next_token_logits(prefix)

    def unroll(
        self,
        sequence: torch.Tensor,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-position predictions over a full sequence.

        For each output position `j`, predicts `sequence[:, context_length + j, :]`
        from context `sequence[:, j : j + context_length, :]`.

        Args:
            sequence: shape (B, L, dim). Must satisfy L >= context_length + 1.
            return_targets: if True, also return the ground-truth next tokens
                shape (B, L - context_length, dim).

        Returns:
            Either logits (B, L - context_length, dim) or a (logits, targets) tuple.
        """
        B, L, D = sequence.shape
        if L <= self.context_length:
            raise ValueError(
                f"sequence length {L} must exceed context_length {self.context_length}"
            )
        contexts, targets = split_into_windows(
            seq=sequence, window=self.context_length, pad=0
        )
        # contexts: (B * (L - ctx), ctx, D); targets: (B * (L - ctx), D)
        logits_flat = self.next_token_logits(contexts)  # (B * (L - ctx), D)
        logits = logits_flat.view(B, L - self.context_length, D)
        if return_targets:
            targets = targets.view(B, L - self.context_length, D)
            return logits, targets
        return logits

    def with_lag_restriction(self, k: int) -> "ARTeacher":
        """Return a shallow-copy teacher with AR context restricted to k lags.

        Only meaningful for teachers with an explicit multi-lag structure
        (e.g. LinearARTeacher). Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support lag restriction"
        )
