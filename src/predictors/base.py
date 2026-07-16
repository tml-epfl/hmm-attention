from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class Predictor(nn.Module, ABC):
    """Turns a teacher into a next-token sampler.

    The dataset never sees the teacher's shape/type — it just calls
    `predictor.sample_next(prefix)` and stores the returned token in the sequence.

    Concrete predictors decide (a) how to invoke the teacher on a growing prefix,
    (b) how to interpret its output (logits vs mean vs log-probs), and (c) how
    to draw a concrete sample.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output dimension of one token."""

    @abstractmethod
    def sample_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Sample the next token given a prefix of shape (T, dim). Returns (dim,).

        Discrete predictors return a one-hot; continuous predictors return a
        real-valued vector.
        """

    def distribution(self, prefix: torch.Tensor) -> torch.distributions.Distribution:
        """Underlying distribution over the next token. Optional; used by the
        trainer for KL / expected-loss metrics. Default: raise."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a distribution."
        )

    def random_burn_in(self, length: int) -> Optional[torch.Tensor]:
        """Optional hook: predictor-specific burn-in prefix (e.g. valid chunk-composed
        tokens for hierarchical predictors). Return None to let the dataset fall
        back to uniform random one-hots."""
        return None
