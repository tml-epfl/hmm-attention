from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.predictors.base import Predictor
from src.teachers import HierarchicalTeacher, MultiLevelHierarchicalTeacher


class ClassificationPredictor(Predictor):
    """One-hot next-token sampler for teachers that expose `predict_next(prefix)`.

    The teacher must return log-probabilities over `dim` classes. Sharpness is
    controlled by the teacher's weight scale — there is no temperature knob.

    Options:
        argmax: if True, deterministic argmax sampling (equivalent to temperature=0).
    """

    def __init__(
        self,
        teacher,
        argmax: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(teacher, "predict_next"):
            raise TypeError(
                f"{type(self).__name__} requires a teacher exposing `predict_next`; "
                f"got {type(teacher).__name__}"
            )
        self.teacher = teacher
        self.argmax = argmax

    @property
    def dim(self) -> int:
        return self.teacher.dim

    def _teacher_log_probs(self, prefix: torch.Tensor) -> torch.Tensor:
        """Return next-token log-probs from the teacher, shape (dim,)."""
        with torch.no_grad():
            device = next(self.teacher.parameters()).device
            x = prefix.to(device)
            if x.ndim == 2:
                x = x.unsqueeze(0)  # (1, T, dim)
            log_probs = self.teacher.predict_next(x)  # (1, dim)
            return log_probs.squeeze(0).to(prefix.device)

    def _teacher_log_probs_batch(self, prefix: torch.Tensor) -> torch.Tensor:
        """Batched next-token log-probs: (B, T, dim) -> (B, dim), returned on
        `prefix.device`. Runs the teacher once on the whole batch."""
        with torch.no_grad():
            device = next(self.teacher.parameters()).device
            log_probs = self.teacher.predict_next(prefix.to(device))
            return log_probs.to(prefix.device)

    def distribution(self, prefix: torch.Tensor) -> Categorical:
        # Categorical treats `logits` as unnormalized log-probs; log-probs are
        # already normalized log-probs (a special case), so this is correct.
        return Categorical(logits=self._teacher_log_probs(prefix))

    def sample_next(self, prefix: torch.Tensor) -> torch.Tensor:
        log_probs = self._teacher_log_probs(prefix)
        if self.argmax:
            idx = log_probs.argmax(dim=-1)
        else:
            idx = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        return F.one_hot(idx, num_classes=self.dim).to(prefix.dtype)

    def sample_next_batch(self, prefix: torch.Tensor) -> torch.Tensor:
        """Batched sampler: (B, T, dim) -> (B, dim). One teacher forward,
        one multinomial draw per batch element."""
        log_probs = self._teacher_log_probs_batch(prefix)  # (B, dim)
        if self.argmax:
            idx = log_probs.argmax(dim=-1)
        else:
            idx = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        return F.one_hot(idx, num_classes=self.dim).to(prefix.dtype)


class HierarchicalPredictor(ClassificationPredictor):
    """Sampler specialized for `HierarchicalTeacher`.

    Identical to `ClassificationPredictor` except for `random_burn_in`, which
    emits valid chunk-composed prefixes (decoding to real hidden ids) instead
    of uniform random one-hots.
    """

    def __init__(self, teacher, argmax: bool = False) -> None:
        if not isinstance(teacher, HierarchicalTeacher):
            raise TypeError(
                f"HierarchicalPredictor requires a HierarchicalTeacher; "
                f"got {type(teacher).__name__}"
            )
        super().__init__(teacher=teacher, argmax=argmax)

    def random_burn_in(self, length: int) -> Optional[torch.Tensor]:
        """Use the teacher's chunk-composed sampler so burn-in tokens are valid
        chunks (decoding to real hidden ids rather than the argmax=0 fallback)."""
        device = next(self.teacher.parameters()).device
        return self.teacher.sample_surface_prefix(length, device=device).cpu()

    def random_burn_in_batch(
        self, batch_size: int, length: int
    ) -> Optional[torch.Tensor]:
        """Vectorized version of `random_burn_in`: draw `batch_size * n_hidden`
        hidden ids in one shot and look them up in the chunk table."""
        if length % self.teacher.chunk_size != 0:
            raise ValueError(
                f"burn-in length ({length}) must be a multiple of "
                f"chunk_size ({self.teacher.chunk_size})"
            )
        device = next(self.teacher.parameters()).device
        table = self.teacher._chunk_table.to(device)
        n_hidden = length // self.teacher.chunk_size
        hidden_ids = torch.randint(
            0, self.teacher.hidden_dim, (batch_size, n_hidden), device=device
        )
        tuple_ids = torch.randint(
            0, self.teacher.num_tuples, (batch_size, n_hidden), device=device
        )
        chunks = table[hidden_ids, tuple_ids]  # (B, n_hidden, chunk_size, chunk_dim)
        return chunks.reshape(batch_size, length, self.teacher.chunk_dim).cpu()


class MultiLevelHierarchicalPredictor(ClassificationPredictor):
    """Sampler specialized for `MultiLevelHierarchicalTeacher`.

    Like `HierarchicalPredictor`, but burn-in prefixes are sampled top-down
    through the full level stack (`sample_surface_prefix`), so every burn-in
    token is a legitimate slot of some base token's nested expansion.
    """

    def __init__(self, teacher, argmax: bool = False) -> None:
        if not isinstance(teacher, MultiLevelHierarchicalTeacher):
            raise TypeError(
                f"MultiLevelHierarchicalPredictor requires a "
                f"MultiLevelHierarchicalTeacher; got {type(teacher).__name__}"
            )
        super().__init__(teacher=teacher, argmax=argmax)

    def random_burn_in(self, length: int) -> Optional[torch.Tensor]:
        return self.teacher.sample_surface_prefix(length).cpu()

    def random_burn_in_batch(
        self, batch_size: int, length: int
    ) -> Optional[torch.Tensor]:
        return self.teacher.sample_surface_prefix(
            length, batch_size=batch_size
        ).cpu()
