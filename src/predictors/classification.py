from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.predictors.base import Predictor
from src.teachers import HierarchicalTeacher


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
