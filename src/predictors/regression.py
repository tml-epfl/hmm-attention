import torch
from torch.distributions import Normal

from src.predictors.base import Predictor


class RegressionPredictor(Predictor):
    """Continuous next-token sampler.

    Treats the teacher's raw output as the mean of an isotropic Gaussian with
    scalar std `noise_std`. Sampling returns a real-valued vector; if
    `noise_std == 0`, the mean is returned deterministically.

    Not exercised by any current experiment — kept as the entry point for future
    regression work sharing the same dataset code path.
    """

    def __init__(self, teacher, noise_std: float = 0.0) -> None:
        super().__init__()
        self.teacher = teacher
        self.noise_std = float(noise_std)

    @property
    def dim(self) -> int:
        return self.teacher.dim

    def _teacher_mean(self, prefix: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            device = next(self.teacher.parameters()).device
            x = prefix.to(device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            mean = self.teacher.predict_next(x)  # (1, dim)
            return mean.squeeze(0).to(prefix.device)

    def distribution(self, prefix: torch.Tensor) -> Normal:
        mean = self._teacher_mean(prefix)
        std = torch.full_like(mean, max(self.noise_std, 1e-8))
        return Normal(loc=mean, scale=std)

    def sample_next(self, prefix: torch.Tensor) -> torch.Tensor:
        mean = self._teacher_mean(prefix)
        if self.noise_std <= 0.0:
            return mean
        noise = torch.randn_like(mean) * self.noise_std
        return mean + noise
