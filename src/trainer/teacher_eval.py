from typing import Dict, List, Tuple

import torch

from src.loss import KLDivergenceLoss
from src.teachers import ARTeacher


class TeacherEvaluator:
    """Runs the teacher (optionally lag-restricted) and updates its metrics.

    Owns the precomputed `teacher.with_lag_restriction(k)` cache so train/val
    loops don't rebuild them each step. Also handles the "teacher-generated
    data has leading prefix tokens" alignment — a lag-restricted teacher
    sees a shorter context, so we trim data from the front so both produce
    the same number of output positions.
    """

    def __init__(self, teacher: torch.nn.Module, device: torch.device) -> None:
        self.teacher = teacher
        self.is_ar = isinstance(teacher, ARTeacher)
        self._teacher_by_k: Dict[int, ARTeacher] = {}
        self.prefix_ks: List[int] = []
        if self.is_ar:
            # k == window is a no-op (same as self.teacher); skip to avoid
            # a redundant params clone.
            for k in range(1, teacher.window):
                self._teacher_by_k[k] = teacher.with_lag_restriction(k).to(device)
            self.prefix_ks = list(range(1, teacher.window + 1))

    def metric_keys(self) -> List[str]:
        if not self.is_ar:
            return []
        return [
            f"{context}/kl/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]

    def context_names(self) -> List[str]:
        if not self.is_ar:
            return []
        return ["teacher"] + [f"teacher_k{k}" for k in self.prefix_ks]

    def loss_metric_keys(self) -> List[str]:
        return [
            f"{context}/loss/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]

    def acc_metric_keys(self) -> List[str]:
        return [
            f"{context}/acc/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]

    @staticmethod
    def context_name(prefix: int) -> str:
        return "teacher" if prefix < 0 else f"teacher_k{prefix}"

    def _resolve(self, prefix: int):
        if prefix > 0 and prefix in self._teacher_by_k:
            return self._teacher_by_k[prefix]
        return self.teacher

    def _align_data(self, data: torch.Tensor, model) -> torch.Tensor:
        if self.teacher.context_length != model.context_length:
            return data[
                :, self.teacher.context_length - model.context_length :, :
            ]
        return data

    def run(
        self,
        data: torch.Tensor,
        prefix: int = -1,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the (optionally lag-restricted) teacher on `data`.

        Returns `(out, log_probs, targets)`. `log_probs` is always the raw
        teacher log-probs; `out` is `exp(log_probs)` (probs) when
        `normalize=True`, else `log_probs`.
        """
        model = self._resolve(prefix)
        data = self._align_data(data, model)
        log_probs, targets = model.unroll(data, return_targets=True)
        out = log_probs.exp() if normalize else log_probs
        return out, log_probs, targets

    def update_kl_metrics(
        self,
        student_out: torch.Tensor,
        data: torch.Tensor,
        split: str,
        metrics: Dict[str, "LossMetric"],
    ) -> None:
        """KL(teacher || student) at full context and each lag restriction."""
        if not self.is_ar:
            return
        kl = KLDivergenceLoss(reduction="mean")
        probs, _, _ = self.run(data, prefix=-1, normalize=True)
        metrics[f"teacher/kl/{split}"].update(
            kl(student_out, probs).item(), data.size(0)
        )
        for k in self.prefix_ks:
            probs_k, _, _ = self.run(data, prefix=k, normalize=True)
            metrics[f"teacher_k{k}/kl/{split}"].update(
                kl(student_out, probs_k).item(), data.size(0)
            )
