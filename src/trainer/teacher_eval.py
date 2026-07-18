from typing import Dict, List, Tuple

import torch

from src.loss import CrossentropyLoss, KLDivergenceLoss
from src.teachers import ARTeacher


class TeacherEvaluator:
    """Runs the teacher (optionally lag-restricted) and updates KL metrics.

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
        keys = ["kl/teacher_train", "kl/teacher_val"]
        for k in self.prefix_ks:
            for split in ("train", "val"):
                keys.append(f"kl/teacher_k{k}_{split}")
        return keys

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
        """KL(student || teacher) at full context and each lag-restriction k."""
        if not self.is_ar:
            return
        kl = KLDivergenceLoss(reduction="mean")
        probs, _, _ = self.run(data, prefix=-1, normalize=True)
        metrics[f"kl/teacher_{split}"].update(
            kl(student_out, probs).item(), data.size(0)
        )
        for k in self.prefix_ks:
            probs_k, _, _ = self.run(data, prefix=k, normalize=True)
            metrics[f"kl/teacher_k{k}_{split}"].update(
                kl(student_out, probs_k).item(), data.size(0)
            )

    def true_loss(
        self,
        student_out: torch.Tensor,
        data: torch.Tensor,
        loss_fn: torch.nn.Module,
    ) -> torch.Tensor:
        """Loss of student vs teacher target.

        Normalizes to probs when the training loss is `CrossentropyLoss`
        (which wants soft-target probabilities); otherwise passes log-probs
        through directly (KL / MSE apply their own transforms).
        """
        normalize = isinstance(loss_fn, CrossentropyLoss)
        with torch.no_grad():
            out, _, _ = self.run(data, prefix=-1, normalize=normalize)
        return loss_fn(student_out, out)
