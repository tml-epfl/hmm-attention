from typing import Dict, List, Optional, Tuple

import torch

from src.loss import KLDivergenceLoss
from src.trainer.teacher_eval import TeacherEvaluator


class NgramEvaluator:
    """Wraps a single ngram model: data slicing + forward + train/eval step.

    The trainer holds one instance per ngram (a `Dict[str, NgramEvaluator]`)
    and iterates outside — keeping "which model" and "how to evaluate a model"
    as separate concerns.
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher: torch.nn.Module,
        teacher_evaluator: TeacherEvaluator,
    ) -> None:
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.teacher = teacher
        self.teacher_evaluator = teacher_evaluator

    def kl_metric_keys(self) -> List[str]:
        return [f"ngram_{self.name}/kl/{split}" for split in ("train", "val")]

    def loss_metric_keys(self) -> List[str]:
        return [f"ngram_{self.name}/loss/{split}" for split in ("train", "val")]

    def acc_metric_keys(self) -> List[str]:
        return [f"ngram_{self.name}/acc/{split}" for split in ("train", "val")]

    def _slice_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, Optional[int]]:
        """Drop the leading context the ngram model doesn't consume."""
        # Teacher's unroll drops `burn_in` leading positions (== context_length
        # for bounded teachers); align the ngram slice to the same offset.
        teacher_context = getattr(
            self.teacher, "burn_in", sum(self.teacher.span_lengths)
        )
        stride: Optional[int] = getattr(self.teacher, "stride", None)
        if stride is not None:
            ngram_context = (
                (self.model.ngram - 1) * stride
                + self.teacher.span_lengths[self.model.ngram - 1]
            )
        else:
            ngram_context = sum(self.teacher.span_lengths[: self.model.ngram])
        return data[:, teacher_context - ngram_context :], stride

    def _forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ngram_data, stride = self._slice_data(data)
        logits, probs, targets = self.model(
            ngram_data,
            span_lengths=self.teacher.span_lengths,
            unroll_sequences=True,
            stride=stride,
        )
        return logits, probs, targets

    def update_kl(
        self,
        student_out: torch.Tensor,
        data: torch.Tensor,
        split: str,
        metrics: Dict[str, "LossMetric"],
    ) -> None:
        """KL(this ngram model || student)."""
        kl = KLDivergenceLoss(reduction="mean")
        _, probs, _ = self._forward(data)
        metrics[f"ngram_{self.name}/kl/{split}"].update(
            kl(student_out, probs).item(), data.size(0)
        )

    def _resolve_target(
        self,
        ngram_target: torch.Tensor,
        data: torch.Tensor,
        use_teacher_target: bool,
    ) -> torch.Tensor:
        if not use_teacher_target:
            return ngram_target
        with torch.no_grad():
            out, _, _ = self.teacher_evaluator.run(data, prefix=-1, normalize=True)
        return out

    def train_step(
        self,
        data: torch.Tensor,
        loss_fn: torch.nn.Module,
        use_teacher_target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.zero_grad()
        self.optimizer.zero_grad()
        logits, _, ngram_target = self._forward(data)
        target = self._resolve_target(ngram_target, data, use_teacher_target)
        loss = loss_fn(logits, target)
        loss.backward()
        self.optimizer.step()
        return logits, target, loss

    def eval_step(
        self,
        data: torch.Tensor,
        loss_fn: torch.nn.Module,
        use_teacher_target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, ngram_target = self._forward(data)
        target = self._resolve_target(ngram_target, data, use_teacher_target)
        loss = loss_fn(logits, target)
        return logits, target, loss
