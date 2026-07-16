from typing import Dict, List, Optional, Tuple

import torch

from src.loss import KLDivergenceLoss
from src.trainer.teacher_eval import TeacherEvaluator


class NgramEvaluator:
    """Ngram model training + KL-vs-ngram metrics.

    Encapsulates the ngram data-slicing logic (leading-context trim based on
    the ngram's shorter window vs the teacher's full context) that was
    previously duplicated four times across the trainer.
    """

    def __init__(
        self,
        ngram_models: Dict[str, torch.nn.Module],
        optim_ngram: Dict[str, torch.optim.Optimizer],
        teacher: torch.nn.Module,
        teacher_evaluator: TeacherEvaluator,
    ) -> None:
        self.ngram_models = ngram_models
        self.optim_ngram = optim_ngram
        self.teacher = teacher
        self.teacher_evaluator = teacher_evaluator

    def kl_metric_keys(self) -> List[str]:
        return [
            f"kl_div_{name}_learned_{split}"
            for name in self.ngram_models
            for split in ("train", "val")
        ]

    def _slice_data(
        self, data: torch.Tensor, model: torch.nn.Module
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Drop the leading context an ngram model doesn't consume."""
        teacher_context = getattr(
            self.teacher, "context_length", sum(self.teacher.span_lengths)
        )
        stride: Optional[int] = getattr(self.teacher, "stride", None)
        if stride is not None:
            ngram_context = (
                (model.ngram - 1) * stride + self.teacher.span_lengths[model.ngram - 1]
            )
        else:
            ngram_context = sum(self.teacher.span_lengths[: model.ngram])
        return data[:, teacher_context - ngram_context :], stride

    def _forward(
        self, model: torch.nn.Module, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ngram_data, stride = self._slice_data(data, model)
        logits, probs, targets = model(
            ngram_data,
            span_lengths=self.teacher.span_lengths,
            unroll_sequences=True,
            stride=stride,
        )
        return logits, probs, targets

    def update_kl_metrics(
        self,
        student_out: torch.Tensor,
        data: torch.Tensor,
        split: str,
        metrics: Dict[str, "LossMetric"],
    ) -> None:
        """KL(student || each ngram model)."""
        kl = KLDivergenceLoss(reduction="mean")
        for name, model in self.ngram_models.items():
            _, probs, _ = self._forward(model, data)
            metrics[f"kl_div_{name}_learned_{split}"].update(
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
        name: str,
        data: torch.Tensor,
        loss_fn: torch.nn.Module,
        use_teacher_target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.ngram_models[name]
        opt = self.optim_ngram[name]
        model.zero_grad()
        opt.zero_grad()
        logits, _, ngram_target = self._forward(model, data)
        target = self._resolve_target(ngram_target, data, use_teacher_target)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        return logits, target, loss

    def eval_step(
        self,
        name: str,
        data: torch.Tensor,
        loss_fn: torch.nn.Module,
        use_teacher_target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.ngram_models[name]
        logits, _, ngram_target = self._forward(model, data)
        target = self._resolve_target(ngram_target, data, use_teacher_target)
        loss = loss_fn(logits, target)
        return logits, target, loss
