import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
import wandb

from src.loss import CrossentropyLoss
from src.metrics import (
    AccuracyMetric,
    ConstantAccuracyMetric,
    ConstantLossMetric,
    LossMetric,
    MinMetric,
    RelativeMetric,
)
from src.teachers import ARTeacher
from src.trainer.attention_logger import AttentionLogger
from src.trainer.ngram_eval import NgramEvaluator
from src.trainer.teacher_eval import TeacherEvaluator


class Trainer(ABC):
    """Abstract trainer with shared step / eval / logging plumbing.

    Concrete subclasses only need to define how the student is optimized —
    typically by overriding `_train_loop` (per-step SGD, custom schedules, etc.)
    and `_train_ngram` (auxiliary-model training phase).
    """

    def __init__(
        self,
        steps: int,
        ngram_steps: int,
        device: torch.device,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        ngram_models: Dict[str, torch.nn.Module],
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optim_ngram: Dict[str, torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        pass_sched_metric: bool = False,
        update_sched_on_iter: bool = False,
        max_grad_norm: Optional[float] = None,
        max_student_norm: Optional[float] = None,
        writer: Optional[wandb.run] = None,
        log_attention_frequency: int = 100,
    ) -> None:
        self.steps = steps
        self.ngram_steps = ngram_steps
        self.current_step = 0

        self.device = device
        self.teacher = teacher
        self.student = student
        self.ngram_models = ngram_models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optim_ngram = optim_ngram
        self.scheduler = scheduler
        self.pass_sched_metric = pass_sched_metric
        self.update_sched_on_iter = update_sched_on_iter
        self.max_grad_norm = max_grad_norm
        self.max_student_norm = max_student_norm
        self.writer = writer
        self.log_attention_frequency = log_attention_frequency
        self.logger = logging.getLogger()

        # Populated in _init_loop after models are moved to device.
        self.teacher_eval: Optional[TeacherEvaluator] = None
        self.ngram_eval: Optional[NgramEvaluator] = None
        self.attention_logger: Optional[AttentionLogger] = None

    # --- Entry point ---
    def train(self) -> None:
        self._init_loop()
        if isinstance(self.teacher, ARTeacher):
            self._dry_loop()  # TODO: adapt for TransformerDecoder

        self.logger.info("Beginning training")
        start = time.time()
        while self.current_step < self.steps:
            # Two phases: first ngram models, then the student.
            if self.current_step < self.ngram_steps:
                self._train_ngram()
            else:
                self._train_loop()
        self.logger.info(
            f"Finished training! Total time: {(time.time() - start) / 3600:.2f}h"
        )

    # --- LR scheduler ---
    def _call_lr_sched(self, metric: torch.Tensor) -> None:
        if self.pass_sched_metric:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _update_lr_sched(self, metric: torch.Tensor, epoch_end: bool) -> None:
        if self.update_sched_on_iter and not epoch_end:
            self._call_lr_sched(metric)
        if not self.update_sched_on_iter and epoch_end:
            self._call_lr_sched(metric)

    # --- Student forward + per-batch metrics ---
    def _run_student(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive student forward, with the leading prefix trimmed off.

        Returns `(out, target, attn_weights)`. The student is not scored on the
        first `prefix_length - 1` output positions (those live inside the
        burn-in region).
        """
        out, attn_weights = self.student(data[:, :-1, :])
        target = data[:, 1:, :]
        ctx = self.train_loader.dataset.prefix_length
        return (
            out[:, ctx - 1 :, :],
            target[:, ctx - 1 :, :],
            attn_weights[:, :, :, ctx - 1 :, ctx - 1 :],
        )

    def _forward_and_metrics(
        self, data: torch.Tensor, split: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Student forward + loss/acc + teacher/ngram KL + teacher-target loss.

        Returns `(out, target, loss, attn_weights)`. Shared by train and val —
        the training loop only adds `backward` + `optimizer.step` around this.
        """
        out, target, attn_weights = self._run_student(data)
        loss = self.loss_fn(out, target)

        self.metrics[f"{split}_loss"].update(loss.item(), data.size(0))
        self.metrics[f"{split}_acc"].update(out, target)

        if isinstance(self.teacher, ARTeacher):
            self.teacher_eval.update_kl_metrics(out, data, split, self.metrics)
            self.ngram_eval.update_kl_metrics(out, data, split, self.metrics)
            true_loss = self.teacher_eval.true_loss(out, data, self.loss_fn)
            self.metrics[f"{split}_true_loss"].update(true_loss.item(), data.size(0))

        return out, target, loss, attn_weights

    # --- Init + dry ---
    def _init_loop(self) -> None:
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        for m in self.ngram_models.values():
            m.to(self.device)

        self.teacher_eval = TeacherEvaluator(self.teacher, self.device)
        self.ngram_eval = NgramEvaluator(
            self.ngram_models,
            self.optim_ngram,
            self.teacher,
            self.teacher_eval,
        )
        self.attention_logger = AttentionLogger(
            writer=self.writer,
            teacher=self.teacher,
            student=self.student,
            frequency=self.log_attention_frequency,
        )

        self.history = {
            "train_loss": [],
            "train_true_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_true_loss": [],
            "val_acc": [],
            "val_best": [],
            "grad_norm": [],
        }
        self.metrics = {
            "train_loss": LossMetric(),
            "train_true_loss": LossMetric(),
            "train_acc": AccuracyMetric(k=1),
            "val_loss": LossMetric(),
            "val_true_loss": LossMetric(),
            "val_acc": AccuracyMetric(k=1),
            "val_best": MinMetric(),
            "grad_norm": LossMetric(),
        }

        if isinstance(self.teacher, ARTeacher):
            self._register_metrics(self.teacher_eval.metric_keys(), LossMetric)
            self._register_metrics(self.ngram_eval.kl_metric_keys(), LossMetric)
            self._register_metrics(
                ["teacher_train_loss", "teacher_val_loss"], ConstantLossMetric
            )
            self._register_metrics(
                ["teacher_train_acc", "teacher_val_acc"], ConstantAccuracyMetric, k=1
            )
            for name in self.ngram_models:
                self._register_metrics([f"{name}_train_loss"], LossMetric)
                self._register_metrics([f"{name}_train_acc"], AccuracyMetric, k=1)
                self._register_metrics([f"{name}_val_loss"], LossMetric)
                self._register_metrics([f"{name}_val_acc"], AccuracyMetric, k=1)

        for key, metric in self.metrics.items():
            setattr(self, key, metric)

        self.lr_metric = self.train_loss if self.update_sched_on_iter else self.val_loss

    def _register_metrics(
        self, metrics: List[str], metric_class, **constructor_kwargs
    ) -> None:
        self.history.update({key: [] for key in metrics})
        self.metrics.update(
            {key: metric_class(**constructor_kwargs) for key in metrics}
        )

    def _dry_loop(self) -> None:
        """One-shot pass over train + val to populate the constant teacher metrics."""
        self.teacher.eval()
        normalize = isinstance(self.loss_fn, CrossentropyLoss)

        for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
            pbar = tqdm.tqdm(total=len(loader), leave=False)
            pbar.set_description(f"Dry run | {split.capitalize()}")
            for data in loader:
                data = data.to(self.device)
                _, log_probs, target = self.teacher_eval.run(
                    data, prefix=-1, normalize=normalize
                )
                # CE and KL both apply log_softmax internally, which is a
                # near-identity on log-probs (see src/loss.py:55).
                self.metrics[f"teacher_{split}_loss"].update(
                    self.loss_fn(log_probs, target).item(), target.shape[0]
                )
                self.metrics[f"teacher_{split}_acc"].update(log_probs, target)
                pbar.update()
            pbar.close()

    # --- Loops that subclasses may override ---
    @abstractmethod
    def _train_loop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_ngram(self) -> None:
        raise NotImplementedError

    def _val_loop(self, step: int) -> None:
        self.student.eval()
        attn_batches: List[torch.Tensor] = []
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)
                _, _, _, attn_weights = self._forward_and_metrics(data, split="val")
                attn_batches.append(attn_weights)
        self.attention_logger.log(step, "val", attn_batches)

    def _val_ngram(self) -> None:
        for model in self.ngram_models.values():
            model.eval()
        self.teacher.eval()
        use_teacher_target = self.student.teacher_target
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)
                for name in self.ngram_models:
                    logits, target, loss = self.ngram_eval.eval_step(
                        name, data, self.loss_fn, use_teacher_target
                    )
                    self.metrics[f"{name}_val_loss"].update(loss.item(), data.size(0))
                    self.metrics[f"{name}_val_acc"].update(logits, target)

    # --- End-of-step book-keeping ---
    def _end_step(
        self, step: int, step_time: Optional[float], ngram: bool = False
    ) -> None:
        self.logger.info(self._step_str(step, step_time, ngram))

        if self.val_loader is not None:
            loss_avg = self.val_loss.compute()
            if loss_avg < self.val_best.compute():
                self.val_best.update(loss_avg)

        if ngram:
            ngram_metrics = {
                f"{name}_{split}_{suffix}": self.metrics[f"{name}_{split}_{suffix}"]
                for name in self.ngram_models
                for split in ("train", "val")
                for suffix in ("loss", "acc")
            }
        else:
            ngram_metrics = None

        if self.writer is not None:
            self._write_to_wandb(step, ngram_metrics)

        metrics = ngram_metrics or self.metrics
        for metric in metrics.values():
            if isinstance(
                metric,
                (RelativeMetric, MinMetric, ConstantLossMetric, ConstantAccuracyMetric),
            ):
                continue
            metric.reset()

    def _step_str(
        self, step: int, step_time: Optional[float], ngram: bool = False
    ) -> str:
        s = f"Step {step} "
        if step_time:
            s += f"| Step time: {step_time:.1f}s"
        if ngram:
            for name in self.ngram_models:
                s += (
                    f"| {name} Train loss: {self.metrics[f'{name}_train_loss'].compute():.3f} "
                    f"| {name} Train acc: {self.metrics[f'{name}_train_acc'].compute():.3f} "
                )
        else:
            s += f"| Train loss: {self.train_loss.compute():.3f} "
            if isinstance(self.loss_fn, CrossentropyLoss):
                s += f"| Train acc: {self.train_acc.compute():.3f} "
                if isinstance(self.teacher, ARTeacher):
                    s += f"| Train teacher acc: {self.teacher_train_acc.compute():.3f} "
        return s

    def _write_to_wandb(self, step: int, ngram_metrics=None) -> None:
        metrics = ngram_metrics or self.metrics
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}
        if self.scheduler is None:
            raise ValueError("Scheduler is None, cannot log learning rate.")
        log_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        self.writer.log(log_metrics, step=step)

    def _write_to_json(self, path: str, epoch: int) -> None:
        for key, metric in self.metrics.items():
            self.history[key].append(metric.compute())
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)
