import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import tqdm

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
from src.trainer.config import LoggingConfig, NgramConfig, SchedulerConfig
from src.trainer.ngram_eval import NgramEvaluator
from src.trainer.registry import MetricRegistry
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
        device: torch.device,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: SchedulerConfig,
        ngram_cfg: NgramConfig,
        logging_cfg: LoggingConfig,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.steps = steps
        self.current_step = 0

        self.device = device
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler_cfg = scheduler_cfg
        self.ngram_cfg = ngram_cfg
        self.logging_cfg = logging_cfg
        self.max_grad_norm = max_grad_norm
        self.logger = logging.getLogger()

        # Populated in _init_loop after models are moved to device.
        self.teacher_eval: Optional[TeacherEvaluator] = None
        self.ngram_evals: Dict[str, NgramEvaluator] = {}
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
            if self.current_step < self.ngram_cfg.steps:
                self._train_ngram()
            else:
                self._train_loop()
        self.logger.info(
            f"Finished training! Total time: {(time.time() - start) / 3600:.2f}h"
        )

    # --- LR scheduler ---
    def _call_lr_sched(self, metric: torch.Tensor) -> None:
        if self.scheduler_cfg.pass_metric:
            self.scheduler_cfg.scheduler.step(metric)
        else:
            self.scheduler_cfg.scheduler.step()

    def _update_lr_sched(self, metric: torch.Tensor, epoch_end: bool) -> None:
        if self.scheduler_cfg.update_on_iter and not epoch_end:
            self._call_lr_sched(metric)
        if not self.scheduler_cfg.update_on_iter and epoch_end:
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

        self.metrics[f"student/{split}_loss"].update(loss.item(), data.size(0))
        self.metrics[f"student/{split}_acc"].update(out, target)

        if isinstance(self.teacher, ARTeacher):
            self.teacher_eval.update_kl_metrics(out, data, split, self.metrics)
            for ne in self.ngram_evals.values():
                ne.update_kl(out, data, split, self.metrics)
            true_loss = self.teacher_eval.true_loss(out, data, self.loss_fn)
            self.metrics[f"student/{split}_true_loss"].update(
                true_loss.item(), data.size(0)
            )

        return out, target, loss, attn_weights

    # --- Init + dry ---
    def _init_loop(self) -> None:
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        for m in self.ngram_cfg.models.values():
            m.to(self.device)

        self.teacher_eval = TeacherEvaluator(self.teacher, self.device)
        self.ngram_evals = {
            name: NgramEvaluator(
                name=name,
                model=model,
                optimizer=self.ngram_cfg.optimizers[name],
                teacher=self.teacher,
                teacher_evaluator=self.teacher_eval,
            )
            for name, model in self.ngram_cfg.models.items()
        }
        self.attention_logger = AttentionLogger(
            writer=self.logging_cfg.writer,
            teacher=self.teacher,
            student=self.student,
            frequency=self.logging_cfg.attention_frequency,
        )

        self.history: Dict[str, List[float]] = {}
        self.metrics = MetricRegistry()
        for key, metric in (
            ("student/train_loss", LossMetric()),
            ("student/train_true_loss", LossMetric()),
            ("student/train_acc", AccuracyMetric(k=1)),
            ("student/val_loss", LossMetric()),
            ("student/val_true_loss", LossMetric()),
            ("student/val_acc", AccuracyMetric(k=1)),
            ("student/val_best", MinMetric()),
            ("student/grad_norm", LossMetric()),
        ):
            self.history[key] = []
            self.metrics.register(key, metric)

        if isinstance(self.teacher, ARTeacher):
            self._register_metrics(self.teacher_eval.metric_keys(), LossMetric)
            self._register_metrics(
                ["teacher/train_loss", "teacher/val_loss"], ConstantLossMetric
            )
            self._register_metrics(
                ["teacher/train_acc", "teacher/val_acc"], ConstantAccuracyMetric, k=1
            )
            for ne in self.ngram_evals.values():
                self._register_metrics(ne.kl_metric_keys(), LossMetric)
                self._register_metrics(ne.loss_metric_keys(), LossMetric)
                self._register_metrics(ne.acc_metric_keys(), AccuracyMetric, k=1)

        self.lr_metric = (
            self.metrics["student/train_loss"]
            if self.scheduler_cfg.update_on_iter
            else self.metrics["student/val_loss"]
        )

    def _register_metrics(
        self, metrics: List[str], metric_class, **constructor_kwargs
    ) -> None:
        for key in metrics:
            self.history[key] = []
            self.metrics.register(key, metric_class(**constructor_kwargs))

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
                self.metrics[f"teacher/{split}_loss"].update(
                    self.loss_fn(log_probs, target).item(), target.shape[0]
                )
                self.metrics[f"teacher/{split}_acc"].update(log_probs, target)
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
        for ne in self.ngram_evals.values():
            ne.model.eval()
        self.teacher.eval()
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)
                for ne in self.ngram_evals.values():
                    logits, target, loss = ne.eval_step(
                        data, self.loss_fn, self.ngram_cfg.use_teacher_target
                    )
                    self.metrics[f"ngram_{ne.name}/val_loss"].update(
                        loss.item(), data.size(0)
                    )
                    self.metrics[f"ngram_{ne.name}/val_acc"].update(logits, target)

    # --- End-of-step book-keeping ---
    def _end_step(
        self, step: int, step_time: Optional[float], ngram: bool = False
    ) -> None:
        self.logger.info(self._step_str(step, step_time, ngram))

        if self.val_loader is not None:
            loss_avg = self.metrics["student/val_loss"].compute()
            if loss_avg < self.metrics["student/val_best"].compute():
                self.metrics["student/val_best"].update(loss_avg)

        if ngram:
            ngram_metrics = {
                key: self.metrics[key]
                for ne in self.ngram_evals.values()
                for key in ne.loss_metric_keys() + ne.acc_metric_keys()
            }
        else:
            ngram_metrics = None

        if self.logging_cfg.writer is not None:
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
            for ne in self.ngram_evals.values():
                s += (
                    f"| {ne.name} Train loss: {self.metrics[f'ngram_{ne.name}/train_loss'].compute():.3f} "
                    f"| {ne.name} Train acc: {self.metrics[f'ngram_{ne.name}/train_acc'].compute():.3f} "
                )
        else:
            s += f"| Train loss: {self.metrics['student/train_loss'].compute():.3f} "
            if isinstance(self.loss_fn, CrossentropyLoss):
                s += f"| Train acc: {self.metrics['student/train_acc'].compute():.3f} "
                if isinstance(self.teacher, ARTeacher):
                    s += f"| Train teacher acc: {self.metrics['teacher/train_acc'].compute():.3f} "
        return s

    def _write_to_wandb(self, step: int, ngram_metrics=None) -> None:
        metrics = ngram_metrics or self.metrics
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}
        if self.scheduler_cfg.scheduler is None:
            raise ValueError("Scheduler is None, cannot log learning rate.")
        log_metrics["system/learning_rate"] = (
            self.scheduler_cfg.scheduler.get_last_lr()[0]
        )
        self.logging_cfg.writer.log(log_metrics, step=step)

    def _write_to_json(self, path: str, epoch: int) -> None:
        for key, metric in self.metrics.items():
            self.history[key].append(metric.compute())
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)
