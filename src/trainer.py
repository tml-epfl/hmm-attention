import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import tqdm
import wandb

from src.loss import CrossentropyLoss, KLDivergenceLoss
from src.metrics import (
    AccuracyMetric,
    ConstantAccuracyMetric,
    ConstantLossMetric,
    LossMetric,
    MinMetric,
    RelativeMetric,
)
from src.model import TransformerDecoder
from src.teachers import ARTeacher, HierarchicalTeacher, LinearARTeacher
from src.visualizer import (
    log_attention_alignment,
    log_attention_heatmap,
    log_attention_span_mass,
    log_attention_table,
    log_value_alignment_scalars,
    log_value_matrix_alignment,
)


class Trainer(ABC):
    """Abstract model trainer

    Args:
        steps: number of training steps
        ngram_steps: number of steps for ngram training
        device: device to train the model on
        teacher: ground truth model
        student: model to train
        train_loader: training dataloader
        val_loader: validation dataloader
        loss_fn: loss function
        optimizer: model optimizer
        scheduler: learning rate scheduler
        pass_sched_metric: whether to pass a metric to scheduler
        update_sched_on_iter: whether to call the scheduler every iter
        max_grad_norm: gradient clipping max norm (disabled if None)
        max_student_norm: student clipping max norm (disabled if None)
        writer: writer which logs metrics to wandb
        log_attention_frequency: frequency for logging attention heatmaps
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

    def train(self) -> None:
        """Trains the model"""

        # define metrics and move models to device
        self._init_loop()

        # dry run to check for errors and to initialize metrics
        if isinstance(self.teacher, ARTeacher):
            self._dry_loop()  # TODO: adapt for TransformerDecoder

        # train loop
        self.logger.info("Beginning training")
        start_time = time.time()

        # Step-based training loop
        while self.current_step < self.steps:
            # Split training into two phases:
            # first phase: training the ngram models,
            # second phase: training the student model.
            if self.current_step < self.ngram_steps:
                self._train_ngram()
            else:
                self._train_loop()

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")

    def _run_ar_model(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        normalize: bool = False,
        return_logits: bool = False,
        return_targets: bool = False,
        return_attn_weights: bool = False,
        prefix: int = -1,
    ) -> torch.Tensor:
        assert not normalize or isinstance(model, ARTeacher), (
            "Normalization is only supported for ARTeacher models"
        )

        if isinstance(model, ARTeacher):
            # `prefix > 0` in the old API meant "restrict AR weights to first k
            # lags." The new API exposes this via with_lag_restriction, and we
            # precompute the restricted variants once at init in _teacher_by_k.
            # For prefix == self.teacher.window, restriction is a no-op; fall
            # through to the unrestricted `model` (typically self.teacher).
            if prefix > 0 and prefix in self._teacher_by_k:
                model = self._teacher_by_k[prefix]
            # Align data length: teacher-generated data has context_length_teacher
            # leading prefix tokens; a lag-restricted model sees a shorter context,
            # so trim data from the front so both produce the same # of output positions.
            if self.teacher.context_length != model.context_length:
                data = data[
                    :, self.teacher.context_length - model.context_length :, :
                ]

            out, target = model.unroll(data, return_targets=True)
            if normalize:
                if isinstance(model, HierarchicalTeacher):
                    # Wrapper already applied temperature internally and returns
                    # log(surface_probs). exp() recovers probs; out_logits stays
                    # as log-probs so downstream CE/KL losses (which internally
                    # log_softmax) remain correct — log_softmax(log p) = log p
                    # when p sums to 1.
                    out_logits = out
                    out = out.exp()
                else:
                    out_logits = out / self.train_loader.dataset.temperature
                    if self.train_loader.dataset.softmax:
                        out = F.softmax(out_logits, dim=-1)
                    else:
                        out = out_logits / torch.linalg.norm(
                            out_logits, ord=1, dim=-1, keepdim=True
                        )
        elif isinstance(model, TransformerDecoder):
            out, attn_weights = model(data[:, :-1, :])
            target = data[:, 1:, :]

            # ignore prefix length when computing the loss.
            ctx = self.train_loader.dataset.prefix_length
            out, attn_weights = (
                out[:, ctx - 1 :, :],
                attn_weights[:, :, :, ctx - 1 :, ctx - 1 :],
            )
            target = target[:, ctx - 1 :, :]

        output = [out]
        if return_logits:
            output.append(out_logits if normalize else out)
        if return_targets:
            output.append(target)
        if return_attn_weights:
            output.append(
                attn_weights if isinstance(model, TransformerDecoder) else None
            )

        return tuple(output) if len(output) > 1 else output[0]

    def _register_metrics(
        self, metrics: List[str], metric_class, **constructor_kwargs
    ) -> None:
        self.history.update({key: [] for key in metrics})
        self.metrics.update(
            {key: metric_class(**constructor_kwargs) for key in metrics}
        )

    def _dry_loop(self) -> None:
        """Dry run to check for errors and to initialize metrics."""
        self.teacher.eval()

        # training
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description("Dry run | Training")

        for data in self.train_loader:
            data = data.to(self.device)
            # Full teacher (prefix: -1)
            kwargs = {"prefix": -1} if isinstance(self.teacher, ARTeacher) else {}
            _, out_logits, target = self._run_ar_model(
                self.teacher,
                data,
                return_targets=True,
                return_logits=True,
                normalize=isinstance(self.loss_fn, CrossentropyLoss),
                **kwargs,
            )

            # Update metrics
            self.teacher_train_loss.update(
                self.loss_fn(out_logits, target).item(), target.shape[0]
            )
            self.teacher_train_acc.update(out_logits, target)

            # update progress bar
            pbar.update()

        pbar.close()

        # evaluation
        pbar = tqdm.tqdm(total=len(self.val_loader), leave=False)
        pbar.set_description("Dry run | Evaluation")

        for data in self.val_loader:
            data = data.to(self.device)

            _, out_logits, target = self._run_ar_model(
                self.teacher,
                data,
                return_targets=True,
                return_logits=True,
                normalize=isinstance(self.loss_fn, CrossentropyLoss),
                **kwargs,
            )

            # Update metrics
            self.teacher_val_loss.update(
                self.loss_fn(out_logits, target).item(), target.shape[0]
            )
            self.teacher_val_acc.update(out_logits, target)

            # update progress bar
            pbar.update()

        pbar.close()


    def _init_loop(self) -> None:
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        for m in self.ngram_models.values():
            m.to(self.device)

        # Precompute lag-restricted teacher variants once, so val/train loops
        # don't rebuild them on every step. Key = k (1..window-1); k == window
        # is equivalent to self.teacher, so skip it to avoid a redundant params clone.
        self._teacher_by_k = {}
        if isinstance(self.teacher, ARTeacher):
            for k in range(1, self.teacher.window):
                self._teacher_by_k[k] = self.teacher.with_lag_restriction(k).to(
                    self.device
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
            teacher_loss_metrics = [
                "teacher_train_loss",
                "teacher_val_loss",
            ]
            teacher_acc_metrics = [
                "teacher_train_acc",
                "teacher_val_acc",
            ]
            kl_divergence_metrics = ["kl_div_teacher_train", "kl_div_teacher_val"]
            for name in self.ngram_models:
                for split in ["train", "val"]:
                    kl_divergence_metrics.append(f"kl_div_{name}_learned_{split}")
            self.prefix_ks = range(1, self.teacher.window + 1)
            for k in self.prefix_ks:
                for split in ["train", "val"]:
                    kl_divergence_metrics.append(f"kl_div_prefix_{k}_teacher_{split}")

            self._register_metrics(kl_divergence_metrics, LossMetric)
            self._register_metrics(teacher_loss_metrics, ConstantLossMetric)
            self._register_metrics(teacher_acc_metrics, ConstantAccuracyMetric, k=1)

            for name in self.ngram_models:
                self._register_metrics([f"{name}_train_loss"], LossMetric)
                self._register_metrics([f"{name}_train_acc"], AccuracyMetric, k=1)
                self._register_metrics([f"{name}_val_loss"], LossMetric)
                self._register_metrics([f"{name}_val_acc"], AccuracyMetric, k=1)

        # make all metrics accessible
        for key, metric in self.metrics.items():
            setattr(self, key, metric)

        if self.update_sched_on_iter:
            self.lr_metric =  self.train_loss
        else:
            self.lr_metric = self.val_loss

    @abstractmethod
    def _train_loop(self, epoch: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_ngram(self) -> None:
        raise NotImplementedError

    def _val_loop(self, step: int) -> None:
        # set to eval
        self.student.eval()

        # loop
        val_attn_weights = []
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)

                # forward
                out, target, attn_weights = self._run_ar_model(
                    self.student,
                    data,
                    return_targets=True,
                    return_attn_weights=True,
                )
                loss = self.loss_fn(out, target)
                val_attn_weights.append(attn_weights)

                # update metrics
                self.val_loss.update(loss.item(), data.shape[0])
                self.val_acc.update(out, target)

                # ----------------------------------------------------------------------
                # student forward pass already computed: `out`
                # ----------------------------------------------------------------------
                if isinstance(self.teacher, ARTeacher):
                    kl_divergence = KLDivergenceLoss(reduction="mean")
                    out_teacher = self._run_ar_model(
                        self.teacher, data, normalize=True, prefix=-1
                    )
                    kl_val = kl_divergence(
                        out, out_teacher
                    )  # raw logits vs. prob-targets
                    self.metrics["kl_div_teacher_val"].update(
                        kl_val.item(), data.size(0)
                    )

                    for k in self.prefix_ks:  # 1 … window
                        out_k = self._run_ar_model(
                            self.teacher, data, normalize=True, prefix=k
                        )
                        kl_k = kl_divergence(out, out_k)
                        self.metrics[f"kl_div_prefix_{k}_teacher_val"].update(
                            kl_k.item(), data.size(0)
                        )

                    for name, model in self.ngram_models.items():
                        # drop the extra initial context for n-gram
                        teacher_context = getattr(self.teacher, 'context_length', sum(self.teacher.span_lengths))
                        stride = getattr(self.teacher, 'stride', None)
                        if stride is not None:
                            ngram_context = (model.ngram - 1) * stride + self.teacher.span_lengths[model.ngram - 1]
                        else:
                            ngram_context = sum(self.teacher.span_lengths[: model.ngram])
                        ngram_data = data[
                            :,
                            teacher_context - ngram_context :,
                        ]
                        _, aux_probs, _ = model(
                            ngram_data,
                            span_lengths=self.teacher.span_lengths,
                            unroll_sequences=True,
                            stride=stride,
                        )
                        kl = kl_divergence(out, aux_probs)
                        self.metrics[f"kl_div_{name}_learned_val"].update(
                            kl.item(), data.size(0)
                        )

                    with torch.no_grad():
                        out_true = self._run_ar_model(
                            self.teacher,
                            data,
                            normalize=isinstance(self.loss_fn, CrossentropyLoss),
                        )
                        loss_true = self.loss_fn(out, out_true)
                        self.val_true_loss.update(loss_true.item(), data.shape[0])

        # Log attention with configurable frequency
        if isinstance(self.student, TransformerDecoder):
            # attention weights (layer, batch, head, seq_len, seq_len), combine across batches.
            val_attn_weights = torch.cat(val_attn_weights, dim=1)

            # Extract first layer and average over batch for numpy saving
            attn_np = (
                val_attn_weights[0].detach().cpu().numpy()
            )  # First layer: (batch, heads, seq_len, seq_len)
            attn_avg = attn_np.mean(axis=0)  # (heads, seq_len, seq_len)

            # Only log attention at specified frequency
            if self.writer is not None and step % self.log_attention_frequency == 0:
                # Log structured attention table
                log_attention_table(
                    run=self.writer,
                    attn_weights=val_attn_weights,
                    layer=0,
                    batch_idx=-1,
                    step=step,
                    table_key="val_attention_weights",
                )

                # Create and log heatmaps using the averaged attention
                log_attention_heatmap(
                    run=self.writer,
                    attn_weights=attn_avg,
                    log_key="val_attention_heatmaps",
                    step=step,
                )

                # Alignment: value matrix and attention pattern vs. ground truth
                if isinstance(self.teacher, LinearARTeacher):
                    _stride = getattr(self.teacher, "stride", None)
                    _ctx_len = getattr(
                        self.teacher, "context_length", sum(self.teacher.span_lengths)
                    )
                    log_attention_alignment(
                        run=self.writer,
                        attn_avg=attn_avg,
                        span_lengths=self.teacher.span_lengths,
                        context_length=_ctx_len,
                        step=step,
                        split="val",
                        stride=_stride,
                    )
                    # Time-series scalars for tracking head collaboration:
                    # - attention span mass: how much each head attends to
                    #   each teacher's position group (collaborative phases)
                    # - value alignment: how each head's value matrix aligns
                    #   with each teacher feature (cooperative offset dynamics)
                    log_attention_span_mass(
                        run=self.writer,
                        attn_avg=attn_avg,
                        span_lengths=self.teacher.span_lengths,
                        context_length=_ctx_len,
                        step=step,
                        split="val",
                        stride=_stride,
                    )
                    log_value_matrix_alignment(
                        run=self.writer,
                        teacher_matrices=self.teacher._params,
                        student=self.student,
                        dim=self.teacher.dim,
                        step=step,
                        split="val",
                        layer=0,
                    )
                    log_value_alignment_scalars(
                        run=self.writer,
                        teacher_matrices=self.teacher._params,
                        student=self.student,
                        dim=self.teacher.dim,
                        step=step,
                        split="val",
                        layer=0,
                    )

    def _end_step(self, step: int, step_time: float, ngram: bool = False) -> None:
        self.logger.info(self._step_str(step, step_time, ngram))

        # save best validation loss
        if self.val_loader is not None:
            loss_avg = self.val_loss.compute()
            if loss_avg < self.val_best.compute():
                self.val_best.update(loss_avg)

        if ngram:
            ngram_metrics = {}
            for name in self.ngram_models:
                ngram_metrics[f"{name}_train_loss"] = self.metrics[f"{name}_train_loss"]
                ngram_metrics[f"{name}_train_acc"] = self.metrics[f"{name}_train_acc"]
                ngram_metrics[f"{name}_val_loss"] = self.metrics[f"{name}_val_loss"]
                ngram_metrics[f"{name}_val_acc"] = self.metrics[f"{name}_val_acc"]
        else:
            ngram_metrics = None

        # write to wandb
        if self.writer is not None:
            self._write_to_wandb(step, ngram_metrics)

        # clear metrics
        metrics = ngram_metrics or self.metrics
        for metric in metrics.values():
            if (
                isinstance(metric, RelativeMetric)
                or isinstance(metric, MinMetric)
                or isinstance(metric, ConstantLossMetric)
                or isinstance(metric, ConstantAccuracyMetric)
            ):
                continue
            metric.reset()

    def _step_str(self, step: int, step_time: float, ngram: bool = False):
        s = f"Step {step} "
        if step_time:
            s += f"| Step time: {step_time:.1f}s"

        if ngram:
            for name in self.ngram_models:
                s += f"| {name} Train loss: {self.metrics[f'{name}_train_loss'].compute():.3f} "
                s += f"| {name} Train acc: {self.metrics[f'{name}_train_acc'].compute():.3f} "
        else:
            s += f"| Train loss: {self.train_loss.compute():.3f} "
            if isinstance(self.loss_fn, CrossentropyLoss):
                s += f"| Train acc: {self.train_acc.compute():.3f} "
                if isinstance(self.teacher, ARTeacher):
                    s += f"| Train teacher acc: {self.teacher_train_acc.compute():.3f} "
            # if self.val_loader is not None:
            #     s += f"| Val loss: {self.val_loss.compute():.4f} "
            #     if isinstance(self.loss_fn, CrossentropyLoss):
            #         s += f"| Val acc: {self.val_acc.compute():.4f} "
            # s += f"| Grad norm: {self.grad_norm.compute():.4f} "

        return s

    def _write_to_wandb(self, step: int, ngram_metrics=None) -> None:
        metrics = ngram_metrics or self.metrics
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}

        # Add learning rate to logs
        if self.scheduler is not None:
            log_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        else:
            raise ValueError("Scheduler is None, cannot log learning rate.")

        self.writer.log(log_metrics, step=step)

    def _write_to_json(self, path: str, epoch: int):
        # save the metrics
        for key, metric in self.metrics.items():
            self.history[key].append(metric.compute())

        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)


class SGDTrainer(Trainer):
    def _train_step(self, data: torch.Tensor) -> None:
        self.student.zero_grad()

        # forward + backward
        self.optimizer.zero_grad()
        out, target, attn_weights = self._run_ar_model(
            self.student, data, return_targets=True, return_attn_weights=True
        )
        loss = self.loss_fn(out, target)
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.max_grad_norm
            )

        # take the optimization step
        self.optimizer.step()

        return out, target, loss, attn_weights

    def _clear_aux_metrics(self):
        for name in self.ngram_models:
            for suffix in ["train_loss", "train_acc", "val_loss", "val_acc"]:
                key = f"{name}_{suffix}"
                self.metrics.pop(key, None)
                self.history.pop(key, None)

    def _train_loop(self) -> None:
        """Train student model for one step (one batch)."""
        self._clear_aux_metrics()

        # set to train
        self.student.train()
        self.teacher.eval()

        # start loop - process one batch
        train_attn_weights = []
        for data in self.train_loader:
            if self.current_step >= self.steps:
                break
            data = data.to(self.device)
            out, target, loss, attn_weights = self._train_step(data)
            train_attn_weights.append(attn_weights)

            # update metrics
            self.train_loss.update(loss.item(), data.shape[0])
            self.train_acc.update(out, target)

            # update learning scheduler
            self._update_lr_sched(self.lr_metric.compute(), epoch_end=False)

            if isinstance(self.teacher, ARTeacher):
                kl_divergence = KLDivergenceLoss(reduction="mean")

                out_teacher = self._run_ar_model(
                    self.teacher,
                    data,
                    normalize=True,
                    prefix=-1,  # full teacher
                )
                kl_div_loss_teacher = kl_divergence(out, out_teacher)
                self.kl_div_teacher_train.update(
                    kl_div_loss_teacher.item(), data.shape[0]
                )

                for k in self.prefix_ks:  # 1 … window
                    out_teacher_k = self._run_ar_model(
                        self.teacher,
                        data,
                        normalize=True,  # gives us *probabilities*
                        prefix=k,  # look only at first k lags
                    )
                    kl_k = kl_divergence(
                        out, out_teacher_k
                    )  # raw logits vs. prob-targets
                    self.metrics[f"kl_div_prefix_{k}_teacher_train"].update(
                        kl_k.item(), data.size(0)
                    )

                for name, model in self.ngram_models.items():
                    teacher_context = getattr(self.teacher, 'context_length', sum(self.teacher.span_lengths))
                    stride = getattr(self.teacher, 'stride', None)
                    if stride is not None:
                        ngram_context = (model.ngram - 1) * stride + self.teacher.span_lengths[model.ngram - 1]
                    else:
                        ngram_context = sum(self.teacher.span_lengths[: model.ngram])
                    ngram_data = data[
                        :,
                        teacher_context - ngram_context :,
                    ]
                    _, aux_probs, _ = model(
                        ngram_data,
                        span_lengths=self.teacher.span_lengths,
                        unroll_sequences=True,
                        stride=stride,
                    )
                    kl = kl_divergence(out, aux_probs)
                    self.metrics[f"kl_div_{name}_learned_train"].update(
                        kl.item(), data.size(0)
                    )

                with torch.no_grad():
                    out_true = self._run_ar_model(
                        self.teacher,
                        data,
                        normalize=isinstance(self.loss_fn, CrossentropyLoss),
                    )
                    loss_true = self.loss_fn(out, out_true)
                    self.train_true_loss.update(loss_true.item(), data.shape[0])

            grad_norm = torch.sqrt(
                sum(
                    [
                        torch.norm(p.grad) ** 2
                        for p in self.student.parameters()
                        if p.grad is not None
                    ]
                )
            )
            self.grad_norm.update(grad_norm.item(), data.shape[0])

            # Log attention with configurable frequency
            if isinstance(self.student, TransformerDecoder):
                # attention weights (layer, batch, head, seq_len, seq_len), combine across batches.
                train_attn_weights_cat = torch.cat(train_attn_weights, dim=1)

                # Extract first layer and average over batch for numpy saving
                attn_np = (
                    train_attn_weights_cat[0].detach().cpu().numpy()
                )  # First layer: (batch, heads, seq_len, seq_len)
                attn_avg = attn_np.mean(axis=0)  # (heads, seq_len, seq_len)

                # Only log attention at specified frequency
                if self.writer is not None and self.current_step % self.log_attention_frequency == 0:
                    # Log structured attention table
                    log_attention_table(
                        run=self.writer,
                        attn_weights=train_attn_weights_cat,
                        layer=0,
                        batch_idx=-1,
                        step=self.current_step,
                        table_key="train_attention_weights",
                    )

                    # Create and log heatmaps using the averaged attention
                    log_attention_heatmap(
                        run=self.writer,
                        attn_weights=attn_avg,
                        log_key="train_attention_heatmaps",
                        step=self.current_step,
                    )

                    # Alignment: value matrix and attention pattern vs. ground truth
                    if isinstance(self.teacher, LinearARTeacher):
                        _stride = getattr(self.teacher, "stride", None)
                        _ctx_len = getattr(
                            self.teacher, "context_length", sum(self.teacher.span_lengths)
                        )
                        log_attention_alignment(
                            run=self.writer,
                            attn_avg=attn_avg,
                            span_lengths=self.teacher.span_lengths,
                            context_length=_ctx_len,
                            step=self.current_step,
                            split="train",
                            stride=_stride,
                        )
                        # Time-series scalars for tracking head collaboration:
                        # - attention span mass: how much each head attends to
                        #   each teacher's position group (collaborative phases)
                        # - value alignment: how each head's value matrix aligns
                        #   with each teacher feature (cooperative offset dynamics)
                        log_attention_span_mass(
                            run=self.writer,
                            attn_avg=attn_avg,
                            span_lengths=self.teacher.span_lengths,
                            context_length=_ctx_len,
                            step=self.current_step,
                            split="train",
                            stride=_stride,
                        )
                        log_value_matrix_alignment(
                            run=self.writer,
                            teacher_matrices=self.teacher._params,
                            student=self.student,
                            dim=self.teacher.dim,
                            step=self.current_step,
                            split="train",
                            layer=0,
                        )
                        log_value_alignment_scalars(
                            run=self.writer,
                            teacher_matrices=self.teacher._params,
                            student=self.student,
                            dim=self.teacher.dim,
                            step=self.current_step,
                            split="train",
                            layer=0,
                        )

                # Reset for next iteration
                train_attn_weights = []

            # Run validation and log metrics after each step
            self._val_loop(self.current_step)
            self._end_step(
                self.current_step,
                step_time=None,
                ngram=False,
            )
            self.current_step += 1
            if self.current_step >= self.steps:
                break

    def _train_ngram_step(self, data: torch.Tensor, name: str):
        """Train ONE auxiliary n-gram model."""
        model = self.ngram_models[name]
        optimizer = self.optim_ngram[name]

        model.zero_grad()
        optimizer.zero_grad()

        # drop the extra initial context for n-gram
        teacher_context = getattr(self.teacher, 'context_length', sum(self.teacher.span_lengths))
        stride = getattr(self.teacher, 'stride', None)
        if stride is not None:
            ngram_context = (model.ngram - 1) * stride + self.teacher.span_lengths[model.ngram - 1]
        else:
            ngram_context = sum(self.teacher.span_lengths[: model.ngram])
        ngram_data = data[
            :,
            teacher_context - ngram_context :,
        ]
        logits, _, ngram_target = model(
            ngram_data,
            span_lengths=self.teacher.span_lengths,
            unroll_sequences=True,
            stride=stride,
        )

        if self.student.teacher_target:
            with torch.no_grad():
                target = self._run_ar_model(
                    self.teacher, data, normalize=True, prefix=-1
                )
        else:
            target = ngram_target

        loss = self.loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        return logits, target, loss

    def _train_ngram(self) -> None:
        """Train ngram models for one step (one batch)."""
        for name, model in self.ngram_models.items():
            model.train()
        self.teacher.eval()

        # Process one batch
        for data in self.train_loader:
            if self.current_step >= self.ngram_steps:
                break
            data = data.to(self.device)
            for name in self.ngram_models:
                out, tgt, loss = self._train_ngram_step(data, name)
                self.metrics[f"{name}_train_loss"].update(loss.item(), data.size(0))
                self.metrics[f"{name}_train_acc"].update(out, tgt)

            # Evaluate and log after each step
            self._evaluate_ngram()
            self._end_step(self.current_step, step_time=None, ngram=True)
            self.current_step += 1
            if self.current_step >= self.ngram_steps:
                break

    def _evaluate_ngram(self):
        for _, model in self.ngram_models.items():
            model.eval()
        self.teacher.eval()
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)
                for name in self.ngram_models:
                    model = self.ngram_models[name]
                    # drop the extra initial context for n-gram
                    teacher_context = getattr(self.teacher, 'context_length', sum(self.teacher.span_lengths))
                    stride = getattr(self.teacher, 'stride', None)
                    if stride is not None:
                        ngram_context = (model.ngram - 1) * stride + self.teacher.span_lengths[model.ngram - 1]
                    else:
                        ngram_context = sum(self.teacher.span_lengths[: model.ngram])
                    ngram_data = data[
                        :,
                        teacher_context - ngram_context :,
                    ]
                    logits, _, ngram_target = model(
                        ngram_data,
                        span_lengths=self.teacher.span_lengths,
                        unroll_sequences=True,
                        stride=stride,
                    )
                    if self.student.teacher_target:
                        target = self._run_ar_model(
                            self.teacher, data, normalize=True, prefix=-1
                        )
                    else:
                        target = ngram_target
                    loss = self.loss_fn(logits, target)
                    self.metrics[f"{name}_val_loss"].update(loss.item(), data.size(0))
                    self.metrics[f"{name}_val_acc"].update(logits, target)
