from typing import Tuple

import torch

from src.profiling import get_profiler
from src.trainer.base import Trainer


class SGDTrainer(Trainer):
    """Vanilla SGD training with optional gradient clipping.

    Per step: one batch through the student (forward, loss, backward, clip,
    optimizer step), then a full validation pass, then end-of-step bookkeeping.
    """

    def _train_step(
        self, data: torch.Tensor, run_teacher_metrics: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prof = get_profiler()
        self.student.zero_grad()
        self.optimizer.zero_grad()
        self.probe_logger.before_forward("train")
        try:
            out, target, loss, attn_weights = self._forward_and_metrics(
                data, split="train", run_teacher_metrics=run_teacher_metrics
            )
        finally:
            self.probe_logger.after_forward(data)
        with prof.cuda("student_backward"):
            loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.max_grad_norm
            )
        with prof.cuda("optimizer_step"):
            self.optimizer.step()
        with prof.cuda("probe_sgd_step"):
            self.probe_logger.sgd_step()
        return out, target, loss, attn_weights

    def _is_log_step(self) -> bool:
        """Log this step? True on the last step and every `log_frequency`-th step."""
        freq = self.logging_cfg.log_frequency
        return (self.current_step + 1 >= self.steps) or (
            self.current_step % freq == 0
        )

    def _train_loop(self) -> None:
        self._clear_aux_metrics()
        self.student.train()
        self.teacher.eval()
        prof = get_profiler()

        for data in self.train_loader:
            if self.current_step >= self.steps:
                break
            with prof.cuda("data_to_device_train"):
                data = data.to(self.device)
            log_this_step = self._is_log_step()
            _, _, _, attn_weights = self._train_step(data, log_this_step)

            self._update_lr_sched(self.lr_metric.compute(), epoch_end=False)

            grad_norm = torch.sqrt(
                sum(
                    torch.norm(p.grad) ** 2
                    for p in self.student.parameters()
                    if p.grad is not None
                )
            )
            self.metrics["student/grad_norm"].update(grad_norm.item(), data.shape[0])

            self.attention_logger.log(self.current_step, "train", [attn_weights])
            if log_this_step:
                with prof.cuda("val_loop"):
                    self._val_loop(self.current_step)
                self._end_step(self.current_step, step_time=None, ngram=False)
            self.current_step += 1
            if self.current_step >= self.steps:
                break

    def _train_ngram(self) -> None:
        for ne in self.ngram_evals.values():
            ne.model.train()
        self.teacher.eval()

        for data in self.train_loader:
            if self.current_step >= self.ngram_cfg.steps:
                break
            data = data.to(self.device)
            for ne in self.ngram_evals.values():
                logits, target, loss = ne.train_step(
                    data, self.loss_fn, self.ngram_cfg.use_teacher_target
                )
                self.metrics[f"ngram_{ne.name}/loss/train"].update(
                    loss.item(), data.size(0)
                )
                self.metrics[f"ngram_{ne.name}/acc/train"].update(logits, target)

            self._val_ngram()
            self._end_step(self.current_step, step_time=None, ngram=True)
            self.current_step += 1
            if self.current_step >= self.ngram_cfg.steps:
                break

    def _clear_aux_metrics(self) -> None:
        """Ngram-training-only metrics — drop before student training starts."""
        for ne in self.ngram_evals.values():
            for key in ne.loss_metric_keys() + ne.acc_metric_keys():
                self.metrics.pop(key, None)
                self.history.pop(key, None)
