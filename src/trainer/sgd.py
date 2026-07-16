from typing import Tuple

import torch

from src.trainer.base import Trainer


class SGDTrainer(Trainer):
    """Vanilla SGD training with optional gradient clipping.

    Per step: one batch through the student (forward, loss, backward, clip,
    optimizer step), then a full validation pass, then end-of-step bookkeeping.
    """

    def _train_step(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.student.zero_grad()
        self.optimizer.zero_grad()
        out, target, loss, attn_weights = self._forward_and_metrics(data, split="train")
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        return out, target, loss, attn_weights

    def _train_loop(self) -> None:
        self._clear_aux_metrics()
        self.student.train()
        self.teacher.eval()

        for data in self.train_loader:
            if self.current_step >= self.steps:
                break
            data = data.to(self.device)
            _, _, _, attn_weights = self._train_step(data)

            self._update_lr_sched(self.lr_metric.compute(), epoch_end=False)

            grad_norm = torch.sqrt(
                sum(
                    torch.norm(p.grad) ** 2
                    for p in self.student.parameters()
                    if p.grad is not None
                )
            )
            self.grad_norm.update(grad_norm.item(), data.shape[0])

            self.attention_logger.log(self.current_step, "train", [attn_weights])
            self._val_loop(self.current_step)
            self._end_step(self.current_step, step_time=None, ngram=False)
            self.current_step += 1
            if self.current_step >= self.steps:
                break

    def _train_ngram(self) -> None:
        for model in self.ngram_models.values():
            model.train()
        self.teacher.eval()
        use_teacher_target = self.student.teacher_target

        for data in self.train_loader:
            if self.current_step >= self.ngram_steps:
                break
            data = data.to(self.device)
            for name in self.ngram_models:
                logits, target, loss = self.ngram_eval.train_step(
                    name, data, self.loss_fn, use_teacher_target
                )
                self.metrics[f"{name}_train_loss"].update(loss.item(), data.size(0))
                self.metrics[f"{name}_train_acc"].update(logits, target)

            self._val_ngram()
            self._end_step(self.current_step, step_time=None, ngram=True)
            self.current_step += 1
            if self.current_step >= self.ngram_steps:
                break

    def _clear_aux_metrics(self) -> None:
        """Ngram-training-only metrics — drop before student training starts."""
        for name in self.ngram_models:
            for suffix in ("train_loss", "train_acc", "val_loss", "val_acc"):
                key = f"{name}_{suffix}"
                self.metrics.pop(key, None)
                self.history.pop(key, None)
