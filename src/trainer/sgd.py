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
            self.metrics["student/grad_norm"].update(grad_norm.item(), data.shape[0])

            self.attention_logger.log(self.current_step, "train", [attn_weights])
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
                self.metrics[f"ngram_{ne.name}/train_loss"].update(
                    loss.item(), data.size(0)
                )
                self.metrics[f"ngram_{ne.name}/train_acc"].update(logits, target)

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
