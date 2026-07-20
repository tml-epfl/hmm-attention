from typing import List, Optional

import torch

from src.model import TransformerDecoder
from src.teachers import LinearARTeacher
from src.visualizer import (
    log_attention_alignment,
    log_attention_heatmap,
    log_attention_span_mass,
    log_attention_table,
    log_value_alignment_scalars,
    log_value_matrix_alignment,
)


class AttentionLogger:
    """Owns all attention visualization for train/val loops.

    No-ops unless `student` is a `TransformerDecoder`, `writer` is not None,
    and `step % frequency == 0`. Alignment-based logs (attention pattern,
    value-matrix alignment) additionally require a `LinearARTeacher` since
    they read `teacher._params` and `span_lengths`.
    """

    def __init__(
        self,
        writer,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        frequency: int,
    ) -> None:
        self.writer = writer
        self.teacher = teacher
        self.student = student
        self.frequency = frequency

    def log(
        self,
        step: int,
        split: str,
        attn_weight_batches: List[torch.Tensor],
    ) -> None:
        if not isinstance(self.student, TransformerDecoder):
            return
        if self.writer is None or step % self.frequency != 0:
            return
        if not attn_weight_batches:
            return

        # (layer, batch, head, seq_len, seq_len); concat along batch.
        attn_weights = torch.cat(attn_weight_batches, dim=1)
        n_layers = attn_weights.shape[0]

        for layer in range(n_layers):
            # Always prefix with `/L{layer}` — even for single-block runs — so
            # wandb groups per-layer visualizations into a dedicated tab. This
            # also keeps key structure stable when you flip `num_blocks` in an
            # experiment.
            suffix = f"/L{layer}"

            # Batch-averaged per-layer attention: (heads, seq_len, seq_len)
            attn_avg = (
                attn_weights[layer].detach().cpu().numpy().mean(axis=0)
            )

            log_attention_table(
                run=self.writer,
                attn_weights=attn_weights,
                layer=layer,
                batch_idx=-1,
                step=step,
                table_key=f"attn/{split}{suffix}/weights",
            )
            log_attention_heatmap(
                run=self.writer,
                attn_weights=attn_avg,
                log_key=f"attn/{split}{suffix}/heatmaps",
                step=step,
            )

            if isinstance(self.teacher, LinearARTeacher):
                stride: Optional[int] = getattr(self.teacher, "stride", None)
                ctx_len = getattr(
                    self.teacher, "context_length", sum(self.teacher.span_lengths)
                )
                split_layer = f"{split}{suffix}"
                log_attention_alignment(
                    run=self.writer,
                    attn_avg=attn_avg,
                    span_lengths=self.teacher.span_lengths,
                    context_length=ctx_len,
                    step=step,
                    split=split_layer,
                    stride=stride,
                )
                log_attention_span_mass(
                    run=self.writer,
                    attn_avg=attn_avg,
                    span_lengths=self.teacher.span_lengths,
                    context_length=ctx_len,
                    step=step,
                    split=split_layer,
                    stride=stride,
                )
                log_value_matrix_alignment(
                    run=self.writer,
                    teacher_matrices=self.teacher._params,
                    student=self.student,
                    dim=self.teacher.dim,
                    step=step,
                    split=split_layer,
                    layer=layer,
                )
                log_value_alignment_scalars(
                    run=self.writer,
                    teacher_matrices=self.teacher._params,
                    student=self.student,
                    dim=self.teacher.dim,
                    step=step,
                    split=split_layer,
                    layer=layer,
                )
