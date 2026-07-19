from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import wandb


@dataclass
class SchedulerConfig:
    """LR scheduler + calling convention (pass metric? step every iter?)."""

    scheduler: torch.optim.lr_scheduler._LRScheduler
    pass_metric: bool = False
    update_on_iter: bool = False


@dataclass
class NgramConfig:
    """Auxiliary ngram models + their optimizers + training-phase knobs.

    `use_teacher_target=True` swaps the ngram's own targets for the teacher's
    soft distribution during training — useful when the teacher is the
    supervising signal rather than the raw sequence.
    """

    models: Dict[str, torch.nn.Module] = field(default_factory=dict)
    optimizers: Dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    steps: int = 0
    use_teacher_target: bool = False


@dataclass
class LoggingConfig:
    """W&B writer + logging cadence.

    `log_frequency` controls how often the val loop runs, how often teacher
    KL / true-loss metrics are computed against the current train batch, and
    how often we write to W&B + reset accumulator metrics. Student loss/acc
    still update every step (accumulated between logs). Default 1 preserves
    the pre-existing "every step" behavior.
    """

    writer: Optional[wandb.run] = None
    attention_frequency: int = 100
    log_frequency: int = 1
