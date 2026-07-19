from typing import Dict, Optional, Tuple

import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from src.predictors import build_predictor
from src.runner.data import get_loaders, get_optimizer
from src.runner.preprocess import configure_positional_encoding, preprocess_cfg
from src.runner.verbose import log_student_summary, log_teacher_summary
from src.trainer import LoggingConfig, NgramConfig, SchedulerConfig, Trainer


def _pass_sched_metric(scheduler: torch.optim.lr_scheduler._LRScheduler) -> bool:
    """True if the scheduler expects a metric on `.step(metric)` (e.g. plateau)."""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _update_scheduler_each_iter(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> bool:
    """True if the scheduler advances per iter rather than per epoch."""
    return isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def _init_wandb(cfg: DictConfig):
    return wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.misc.wandb.entity,
        project=cfg.misc.wandb.project,
        tags=cfg.misc.wandb.tags,
        settings=wandb.Settings(start_method="thread"),
    )


def _build_student(
    cfg: DictConfig, teacher: torch.nn.Module
) -> Tuple[torch.nn.Module, Optional[int], Optional[torch.Tensor]]:
    """Instantiate the student. When `teacher_readout` is set, warm-start its
    value projections from the teacher weights.

    Returns `(student, window, teacher_matrices)`; the latter two are forwarded
    to ngram model instantiation so all auxiliary models share the same
    readout configuration.
    """
    if "teacher_readout" in cfg.student and cfg.student.teacher_readout:
        window = teacher.window
        teacher_matrices = teacher._get_weights().detach().clone()
        student = instantiate(
            cfg.student, window=window, teacher_matrices=teacher_matrices
        )
        return student, window, teacher_matrices
    return instantiate(cfg.student), None, None


def _build_ngram_models(
    cfg: DictConfig, window: Optional[int], teacher_matrices: Optional[torch.Tensor]
) -> Dict[str, torch.nn.Module]:
    if "ngrams" not in cfg:
        return {}
    return {
        name: instantiate(m_cfg, window=window, teacher_matrices=teacher_matrices)
        for name, m_cfg in cfg.ngrams.items()
    }


def get_trainer(cfg: DictConfig) -> Trainer:
    cfg = preprocess_cfg(cfg)

    teacher = instantiate(cfg.teacher)
    prefix_length = getattr(teacher, "context_length", sum(teacher.span_lengths))
    configure_positional_encoding(cfg, teacher)

    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    writer = _init_wandb(cfg) if cfg.misc.wandb.use else None

    student, window, teacher_matrices = _build_student(cfg, teacher)

    predictor_cfg = (
        OmegaConf.to_container(cfg.predictor, resolve=True) if "predictor" in cfg else {}
    )
    predictor = build_predictor(teacher, **predictor_cfg)

    train_loader, val_loader = get_loaders(
        cfg.dataset, cfg.trainer, predictor=predictor, prefix_length=prefix_length
    )
    loss_fn = instantiate(cfg.loss)
    optimizer = get_optimizer(cfg.optimizer, student)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    ngram_models = _build_ngram_models(cfg, window, teacher_matrices)
    optim_ngram = {
        name: get_optimizer(cfg.optimizer, model)
        for name, model in ngram_models.items()
    }

    if cfg.misc.verbose:
        log_teacher_summary(teacher)
        log_student_summary(student)

    scheduler_cfg = SchedulerConfig(
        scheduler=scheduler,
        pass_metric=_pass_sched_metric(scheduler),
        update_on_iter=_update_scheduler_each_iter(scheduler),
    )
    ngram_cfg = NgramConfig(
        models=ngram_models,
        optimizers=optim_ngram,
        steps=cfg.trainer.ngram_steps,
        use_teacher_target=(
            cfg.trainer.use_teacher_target
            if "use_teacher_target" in cfg.trainer
            else False
        ),
    )
    logging_cfg = LoggingConfig(
        writer=writer,
        attention_frequency=cfg.misc.log_attention_frequency,
        log_frequency=cfg.misc.get("log_frequency", 1),
    )

    # Construct the trainer class directly. Hydra's `instantiate` cannot merge
    # kwargs containing `torch.nn.Module` (or dataclasses wrapping them) into a
    # DictConfig, so we resolve `_target_` manually and call the constructor.
    trainer_cls = get_class(cfg.trainer._target_)
    max_grad_norm = (
        cfg.trainer.max_grad_norm if "max_grad_norm" in cfg.trainer else None
    )
    return trainer_cls(
        steps=cfg.trainer.steps,
        max_grad_norm=max_grad_norm,
        device=cfg.misc.device,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler_cfg=scheduler_cfg,
        ngram_cfg=ngram_cfg,
        logging_cfg=logging_cfg,
    )
