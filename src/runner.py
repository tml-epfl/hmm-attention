from typing import Tuple, Dict
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import logging
import torch, wandb, copy
from torch.utils.data import DataLoader, RandomSampler

from src.trainer import Trainer
from src.model import LinearARModel
from src.data import HierarchicalGaussianARClassification

def _pass_sched_metric(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> bool:
    """Returns true if the scheduler should receive a metric."""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _update_scheduler_each_iter(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> bool:
    """Returns true if the scheduler should be updated each iteration and
    false if the scheduler should be updated each epoch."""
    return isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def _calculate_context_length(span_lengths, stride=None):
    """Calculate total context length accounting for stride."""
    if stride is not None:
        return (len(span_lengths) - 1) * stride + span_lengths[-1]
    return sum(span_lengths)


def _configure_positional_encoding(cfg):
    """Configure positional encoding."""
    if "student" in cfg and "pe_type" in cfg.student:
        pe_type = cfg.student["pe_type"]

        # Get stride from teacher config (may be None)
        stride = cfg.teacher.get("stride", None)

        if pe_type == "one_hot":
            prefix_length = _calculate_context_length(cfg.teacher.span_lengths, stride)
            embedding_dim = prefix_length + cfg.dataset.length - 1
            cfg.student.hidden_dim = cfg.dataset.dim + embedding_dim
            cfg.student.pe_embedding_dim = embedding_dim
        if pe_type == "absolute":
            # For absolute PE, embedding_dim should match hidden_dim
            cfg.student.pe_embedding_dim = cfg.student.hidden_dim

        # Configure ngram models
        for name, ngram_cfg in cfg.ngrams.items():
            if pe_type == "one_hot":
                # ngram use unroll sequences True
                # For ngrams, calculate context length for first n spans
                ngram = ngram_cfg.ngram
                if stride is not None:
                    # With stride: (ngram - 1) * stride + last_span_length
                    embedding_dim = (ngram - 1) * stride + cfg.teacher.span_lengths[ngram - 1]
                else:
                    # Without stride: sum of first ngram spans
                    embedding_dim = sum(cfg.teacher.span_lengths[:ngram])
                hidden_dim = cfg.dataset.dim + embedding_dim
            else:
                embedding_dim = cfg.student.pe_embedding_dim
                hidden_dim = cfg.student.hidden_dim
            cfg.ngrams[name].pe_embedding_dim = embedding_dim
            cfg.ngrams[name].hidden_dim = hidden_dim


def _preprocess_integer(i: int, default: int) -> int:
    if i == -1:
        return default
    return i


def _preprocess_cfg(cfg: DictConfig) -> DictConfig:
    # process teach dim, rank, window
    if "dim" in cfg.teacher:
        cfg.teacher.dim = _preprocess_integer(cfg.teacher.dim, default=cfg.dataset.dim)
    if "rank" in cfg.teacher:
        cfg.teacher.rank = _preprocess_integer(cfg.teacher.rank, default=cfg.teacher.dim)
    if "window" in cfg.teacher:
        cfg.teacher.window = _preprocess_integer(
            cfg.teacher.window, default=cfg.dataset.window
        )
    if "hidden_dim" in cfg.teacher:
        cfg.teacher.hidden_dim = _preprocess_integer(
            cfg.teacher.hidden_dim, default=cfg.teacher.dim
        )

    # process student dim, rank, window
    if "student" in cfg:
        cfg.student.dim = _preprocess_integer(cfg.student.dim, default=cfg.dataset.dim)
        if "rank" in cfg.student:
            cfg.student.rank = _preprocess_integer(
                cfg.student.rank, default=cfg.student.dim
            )
        if "window" in cfg.student:
            cfg.student.window = _preprocess_integer(
                cfg.student.window, default=cfg.teacher.window
            )
        if "hidden_dim" in cfg.student:
            cfg.student.hidden_dim = _preprocess_integer(
                cfg.student.hidden_dim, default=cfg.student.dim
            )

    if "ngrams" in cfg:
        for name, ngram_cfg in cfg.ngrams.items():
            # Start with student config as base
            merged_cfg = OmegaConf.create(cfg.student)
            # Override with ngram-specific settings
            merged_cfg._target_ = ngram_cfg._target_
            merged_cfg.ngram = ngram_cfg.ngram
            # Add conf
            cfg.ngrams[name] = merged_cfg

    _configure_positional_encoding(cfg)

    return cfg


def _get_loaders(
    cfg: DictConfig,
    trainer_cfg: DictConfig,
    teacher: torch.nn.Module,
    prefix_length: int,
) -> Tuple[DataLoader, DataLoader]:
    # init batch size
    if trainer_cfg.batch_size == -1:
        batch_size = cfg.number.train
    else:
        batch_size = trainer_cfg.batch_size

    train_size = cfg.number.train
    val_size = cfg.number.val
    del cfg.number

    # initialize datasets
    train_dataset = instantiate(
        cfg, teacher=teacher, number=train_size, prefix_length=prefix_length
    )
    validation_dataset = (
        copy.deepcopy(train_dataset)
        if val_size == 0
        else instantiate(
            cfg, teacher=teacher, number=val_size, prefix_length=prefix_length
        )
    )

    # initialize dataloaders
    train_sampler = RandomSampler(
        train_dataset, replacement=trainer_cfg.replacement, num_samples=abs(train_size)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size
    )

    validation_sampler = RandomSampler(
        validation_dataset,
        replacement=False,
        num_samples=abs(train_size if val_size == 0 else val_size),
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        batch_size=batch_size,
    )

    del trainer_cfg.batch_size, trainer_cfg.replacement
    return (train_dataloader, validation_dataloader)


def _get_optimizer(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    return instantiate(cfg, params=model.parameters())


def get_trainer(cfg: DictConfig) -> Trainer:
    # preprocess and save config
    cfg = _preprocess_cfg(cfg)

    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.misc.wandb.use:
        writer = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            entity=cfg.misc.wandb.entity,
            project=cfg.misc.wandb.project,
            tags=cfg.misc.wandb.tags,
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        writer = None

    teacher = instantiate(cfg.teacher)
    # Use context_length if available (accounts for stride), otherwise fall back to sum
    prefix_length = getattr(teacher, 'context_length', sum(teacher.span_lengths))

    if "teacher_readout" in cfg.student and cfg.student.teacher_readout:
        window = teacher.window
        teacher_matrices = teacher._get_weights().detach().clone()
        student = instantiate(cfg.student, window=window, teacher_matrices=teacher_matrices)
    else:
        window, teacher_matrices = None, None
        student = instantiate(cfg.student)

    train_loader, val_loader = _get_loaders(
        cfg.dataset, cfg.trainer, teacher=teacher, prefix_length=prefix_length
    )
    loss_fn = instantiate(cfg.loss)
    optimizer = _get_optimizer(cfg.optimizer, student)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    ngram_models: Dict[str, nn.Module] = {}
    if "ngrams" in cfg:
        for name, m_cfg in cfg.ngrams.items():
            ngram_models[name] = instantiate(
                m_cfg,
                window=window,
                teacher_matrices=teacher_matrices,
            )
    optim_ngram = {
        name: _get_optimizer(cfg.optimizer, model)
        for name, model in ngram_models.items()
    }

    if cfg.misc.verbose:
        if isinstance(teacher, LinearARModel):
            logger = logging.getLogger()
            logger.info(f"===== Teacher =====")
            logger.info(f"Teacher rank: {teacher.rank}")
            logger.info(f"Teacher dim: {teacher.dim}")
            logger.info(f"Teacher window: {teacher.window}")
            logger.info(f"Teacher scale: {teacher.scale}")
            logger.info(f"Teacher weights: {teacher._get_weights().shape}")

            params = teacher._params
            params = params.view(-1, params.size(-1))
            logger.info(
                f"Frobenius norm/norm^2: {torch.linalg.norm(params)}, {torch.linalg.norm(params) ** 2}"
            )
            logger.info(
                f"Operator norm/norm^2: {torch.linalg.norm(params, ord=2)}, {torch.linalg.norm(params, ord=2) ** 2}"
            )

        if isinstance(student, LinearARModel):
            logger = logging.getLogger()
            logger.info(f"===== Student =====")
            logger.info(f"Student rank: {student.rank}")
            logger.info(f"Student dim: {student.dim}")
            logger.info(f"Student window: {student.window}")
            logger.info(f"Student scale: {student.scale}")
            logger.info(f"Student weights: {student._get_weights().shape}")

            params = student._get_weights()
            params = params.view(-1, params.size(-1))
            logger.info(
                f"Frobenius norm/norm^2: {torch.linalg.norm(params)}, {torch.linalg.norm(params) ** 2}"
            )
            logger.info(
                f"Operator norm/norm^2: {torch.linalg.norm(params, ord=2)}, {torch.linalg.norm(params, ord=2) ** 2}"
            )

    if isinstance(train_loader.dataset, HierarchicalGaussianARClassification):
        teacher = None

    return instantiate(
        cfg.trainer,
        device=cfg.misc.device,
        teacher=teacher,
        student=student,
        ngram_models=ngram_models,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optim_ngram=optim_ngram,
        scheduler=scheduler,
        pass_sched_metric=_pass_sched_metric(scheduler),
        update_sched_on_iter=_update_scheduler_each_iter(scheduler),
        writer=writer,
        log_attention_frequency=cfg.misc.log_attention_frequency,
    )
