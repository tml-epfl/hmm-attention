import copy
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

from src.data import ar_batch_collate


def get_loaders(
    dataset_cfg: DictConfig,
    trainer_cfg: DictConfig,
    predictor: torch.nn.Module,
    prefix_length: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders.

    Mutates `dataset_cfg` (strips `number`) and `trainer_cfg` (strips
    `batch_size`, `replacement`, `num_workers`) so downstream Hydra
    instantiation of the trainer does not receive fields it doesn't accept.
    """
    batch_size = (
        dataset_cfg.number.train
        if trainer_cfg.batch_size == -1
        else trainer_cfg.batch_size
    )
    num_workers = trainer_cfg.get("num_workers", 0)

    train_size = dataset_cfg.number.train
    val_size = dataset_cfg.number.val
    del dataset_cfg.number

    train_dataset = instantiate(
        dataset_cfg,
        predictor=predictor,
        number=train_size,
        prefix_length=prefix_length,
    )
    val_dataset = (
        copy.deepcopy(train_dataset)
        if val_size == 0
        else instantiate(
            dataset_cfg,
            predictor=predictor,
            number=val_size,
            prefix_length=prefix_length,
        )
    )

    train_sampler = RandomSampler(
        train_dataset, replacement=trainer_cfg.replacement, num_samples=abs(train_size)
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "collate_fn": ar_batch_collate,
    }
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True,
        )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_sampler = RandomSampler(
        val_dataset,
        replacement=False,
        num_samples=abs(train_size if val_size == 0 else val_size),
    )
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    del trainer_cfg.batch_size, trainer_cfg.replacement
    if "num_workers" in trainer_cfg:
        del trainer_cfg.num_workers
    return train_loader, val_loader


def get_optimizer(
    cfg: DictConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return instantiate(cfg, params=model.parameters())
