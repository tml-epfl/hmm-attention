import copy
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler


def get_loaders(
    dataset_cfg: DictConfig,
    trainer_cfg: DictConfig,
    predictor: torch.nn.Module,
    prefix_length: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders.

    Mutates `dataset_cfg` (strips `number`) and `trainer_cfg` (strips
    `batch_size`, `replacement`) so downstream Hydra instantiation of the
    trainer does not receive fields it doesn't accept.
    """
    batch_size = (
        dataset_cfg.number.train
        if trainer_cfg.batch_size == -1
        else trainer_cfg.batch_size
    )

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
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size
    )
    val_sampler = RandomSampler(
        val_dataset,
        replacement=False,
        num_samples=abs(train_size if val_size == 0 else val_size),
    )
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=batch_size
    )

    del trainer_cfg.batch_size, trainer_cfg.replacement
    return train_loader, val_loader


def get_optimizer(
    cfg: DictConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return instantiate(cfg, params=model.parameters())
