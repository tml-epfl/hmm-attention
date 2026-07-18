"""End-to-end smoke test: build a `SGDTrainer` from fixtures and run 2 steps.

Exercises the whole assembled system (student forward, TeacherEvaluator KL
metrics, MetricRegistry lookups, AttentionLogger construction, train/val
loops) — no Hydra needed.
"""

import math

import pytest
import torch

from src.loss import CrossentropyLoss
from src.trainer import LoggingConfig, NgramConfig, SchedulerConfig, SGDTrainer


@pytest.fixture()
def smoke_trainer(
    tiny_teacher, tiny_student, tiny_loaders, device
) -> SGDTrainer:
    train_loader, val_loader = tiny_loaders

    optimizer = torch.optim.SGD(tiny_student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    return SGDTrainer(
        steps=2,
        device=device,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=CrossentropyLoss(),
        optimizer=optimizer,
        scheduler_cfg=SchedulerConfig(scheduler=scheduler),
        ngram_cfg=NgramConfig(steps=0),  # no ngram phase
        logging_cfg=LoggingConfig(writer=None, attention_frequency=100),
    )


def test_trainer_train_runs_two_steps_without_crash(smoke_trainer):
    smoke_trainer.train()
    assert smoke_trainer.current_step == 2


def test_trainer_records_finite_val_best(smoke_trainer):
    # `student/val_loss` is reset in `_end_step`, but `student/val_best`
    # (MinMetric) persists across steps — proves val actually ran.
    smoke_trainer.train()
    best = smoke_trainer.metrics["student/val_best"].compute()
    assert math.isfinite(best)
    assert best >= 0


def test_trainer_populates_constant_teacher_metrics(smoke_trainer):
    # ConstantLossMetric populated once in `_dry_loop`; never reset.
    smoke_trainer.train()
    tl = smoke_trainer.metrics["teacher/train_loss"].compute()
    ta = smoke_trainer.metrics["teacher/train_acc"].compute()
    assert math.isfinite(tl) and tl >= 0
    assert 0 <= ta <= 1


def test_trainer_has_no_ngram_evaluators_when_disabled(smoke_trainer):
    # NgramConfig(steps=0, models={}) — no evaluators built.
    smoke_trainer.train()
    assert smoke_trainer.ngram_evals == {}


def test_trainer_constructs_attention_logger(smoke_trainer):
    smoke_trainer.train()
    assert smoke_trainer.attention_logger is not None
    # No writer → all log calls are no-ops, but the logger object still exists.
    assert smoke_trainer.attention_logger.writer is None


def test_trainer_metrics_registry_rejects_typos_after_init(smoke_trainer):
    smoke_trainer.train()
    with pytest.raises(KeyError):
        smoke_trainer.metrics["student/train_los"]  # noqa: B015
