"""Focused tests on NgramEvaluator's pure logic.

Full train_step + eval_step end-to-end coverage lives in the trainer smoke
test — here we only nail down the data-slicing arithmetic + metric key naming.
"""
from unittest.mock import MagicMock

import torch

from src.trainer import NgramEvaluator


class _StubNgramModel:
    """Duck-typed stand-in — NgramEvaluator only reads `.ngram` for slicing."""

    def __init__(self, ngram: int):
        self.ngram = ngram


class _StubTeacher:
    def __init__(self, span_lengths, stride=None, context_length=None):
        self.span_lengths = span_lengths
        self.stride = stride
        if context_length is not None:
            self.context_length = context_length


def _make_evaluator(teacher, model, name="bigram"):
    return NgramEvaluator(
        name=name,
        model=model,
        optimizer=MagicMock(),
        teacher=teacher,
        teacher_evaluator=MagicMock(),
    )


# ---- _slice_data -------------------------------------------------------------


def test_slice_data_no_stride():
    # teacher spans [1,1,1] → context_length=3; bigram (ngram=2) → ngram_context=2.
    # Trim leading 3 - 2 = 1 token.
    teacher = _StubTeacher(span_lengths=[1, 1, 1], context_length=3)
    ev = _make_evaluator(teacher, _StubNgramModel(ngram=2))
    data = torch.zeros(1, 10, 4)
    sliced, stride = ev._slice_data(data)
    assert sliced.shape[1] == 10 - 1
    assert stride is None


def test_slice_data_with_stride():
    # stride=2, span_lengths=[1,1,1], context_length=5.
    # bigram (ngram=2): (2-1)*2 + span[1] = 2 + 1 = 3.
    # Trim leading 5 - 3 = 2 tokens.
    teacher = _StubTeacher(span_lengths=[1, 1, 1], stride=2, context_length=5)
    ev = _make_evaluator(teacher, _StubNgramModel(ngram=2))
    data = torch.zeros(1, 10, 4)
    sliced, stride = ev._slice_data(data)
    assert sliced.shape[1] == 10 - 2
    assert stride == 2


def test_slice_data_falls_back_to_sum_when_no_context_length_attr():
    # Older teacher-like objects without context_length: fallback to sum(span_lengths).
    teacher = _StubTeacher(span_lengths=[2, 1, 3])
    # Ensure fallback path is hit: object shouldn't have `context_length`.
    assert not hasattr(teacher, "context_length")
    ev = _make_evaluator(teacher, _StubNgramModel(ngram=1))
    data = torch.zeros(1, 10, 4)
    sliced, _ = ev._slice_data(data)
    # teacher_context = 6, ngram_context = span_lengths[0] = 2 → trim 4.
    assert sliced.shape[1] == 10 - 4


# ---- metric key naming -------------------------------------------------------


def test_metric_keys_use_correct_grouping():
    teacher = _StubTeacher(span_lengths=[1, 1], context_length=2)
    ev = _make_evaluator(teacher, _StubNgramModel(ngram=1), name="bigram")
    assert ev.kl_metric_keys() == ["kl/bigram_train", "kl/bigram_val"]
    assert ev.loss_metric_keys() == ["ngram_bigram/train_loss", "ngram_bigram/val_loss"]
    assert ev.acc_metric_keys() == ["ngram_bigram/train_acc", "ngram_bigram/val_acc"]
