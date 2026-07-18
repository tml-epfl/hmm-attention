import pytest
import torch

from src.teachers import LinearARTeacher


# ---- context_length ----------------------------------------------------------


def test_context_length_no_stride():
    t = LinearARTeacher.from_parameters(dim=4, window=3, span_lengths=[2, 1, 3])
    assert t.context_length == 6


def test_context_length_with_stride():
    t = LinearARTeacher.from_parameters(
        dim=4, window=3, span_lengths=[1, 1, 1], stride=2
    )
    # (n-1)*stride + last_span = 2*2 + 1 = 5
    assert t.context_length == 5


# ---- next_token_log_probs ----------------------------------------------------


def test_next_token_log_probs_shape_and_normalization(tiny_teacher):
    B = 3
    context = torch.zeros(B, tiny_teacher.context_length, tiny_teacher.dim)
    context[..., 0] = 1.0
    log_probs = tiny_teacher.next_token_log_probs(context)
    assert log_probs.shape == (B, tiny_teacher.dim)
    # exp() sums to 1 along last dim.
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(B), atol=1e-5)


def test_next_token_log_probs_rejects_wrong_context_length(tiny_teacher):
    wrong_ctx = torch.zeros(1, tiny_teacher.context_length + 1, tiny_teacher.dim)
    with pytest.raises(ValueError, match="expected exactly"):
        tiny_teacher.next_token_log_probs(wrong_ctx)


# ---- predict_next slicing ----------------------------------------------------


def test_predict_next_auto_slices_long_prefix(tiny_teacher):
    long_prefix = torch.zeros(1, tiny_teacher.context_length + 5, tiny_teacher.dim)
    long_prefix[..., 0] = 1.0
    out = tiny_teacher.predict_next(long_prefix)
    assert out.shape == (1, tiny_teacher.dim)


def test_predict_next_rejects_short_prefix(tiny_teacher):
    short = torch.zeros(1, tiny_teacher.context_length - 1, tiny_teacher.dim)
    with pytest.raises(ValueError, match="context_length"):
        tiny_teacher.predict_next(short)


# ---- with_lag_restriction ----------------------------------------------------


def test_with_lag_restriction_reduces_context():
    t = LinearARTeacher.from_parameters(dim=4, window=3, span_lengths=[1, 1, 1])
    restricted = t.with_lag_restriction(2)
    assert restricted.context_length == 2  # first 2 spans of length 1 each


def test_with_lag_restriction_produces_valid_log_probs():
    t = LinearARTeacher.from_parameters(dim=4, window=3, span_lengths=[1, 1, 1], scale=5.0)
    restricted = t.with_lag_restriction(2)
    ctx = torch.zeros(1, restricted.context_length, restricted.dim)
    ctx[..., 0] = 1.0
    log_probs = restricted.next_token_log_probs(ctx)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.tensor([1.0]), atol=1e-5)


def test_with_lag_restriction_rejects_out_of_range():
    t = LinearARTeacher.from_parameters(dim=4, window=3, span_lengths=[1, 1, 1])
    with pytest.raises(ValueError, match=r"k=0"):
        t.with_lag_restriction(0)
    with pytest.raises(ValueError, match=r"k=4"):
        t.with_lag_restriction(4)
