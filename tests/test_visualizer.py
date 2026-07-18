import numpy as np

from src.visualizer import _span_column_ranges, compute_gt_attention_row


def test_span_column_ranges_no_stride():
    # span_lengths=[2,1,3], context_length=6, seq_len=10 → context starts at 4.
    # Spans laid out contiguously inside the context window.
    ranges = _span_column_ranges(
        span_lengths=[2, 1, 3], context_length=6, seq_len=10, stride=None
    )
    assert ranges == [(4, 6), (6, 7), (7, 10)]


def test_span_column_ranges_with_stride():
    # stride=2, span_lengths=[1,1,1], context_length=5, seq_len=8.
    # Context starts at 3; spans at columns 0, 2, 4 within the context.
    ranges = _span_column_ranges(
        span_lengths=[1, 1, 1], context_length=5, seq_len=8, stride=2
    )
    assert ranges == [(3, 4), (5, 6), (7, 8)]


def test_span_column_ranges_clips_when_context_exceeds_seq():
    # context_length=10 > seq_len=4 → context_start would be -6, must clip.
    ranges = _span_column_ranges(
        span_lengths=[3, 3], context_length=10, seq_len=4, stride=None
    )
    for start, end in ranges:
        assert 0 <= start <= end <= 4


def test_compute_gt_attention_row_rows_sum_to_one():
    gt = compute_gt_attention_row(
        span_lengths=[2, 1, 3], context_length=6, seq_len=10, stride=None
    )
    assert gt.shape == (3, 10)
    assert np.allclose(gt.sum(axis=1), 1.0)


def test_compute_gt_attention_row_uniform_within_span():
    # Row 0 supports span of length 2 → each entry 0.5; elsewhere 0.
    gt = compute_gt_attention_row(
        span_lengths=[2, 1, 3], context_length=6, seq_len=10, stride=None
    )
    assert np.count_nonzero(gt[0]) == 2
    nonzero = gt[0][gt[0] > 0]
    assert np.allclose(nonzero, 0.5)
