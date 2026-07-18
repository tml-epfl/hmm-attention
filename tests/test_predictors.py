import pytest
import torch
import torch.nn.functional as F

from src.predictors import (
    ClassificationPredictor,
    HierarchicalPredictor,
    build_predictor,
)


# ---- enforcement -------------------------------------------------------------


def test_classification_predictor_rejects_teacher_without_predict_next():
    with pytest.raises(TypeError, match="predict_next"):
        ClassificationPredictor(object())


def test_hierarchical_predictor_rejects_non_hierarchical_teacher(tiny_teacher):
    with pytest.raises(TypeError, match="HierarchicalTeacher"):
        HierarchicalPredictor(tiny_teacher)


def test_hierarchical_predictor_accepts_hierarchical_teacher(tiny_hier_teacher):
    HierarchicalPredictor(tiny_hier_teacher)  # no raise


# ---- factory -----------------------------------------------------------------


def test_build_predictor_returns_classification_for_linear(tiny_teacher):
    p = build_predictor(tiny_teacher, kind="classification")
    assert isinstance(p, ClassificationPredictor)
    assert not isinstance(p, HierarchicalPredictor)


def test_build_predictor_returns_hierarchical_for_hier(tiny_hier_teacher):
    p = build_predictor(tiny_hier_teacher, kind="classification")
    assert isinstance(p, HierarchicalPredictor)


# ---- sampling ----------------------------------------------------------------


def test_sample_next_returns_one_hot(tiny_predictor, tiny_teacher):
    prefix = torch.zeros(tiny_teacher.context_length, tiny_teacher.dim)
    prefix[..., 0] = 1.0
    sample = tiny_predictor.sample_next(prefix)
    assert sample.shape == (tiny_teacher.dim,)
    assert torch.isclose(sample.sum(), torch.tensor(1.0))
    assert ((sample == 0) | (sample == 1)).all()


def test_argmax_sample_is_deterministic(tiny_teacher):
    p = ClassificationPredictor(tiny_teacher, argmax=True)
    prefix = torch.zeros(tiny_teacher.context_length, tiny_teacher.dim)
    prefix[..., 0] = 1.0
    a = p.sample_next(prefix)
    b = p.sample_next(prefix)
    assert torch.equal(a, b)


# ---- random_burn_in ----------------------------------------------------------


def test_hierarchical_burn_in_produces_valid_chunks(tiny_hier_predictor, tiny_hier_teacher):
    # Must be a multiple of chunk_size.
    length = tiny_hier_teacher.chunk_size * 3
    burn = tiny_hier_predictor.random_burn_in(length)
    assert burn.shape == (length, tiny_hier_teacher.chunk_dim)
    # Each row is a one-hot.
    assert torch.allclose(burn.sum(dim=-1), torch.ones(length))
    # Decoding back to hidden ids should succeed (i.e. every chunk matches
    # some row in the chunk table).
    decoded = tiny_hier_teacher._decode_chunk_aligned(burn.unsqueeze(0))
    assert decoded.shape[1] == length // tiny_hier_teacher.chunk_size


def test_classification_predictor_has_no_burn_in_override(tiny_predictor):
    assert tiny_predictor.random_burn_in(4) is None
