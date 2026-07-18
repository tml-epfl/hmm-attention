import torch

from src.trainer import MetricRegistry, TeacherEvaluator


def _make_data(teacher, batch=2, length=None):
    """Build a valid one-hot batch of shape (batch, teacher.context_length + 4, dim)."""
    T = length if length is not None else teacher.context_length + 4
    ids = torch.randint(0, teacher.dim, (batch, T))
    return torch.nn.functional.one_hot(ids, num_classes=teacher.dim).float()


# ---- construction ------------------------------------------------------------


def test_evaluator_for_non_ar_teacher_is_empty(device):
    ev = TeacherEvaluator(teacher=object(), device=device)  # non-ARTeacher
    assert ev.is_ar is False
    assert ev._teacher_by_k == {}
    assert ev.prefix_ks == []
    assert ev.metric_keys() == []


def test_evaluator_builds_lag_restricted_cache(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    # window=2 → cache holds only k=1 (k=window is a no-op, skipped).
    assert set(ev._teacher_by_k.keys()) == {1}
    assert ev.prefix_ks == [1, 2]


def test_metric_keys_include_kl_teacher_and_per_k(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    keys = set(ev.metric_keys())
    assert "kl/teacher_train" in keys
    assert "kl/teacher_val" in keys
    assert "kl/teacher_k1_train" in keys
    assert "kl/teacher_k2_val" in keys


# ---- _align_data -------------------------------------------------------------


def test_align_data_trims_by_context_length_diff(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    restricted = ev._teacher_by_k[1]
    trimmed = ev._align_data(data, restricted)
    diff = tiny_teacher.context_length - restricted.context_length
    assert trimmed.shape[1] == data.shape[1] - diff


def test_align_data_noop_when_full_teacher(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    assert ev._align_data(data, tiny_teacher).shape == data.shape


# ---- run ---------------------------------------------------------------------


def test_run_unnormalized_returns_log_probs(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    out, log_probs, targets = ev.run(data, prefix=-1, normalize=False)
    # Unnormalized: out is log-probs (same object as log_probs).
    assert torch.equal(out, log_probs)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones_like(log_probs[..., 0]), atol=1e-5)


def test_run_normalized_returns_probs_and_log_probs(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    probs, log_probs, _ = ev.run(data, prefix=-1, normalize=True)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5)
    assert torch.allclose(probs, log_probs.exp(), atol=1e-5)


# ---- update_kl_metrics -------------------------------------------------------


def test_update_kl_metrics_populates_registry(tiny_teacher, tiny_student, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    reg = MetricRegistry()
    for key in ev.metric_keys():
        from src.metrics import LossMetric
        reg.register(key, LossMetric())

    data = _make_data(tiny_teacher)
    # Student output shape must match teacher output shape.
    out_teacher, _, _ = ev.run(data, prefix=-1, normalize=False)
    fake_student_out = out_teacher.clone()  # perfect student → KL = 0.

    ev.update_kl_metrics(fake_student_out, data, split="train", metrics=reg)
    assert reg["kl/teacher_train"].compute() >= 0
    assert reg["kl/teacher_k1_train"].compute() >= 0
