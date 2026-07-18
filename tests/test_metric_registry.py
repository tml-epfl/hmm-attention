import pytest

from src.trainer import MetricRegistry


def test_register_adds_key():
    reg = MetricRegistry()
    reg.register("student/train_loss", object())
    assert "student/train_loss" in reg
    assert len(reg) == 1


def test_register_duplicate_raises_value_error():
    reg = MetricRegistry()
    reg.register("k", object())
    with pytest.raises(ValueError, match="already registered"):
        reg.register("k", object())


def test_getitem_unknown_raises_key_error_with_suggestion():
    reg = MetricRegistry()
    reg.register("student/train_loss", object())
    with pytest.raises(KeyError, match="Did you mean.*student/train_loss"):
        reg["student/train_los"]  # noqa: B015


def test_getitem_unknown_no_close_match_omits_hint():
    reg = MetricRegistry()
    reg.register("student/train_loss", object())
    with pytest.raises(KeyError) as excinfo:
        reg["totally_unrelated_key_1234567890"]  # noqa: B015
    assert "Did you mean" not in str(excinfo.value)


def test_dict_compat_interface():
    reg = MetricRegistry()
    a, b = object(), object()
    reg.register("a", a)
    reg.register("b", b)

    assert list(reg) == ["a", "b"]
    assert dict(reg.items()) == {"a": a, "b": b}
    assert set(reg.values()) == {a, b}


def test_pop_returns_default_when_missing():
    reg = MetricRegistry()
    sentinel = object()
    assert reg.pop("missing", sentinel) is sentinel


def test_pop_removes_and_returns_value():
    reg = MetricRegistry()
    a = object()
    reg.register("a", a)
    assert reg.pop("a") is a
    assert "a" not in reg
