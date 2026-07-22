"""Sanity checks for AttentionARTeacher + the adaptive/bounded ARTeacher API.

Verifies:
  1. next_token_log_probs returns normalized log-probs of the right shape.
  2. predict_next is the *inherited* base method (uniform API, no override).
  3. context_length/burn_in report the right thing in each mode.
  4. unbounded predict_next depends on far-back context (not a fixed window).
  5. unroll is consistent with per-position predict_next (unbounded).
  6. bounded mode agrees with the base sliding-window unroll.
  7. It composes as a base_teacher inside HierarchicalTeacher, and an unbounded
     base actually sees full history through the wrapper (no neutralization).
"""
import torch

from src.teachers import AttentionARTeacher, HierarchicalTeacher
from src.teachers.base import ADAPTIVE, ARTeacher


def _log_probs_normalized(lp: torch.Tensor) -> bool:
    return torch.allclose(lp.exp().sum(-1), torch.ones(lp.shape[:-1]), atol=1e-5)


def _flip(one_hot_token: torch.Tensor) -> torch.Tensor:
    dim = one_hot_token.shape[-1]
    return torch.eye(dim)[(one_hot_token.argmax() + 1) % dim]


def test_uniform_predict_next_and_reported_lengths():
    """predict_next is inherited (not overridden); lengths report per mode."""
    assert AttentionARTeacher.predict_next is ARTeacher.predict_next

    adaptive = AttentionARTeacher(dim=6, hidden_dim=16, unbounded=True, burn_in=2, seed=0)
    assert adaptive.is_adaptive and adaptive.context_length == ADAPTIVE
    assert adaptive.burn_in == 2

    bounded = AttentionARTeacher(dim=6, hidden_dim=16, unbounded=False, window=3, seed=0)
    assert not bounded.is_adaptive and bounded.context_length == 3
    # Invariant: burn_in == context_length for non-adaptive teachers.
    assert bounded.burn_in == bounded.context_length == 3


def test_bounded_burn_in_pinned_to_window():
    """A bounded teacher may not diverge burn_in from context_length."""
    import pytest

    with pytest.raises(ValueError, match="burn_in"):
        AttentionARTeacher(dim=6, unbounded=False, window=3, burn_in=5)
    # Matching burn_in is fine (redundant but allowed).
    t = AttentionARTeacher(dim=6, unbounded=False, window=3, burn_in=3)
    assert t.burn_in == t.context_length == 3


def test_shapes_and_normalization():
    torch.manual_seed(0)
    dim = 6
    t = AttentionARTeacher(dim=dim, hidden_dim=16, num_heads=2, seed=0)
    ctx = torch.eye(dim)[torch.randint(0, dim, (4, 5))]  # (B=4, T=5, dim)
    lp = t.next_token_log_probs(ctx)
    assert lp.shape == (4, dim)
    assert _log_probs_normalized(lp)


def test_unbounded_uses_far_context():
    """Changing a *distant* token must change the prediction under unbounded."""
    torch.manual_seed(0)
    dim = 6
    t = AttentionARTeacher(dim=dim, hidden_dim=16, unbounded=True, seed=1)
    seq = torch.eye(dim)[torch.randint(0, dim, (1, 12))]
    lp0 = t.predict_next(seq)
    seq2 = seq.clone()
    seq2[0, 0] = _flip(seq[0, 0])  # flip the very first token
    lp1 = t.predict_next(seq2)
    assert not torch.allclose(lp0, lp1, atol=1e-4), "far token had no effect"


def test_unroll_matches_predict_next_unbounded():
    torch.manual_seed(0)
    dim, bi = 5, 2
    t = AttentionARTeacher(dim=dim, hidden_dim=12, unbounded=True, burn_in=bi, seed=2)
    seq = torch.eye(dim)[torch.randint(0, dim, (3, 9))]  # (B=3, L=9, dim)
    unrolled = t.unroll(seq)  # (B, L - burn_in, dim)
    assert unrolled.shape == (3, 9 - bi, dim)
    for j in range(9 - bi):
        step = t.predict_next(seq[:, : bi + j, :])
        assert torch.allclose(unrolled[:, j, :], step, atol=1e-5), f"mismatch at j={j}"


def test_bounded_matches_base_unroll():
    torch.manual_seed(0)
    dim, w = 5, 3
    t = AttentionARTeacher(dim=dim, hidden_dim=12, unbounded=False, window=w, seed=3)
    seq = torch.eye(dim)[torch.randint(0, dim, (2, 8))]
    lp = t.unroll(seq)
    assert lp.shape == (2, 8 - w, dim)
    assert _log_probs_normalized(lp)
    step = t.predict_next(seq[:, :w, :])  # bounded: only the last w tokens matter
    assert torch.allclose(lp[:, 0, :], step, atol=1e-5)


def test_composes_with_hierarchical_adaptive_base():
    torch.manual_seed(0)
    dim = 8
    base = AttentionARTeacher(dim=dim, hidden_dim=16, unbounded=True, burn_in=2, seed=4)
    ht = HierarchicalTeacher(base_teacher=base, chunk_dim=8, chunk_size=2, chunk_seed=0)
    assert ht.is_adaptive and ht.context_length == ADAPTIVE
    assert ht.burn_in == base.burn_in * ht.chunk_size

    surf = ht.sample_surface_prefix(ht.burn_in + 3 * ht.chunk_size)  # chunk-aligned
    lp = ht.predict_next(surf.unsqueeze(0))
    assert lp.shape == (1, ht.chunk_dim)
    assert _log_probs_normalized(lp)

    # Full history reaches the unbounded base through the wrapper: replacing the
    # distant first chunk with a *different* hidden id's chunk must change the
    # surface prediction (this fails if the wrapper truncates context).
    h0 = ht.decode_chunk_aligned(surf.unsqueeze(0))[0, 0].argmax().item()
    new_h0 = (h0 + 1) % ht.hidden_dim
    surf2 = surf.clone()
    surf2[: ht.chunk_size] = ht._chunk_table[new_h0, 0]  # valid chunk, different hidden id
    lp2 = ht.predict_next(surf2.unsqueeze(0))
    assert not torch.allclose(lp, lp2, atol=1e-4), "distant chunk had no effect through wrapper"


if __name__ == "__main__":
    test_uniform_predict_next_and_reported_lengths()
    test_bounded_burn_in_pinned_to_window()
    test_shapes_and_normalization()
    test_unbounded_uses_far_context()
    test_unroll_matches_predict_next_unbounded()
    test_bounded_matches_base_unroll()
    test_composes_with_hierarchical_adaptive_base()
    print("all AttentionARTeacher sanity checks passed")
