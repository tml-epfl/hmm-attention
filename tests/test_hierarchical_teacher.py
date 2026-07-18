"""Sanity check for HierarchicalTeacher.

Verifies (in isolation from the trainer/dataset):
  1. Chunk table is invertible (encode -> decode is exact).
  2. Wrapper forward produces normalized surface distributions.
  3. Slot-1 predictions match slot-1 targets ~100% of the time (deterministic
     given slot 0 when the observed slot 0 uniquely identifies the hidden id).
  4. Slot-0 predictions align with the base teacher's argmax hidden id
     (via chunk_table[argmax_h, 0]).
  5. Autoregressive path is consistent with unrolled path at chunk-aligned positions.
"""
import torch
import torch.nn.functional as F

from src.teachers import HierarchicalTeacher, LinearARTeacher


def _hr(title: str) -> None:
    """Section header for readable log output."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def test_hierarchical_teacher():
    torch.manual_seed(0)
    hidden_dim = 8
    chunk_dim = 8
    chunk_size = 2
    window = 3
    span_lengths = [1, 1, 1]

    # Sharpness (formerly `temperature`) is now encoded via the base teacher's
    # weight scale: softmax(W·x / T) == softmax((W/T)·x), so scale = 1/T.
    base = LinearARTeacher.from_parameters(
        dim=hidden_dim,
        span_lengths=span_lengths,
        rank=hidden_dim,
        window=window,
        multiplicative_constant=1.7,
        scale=10.0,  # equivalent to temperature=0.1 in the old design
    )
    ht = HierarchicalTeacher(
        base_teacher=base,
        chunk_dim=chunk_dim,
        chunk_size=chunk_size,
        chunk_seed=0,
    )

    _hr("SETUP")
    print(f"hidden_dim = {hidden_dim}, chunk_dim = {chunk_dim}, chunk_size = {chunk_size}")
    print(f"window = {window}, span_lengths = {span_lengths}, base.scale = 10.0")
    print(f"base.context_length = {base.context_length}, ht.context_length = {ht.context_length}")
    print(f"base.span_lengths   = {base.span_lengths}, ht.span_lengths   = {ht.span_lengths}")

    _hr("CHUNK TABLE (hidden id -> chunk slot indices)")
    print("       slot0  slot1")
    for hid in range(hidden_dim):
        s0, s1 = ht._chunk_slot_indices[hid].tolist()
        print(f"  h={hid}:   {s0:>3}    {s1:>3}")
    # Slot-0 collisions: which hidden ids share slot-0 value?
    from collections import defaultdict
    slot0_groups: defaultdict = defaultdict(list)
    for hid in range(hidden_dim):
        slot0_groups[int(ht._chunk_slot_indices[hid, 0])].append(hid)
    collisions = {v: ids for v, ids in slot0_groups.items() if len(ids) > 1}
    print(f"slot-0 collisions: {dict(collisions) if collisions else 'none — every slot-0 value is unique'}")

    # ---- 1. chunk table invertibility ----
    _hr("[1] CHUNK TABLE INVERTIBILITY")
    hidden_ids = torch.arange(hidden_dim)
    surface = ht._chunk_table[hidden_ids]
    surface_flat = surface.reshape(1, hidden_dim * chunk_size, chunk_dim)
    decoded = ht._decode_chunk_aligned(surface_flat)
    decoded_ids = decoded.argmax(dim=-1).squeeze(0)
    print(f"input hidden ids : {hidden_ids.tolist()}")
    print(f"decoded hidden ids: {decoded_ids.tolist()}")
    assert torch.equal(decoded_ids, hidden_ids), (
        f"[FAIL] chunk table not invertible; got {decoded_ids}, expected {hidden_ids}"
    )
    print("[OK] round-trip exact")

    # ---- 2. build a valid surface batch by argmax-generating from base ----
    _hr("[2] BUILD DETERMINISTIC HIDDEN + SURFACE SEQUENCES")
    B = 4
    L_h = 10
    hidden = torch.zeros(B, L_h, hidden_dim)
    prefix_ids = torch.randint(0, hidden_dim, (B, window))
    hidden[:, :window, :] = F.one_hot(prefix_ids, num_classes=hidden_dim).float()
    for i in range(window, L_h):
        ctx = hidden[:, i - window : i, :]
        log_probs = base.next_token_log_probs(ctx)
        next_ids = log_probs.argmax(dim=-1)
        hidden[:, i, :] = F.one_hot(next_ids, num_classes=hidden_dim).float()

    hidden_ids_batched = hidden.argmax(dim=-1)  # (B, L_h)
    L_surf = L_h * chunk_size
    surface_full = ht._chunk_table[hidden_ids_batched]
    surface_full = surface_full.reshape(B, L_surf, chunk_dim)
    surface_slot_ids = surface_full.argmax(dim=-1)  # (B, L_surf)

    print(f"B = {B}, L_h = {L_h}, L_surf = {L_surf}")
    for b in range(B):
        print(f"  seq {b} hidden ids:  {hidden_ids_batched[b].tolist()}")
        print(f"  seq {b} surface ids: {surface_slot_ids[b].tolist()}")

    # ---- 3. run wrapper unrolled and check outputs ----
    _hr("[3] WRAPPER UNROLL")
    with torch.no_grad():
        log_probs, targets = ht.unroll(surface_full, return_targets=True)
    surface_probs = log_probs.exp()
    print(f"log_probs shape: {tuple(log_probs.shape)}   targets shape: {tuple(targets.shape)}")
    assert log_probs.shape == targets.shape, "[FAIL] shape mismatch"

    prob_sums = surface_probs.sum(dim=-1)
    print(f"per-position prob sum: min={prob_sums.min():.6f}  max={prob_sums.max():.6f}")
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4), (
        "[FAIL] surface probs don't sum to 1"
    )

    L_out = log_probs.shape[1]
    L_out_h = L_out // chunk_size
    pred_ids = log_probs.argmax(dim=-1)
    target_ids = targets.argmax(dim=-1)

    # per-example, per-position table
    _hr("[3b] PER-POSITION PREDICTIONS (batch 0)")
    print("pos | slot | target | argmax | top-prob | top-3 probs")
    for j in range(L_out):
        s = j % chunk_size
        top_probs, top_ids = surface_probs[0, j].topk(3)
        top_str = ", ".join(
            f"{int(i)}={float(p):.3f}" for i, p in zip(top_ids.tolist(), top_probs.tolist())
        )
        print(
            f" {j:>2} | {s:>4} | {int(target_ids[0, j]):>6} | {int(pred_ids[0, j]):>6} | "
            f"{float(surface_probs[0, j].max()):>8.4f} | {top_str}"
        )

    # ---- 4. per-slot accuracy ----
    _hr("[4] PER-SLOT ACCURACY (across all batch, all output positions)")
    slot_ids = torch.arange(L_out) % chunk_size
    for s in range(chunk_size):
        mask = slot_ids == s
        correct = (pred_ids[:, mask] == target_ids[:, mask]).float().mean().item()
        print(f"  slot {s}: {correct:.4f}   ({int((pred_ids[:, mask] == target_ids[:, mask]).sum())}/{int(mask.sum()) * B} matches)")
        if s == chunk_size - 1:
            assert correct > 0.95, f"[FAIL] slot {s} accuracy {correct:.3f} < 0.95"

    # ---- 5. slot-0 check against base teacher argmax ----
    _hr("[5] SLOT-0 CROSS-CHECK vs base_teacher argmax")
    with torch.no_grad():
        base_log_probs, _ = base.unroll(hidden, return_targets=True)
    base_argmax_h = base_log_probs.argmax(dim=-1)  # (B, L_out_h)
    expected_slot0 = ht._chunk_slot_indices[base_argmax_h, 0]

    slot0_positions = torch.arange(0, L_out, chunk_size)
    wrapper_slot0 = pred_ids[:, slot0_positions]
    print("For batch 0, per output chunk:")
    print("  chunk | base argmax_h | chunk_table[h,0] | wrapper argmax | target slot0")
    for k in range(L_out_h):
        h_star = int(base_argmax_h[0, k])
        exp0 = int(ht._chunk_slot_indices[h_star, 0])
        wrap = int(wrapper_slot0[0, k])
        tgt = int(target_ids[0, k * chunk_size])
        marker = "✓" if wrap == exp0 == tgt else ("~" if wrap == exp0 else "✗")
        print(f"    {k:>3}   |     {h_star:>3}       |       {exp0:>3}         |      {wrap:>3}        |     {tgt:>3}   {marker}")

    match = (wrapper_slot0 == expected_slot0).float().mean().item()
    print(f"\nwrapper-slot0 == chunk_table[base_argmax_h, 0]: {match:.4f}")
    slot0_correct = (pred_ids[:, slot0_positions] == target_ids[:, slot0_positions]).float().mean().item()
    print(f"wrapper-slot0 == target                       : {slot0_correct:.4f}")
    assert match > 0.95, f"[FAIL] wrapper slot-0 argmax disagrees with base teacher argmax: {match:.3f}"
    assert slot0_correct > 0.95, f"[FAIL] slot-0 vs deterministic target: {slot0_correct:.3f}"

    # ---- 6. autoregressive path consistency ----
    _hr("[6] AR vs UNROLLED CONSISTENCY (batch 0, chunk-boundary positions)")
    b = 0
    print("boundary | AR vs unrolled max|diff|")
    for k in range(3):
        boundary = (window + k) * chunk_size
        pref = surface_full[b : b + 1, :boundary, :]
        with torch.no_grad():
            ar_out = ht.predict_next(pref)
        unrolled_pred = log_probs[b, k * chunk_size]
        max_diff = (ar_out.squeeze(0) - unrolled_pred).abs().max().item()
        print(f"   {boundary:>4}   |   {max_diff:.6e}")
        assert max_diff < 1e-4, f"[FAIL] AR/unrolled mismatch at boundary {boundary}: {max_diff}"

    print("\n[ALL PASS]")
