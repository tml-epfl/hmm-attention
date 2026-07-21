"""Flock-based work coordination tests.

Verifies the primitives one process at a time (single-process tests) plus a
cross-process contention test using `multiprocessing` to prove flock actually
serializes access across processes.
"""

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from src.trainer import lock as work_lock


@pytest.fixture(autouse=True)
def _reset_lock_state():
    """Reset the module-level fd cache between tests so each starts fresh."""
    work_lock.release()
    yield
    work_lock.release()


def test_acquire_on_empty_dir_succeeds(tmp_path):
    assert work_lock.try_acquire(tmp_path) is True


def test_second_acquire_in_same_process_raises(tmp_path):
    work_lock.try_acquire(tmp_path)
    with pytest.raises(RuntimeError, match="already holds a lock"):
        work_lock.try_acquire(tmp_path)


def test_release_then_reacquire(tmp_path):
    assert work_lock.try_acquire(tmp_path)
    work_lock.release()
    assert work_lock.try_acquire(tmp_path)


def test_is_completed_absent_by_default(tmp_path):
    assert work_lock.is_completed(tmp_path) is False


def test_mark_completed_creates_done_marker(tmp_path):
    work_lock.mark_completed(tmp_path)
    assert work_lock.is_completed(tmp_path)
    assert (tmp_path / "done").exists()


def test_mark_completed_is_idempotent(tmp_path):
    work_lock.mark_completed(tmp_path)
    work_lock.mark_completed(tmp_path)  # should not raise
    assert work_lock.is_completed(tmp_path)


# ---- cross-process contention ------------------------------------------------


def _child_acquire(ckpt_dir_str: str, hold_seconds: float, result_queue):
    """Run in a child process: try to acquire, hold, then report result."""
    from src.trainer import lock as work_lock

    got = work_lock.try_acquire(Path(ckpt_dir_str))
    result_queue.put(("acquired", got))
    if got:
        time.sleep(hold_seconds)


def test_flock_blocks_other_process(tmp_path):
    """The core guarantee: two processes cannot both hold the lock."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    child = ctx.Process(target=_child_acquire, args=(str(tmp_path), 2.0, q))
    child.start()
    # Wait for child to acquire.
    tag, got = q.get(timeout=5)
    assert tag == "acquired" and got is True

    # Now parent tries — must fail because child holds it.
    assert work_lock.try_acquire(tmp_path) is False

    child.join(timeout=5)


def test_flock_releases_on_process_death(tmp_path):
    """Kernel releases the fd when the holding process exits, no cleanup."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # Child acquires and holds for 0.1s then exits normally.
    child = ctx.Process(target=_child_acquire, args=(str(tmp_path), 0.1, q))
    child.start()
    tag, got = q.get(timeout=5)
    assert got is True
    child.join(timeout=5)
    assert not child.is_alive()

    # Parent should now be able to take it — no timeout, no cleanup.
    assert work_lock.try_acquire(tmp_path) is True
