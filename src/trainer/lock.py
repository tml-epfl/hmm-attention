"""Per-config work coordination for parallel workers.

Multiple Runai pods / ray workers can launch the same sweep simultaneously.
Each config's checkpoint dir gets:

- `lock` — held via `fcntl.flock` (kernel-level exclusive lock) for as long
  as the working process is alive. On process death (including SIGKILL), the
  kernel releases the fd → lock frees automatically. No heartbeat, no
  timeout, no stale-detection code.
- `done` — plain file, present iff the config has completed successfully.

Acquire policy:
1. If `done` exists → skip this config (already finished by someone).
2. Else try `flock(LOCK_EX | LOCK_NB)`. If it fails with `BlockingIOError`,
   another live worker holds it → skip.
3. Else we hold the lock; proceed with training. On graceful completion, write
   `done`.

Skipped workers return early and exit cleanly (no `wandb.init`, no state).
Ray / Hydra multirun sees the task complete as a no-op and moves on.
"""

import errno
import fcntl
import logging
import os
from pathlib import Path
from typing import Optional

LOCK_FILENAME = "lock"
DONE_FILENAME = "done"

# Module-level cache prevents the fd from being garbage-collected (which
# would close it and release the lock while training is running). One process
# only ever holds one config's lock, so a single slot suffices.
_held_lock_fd: Optional[int] = None


def is_completed(ckpt_dir: Path) -> bool:
    return (ckpt_dir / DONE_FILENAME).exists()


def try_acquire(ckpt_dir: Path) -> bool:
    """Return True if we now hold the lock; False if another worker has it.

    The fd is kept alive at module scope; the kernel releases the lock when
    the process exits (or when `release()` is called explicitly).
    """
    global _held_lock_fd
    if _held_lock_fd is not None:
        raise RuntimeError(
            "This process already holds a lock. `try_acquire` is single-shot."
        )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lock_path = ckpt_dir / LOCK_FILENAME

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    except OSError as e:
        # ENOLCK / EACCES / etc. — filesystem doesn't support flock properly.
        # Fail loudly rather than silently corrupting the sweep by letting
        # two workers on one config.
        os.close(fd)
        raise RuntimeError(
            f"flock failed on {lock_path} (errno={errno.errorcode.get(e.errno, e.errno)}). "
            "Check that the filesystem supports advisory locks."
        ) from e

    # Write pid + hostname for post-mortem debugging (not used for acquire).
    os.write(fd, f"{os.getpid()}@{os.uname().nodename}\n".encode())
    os.fsync(fd)
    _held_lock_fd = fd
    return True


def release() -> None:
    """Explicit release. Optional — process exit releases automatically."""
    global _held_lock_fd
    if _held_lock_fd is None:
        return
    try:
        fcntl.flock(_held_lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(_held_lock_fd)
        _held_lock_fd = None


def mark_completed(ckpt_dir: Path) -> None:
    """Write the `done` marker atomically. Idempotent."""
    done_path = ckpt_dir / DONE_FILENAME
    tmp = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp.write_text("")
    os.replace(tmp, done_path)
    logging.getLogger().info(f"Marked config as completed: {done_path}")
