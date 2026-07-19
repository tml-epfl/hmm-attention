"""Opt-in section timer for identifying training-loop bottlenecks.

Enable by setting `HMM_PROFILE=1` in the environment. When disabled, every
context manager is a cheap no-op so instrumented code paths pay nothing in
normal runs.

GPU sections are timed with `torch.cuda.Event`s recorded asynchronously — we
never synchronize per section (that would serialize the work we are trying to
measure). `report()` performs a single `torch.cuda.synchronize()` before
reading elapsed times.
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


class Profiler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._cuda_events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        self._cpu_ms: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def cuda(self, name: str):
        if not self.enabled or not torch.cuda.is_available():
            # Fall back to CPU timing when CUDA is not available so the
            # section is still visible in the report.
            if self.enabled:
                with self.cpu(name):
                    yield
            else:
                yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda_events[name].append((start, end))

    @contextmanager
    def cpu(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_ms[name].append((time.perf_counter() - t0) * 1000.0)

    def report(self, reset: bool = True) -> Dict[str, dict]:
        if not self.enabled:
            return {}
        if torch.cuda.is_available() and self._cuda_events:
            torch.cuda.synchronize()
        summary: Dict[str, dict] = {}
        for name, pairs in self._cuda_events.items():
            times_ms = [s.elapsed_time(e) for s, e in pairs]
            summary[name] = self._stats(times_ms)
        for name, times_ms in self._cpu_ms.items():
            # If a name was recorded on both paths (CUDA + CPU fallback),
            # keep them separate by suffixing — but in practice a single run
            # only hits one branch, so we merge by summing.
            if name in summary:
                summary[name]["count"] += len(times_ms)
                summary[name]["total_ms"] += sum(times_ms)
                summary[name]["mean_ms"] = (
                    summary[name]["total_ms"] / summary[name]["count"]
                )
            else:
                summary[name] = self._stats(times_ms)
        if reset:
            self._cuda_events.clear()
            self._cpu_ms.clear()
        return summary

    @staticmethod
    def _stats(times_ms: List[float]) -> dict:
        n = len(times_ms)
        total = sum(times_ms)
        return {
            "count": n,
            "total_ms": total,
            "mean_ms": total / n if n else 0.0,
        }


_singleton: Optional[Profiler] = None


def get_profiler() -> Profiler:
    global _singleton
    if _singleton is None:
        flag = os.environ.get("HMM_PROFILE", "0")
        enabled = flag not in ("", "0", "false", "False")
        _singleton = Profiler(enabled=enabled)
    return _singleton


def format_report(summary: Dict[str, dict]) -> str:
    if not summary:
        return "(profiler disabled or no data collected)"
    total_all = sum(v["total_ms"] for v in summary.values())
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["total_ms"])
    header = (
        f"{'section':<32} {'count':>7} {'mean_ms':>10} "
        f"{'total_ms':>12} {'share':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, s in rows:
        share = s["total_ms"] / total_all * 100 if total_all else 0.0
        lines.append(
            f"{name:<32} {s['count']:>7d} {s['mean_ms']:>10.3f} "
            f"{s['total_ms']:>12.1f} {share:>6.1f}%"
        )
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':<32} {'':>7} {'':>10} {total_all:>12.1f} {100.0:>6.1f}%")
    return "\n".join(lines)
