"""
Lightweight per-phase profiling utilities for the GRPO training loop.

`CudaPhaseTimer` accumulates wall-clock per named phase using cuda.Event
timers. Events are submitted to the stream non-blockingly and only
synchronized when the caller requests a flush (e.g. once per logging step),
so the per-call overhead during training is negligible.

Usage:

    timer = CudaPhaseTimer()
    with timer.time("rollout_text"):
        ...
    # ...later, at log time:
    stats = timer.flush()   # dict of phase -> (total_ms, count, mean_ms)

The class also supports `enabled=False` for a true zero-cost no-op so the
same call sites can be left in place when profiling is off.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


class CudaPhaseTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )

    @contextlib.contextmanager
    def time(self, phase: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events[phase].append((start, end))

    def flush(self) -> Dict[str, Tuple[float, int, float]]:
        """Synchronize, accumulate, and clear. Returns {phase: (total_ms, count, mean_ms)}."""
        if not self.enabled or not self._events:
            return {}
        torch.cuda.synchronize()
        out: Dict[str, Tuple[float, int, float]] = {}
        for phase, pairs in self._events.items():
            total = 0.0
            for s, e in pairs:
                total += s.elapsed_time(e)
            n = len(pairs)
            out[phase] = (total, n, total / max(n, 1))
        self._events.clear()
        return out

    def reset(self):
        self._events.clear()


def make_profiler(
    output_dir: str,
    active_steps: int,
    wait_steps: int = 1,
    warmup_steps: int = 1,
):
    """Build a `torch.profiler.profile` configured to save a Chrome trace.

    The schedule is `wait → warmup → active → done` (single repeat). The
    caller must call `prof.step()` once per training step.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    def _on_trace_ready(prof):
        path = os.path.join(output_dir, f"profile_trace.json")
        prof.export_chrome_trace(path)

    schedule = torch.profiler.schedule(
        wait=wait_steps,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=_on_trace_ready,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    return prof


def format_phase_table(stats: Dict[str, Tuple[float, int, float]]) -> str:
    """Human-readable one-line-per-phase summary for the training log."""
    if not stats:
        return "phase_timings: (none)"
    items = sorted(stats.items(), key=lambda kv: -kv[1][0])
    total = sum(v[0] for v in stats.values())
    parts = []
    for phase, (total_ms, count, mean_ms) in items:
        pct = 100.0 * total_ms / max(total, 1e-9)
        parts.append(f"{phase}={total_ms:.0f}ms({pct:.0f}%,n={count})")
    return "phase_timings: " + " ".join(parts)
