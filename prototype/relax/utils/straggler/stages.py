# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class StageTimer:
    """Coarse-grained stage timer using CUDA events when available.

    Designed for low overhead:
    - CUDA Event record is cheap; host sync happens only when reading.
    - Non-sampled steps should not instantiate/use this timer at all.
    """

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self._durations_ms: Dict[str, float] = {}
        self._active: Dict[str, tuple] = {}
        self._use_cuda = bool(
            self.sync_cuda and torch.cuda.is_available() and torch.cuda.current_device() >= 0
        )

    def reset(self) -> None:
        self._durations_ms.clear()
        self._active.clear()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._active[name] = ("cuda", start, end)
            try:
                yield
            finally:
                end.record()
        else:
            start_ts = time.perf_counter()
            self._active[name] = ("cpu", start_ts, None)
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                self._durations_ms[name] = self._durations_ms.get(name, 0.0) + elapsed_ms
                self._active.pop(name, None)

    def finish(self) -> Dict[str, float]:
        """Synchronize pending CUDA events and return accumulated ms per stage."""
        for name, payload in list(self._active.items()):
            kind = payload[0]
            if kind == "cuda":
                _, start, end = payload
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                self._durations_ms[name] = self._durations_ms.get(name, 0.0) + elapsed_ms
            self._active.pop(name, None)
        return dict(self._durations_ms)

    def get_durations_ms(self) -> Dict[str, float]:
        return dict(self._durations_ms)
