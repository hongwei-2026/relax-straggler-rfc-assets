# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StragglerConfig:
    """Configuration for online straggler analysis.

    Attributes:
        enabled: Master switch. When False, all APIs become no-ops.
        sample_interval: Profile one step every N training steps. Larger values
            reduce overhead; 1 profiles every step.
        relative_threshold: A rank is marked as straggler when its stage time
            exceeds ``mean * relative_threshold``.
        absolute_ms_threshold: Optional absolute delta over the mean (ms). A
            rank is marked if either relative or absolute threshold trips.
        sync_cuda: If True, record CUDA events and synchronize when reading
            durations. Required for accurate GPU-side stage timing.
        report_top_k: Number of slowest ranks to include in the report.
        stages: Coarse stage names to track.
    """

    enabled: bool = True
    sample_interval: int = 10
    relative_threshold: float = 1.2
    absolute_ms_threshold: float = 5.0
    sync_cuda: bool = True
    report_top_k: int = 3
    stages: tuple[str, ...] = (
        "forward",
        "backward",
        "communication",
        "optimizer",
        "attention",
        "moe",
    )

    def should_sample(self, step: int) -> bool:
        if not self.enabled:
            return False
        if self.sample_interval <= 1:
            return True
        return step >= 0 and (step % self.sample_interval == 0)
