# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lightweight online straggler analysis for Relax training.

This package provides coarse-grained, low-overhead stage timing and cross-rank
straggler detection suitable for continuous production training.
"""

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import StragglerAnalyzer, StragglerReport
from relax.utils.straggler.stages import StageTimer

__all__ = [
    "StragglerConfig",
    "StragglerAnalyzer",
    "StragglerReport",
    "StageTimer",
]
