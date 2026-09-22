# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import Any, Dict

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.detector import StragglerReport


logger = get_logger(__name__)


def report_to_tracking(
    args: Any,
    report: StragglerReport,
    step_key: str = "rollout/step",
) -> Dict[str, Any]:
    """Publish straggler metrics through Relax tracking_utils when available."""
    metrics = report.to_metrics()
    metrics[step_key] = report.step
    try:
        from relax.utils import tracking_utils

        tracking_utils.log(args, metrics, step_key=step_key)
    except Exception as exc:  # pragma: no cover - best-effort sink
        logger.warning(f"failed to publish straggler metrics via tracking_utils: {exc}")
    return metrics


def format_human_summary(report: StragglerReport) -> str:
    if not report.hints:
        return f"step={report.step} no straggler above threshold"
    return f"step={report.step} " + " | ".join(report.hints)
