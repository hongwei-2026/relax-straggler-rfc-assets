# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.stages import StageTimer


logger = get_logger(__name__)


@dataclass
class StragglerReport:
    """Cross-rank straggler analysis result for one sampled step."""

    step: int
    world_size: int
    rank: int
    stage_ms: Dict[str, float]
    stage_mean_ms: Dict[str, float]
    stage_max_ms: Dict[str, float]
    straggler_ranks: Dict[str, List[int]] = field(default_factory=dict)
    slowdown_ratio: Dict[str, float] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)

    def to_metrics(self) -> Dict[str, Any]:
        """Flatten into MetricsClient-friendly metric dict."""
        metrics: Dict[str, Any] = {
            "straggler/step": self.step,
            "straggler/world_size": self.world_size,
            "straggler/local_rank": self.rank,
        }
        for stage, ms in self.stage_ms.items():
            metrics[f"straggler/{stage}/local_ms"] = ms
            metrics[f"straggler/{stage}/mean_ms"] = self.stage_mean_ms.get(stage, 0.0)
            metrics[f"straggler/{stage}/max_ms"] = self.stage_max_ms.get(stage, 0.0)
            metrics[f"straggler/{stage}/slowdown_ratio"] = self.slowdown_ratio.get(stage, 1.0)
            ranks = self.straggler_ranks.get(stage, [])
            metrics[f"straggler/{stage}/num_stragglers"] = len(ranks)
            if ranks:
                metrics[f"straggler/{stage}/top_rank"] = ranks[0]
        metrics["straggler/hints"] = "; ".join(self.hints) if self.hints else ""
        return metrics


class StragglerAnalyzer:
    """Online straggler analyzer for continuous training."""

    def __init__(self, config: Optional[StragglerConfig] = None):
        self.config = config or StragglerConfig()
        self._timer: Optional[StageTimer] = None
        self._sampling = False

    @property
    def is_sampling(self) -> bool:
        return self._sampling

    def begin_step(self, step: int) -> bool:
        self._sampling = self.config.should_sample(step)
        if not self._sampling:
            self._timer = None
            return False
        self._timer = StageTimer(sync_cuda=self.config.sync_cuda)
        return True

    def stage(self, name: str):
        if not self._sampling or self._timer is None:
            from contextlib import nullcontext

            return nullcontext()
        return self._timer.stage(name)

    def end_step(self, step: int) -> Optional[StragglerReport]:
        if not self._sampling or self._timer is None:
            return None

        local = self._timer.finish()
        for stage in self.config.stages:
            local.setdefault(stage, 0.0)

        gathered = self._all_gather_stage_ms(local)
        report = self._build_report(step=step, local=local, gathered=gathered)
        self._timer = None
        self._sampling = False

        if report.hints and self._is_rank0():
            logger.info(f"straggler step={step}: {'; '.join(report.hints)}")
        return report

    def analyze_local_only(self, step: int, stage_ms: Mapping[str, float]) -> StragglerReport:
        """Build a report without distributed collectives (for tests/offline)."""
        local = {k: float(stage_ms.get(k, 0.0)) for k in self.config.stages}
        for k, v in stage_ms.items():
            local.setdefault(k, float(v))
        gathered = [local]
        return self._build_report(step=step, local=local, gathered=gathered)

    def _all_gather_stage_ms(self, local: Dict[str, float]) -> List[Dict[str, float]]:
        if not (dist.is_available() and dist.is_initialized()):
            return [local]

        stages = list(self.config.stages)
        # NCCL process groups require CUDA tensors for all_gather.
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        values = torch.tensor(
            [local.get(s, 0.0) for s in stages],
            dtype=torch.float64,
            device=device,
        )
        world = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(values) for _ in range(world)]
        dist.all_gather(gathered_tensors, values)
        result: List[Dict[str, float]] = []
        for tensor in gathered_tensors:
            arr = tensor.detach().cpu().tolist()
            result.append({stages[i]: float(arr[i]) for i in range(len(stages))})
        return result

    def _build_report(
        self,
        step: int,
        local: Mapping[str, float],
        gathered: Sequence[Mapping[str, float]],
    ) -> StragglerReport:
        world = len(gathered)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        stage_mean: Dict[str, float] = {}
        stage_max: Dict[str, float] = {}
        straggler_ranks: Dict[str, List[int]] = {}
        slowdown_ratio: Dict[str, float] = {}
        hints: List[str] = []

        for stage in self.config.stages:
            vals = [float(g.get(stage, 0.0)) for g in gathered]
            mean = sum(vals) / max(len(vals), 1)
            mx = max(vals) if vals else 0.0
            stage_mean[stage] = mean
            stage_max[stage] = mx
            slowdown_ratio[stage] = (mx / mean) if mean > 1e-9 else 1.0

            flagged: List[tuple[int, float]] = []
            for r, v in enumerate(vals):
                if mean <= 1e-9:
                    continue
                rel = v >= mean * self.config.relative_threshold
                abs_hit = (v - mean) >= self.config.absolute_ms_threshold
                if rel or abs_hit:
                    flagged.append((r, v))
            flagged.sort(key=lambda x: x[1], reverse=True)
            top = [r for r, _ in flagged[: self.config.report_top_k]]
            straggler_ranks[stage] = top
            if top and mean > 1e-9:
                worst = flagged[0]
                hints.append(
                    f"{stage}: rank{worst[0]} {worst[1]:.2f}ms "
                    f"(mean {mean:.2f}ms, ratio {worst[1] / mean:.2f}x)"
                )

        gaps = {s: stage_max[s] - stage_mean[s] for s in self.config.stages}
        if gaps:
            bottleneck = max(gaps, key=gaps.get)
            if gaps[bottleneck] > self.config.absolute_ms_threshold:
                hints.append(f"likely_bottleneck={bottleneck}")

        return StragglerReport(
            step=step,
            world_size=world,
            rank=rank,
            stage_ms={k: float(local.get(k, 0.0)) for k in self.config.stages},
            stage_mean_ms=stage_mean,
            stage_max_ms=stage_max,
            straggler_ranks=straggler_ranks,
            slowdown_ratio=slowdown_ratio,
            hints=hints,
        )

    @staticmethod
    def _is_rank0() -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
