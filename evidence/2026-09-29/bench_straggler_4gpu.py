# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""4-GPU check of the non-blocking timer plus cross-rank judgement.

Does not start a Relax recipe. Measures a matmul loop with and without the
timer shim, then makes one rank do extra compute and checks the detector.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

from relax.utils.straggler.detector import JudgeState, judge, metrics_from, rows_from_payload
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.stages import STAGES
from relax.utils.straggler.timer_shim import NonBlockingTimers


def loop(steps: int, x: torch.Tensor, weight: torch.Tensor, timers: NonBlockingTimers | None, slow: bool, repeats: int) -> None:
    for _ in range(steps):
        if timers is not None:
            timers("forward-compute").start()
        y = x @ weight
        extra = repeats + (8 if slow and dist.get_rank() == 3 else 0)
        for _inner in range(extra):
            y = y @ weight
        if timers is not None:
            timers("forward-compute").stop()
            timers("all-grads-sync").start()
        dist.all_reduce(y)
        if timers is not None:
            timers("all-grads-sync").stop()
    torch.cuda.synchronize()
    dist.barrier()


def timed(steps: int, x: torch.Tensor, weight: torch.Tensor, timers: NonBlockingTimers | None, slow: bool, repeats: int) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    loop(steps, x, weight, timers, slow, repeats)
    return time.perf_counter() - t0


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    steps = 30
    x = torch.randn(2048, 2048, device=device)
    weight = torch.randn(2048, 2048, device=device)

    loop(5, x, weight, None, slow=False, repeats=16)
    off = timed(40, x, weight, None, slow=False, repeats=16)
    timers = NonBlockingTimers()
    on = timed(40, x, weight, timers, slow=False, repeats=16)
    torch.cuda.synchronize()
    _ = timers.drain()

    detect = NonBlockingTimers()
    loop(8, x, weight, detect, slow=True, repeats=4)
    torch.cuda.synchronize()
    totals = detect.drain()
    payload = {"rank": rank, "pp": 0, "tokens": 2048.0, "gc_ms": 0.0}
    for name in STAGES:
        payload[name] = float(totals.get(name, 0.0))
    world = dist.get_world_size()
    gathered = [None] * world
    dist.all_gather_object(gathered, payload, group=dist.new_group(backend="gloo"))
    result = None
    if rank == 0:
        rows = [rows_from_payload(item) for item in gathered]
        cfg = StragglerConfig(persist_windows=1, relative_threshold=0.10, absolute_ms_threshold=5.0)
        alerts, _ = judge(rows, cfg, JudgeState())
        overhead = (on - off) / off * 100.0
        result = {
            "world_size": world,
            "gpus": [torch.cuda.get_device_name(i) for i in range(world)],
            "steps": 40,
            "repeats_per_step": 16,
            "off_sec": off,
            "on_sec": on,
            "overhead_pct": overhead,
            "fwd_ms_by_rank": [item["fwd"] for item in gathered],
            "alerts": [{"rank": item.rank, "reason": item.reason} for item in alerts],
            "metrics": {k: v for k, v in metrics_from(rows, alerts, 0).items() if not isinstance(v, str)},
            "note": "synthetic matmul loop, not a Relax recipe",
        }
        path = "/root/autodl-tmp/straggler_4gpu.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
