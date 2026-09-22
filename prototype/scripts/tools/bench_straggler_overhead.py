#!/usr/bin/env python3
"""Measure StragglerAnalyzer overhead on a tiny multi-GPU training loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


REPO = "/root/autodl-tmp/work/Relax"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from relax.utils.straggler import StragglerAnalyzer, StragglerConfig  # noqa: E402


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


class TinyBlock(nn.Module):
    def __init__(self, dim=2048, hidden=8192):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def run_loop(steps: int, analyzer: StragglerAnalyzer | None, batch: torch.Tensor, model, optim):
    for _ in range(10):
        optim.zero_grad(set_to_none=True)
        loss = model(batch).pow(2).mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optim.step()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    reports = 0
    for step in range(steps):
        if analyzer is not None:
            analyzer.begin_step(step)
            optim.zero_grad(set_to_none=True)
            with analyzer.stage("forward"):
                out = model(batch)
                loss = out.pow(2).mean()
            with analyzer.stage("backward"):
                loss.backward()
            with analyzer.stage("communication"):
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            with analyzer.stage("optimizer"):
                optim.step()
            report = analyzer.end_step(step)
            if report is not None:
                reports += 1
        else:
            optim.zero_grad(set_to_none=True)
            out = model(batch)
            loss = out.pow(2).mean()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optim.step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    return elapsed, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument(
        "--out",
        type=str,
        default="/root/autodl-tmp/work/straggler_bench/overhead_result.json",
    )
    args = parser.parse_args()

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    model = TinyBlock(args.dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randn(args.batch, args.dim, device=device)

    base_s, _ = run_loop(args.steps, None, batch, model, optim)
    analyzer = StragglerAnalyzer(
        StragglerConfig(
            enabled=True,
            sample_interval=args.interval,
            sync_cuda=True,
            stages=("forward", "backward", "communication", "optimizer", "attention", "moe"),
        )
    )
    prof_s, reports = run_loop(args.steps, analyzer, batch, model, optim)

    detect_ok = False
    detect_detail = {}
    if dist.get_world_size() >= 2:
        analyzer2 = StragglerAnalyzer(
            StragglerConfig(
                sample_interval=1,
                relative_threshold=1.15,
                absolute_ms_threshold=0.5,
            )
        )
        analyzer2.begin_step(0)
        with analyzer2.stage("forward"):
            _ = model(batch)
            # Create a real GPU-side straggler on rank1 (CUDA events measure GPU time).
            if dist.get_rank() == 1:
                for _ in range(40):
                    _ = model(batch)
        with analyzer2.stage("backward"):
            loss = model(batch).pow(2).mean()
            loss.backward()
        with analyzer2.stage("communication"):
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        with analyzer2.stage("optimizer"):
            pass
        report = analyzer2.end_step(0)
        if report is not None:
            detect_detail = {
                "forward_stragglers": report.straggler_ranks.get("forward", []),
                "forward_ratio": report.slowdown_ratio.get("forward", 1.0),
                "hints": report.hints,
            }
            detect_ok = (
                1 in report.straggler_ranks.get("forward", [])
                or report.slowdown_ratio.get("forward", 1.0) > 1.15
            )

    overhead = (prof_s - base_s) / base_s * 100.0
    result = {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "steps": args.steps,
        "sample_interval": args.interval,
        "baseline_sec": base_s,
        "profiled_sec": prof_s,
        "overhead_pct": overhead,
        "sampled_reports": reports,
        "detection_smoke_ok": detect_ok,
        "detection_detail": detect_detail,
        "pass_overhead_lt_0_5": overhead < 0.5,
    }

    if dist.get_rank() == 0:
        print(json.dumps(result, indent=2))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")

    cleanup()


if __name__ == "__main__":
    main()
