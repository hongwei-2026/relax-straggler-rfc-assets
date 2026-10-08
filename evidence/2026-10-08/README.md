# Evidence 2026-10-08 — AutoDL 4× RTX 6000D

Environment: PyTorch `2.12.1+cu130`, CUDA 13.0, host `connect.cqa1.seetacloud.com`.

These runs validate the **mechanism** in RFC #334 (non-blocking timers + Gloo judge). They are **not** official Relax recipe acceptance.

| Artifact | Result |
| --- | --- |
| `unit_all.txt` | 11 passed (detector + config/runtime no-op) |
| `event_smoke.txt` | `fwd_ms ≈ 102.1` |
| `straggler_4gpu.json` | rank3 `slow_device`; overhead ≈ **0.88%** (short loop) |
| `straggler_4gpu_heavy.json` | rank3 `slow_device`; short-loop wall noise → overhead **negative** (noise-dominated) |
| `straggler_4gpu_sm.json` | SM-contention inject on rank3 → `slow_device` |
| `straggler_overhead_ab.json` | paired off/on A/B：**median ≈ 0.025%**, mean noise-dominated |

Earlier TinyBlock mechanism number (2-GPU, `interval=10`): **0.479%** — see repo `overhead_result_heavy.json` on `hongwei-2026/relax-straggler-rfc-assets`.

Still pending for final acceptance: any Relax recipe switch A/B (loss + overhead) and MetricsService/`straggler/*` dashboard screenshot.
