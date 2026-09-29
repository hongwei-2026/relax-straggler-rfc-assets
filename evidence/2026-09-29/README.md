# 2026-09-29 云机检查

机器：AutoDL，4×NVIDIA RTX 6000D。代码在 `/root/autodl-tmp/Relax`，基线 `99a0b6a`。

| 图 | 结果 |
| --- | --- |
| `01-gpus.png` | 4 张 RTX 6000D |
| `02-hooks.png` | `attach_timers` / `on_rollout_end` / `--straggler-analysis` 已接上 |
| `03-unit-ok.png` | 判定单测 `unit_ok` |
| `04-fwd-ms.png` | 单卡 Event 冒烟 `fwd_ms=116.311` |

另外用 4 张卡跑了一次矩阵乘循环：`straggler_4gpu.json`。rank 3 多算后被标成 `slow_device`（forward 约 32.6ms，其余约 13ms）。开关墙钟差约 1.33%。每步只有大约 7ms，百分比会被放大，不能当成 0.5% 验收。

