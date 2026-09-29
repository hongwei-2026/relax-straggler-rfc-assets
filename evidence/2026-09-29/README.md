# 2026-09-29 云机检查

机器：AutoDL，4×NVIDIA RTX 6000D。代码在 `/root/autodl-tmp/Relax`，基线 `99a0b6a`。

| 图 | 结果 |
| --- | --- |
| `01-gpus.png` | 4 张 RTX 6000D |
| `02-hooks.png` | `attach_timers` / `on_rollout_end` / `--straggler-analysis` 已接上 |
| `03-unit-ok.png` | 判定单测 `unit_ok` |
| `04-fwd-ms.png` | 单卡 Event 冒烟 `fwd_ms=116.311` |

这组图不是 Relax recipe 的开关对比，也不是开销验收。
