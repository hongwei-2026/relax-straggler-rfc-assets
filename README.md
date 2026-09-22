# Relax Task 11 RFC assets

Figures + **mechanism prototype** for [#334](https://github.com/redai-studio/Relax/issues/334).

## Prototype

Source under `prototype/` (not yet landed on Relax `main`):

- `relax/utils/straggler/` — Event section timer, analyzer, reporter
- `scripts/tools/bench_straggler_overhead.py` — 2-GPU A/B overhead + injection
- `tests/utils/test_straggler_detector.py` — CPU unit tests

## Overhead numbers

| run | steps | interval | batch | dim | overhead | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| heavy | 80 | 10 | 32 | 4096 | **0.479%** | mechanism feasibility |
| light | 120 | 10 | (smaller) | — | 2.201% | short-step % blown up; reference only |

Env (AutoDL): 2×GPU, PyTorch 2.8, NCCL. Metric = wall time of whole loop `(profiled-baseline)/baseline` after short warmup; sampled steps include Event sync + gather.

Raw: `overhead_result_heavy.json`, `overhead_result_light.json`.

## Claim boundary

- **Is**: CUDA Event + interval sampling + cross-rank detect works; overhead can be <0.5% when step is not tiny.
- **Is not**: Relax `train_one_step` hook; official recipe E2E A/B.
