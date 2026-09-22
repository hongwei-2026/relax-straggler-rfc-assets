# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.straggler import StragglerAnalyzer, StragglerConfig


def test_should_sample_interval():
    cfg = StragglerConfig(enabled=True, sample_interval=10)
    assert cfg.should_sample(0)
    assert not cfg.should_sample(1)
    assert cfg.should_sample(10)
    assert not StragglerConfig(enabled=False).should_sample(0)


def test_detect_straggler_local_only():
    analyzer = StragglerAnalyzer(
        StragglerConfig(
            relative_threshold=1.2,
            absolute_ms_threshold=1.0,
            stages=("forward", "backward", "communication", "optimizer", "attention", "moe"),
        )
    )
    report = analyzer.analyze_local_only(
        step=3,
        stage_ms={"forward": 10.0, "backward": 20.0, "optimizer": 5.0},
    )
    assert report.step == 3
    assert report.world_size == 1
    assert report.stage_ms["forward"] == 10.0
    metrics = report.to_metrics()
    assert metrics["straggler/forward/local_ms"] == 10.0
    assert "straggler/hints" in metrics


def test_build_report_multi_rank_manual():
    analyzer = StragglerAnalyzer(
        StragglerConfig(
            relative_threshold=1.2,
            absolute_ms_threshold=1.0,
            stages=("forward", "backward", "communication", "optimizer", "attention", "moe"),
        )
    )
    gathered = [
        {
            "forward": 10.0,
            "backward": 20.0,
            "communication": 5.0,
            "optimizer": 4.0,
            "attention": 0.0,
            "moe": 0.0,
        },
        {
            "forward": 10.5,
            "backward": 40.0,
            "communication": 5.1,
            "optimizer": 4.1,
            "attention": 0.0,
            "moe": 0.0,
        },
    ]
    report = analyzer._build_report(step=1, local=gathered[0], gathered=gathered)
    assert 1 in report.straggler_ranks["backward"]
    assert report.slowdown_ratio["backward"] > 1.2
    assert any("backward" in h for h in report.hints)
    assert any("likely_bottleneck=backward" in h for h in report.hints)


def test_begin_end_noop_when_not_sampling():
    analyzer = StragglerAnalyzer(StragglerConfig(sample_interval=10))
    assert analyzer.begin_step(1) is False
    assert analyzer.end_step(1) is None
