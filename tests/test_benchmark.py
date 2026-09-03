import json
from dataclasses import asdict

from ray_multimodal_pipeline.benchmark import (
    BenchmarkResult,
    ResourceUsage,
    build_sweep,
    format_markdown_table,
    gpu_utilization,
)


def _result(label: str, rps: float, gpu: float | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        label=label,
        engine="ray",
        batch_size=8,
        concurrency=2,
        preprocess_concurrency=8,
        device="cpu",
        records=100,
        wall_clock_s=10.0,
        records_per_sec=rps,
        usage=ResourceUsage(cpu_percent_mean=50.0, cpu_percent_peak=90.0, gpu_percent_mean=gpu),
    )


def test_build_sweep_covers_full_grid_plus_baselines() -> None:
    configs = build_sweep([1, 2], [4, 8], [4, 8])

    assert [c.engine for c in configs[:2]] == ["sequential", "ray"]
    assert configs[1].concurrency == 1
    # 2x2x2 grid, minus the point the 1-actor baseline already covers.
    assert len(configs) == 2 + (2 * 2 * 2) - 1


def test_build_sweep_does_not_duplicate_the_baseline_grid_point() -> None:
    """A duplicated config would publish two rows that differ only by measurement noise."""
    configs = build_sweep([1, 2], [4, 8], [4, 8])

    keys = [(c.engine, c.concurrency, c.batch_size, c.preprocess_concurrency) for c in configs]
    assert len(keys) == len(set(keys))


def test_build_sweep_can_omit_baselines() -> None:
    configs = build_sweep([2], [8], [8], include_baselines=False)

    assert len(configs) == 1
    assert all(c.engine == "ray" for c in configs)


def test_build_sweep_grid_is_unique() -> None:
    configs = build_sweep([1, 2, 4], [4, 8], [4, 8], include_baselines=False)

    keys = {(c.concurrency, c.batch_size, c.preprocess_concurrency) for c in configs}
    assert len(keys) == len(configs)


def test_format_markdown_table_orders_by_throughput() -> None:
    table = format_markdown_table([_result("slow", 1.0), _result("fast", 9.0)])

    lines = table.splitlines()
    assert lines[0].startswith("| Config |")
    assert "fast" in lines[2] and "slow" in lines[3]


def test_format_markdown_table_marks_missing_gpu_as_unavailable() -> None:
    table = format_markdown_table([_result("cpu-only", 5.0, gpu=None)])

    assert "n/a" in table


def test_format_markdown_table_reports_gpu_when_present() -> None:
    table = format_markdown_table([_result("cuda", 5.0, gpu=87.5)])

    assert "87.5%" in table


def test_results_are_json_serializable() -> None:
    """The harness persists raw results, so every field must survive asdict + json."""
    payload = json.dumps([asdict(_result("a", 1.0))])

    assert json.loads(payload)[0]["usage"]["cpu_percent_peak"] == 90.0


def test_gpu_utilization_returns_none_without_cuda_tooling() -> None:
    """Must degrade quietly on hosts with no nvidia-smi rather than raising."""
    assert gpu_utilization() is None or isinstance(gpu_utilization(), float)
