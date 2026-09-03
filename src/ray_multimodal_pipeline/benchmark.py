"""Measurement harness: sweep pipeline configurations and record throughput.

Produces the README's before/after table. Every run measures the same unit of work —
ingest through inference, excluding the Parquet write — so the sequential baseline and
the Ray configurations are directly comparable.

GPU utilization is sampled through `nvidia-smi` and is therefore CUDA-only. On Apple
Silicon the figure is reported as unavailable rather than guessed at: macOS exposes no
unprivileged Metal utilization counter, so local runs measure CPU parallelism only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import ray
from nuscenes.nuscenes import NuScenes

from ray_multimodal_pipeline.inference import DEFAULT_SCORE_THRESHOLD, DetectionPredictor
from ray_multimodal_pipeline.manifest import build_manifest
from ray_multimodal_pipeline.preprocess import preprocess_batch

SAMPLE_INTERVAL_S = 0.25
GPU_SAMPLE_EVERY = 4  # sample the GPU every Nth tick; nvidia-smi is comparatively slow
WARMUP_RECORDS = 8


def warm_up(dataroot: str, version: str, *, device: str | None, score_threshold: float) -> None:
    """Pay one-time costs before any timed run.

    Starting the Ray cluster takes tens of seconds, and the first inference actor also
    pulls model weights off disk. Absorbed into the first timed configuration, that cost
    reads as a throughput collapse in whichever row happens to run first rather than as
    the fixed startup overhead it actually is.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    run_ray(
        dataroot,
        version,
        batch_size=WARMUP_RECORDS,
        concurrency=1,
        preprocess_concurrency=None,
        device=device,
        score_threshold=score_threshold,
        limit=WARMUP_RECORDS,
    )


def gpu_utilization() -> float | None:
    """Mean GPU utilization percentage across visible CUDA devices, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    readings = [float(line) for line in completed.stdout.split() if line.strip()]
    return statistics.fmean(readings) if readings else None


@dataclass
class ResourceUsage:
    """Utilization sampled across a run."""

    cpu_percent_mean: float = 0.0
    cpu_percent_peak: float = 0.0
    gpu_percent_mean: float | None = None
    gpu_percent_peak: float | None = None


class ResourceSampler:
    """Background sampler for system-wide CPU (and CUDA GPU) utilization.

    Samples system-wide rather than per-process on purpose: Ray executes the work in
    separate worker processes, so driver-only accounting would report almost nothing.
    """

    def __init__(self, sample_interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self._interval = sample_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu: list[float] = []
        self._gpu: list[float] = []

    def __enter__(self) -> ResourceSampler:
        psutil.cpu_percent(interval=None)  # prime the counter; first read is meaningless
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        tick = 0
        while not self._stop.is_set():
            self._cpu.append(psutil.cpu_percent(interval=None))
            if tick % GPU_SAMPLE_EVERY == 0:
                reading = gpu_utilization()
                if reading is not None:
                    self._gpu.append(reading)
            tick += 1
            self._stop.wait(self._interval)

    def result(self) -> ResourceUsage:
        return ResourceUsage(
            cpu_percent_mean=round(statistics.fmean(self._cpu), 1) if self._cpu else 0.0,
            cpu_percent_peak=round(max(self._cpu), 1) if self._cpu else 0.0,
            gpu_percent_mean=round(statistics.fmean(self._gpu), 1) if self._gpu else None,
            gpu_percent_peak=round(max(self._gpu), 1) if self._gpu else None,
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """One point in the sweep."""

    label: str
    engine: str  # "sequential" or "ray"
    batch_size: int
    concurrency: int | None = None
    preprocess_concurrency: int | None = None


@dataclass
class BenchmarkResult:
    """Outcome of a single run."""

    label: str
    engine: str
    batch_size: int
    concurrency: int | None
    preprocess_concurrency: int | None
    device: str
    records: int
    wall_clock_s: float
    records_per_sec: float
    usage: ResourceUsage = field(default_factory=ResourceUsage)


def run_sequential(
    dataroot: str,
    version: str,
    *,
    batch_size: int,
    device: str | None,
    score_threshold: float,
    limit: int | None,
) -> int:
    """Single-process baseline: identical work, no Ray, no parallelism.

    Calls the same preprocess and inference code as the Ray path so the comparison
    isolates the execution engine rather than the implementation.
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    manifest = build_manifest(nusc)
    if limit is not None:
        manifest = manifest[:limit]

    predictor = DetectionPredictor(device=device, score_threshold=score_threshold)
    records = 0
    for start in range(0, len(manifest), batch_size):
        chunk = manifest[start : start + batch_size]
        batch = {
            key: [record[key] for record in chunk]
            for key in ("sample_token", "timestamp", "cam_front_path", "lidar_top_path")
        }
        predictor(preprocess_batch(batch))
        records += len(chunk)
    return records


def run_ray(
    dataroot: str,
    version: str,
    *,
    batch_size: int,
    concurrency: int,
    preprocess_concurrency: int | None,
    device: str | None,
    score_threshold: float,
    limit: int | None,
) -> int:
    """Ray Data path, materialized through inference (the Parquet write is excluded)."""
    from ray_multimodal_pipeline.pipeline import run as run_pipeline

    dataset = run_pipeline(
        dataroot,
        version,
        device=device,
        score_threshold=score_threshold,
        batch_size=batch_size,
        concurrency=concurrency,
        preprocess_concurrency=preprocess_concurrency,
        limit=limit,
    )
    return dataset.materialize().count()


def run_config(
    config: BenchmarkConfig,
    dataroot: str,
    version: str,
    *,
    device: str | None,
    score_threshold: float,
    limit: int | None,
) -> BenchmarkResult:
    """Execute one configuration and capture timing plus utilization."""
    from ray_multimodal_pipeline.inference import resolve_device

    with ResourceSampler() as sampler:
        started = time.perf_counter()
        if config.engine == "sequential":
            records = run_sequential(
                dataroot,
                version,
                batch_size=config.batch_size,
                device=device,
                score_threshold=score_threshold,
                limit=limit,
            )
        else:
            records = run_ray(
                dataroot,
                version,
                batch_size=config.batch_size,
                concurrency=config.concurrency or 1,
                preprocess_concurrency=config.preprocess_concurrency,
                device=device,
                score_threshold=score_threshold,
                limit=limit,
            )
        elapsed = time.perf_counter() - started

    return BenchmarkResult(
        label=config.label,
        engine=config.engine,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
        preprocess_concurrency=config.preprocess_concurrency,
        device=resolve_device(device),
        records=records,
        wall_clock_s=round(elapsed, 1),
        records_per_sec=round(records / elapsed, 2) if elapsed else 0.0,
        usage=sampler.result(),
    )


def build_sweep(
    concurrencies: list[int],
    batch_sizes: list[int],
    preprocess_concurrencies: list[int],
    *,
    include_baselines: bool = True,
) -> list[BenchmarkConfig]:
    """Baseline rows followed by the full concurrency x batch-size x preprocess grid."""
    configs: list[BenchmarkConfig] = []
    if include_baselines:
        configs.append(
            BenchmarkConfig(
                label="sequential (no Ray)", engine="sequential", batch_size=batch_sizes[0]
            )
        )
        configs.append(
            BenchmarkConfig(
                label="ray, 1 actor",
                engine="ray",
                batch_size=batch_sizes[0],
                concurrency=1,
                preprocess_concurrency=preprocess_concurrencies[0],
            )
        )

    seen = {
        (config.engine, config.concurrency, config.batch_size, config.preprocess_concurrency)
        for config in configs
    }
    for concurrency, batch_size, preprocess in itertools.product(
        concurrencies, batch_sizes, preprocess_concurrencies
    ):
        # The 1-actor baseline is usually also a grid point; emitting it twice would put
        # two rows with identical settings but different noise into the published table.
        key = ("ray", concurrency, batch_size, preprocess)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            BenchmarkConfig(
                label=f"ray, {concurrency} actors, batch {batch_size}, prep {preprocess}",
                engine="ray",
                batch_size=batch_size,
                concurrency=concurrency,
                preprocess_concurrency=preprocess,
            )
        )
    return configs


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    """Render results as a markdown table ready to paste into the README."""
    header = (
        "| Config | Records/sec | Wall-clock | CPU util (mean/peak) | GPU util | Records |\n"
        "|--------|-------------|------------|----------------------|----------|---------|"
    )
    rows = []
    for result in sorted(results, key=lambda r: r.records_per_sec, reverse=True):
        gpu = (
            f"{result.usage.gpu_percent_mean}%"
            if result.usage.gpu_percent_mean is not None
            else "n/a"
        )
        rows.append(
            f"| {result.label} | {result.records_per_sec} | {result.wall_clock_s}s | "
            f"{result.usage.cpu_percent_mean}% / {result.usage.cpu_percent_peak}% | "
            f"{gpu} | {result.records} |"
        )
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", default="data/benchmarks/results.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None, help="samples per run")
    parser.add_argument("--repeats", type=int, default=1, help="runs per config; results averaged")
    parser.add_argument("--concurrency-grid", default="1,2,4")
    parser.add_argument("--batch-size-grid", default="4,8,16")
    parser.add_argument("--preprocess-grid", default="4,8,15")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    def parse_grid(raw: str) -> list[int]:
        return [int(value) for value in raw.split(",") if value.strip()]

    configs = build_sweep(
        parse_grid(args.concurrency_grid),
        parse_grid(args.batch_size_grid),
        parse_grid(args.preprocess_grid),
        include_baselines=not args.skip_baselines,
    )

    print("warming up (Ray cluster + model weights) ...", flush=True)
    warm_up(
        args.dataroot,
        args.version,
        device=args.device,
        score_threshold=args.score_threshold,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.label} ...", flush=True)
        repeats = [
            run_config(
                config,
                args.dataroot,
                args.version,
                device=args.device,
                score_threshold=args.score_threshold,
                limit=args.limit,
            )
            for _ in range(args.repeats)
        ]
        # Report the median run so a single scheduling hiccup cannot define the number.
        median = sorted(repeats, key=lambda r: r.records_per_sec)[len(repeats) // 2]
        results.append(median)
        print(f"    {median.records_per_sec} rec/s over {median.wall_clock_s}s", flush=True)

        # Persist after every config. A full sweep runs for hours, and a pathological
        # configuration can stretch it far enough that it gets interrupted; writing only
        # at the end would discard every measurement taken up to that point.
        output.write_text(json.dumps([asdict(result) for result in results], indent=2))

    print()
    print(format_markdown_table(results))
    print()
    print(f"raw results: {output}")


if __name__ == "__main__":
    main()
