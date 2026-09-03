"""End-to-end entry point: nuScenes-mini ingest, preprocess, batch inference, Parquet write."""

from __future__ import annotations

import argparse
import os
import time
from urllib.parse import urlparse

import ray.data
from nuscenes.nuscenes import NuScenes

from ray_multimodal_pipeline.inference import DEFAULT_SCORE_THRESHOLD, DetectionPredictor
from ray_multimodal_pipeline.ingest import build_ingest_dataset
from ray_multimodal_pipeline.manifest import build_manifest
from ray_multimodal_pipeline.preprocess import preprocess_batch

DEFAULT_INFERENCE_BATCH_SIZE = 8
DEFAULT_INFERENCE_CONCURRENCY = 2


def resolve_output_uri(output: str) -> str:
    """Absolutize a local output path, passing remote URIs through untouched.

    Ray's write tasks run from their own working directory, so a relative local path
    silently lands inside the worker's runtime dir instead of the project. Remote URIs
    (s3://, gs://, ...) are already unambiguous.
    """
    if urlparse(output).scheme:
        return output
    return os.path.abspath(output)


def run(
    dataroot: str,
    version: str = "v1.0-mini",
    *,
    device: str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    concurrency: int = DEFAULT_INFERENCE_CONCURRENCY,
    preprocess_concurrency: int | None = None,
    limit: int | None = None,
) -> ray.data.Dataset:
    """Ingest nuScenes-mini, preprocess, and run pretrained detection over camera frames.

    `preprocess_concurrency` caps concurrent decode/resize tasks; None lets Ray Data
    size the task pool itself. It is exposed so the benchmark harness can sweep it.
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    manifest = build_manifest(nusc)
    if limit is not None:
        manifest = manifest[:limit]

    dataset = build_ingest_dataset(manifest)
    dataset = dataset.map_batches(
        preprocess_batch, batch_format="numpy", concurrency=preprocess_concurrency
    )
    return dataset.map_batches(
        DetectionPredictor,
        batch_format="numpy",
        batch_size=batch_size,
        concurrency=concurrency,
        fn_constructor_kwargs={"device": device, "score_threshold": score_threshold},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", default="data/outputs/detections")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_INFERENCE_CONCURRENCY)
    parser.add_argument(
        "--preprocess-concurrency",
        type=int,
        default=None,
        help="cap concurrent decode/resize tasks (default: Ray Data decides)",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    args = parser.parse_args()

    dataset = run(
        args.dataroot,
        args.version,
        device=args.device,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        preprocess_concurrency=args.preprocess_concurrency,
        limit=args.limit,
    )

    # Ray Data is lazy, so materialize once up front: calling write_parquet() and then
    # count() on an unexecuted dataset would run every stage twice. Detections are small
    # (the image tensor was dropped in the inference stage), so holding them is cheap.
    output_uri = resolve_output_uri(args.output)
    started = time.perf_counter()
    dataset = dataset.materialize()
    records = dataset.count()
    dataset.write_parquet(output_uri)
    elapsed = time.perf_counter() - started

    print(f"wrote {records} records to {output_uri}")
    print(f"wall-clock: {elapsed:.1f}s  ({records / elapsed:.1f} records/sec)")


if __name__ == "__main__":
    main()
