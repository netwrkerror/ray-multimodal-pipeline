"""End-to-end entry point: nuScenes-mini ingest + decode/resize/normalize preprocessing."""

from __future__ import annotations

import argparse

import ray.data
from nuscenes.nuscenes import NuScenes

from ray_multimodal_pipeline.ingest import build_ingest_dataset
from ray_multimodal_pipeline.manifest import build_manifest
from ray_multimodal_pipeline.preprocess import preprocess_batch


def run(dataroot: str, version: str = "v1.0-mini") -> ray.data.Dataset:
    """Ingest nuScenes-mini and run the decode/resize/normalize preprocessing stage."""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    manifest = build_manifest(nusc)
    dataset = build_ingest_dataset(manifest)
    return dataset.map_batches(preprocess_batch, batch_format="numpy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    # Ray Data is lazy: materialize once so schema() and count() below don't each
    # trigger a separate full execution of the preprocessing stage.
    dataset = run(args.dataroot, args.version).materialize()
    print(dataset.schema())
    print(f"{dataset.count()} records preprocessed")


if __name__ == "__main__":
    main()
