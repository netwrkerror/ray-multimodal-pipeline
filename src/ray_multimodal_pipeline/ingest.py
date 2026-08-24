"""Ingest stage: turn a manifest of nuScenes sample records into a Ray Dataset."""

from __future__ import annotations

import ray.data


def build_ingest_dataset(manifest: list[dict]) -> ray.data.Dataset:
    """Wrap a manifest (list of per-sample record dicts) as a lazy Ray Dataset."""
    return ray.data.from_items(manifest)
