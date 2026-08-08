# CLAUDE.md — ray-multimodal-pipeline

## Overview
A distributed multimodal sensor-data processing pipeline built on Ray. Ingests multimodal
autonomous-driving data (camera + lidar + metadata), preprocesses it, runs pretrained model
inference as a batch stage, and writes structured outputs. Focus: distributed data/compute
engineering with reproducible performance measurement.

## Core constraints
- **Pretrained models only.** Use a pretrained detector (torchvision / YOLO variant); no training
  from scratch. The focus is the pipeline, not the weights.
- **Ray for the compute layer.** Ray Core, Ray Data (primary), Ray Serve, RLlib. No Spark/Dask.
- **Measurement is a deliverable.** Every scaling change records throughput (records/sec),
  GPU utilization, and wall-clock in the README's before/after table.

## Compute & GPU
- Local dev: Apple Silicon GPU via PyTorch MPS for iteration and functional testing.
- Scale-out + measurement: managed Ray on CUDA GPUs.
- Ray's `num_gpus` scheduling is CUDA-only — it does not see the Apple GPU. Therefore:
  - Inference stages are **device-agnostic**: select `cuda` → `mps` → `cpu` via one helper. Never hardcode.
  - Ray GPU scheduling (`num_gpus`) is used only on the CUDA cluster.
  - Local runs yield CPU-parallelism numbers; label them as such (GPU-utilization requires CUDA).

## Tech stack
- Python 3.11+, Ray (Data / Core / Serve / RLlib), PyTorch (inference only), PyArrow/Parquet.
- Dataset: nuScenes mini split (KITTI fallback).
- Tooling: `uv`/`venv`, `ruff`, `pytest`.

## Conventions
- One feature per branch (`feat/…`, `fix/…`); small, atomic commits with Conventional Commits messages.
- Keep pipeline stages small, testable, composable (`map_batches`-friendly).
- Type hints on public functions; docstrings on modules and stages.
- A `pytest` test for each stage's transform logic where practical.
- README stays current: architecture diagram, how-to-run, measurement table.
- Never commit secrets, dataset blobs, or large artifacts; keep `.gitignore` current.

## Local, personal overrides (workflow / positioning) — imported if present, gitignored
@CLAUDE.local.md
