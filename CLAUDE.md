# CLAUDE.md — ray-multimodal-pipeline

## What this project is
A distributed multimodal sensor-data processing pipeline built on **Ray**, as a portfolio piece for
ML platform/infrastructure roles with a physical-AI lean (robotics / AV / drones). It ingests
multimodal autonomous-driving data (camera + lidar + metadata), preprocesses it, runs pretrained
model inference as a batch stage, and writes structured outputs. The point of the project is to
demonstrate **distributed-systems / infra engineering**, not modeling.

Full phased plan lives in `docs/ray-ml-infra-ramp-plan.md`. Read it for phase goals and definitions of done.

## Core constraints (do not drift from these)
- **Orchestration, not modeling.** Always use *pretrained* models (torchvision detection / a YOLO
  variant). Never train a model from scratch — the value story is the pipeline, not the weights.
- **Measurement is a first-class deliverable.** Every scaling change must produce numbers:
  throughput (records/sec), GPU utilization, wall-clock. These go in the README as a before/after table.
- **Stay in Ray's ecosystem** for the compute layer: Ray Core, Ray Data (primary), Ray Serve (later),
  RLlib (final phase). Don't reach for Spark/Dask here — the point is to demonstrate Ray.

## Tech stack
- Python 3.11+, Ray (Data / Core / Serve / RLlib), PyTorch (inference only), PyArrow/Parquet.
- Dataset: nuScenes mini split (or KITTI as a simpler fallback).
- Tooling: `uv` or `venv`, `ruff` for lint/format, `pytest` for tests.

## Git workflow — I run git myself; you provide the artifacts
The human owns all git operations (branching, staging, commits, PRs, merges). Never run
git commands. Instead, for each feature provide:
- A **branch name** (`feat/…`, `fix/…`, etc.) to create from `main`.
- **Conventional Commit message(s)** — if a feature is several logical commits, give the
  sequence with which files go in each.
- A **PR title and description** (what changed, why, and any measurement numbers).
Work one feature at a time. When the code is ready, stop and hand me the branch name +
commit plan + PR text, then wait while I create the branch, commit, and open the PR myself.
Never commit secrets, dataset blobs, or large artifacts. Keep `.gitignore` current
(data/, .venv/, __pycache__/, *.parquet outputs, cloud creds).

## Code standards
- Type hints on public functions. Docstrings on modules and pipeline stages.
- Keep pipeline stages as small, testable, composable functions (`map_batches`-friendly).
- A `pytest` test for each stage's transform logic where it's practical (pure functions first).
- README stays current: architecture diagram, how-to-run, and the measurement table.

## How to work with me on this
- Before starting a feature, restate the plan for it and wait for confirmation if scope is ambiguous.
- Implement one feature at a time; don't jump ahead to later phases.
- After each feature: run lint + tests, update the README if behavior changed, then hand me the
  branch name, commit plan, and PR description (see git workflow above).
- When a design decision has real tradeoffs (e.g., actor vs task, batch size, CPU/GPU split),
  surface it and ask rather than silently picking.
