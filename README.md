# ray-multimodal-pipeline

A distributed multimodal sensor-data processing pipeline built on [Ray](https://www.ray.io/).
It ingests autonomous-driving sensor data (camera + lidar + metadata), preprocesses it,
runs pretrained-model batch inference, and writes structured output — as a demonstration of
distributed-systems / ML-infra engineering on Ray, not of modeling.

**Status:** ingest + preprocess stages implemented (decode, resize, normalize). Batch inference
and Parquet output are next.

## Why this exists

Orchestration, not modeling: every model used is pretrained (torchvision detection / a YOLO
variant). The point is to show how Ray Data/Core scale a real multimodal ingest → preprocess →
inference → write pipeline, with before/after throughput and GPU-utilization numbers to back it up.

## Architecture

```
                 ┌──────────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────────┐
 nuScenes mini → │   Ingest     │ ──▶ │  Preprocess   │ ──▶ │   Inference    │ ──▶ │  Write output │
 (camera+lidar)  │ (Ray Data    │     │ (decode,      │     │ (pretrained    │     │  (Parquet)    │
                 │  read_*)     │     │  resize,      │     │  detection     │     │               │
                 │              │     │  normalize)   │     │  model,        │     │               │
                 │              │     │  map_batches  │     │  map_batches)  │     │               │
                 └──────────────┘     └───────────────┘     └───────────────┘     └──────────────┘
```

_Diagram is a placeholder — will be refined as stages land and scaling is added (Phase 2)._

## Project layout

```
src/ray_multimodal_pipeline/
  manifest.py     # driver-side: nuScenes sample/sample_data tables -> list[dict]
  ingest.py        # manifest -> Ray Dataset (ray.data.from_items)
  preprocess.py    # decode / resize / normalize camera images, load lidar sweeps
  pipeline.py      # wires manifest -> ingest -> preprocess, CLI entry point
tests/                           # pytest — one test per stage's transform logic
data/                            # local dataset cache, gitignored — see Dataset setup below
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # creates .venv and installs runtime + dev dependencies
```

## Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pytest          # tests
```

## Dataset setup

Uses the **nuScenes mini** split (camera + lidar, small enough for local dev). Download it
locally into `data/`; the directory is gitignored so nothing here is ever committed.

1. Register for a free account at [nuscenes.org](https://www.nuscenes.org/sign-up) (required
   before any download link works).
2. From the [nuScenes download page](https://www.nuscenes.org/nuscenes#download), under
   **Full dataset (v1.0) → Mini**, download `v1.0-mini.tgz` (~4 GB: samples, sweeps, and maps).
3. Unpack it into `data/nuscenes`:
   ```bash
   mkdir -p data/nuscenes
   tar -xzf v1.0-mini.tgz -C data/nuscenes
   ```
4. `nuscenes-devkit` is already a project dependency (installed by `uv sync`), so no extra
   install is needed.
5. Sanity-check the download:
   ```bash
   uv run python -c "
   from nuscenes.nuscenes import NuScenes
   nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
   print(len(nusc.sample), 'samples loaded')
   "
   ```
   Expect ~404 samples for `v1.0-mini`.

## How to run

Ingest nuScenes-mini and run the decode/resize/normalize preprocessing stage:

```bash
uv run python -m ray_multimodal_pipeline.pipeline --dataroot data/nuscenes --version v1.0-mini
```

Prints the resulting dataset's schema and record count. Batch inference and Parquet output land
in the next stage.

## Measurement

Phase 2 deliverable — single-node vs Ray-parallelized runs, filled in once scaling work lands.

| Config | Throughput (records/sec) | GPU utilization | Wall-clock | Notes |
|--------|---------------------------|------------------|------------|-------|
| Single-node baseline | — | — | — | |
| Ray-parallelized | — | — | — | |

## Roadmap

- [x] Phase 1: project scaffold
- [x] Phase 1: Ray Data ingest/preprocess pipeline
- [ ] Phase 1: batch-inference stage + Parquet write
- [ ] Phase 2: scale out + measurement
- [ ] Phase 3: RLlib/sim slice + write-up
