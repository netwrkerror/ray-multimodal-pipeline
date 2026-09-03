# ray-multimodal-pipeline

A distributed multimodal sensor-data processing pipeline built on [Ray](https://www.ray.io/).
It ingests autonomous-driving sensor data (camera + lidar + metadata), preprocesses it,
runs pretrained-model batch inference, and writes structured output — as a demonstration of
distributed-systems / ML-infra engineering on Ray, not of modeling.

**Status:** Phase 1 complete — the pipeline reads, preprocesses, runs batch inference, and writes
Parquet. Phase 2 in progress: the measurement harness and single-node baseline are in place; the
CUDA multi-GPU scale-out run is next.

## Why this exists

Orchestration, not modeling: every model used is pretrained (torchvision detection / a YOLO
variant). The point is to show how Ray Data/Core scale a real multimodal ingest → preprocess →
inference → write pipeline, with before/after throughput and GPU-utilization numbers to back it up.

## Architecture

```
  nuScenes mini        Ingest              Preprocess            Inference             Write
 (camera + lidar)   ray.data           map_batches           map_batches         write_parquet
        │           from_items          (stateless tasks)    (actor pool)              │
        │               │                     │                    │                  │
        ▼               ▼                     ▼                    ▼                  ▼
  sample/sample_  ──▶ Dataset of   ──▶  decode → resize  ──▶  Faster R-CNN   ──▶  detections +
  data tables         file paths        → normalize;          per batch,          lidar features
  (driver-side)       (lazy)            summarize lidar       weights loaded      (Parquet)
                                                              once per worker
```

Two deliberate choices in that flow:

- **Tasks for preprocess, actors for inference.** Decode/resize/normalize is stateless, so it runs
  as ordinary Ray Data tasks. Detection weights are ~160MB, so that stage is a callable class on an
  actor pool — the load cost is paid once per worker instead of once per batch.
- **The image tensor never reaches the output.** It is ~4.3MB/record and nothing downstream reads
  it, so the inference stage drops it. Lidar sweeps are likewise reduced to scalar features during
  preprocessing rather than carried (~700KB/record) through a stage that ignores them.

## Project layout

```
src/ray_multimodal_pipeline/
  manifest.py      # driver-side: nuScenes sample/sample_data tables -> list[dict]
  ingest.py        # manifest -> Ray Dataset (ray.data.from_items)
  preprocess.py    # decode / resize / normalize camera images, summarize lidar sweeps
  inference.py     # DetectionPredictor actor: pretrained Faster R-CNN over image batches
  pipeline.py      # wires the stages together, CLI entry point
  benchmark.py     # measurement harness: config sweep, resource sampling, results table
tests/                           # pytest — one test per stage's transform logic
data/                            # dataset + outputs, gitignored — see Dataset setup below
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

Run the full pipeline — ingest, preprocess, detection, Parquet write:

```bash
uv run python -m ray_multimodal_pipeline.pipeline \
  --dataroot data/nuscenes --output data/outputs/detections
```

It prints the record count and wall-clock throughput on completion. The first run downloads
~160MB of pretrained Faster R-CNN weights to `~/.cache/torch`; later runs reuse them.

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--limit N` | all 404 | Process only the first N samples — use for quick iteration |
| `--device` | best available | `cuda` / `mps` / `cpu`; auto-detects otherwise |
| `--batch-size` | 8 | Images per inference batch |
| `--concurrency` | 2 | Inference actors in the pool |
| `--score-threshold` | 0.5 | Minimum detection confidence to keep |

`--batch-size` and `--concurrency` are the knobs Phase 2's scaling measurements sweep.

## Output schema

One Parquet row per sample, with detections nested as a list of structs:

```
sample_token          string          stable join key back to nuScenes
timestamp             int64
lidar_num_points      int64           derived lidar features
lidar_mean_range_m    double
lidar_max_range_m     double
lidar_mean_intensity  double
lidar_z_min_m         double
lidar_z_max_m         double
detections            list<struct<x1, y1, x2, y2: float,
                                  label: int64, label_name: string,
                                  score: float>>
num_detections        int64
```

Preprocessed image tensors are deliberately not persisted — they are recomputable, ~4.3MB/record,
and tied to the current resize/normalize config, so they would go stale the moment preprocessing
changes. The Parquet output is the derived product, keyed by `sample_token` so results join back
to the source data.

The `detections` Arrow type is declared explicitly rather than inferred. Inference would collapse
to `list<null>` for any block where every record happens to have zero detections, leaving the
output files without a single shared schema.

## Measurement

Baseline throughput on a single node (Apple M-series, 15 CPUs, one integrated GPU), full 404-record
dataset, measured ingest-through-inference:

| Config | Records/sec | Wall-clock | CPU util (mean/peak) | GPU util |
|--------|-------------|------------|----------------------|----------|
| Ray, 2 actors | 4.59 | 88.1s | 5.8% / 55.2% | n/a (MPS) |
| Sequential (no Ray) | 4.46 | 90.6s | 5.3% / 34.4% | n/a (MPS) |
| Ray, 1 actor | 4.34 | 93.2s | 7.7% / 89.3% | n/a (MPS) |

Reproduce with:

```bash
uv run python -m ray_multimodal_pipeline.benchmark --limit 404 --repeats 1 \
  --concurrency-grid 1,2,4 --batch-size-grid 4 --preprocess-grid 15
```

**These rows are statistically indistinguishable** — a 5% spread inside a ~27% measured noise
floor. That is the expected result on one GPU: actors can be scheduled independently but all issue
work to the same device, so this establishes a baseline rather than demonstrating scale-out. The
scale-out numbers require the CUDA cluster and are the outstanding Phase 2 item.

One finding is actionable today: **batch size is a cliff, not a curve.** Batch 16 runs 8.8x slower
than batch 4 (778.9s vs 88.1s) as the working set outgrows unified memory, so the default of 8 sits
deliberately below it.

Full methodology, the CPU-device sweep, and caveats are in **[MEASUREMENT.md](MEASUREMENT.md)**.

## Roadmap

- [x] Phase 1: project scaffold
- [x] Phase 1: Ray Data ingest/preprocess pipeline
- [x] Phase 1: batch-inference stage + Parquet write
- [x] Phase 2: measurement harness + single-node baseline
- [ ] Phase 2: scale out on CUDA (multi-GPU) and re-run the sweep
- [ ] Phase 2: one real-world wrinkle (backpressure / failed task / CPU-GPU split)
- [ ] Phase 3: RLlib/sim slice + write-up
