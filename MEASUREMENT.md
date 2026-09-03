# Measurement notes

Detailed methodology and single-node results behind the summary table in the
[README](README.md#measurement). Kept separate so the README stays a description of the pipeline
rather than a benchmarking write-up.

**Scope caveat.** Everything here was measured on one laptop (Apple M-series, 15 CPUs, one
integrated GPU). It establishes a baseline and characterises the workload; it does **not**
demonstrate scale-out, which needs multiple devices. The CUDA cluster run is the outstanding
Phase 2 item.

## Harness

```bash
# headline numbers: full dataset, startup amortized
uv run python -m ray_multimodal_pipeline.benchmark --limit 404 --repeats 1 \
  --concurrency-grid 1,2,4 --batch-size-grid 4 --preprocess-grid 15

# full sweep across all three axes (slow; batch 16 configs are pathological)
uv run python -m ray_multimodal_pipeline.benchmark --limit 120 --repeats 3
```

It sweeps inference-actor count × batch size × preprocess parallelism against two baselines — a
single-process loop with no Ray at all, and Ray restricted to one actor. Every configuration
measures the same unit of work (ingest through inference, excluding the Parquet write), so the
comparison isolates the execution engine rather than the implementation.

Four details make the numbers trustworthy enough to publish:

- **Warm-up before timing.** Starting the Ray cluster costs tens of seconds and the first actor
  also pulls model weights off disk. Left inside the first timed run, that reads as a throughput
  collapse in whichever row happened to go first — it inflated one measurement 7x before being
  fixed.
- **Repeats with a median.** Back-to-back runs of an identical config varied by ~27%, so single
  runs are not evidence. `--repeats` reports the median.
- **CPU sampled system-wide.** Ray executes in separate worker processes, so driver-only
  accounting would report near-zero.
- **Results persisted per configuration.** A full three-axis sweep runs for hours and a
  pathological configuration can stretch it further; writing results only at the end discarded
  70 minutes of measurements once already.

GPU utilization comes from `nvidia-smi` and is therefore CUDA-only. macOS exposes no unprivileged
Metal utilization counter, so local runs report `n/a` rather than a guess.

## MPS results

Full dataset (404 records), inference-device MPS, batch 4, preprocess concurrency 15:

| Config | Records/sec | Wall-clock | CPU util (mean/peak) | GPU util |
|--------|-------------|------------|----------------------|----------|
| Ray, 2 actors | 4.59 | 88.1s | 5.8% / 55.2% | n/a (MPS) |
| Sequential (no Ray) | 4.46 | 90.6s | 5.3% / 34.4% | n/a (MPS) |
| Ray, 4 actors | 4.38 | 92.2s | 9.9% / 54.7% | n/a (MPS) |
| Ray, 1 actor | 4.34 | 93.2s | 7.7% / 89.3% | n/a (MPS) |

The whole spread is 5%, well inside the ~27% run-to-run variance measured on repeated identical
configurations, so these four rows are statistically indistinguishable. Adding inference actors
cannot help when every actor issues work to the *same single* MPS device. CPU utilization stays
near 5-10% throughout, so the pipeline is not CPU-bound either.

> An earlier Phase 1 smoke run reported 3.8 records/sec including the Parquet write. It was a
> single unrepeated measurement over different work, and is superseded by the sweep above.

## CPU-device results

MPS gives every actor the same one device, so the flat result there could be explained away by
hardware. Repeating the sweep with `--device cpu`, where 15 genuinely independent cores exist,
removes that excuse:

| Config | Records/sec | Wall-clock | CPU util (mean/peak) |
|--------|-------------|------------|----------------------|
| Sequential (no Ray) | **2.08** | 194.6s | 20.9% / 47.8% |
| Ray, 2 actors | 1.66 | 243.1s | 15.8% / 46.5% |
| Ray, 1 actor | 1.09 | 371.3s | 10.0% / 86.2% |
| Ray, 4 actors | 0.99 | 406.5s | 29.7% / 75.4% |

Scaling is **non-monotonic**: 1 → 2 actors improves 1.5x, then 2 → 4 regresses below the 1-actor
figure while consuming the most CPU (29.7% mean, ~4.5 cores) — more resource spent for less work
done, which indicates contention rather than saturation. Every Ray configuration loses to a single
process.

Utilization explains part of it. Ray assigns `num_cpus=1` per actor and sets `OMP_NUM_THREADS=1`
accordingly, so each actor's PyTorch is single-threaded: ~1.5 cores at one actor, ~2.4 at two,
~4.5 at four. The sequential baseline runs in the main process where PyTorch multithreads freely
(~3.1 cores) and pays no serialization or per-actor model-load cost.

That does not explain the 2 → 4 regression, and nothing measured here does. Four concurrent
processes each holding a ~160MB model plus activations on unified memory is the obvious suspect,
but it is untested — an open question, not a conclusion. The most promising untested lead is
giving each actor `num_cpus=N` so PyTorch can multithread within it.

## Batch size is a cliff, not a curve

Full 404 records, 2 actors, MPS:

| Batch size | Records/sec | Wall-clock | CPU util (mean/peak) |
|------------|-------------|------------|----------------------|
| 4 | 4.59 | 88.1s | 5.8% / 55.2% |
| 16 | **0.52** | **778.9s** | 15.4% / 99.7% |

Batch 16 is **8.8x slower** — 13 minutes against 88 seconds for identical work. A batch of 16
preprocessed frames is ~69MB of float32 tensors before Faster R-CNN activations, which on unified
memory pushes the working set past what the device holds; the CPU peak near 100% alongside
collapsed throughput is the signature of memory pressure rather than compute. The 120-record sweep
showed the same cliff at 3-7x, so it is not an artifact of one scale.

Batch size is therefore a correctness-adjacent setting on this hardware, not a tuning preference —
the default of 8 sits deliberately below the cliff, and the largest batch is the worst choice
despite being the one intuition suggests.

## Caveats

- **Compare only within a run.** The same sequential baseline measured 2.48 rec/s in one session
  and 4.46 in another — a ~1.8x swing, most plausibly OS page-cache warmth on the JPEG reads.
  Absolute figures are not portable across sessions; ratios within one sweep are.
- **Small `--limit` values measure startup, not throughput.** Each configuration builds fresh
  actors and loads ~160MB of weights per actor inside the timed region, so at 60-120 records the
  higher actor counts are dominated by their own startup. An earlier sweep appeared to show Ray
  losing to sequential by 9%; at the full 404 records that reversed to Ray winning by 3%, and both
  are noise. Headline numbers use the full dataset.
- **A plausible-sounding cause is not a measured one.** The flat scaling was attributed in turn to
  MPS device contention, to PyTorch thread oversubscription, and to Ray's per-actor thread cap
  implying monotonic scaling toward core count. The first two were refuted by forcing CPU inference
  and pinning `OMP_NUM_THREADS=1`; the third predicted 4 actors would beat 2, and the CPU sweep
  showed the opposite. Each explanation sounded mechanistic and was wrong.
