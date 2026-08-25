# Workplan Script Implementation Handoff

Updated: 2026-08-25

## Repository state

- Repository: `GMLDatasets.jl`
- Branch: `revision-measurement-harness`
- Base: PR #4 branch `pendulum-dataset`
- Base commit: `2d4a40eec0f0a5a7bf41f8ecb2a64c23792766a3`
- The branch has not yet been committed, pushed, or opened as a pull request.

## Implemented

- Extended `scripts/geometric_optimizers/mnist_cuda_repetitions.jl`:
  - Full-run default increased from 5 to 10 repetitions.
  - Added explicit comma-separated `MNIST_SEEDS` support.
  - Added `MNIST_DATASET=mnist|fashion-mnist` selection.
  - Added machine-readable per-run CSV output through `MNIST_RECORDS`.
  - Added per-run peak device-memory sampling to emitted records.
- Updated `scripts/geometric_optimizers/run_mnist_repetitions.sh`:
  - Uses 10 repetitions by default.
  - Produces and reports the new run-record CSV.
- Extended `scripts/pendulum/train_sae.jl`:
  - Reads epochs, batch size, step size, seed, reduced dimension, output path, and CUDA requirement from environment variables.
  - Measures elapsed time, host allocations, and GC time.
  - Stores timing metadata in the HDF5 checkpoint.
  - Appends one machine-readable CSV record per run through `SAE_RECORD`.
- Added `scripts/revision/check_environment.jl`:
  - Validates CUDA and the expected RTX 4090 device unless explicitly relaxed for local smoke testing.
  - Records Julia, active project, package manifest status, CUDA versions, device, and thread count.
- Added `scripts/revision/run_experiments.sh`:
  - Supports `--smoke` and `--full` modes.
  - Supports stage, seed, configuration, output-directory, dirty-tree, GPU, and retraction-repository filters.
  - Runs MNIST, Fashion-MNIST, pendulum SAE, and the sibling `GeometricOptimizers.jl` retraction benchmark.
  - Performs warm-up runs before full image/SAE runs.
  - Streams per-stage stdout/stderr to durable files and records stage status.
  - Preserves partial outputs after failure through an EXIT trap.
  - Captures both repositories' SHAs, status, and binary diffs.
  - Copies root and scripts Project/Manifest files without filename collisions.
  - Produces a timestamped `.tar.gz` and SHA-256 checksum.
- Added `scripts/revision/README.md` with setup, smoke, full-run, monitoring, restart, runtime/disk planning, transfer, and checksum instructions.

## Validation completed

- `bash -n` passes for both shell runners.
- `Meta.parseall` passes for all modified/added Julia scripts.
- `git diff --check` passes.
- Direct one-epoch CPU SAE smoke passed:
  - Checkpoint: `/tmp/gmldatasets-sae-smoke.h5`
  - Record: `/tmp/gmldatasets-sae-smoke.csv`
- End-to-end orchestration smoke passed for the pendulum stage:
  - Directory: `/tmp/gmldatasets-revision-smoke/20260825T093545Z_smoke`
  - Archive: `/tmp/gmldatasets-revision-smoke/20260825T093545Z_smoke.tar.gz`
  - Checksum: `/tmp/gmldatasets-revision-smoke/20260825T093545Z_smoke.tar.gz.sha256`

## Important limitations / next work

- The current MNIST configuration set is still the pre-existing four configurations. The workplan's matched Cayley-ADAM and scalar-moment ablations still need implementation after confirming the exact supported parameter-tree arrangement.
- The SAE runner currently records geometric Adam only. Matched baseline optimizer variants and structure-preservation evaluation metrics still need to be added.
- The retraction benchmark is invoked and logged, but its human-readable output is not yet converted into the common machine-readable record schema.
- The image runner records total/epoch timing and peak device usage; gradient, optimizer-update, and retraction timing still need separate instrumentation.
- Resume is checkpoint-aware for completed pendulum seeds. MNIST/Fashion-MNIST preserve partial artifacts but rerun an interrupted matrix because the legacy trainer does not yet resume individual completed jobs.
- A full MNIST/Fashion smoke was not run locally; it may require dataset availability/network and is much more expensive than the SAE smoke.

## Resume commands

```bash
cd /Users/benbradmin/Documents/GMLDataSets
git status --short --branch
bash -n scripts/revision/run_experiments.sh scripts/geometric_optimizers/run_mnist_repetitions.sh
julia --startup-file=no -e 'for path in ARGS; Meta.parseall(read(path, String)); println("parsed ", path); end' \
  scripts/revision/check_environment.jl \
  scripts/pendulum/train_sae.jl \
  scripts/geometric_optimizers/mnist_cuda_repetitions.jl
git diff --check
```

After reviewing the diff, commit and push `revision-measurement-harness`, then open a **draft** pull request targeting `pendulum-dataset` (not `main`).

