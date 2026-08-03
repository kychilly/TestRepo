# Week 1 scGPT setup and benchmark

## Findings from this repository

The repository contained no preprocessing dataset object, patient split, scGPT checkpoint, vocabulary, or dependency lock at implementation time. The selected host is macOS with Python 3.11.9 and PyTorch 2.9.1 without CUDA or GPUs. No checkpoint was downloaded or substituted.

The checked-in `results/compute/week1_scgpt_benchmark.json` is therefore explicitly `blocked`; it is not a benchmark result.

## Recreate the environment

Use Python 3.11.x in a fresh virtual environment:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --config config/model.yaml --environment-export results/compute/environment.json
```

`requirements.txt` pins the Python packages, but it is not an exact reproducibility guarantee. The successful `environment.json` export, OS, Python executable, package index/wheel provenance, hardware, CUDA driver, and the checkpoint/vocabulary hashes are also required. On a CUDA host, install the compatible PyTorch CUDA wheel before the remaining requirements and record the exact command and export; do not replace the configured checkpoint.

## Required Data Lead inputs

Populate `config/model.yaml` with the actual preprocessed AnnData path, patient-ID column, gene-ID column and namespace, Data Lead split file, checkpoint path, vocabulary path, and checkpoint-specific token length. The split must provide a `train` patient list. The benchmark filters cells by that list before seeded sampling, so cells are never sampled independently across patient partitions.

The vocabulary must be a JSON mapping from the declared gene identifier namespace to integer token IDs and must contain the configured special tokens. Gene IDs are matched exactly. The adapter reports retained, dropped, duplicated, and unmapped genes and fails if none map.

## Checks and benchmark

```sh
python scripts/check_environment.py --config config/model.yaml --json-out results/compute/environment.json --environment-export results/compute/environment_export.json
python scripts/benchmark_scgpt.py --config config/model.yaml --output results/compute/week1_scgpt_benchmark.json
```

The benchmark requires a verified checkpoint-specific scGPT model loader. The generic adapter validates mapping, deterministic sampling, input/output shapes, finite embeddings, asset hashes, and timing primitives, but it does not guess the architecture or silently load another checkpoint. A successful implementation must add the loader for the exact configured checkpoint and then record warm-up count, batch size, token length, precision, CUDA synchronization policy, wall time, peak allocated/reserved memory, cells per second, GPU-seconds per 1,000 cells, and projected GPU-seconds per 10,000 cells.

## Scientific and operational limits

A CPU-only host cannot satisfy the requested CUDA benchmark. The checker exits nonzero for `requested_device: cuda` when CUDA is unavailable. No compute cost or forward-pass success is claimed until 1,000 real cells from training patients are processed with the configured checkpoint and vocabulary.
