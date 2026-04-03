# Graphics-LPIPS for 3D Mesh Quality Assessment

This document is the public-facing README for the current state of this repository.

The root [`README.md`](../README.md) is intentionally kept as the legacy upstream-style document inherited from the original project history. It is still useful for background, but it does not reflect the full workflow, file layout, and scripts currently used in this fork.

Graphics-LPIPS extends the LPIPS idea to **textured 3D mesh quality assessment**. Instead of comparing isolated 2D image distortions only, this repository focuses on rendered views of 3D objects, patch-based perceptual scoring, MOS-based supervision, and experiment pipelines for training, evaluation, and correlation analysis.

## What This Repository Adds Compared to the Original Project

Compared to the original Graphics-LPIPS / LPIPS-style workflow, this repository now includes:

- a workflow centered on **textured mesh quality assessment** rather than generic image similarity alone
- support for **MOS / DSIS-style supervision** instead of only preference-style formulations
- **multi-view evaluation** of 3D objects
- a newer evaluation path with **in-memory patch extraction** through `Light_GraphicsLPIPS_csv.py`, avoiding the need to keep a fully patchified dataset on disk
- **k-fold evaluation** support
- structured experiment outputs under `out/...`
- utilities for **correlation plots**, **per-fold summaries**, and **experiment-level CSV reports**
- training and evaluation code adapted to the current research workflow used in this repository

In short: the project is no longer just a small adaptation of LPIPS. It is now a fuller experimental codebase for perceptual quality prediction on rendered 3D content.

## Recommended Current Pipeline

If you are new to this repository, start with this mental model:

1. Train a model with `train.py` or reuse an existing checkpoint from `checkpoints/`.
2. Evaluate a dataset or experiment with `Light_GraphicsLPIPS_csv.py`.
3. Compute correlation summaries and plots with `correlation_VP.py`.

Legacy scripts such as `GraphicsLpips_csvFile.py` and `GraphicsLpips_2imgs.py` are still present, but they should be treated as compatibility/reference utilities rather than the main public entrypoint.

## Repository Map

The most important files and folders are:

- `train.py`: main training entrypoint for Graphics-LPIPS models
- `Light_GraphicsLPIPS_csv.py`: recommended evaluation script for current multi-view experiments
- `correlation_VP.py`: computes Pearson/Spearman-style summaries, regression plots, and fold statistics
- `GraphicsLpips_csvFile.py`: historical evaluation script based on a fully patchified dataset already stored on disk
- `GraphicsLpips_2imgs.py`: quick patch-to-patch distance example
- `checkpoints/`: trained model weights
- `dataset/`: CSV splits, folds, judges/MOS-related files, and optional patchified data
- `out/`: generated experiment outputs, metric CSVs, plots, and correlation summaries
- `lpips/`: model code, pretrained backbones, and training utilities
- `data/`: PyTorch data loading pipeline

## Installation

Install PyTorch and torchvision first, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone <YOUR_PUBLIC_REPOSITORY_URL>
cd Graphics-LPIPS
```

Notes:

- This repository currently contains several Windows-specific assumptions in some scripts.
- Some paths are still hardcoded in the codebase and may need cleanup before a fully portable public release.
- The `requirements.txt` file should be reviewed before release if you plan to publish a clean reproducible environment.

## Expected Data Layout

There are two main evaluation styles in this repository.

### 1. Legacy patchified workflow

This is the workflow used by `GraphicsLpips_csvFile.py`.

It expects:

- a CSV describing the test stimuli
- a directory of reference patches already exported to disk
- a directory of distorted patches already exported to disk

This is close to the original historical setup.

### 2. Current multi-view workflow

This is the workflow used by `Light_GraphicsLPIPS_csv.py`.

It expects rendered views and patch metadata rather than a fully materialized patch dataset:

- reference objects stored under a `Source/...` tree
- distorted objects stored under a `Distorted/...` tree
- `views/` folders containing images such as `view_1.png`
- a patch list CSV for each reference object describing patch coordinates

Typical structures look like:

```text
<SRC_ROOT>/
  Source/
    <N>VP/
      <REF_OBJECT>/
        views/
          view_1.png
          view_2.png
        patchs/
          <REF_OBJECT>_patchlist.csv
  Distorted/
    <N>VP/
      <DISTORTED_OBJECT>/
        views/
          view_1.png
          view_2.png
```

Important note: some scripts use the folder name `patchs` instead of `patches`. This is a historical convention in the current codebase and an easy source of confusion.

## Quick Start

If you have just cloned the repository and want to avoid getting lost, use this rule of thumb:

- If you want to compare two image patches: use `GraphicsLpips_2imgs.py`
- If you want to evaluate a modern multi-view experiment: use `Light_GraphicsLPIPS_csv.py`
- If you want to train a model: use `train.py`
- If you are reading old notes, scripts, or papers that mention patchified datasets on disk: check `GraphicsLpips_csvFile.py`

Generated results usually end up in:

- `checkpoints/<EXPERIMENT_NAME>/` for trained weights
- `out/<DATABASE>/<RENDER_METHOD>/<VIEW_METHOD>/<MODEL>/<N>VP/` for evaluation outputs

## Typical Use Cases

### 1. Compare two patches directly

This is the smallest possible sanity check:

```bash
python GraphicsLpips_2imgs.py \
  -p0 ./imgs/ex_ref.png \
  -p1 ./imgs/ex_p0.png \
  -m ./checkpoints/<MODEL_NAME>/latest_net_.pth \
  --use_gpu
```

What it does:

- loads a reference patch and a distorted patch
- runs the Graphics-LPIPS network
- prints a single perceptual distance score

### 2. Run the legacy patchified-dataset evaluation

Use this only if your data is already exported as patch images on disk:

```bash
python GraphicsLpips_csvFile.py \
  -f ./dataset/<TEST_LIST_CSV>.csv \
  -m ./checkpoints/<MODEL_NAME>/latest_net_.pth \
  -o ./GLPIPS_scores.csv \
  --use_gpu
```

What it does:

- reads a test CSV listing stimuli and patch counts
- loads reference and distorted patches from disk
- computes an average score per stimulus
- writes a CSV of results and produces simple regression outputs

This is the historical pipeline and is no longer the best reflection of the current repository workflow.

### 3. Run the recommended multi-view evaluation

This is the main evaluation script for the current codebase:

```bash
python Light_GraphicsLPIPS_csv.py \
  -m <MODEL_NAME> \
  -v <N_VIEWS> \
  -vm <VIEW_METHOD> \
  -rm <RENDER_METHOD> \
  -db <DATABASE_NAME> \
  -mos <MOS_CSV> \
  -testlist <TEST_LIST_CSV> \
  --use_gpu
```

Example:

```bash
python Light_GraphicsLPIPS_csv.py \
  -m TMQ_NR_4VP_yf03_kfolds \
  --use_folds \
  -v 4 \
  -vm Y_fixed_0.3 \
  -rm New_Render \
  -db TMQ \
  -mos ./dataset/TMQ/<MOS_FILE>.csv \
  -testlist ./dataset/TMQ/<TEST_LIST>.csv \
  --use_gpu
```

What it does:

- loads the checkpoint from `./checkpoints/<MODEL_NAME>/...`
- reads the test list and MOS file
- reconstructs patches in memory from rendered views
- computes scores per view and per distorted object
- writes per-object `GLPIPS_results_testset.csv` files under `out/...`

### 4. Compute correlation summaries and plots

After evaluation, run:

```bash
python correlation_VP.py \
  -m <MODEL_NAME> \
  -v <N_VIEWS> \
  -vm <VIEW_METHOD> \
  -rm <RENDER_METHOD> \
  -db <DATABASE_NAME>
```

If your model is evaluated with folds:

```bash
python correlation_VP.py \
  -m <MODEL_NAME> \
  --use_folds \
  -v <N_VIEWS> \
  -vm <VIEW_METHOD> \
  -rm <RENDER_METHOD> \
  -db <DATABASE_NAME>
```

What it does:

- reads the `GLPIPS_results_testset.csv` files produced during evaluation
- averages viewpoint scores when needed
- computes correlations
- writes fold-level and experiment-level summary CSVs
- saves plots in the experiment output folder

### 5. Train a new model

Training is driven by `train.py`.

Minimal command shape:

```bash
python train.py \
  --datasets <TRAIN_CSV> \
  --testcsv <TEST_CSV> \
  --src_root <SRC_ROOT> \
  --root_refPatches <REF_RELATIVE_PATH> \
  --root_distPatches <DIST_RELATIVE_PATH> \
  --name <EXPERIMENT_NAME> \
  --target mos \
  --net alex \
  --npatches 150 \
  --nInputImg 4 \
  --nepoch 5 \
  --nepoch_decay 5 \
  --use_gpu
```

Example:

```bash
python train.py \
  --datasets ./dataset/TMQ/folds/<TRAIN_SPLIT>.csv \
  --testcsv ./dataset/TMQ/folds/<TEST_SPLIT>.csv \
  --src_root <SRC_ROOT> \
  --root_refPatches Source/4VP \
  --root_distPatches Distorted/4VP \
  --name TMQ_NR_4VP_example \
  --target mos \
  --net alex \
  --npatches 150 \
  --nInputImg 4 \
  --nepoch 5 \
  --nepoch_decay 5 \
  --use_folds \
  --use_gpu
```

What it does:

- loads stimuli from CSV files
- samples patches per stimulus
- aggregates patch scores into a stimulus-level prediction
- trains the linear calibration layers on top of the selected backbone
- saves checkpoints under `checkpoints/<EXPERIMENT_NAME>/`

## Outputs and Results

Depending on the workflow, the most useful outputs are:

- `checkpoints/<EXPERIMENT_NAME>/`: saved model weights
- `out/.../_METRIC_RESULTS_TESTSET_/.../GLPIPS_results_testset.csv`: per-object evaluation results
- `out/.../correlation_folds_stats.csv`: fold-level summary
- `out/.../correlation_summary_kfolds.csv`: experiment-level summary

When debugging a run, the first question is usually: did the expected files appear under `out/...` or `checkpoints/...`?

## Legacy vs Current Scripts

### Recommended for new users

- `train.py`
- `Light_GraphicsLPIPS_csv.py`
- `correlation_VP.py`

### Useful wrappers or helpers

- `GraphicsLpips_2imgs.py`

### Historical / compatibility-oriented

- `GraphicsLpips_csvFile.py`

If you are unsure where to start, choose the recommended path unless you already know you need the legacy patchified workflow.

## Current Limitations and Important Notes

Before publishing or reusing this repository on another machine, keep the following in mind:

- several scripts still contain **Windows-specific absolute paths**
- some scripts still assume a specific local research environment
- not all datasets, checkpoints, generated plots, and `out/` files should be committed to a public repository
- some identifiers are historical:
  - `2afc` in the data loader naming
  - `judge` in places where the target may actually be MOS
- the `patchs` / `patches` naming mismatch can break setups if not handled carefully
- some scripts still mix older and newer workflow assumptions

This means the repository is already very useful for research and experimentation, but still benefits from cleanup before being treated as a turnkey public package.

## Notes Before Public Release

If you plan to publish this repository as a clean public version, the most useful next steps are:

- remove or generalize hardcoded local paths
- exclude `dataset/`, `out/`, `checkpoints/`, caches, and generated artifacts from the public repository unless they are intentionally shared
- keep only lightweight examples and documentation in version control
- document how to obtain datasets and how to reproduce experiments without bundling large private or generated files

## Related Reference

This repository is based on the Graphics-LPIPS / LPIPS research line and is connected to the textured mesh quality assessment work described in the project history and the legacy root `README.md`.

For the original historical description of the project, see the root [`README.md`](../README.md).
