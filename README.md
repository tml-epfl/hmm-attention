# Incremental Learning of Sparse Attention Patterns in Transformers

This is the official code for the paper on [Incremental Learning of Sparse Attention Patterns in Transformers](https://arxiv.org/pdf/2602.19143) presented at [EurIPS 2025 Workshop
on Principles of Generative Modeling](https://okyksl.github.io/slides/prigm-2025/#/1) and accepted to ICML 2026 Main Conference.

The `analysis/` folder contains notebooks for regenerating the paper plots from
the W&B project `r-alvarezlucendo16/incremental-learning`.

## Installation

```bash
uv sync
```

## Running Experiments

```bash
# List available experiments
bash run.sh

# Run a specific experiment
bash run.sh <experiment_name>
```

## Configuration

Experiments are configured using [Hydra](https://hydra.cc/) with configs located in `conf/`.

- **Experiment configs** in `conf/experiments/` override base settings from `conf/train.yaml`
- **Component configs** can be customized: `model/`, `dataset/`, `optimizer/`, `scheduler/`, `loss/`
