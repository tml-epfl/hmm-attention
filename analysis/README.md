# Analysis Notebooks

This folder contains cleaned notebooks for reproducing the plots in the report
`gfe23Z-2602.19143v1.pdf`. Each notebook states the report figure it supports,
the W&B tag or run ID used, and a short explanation of how the plot relates to
the paper.

Most paper plots in this folder are pulled from the W&B project
`r-alvarezlucendo16/incremental-learning` and regenerated locally by the
corresponding notebook.

## Figure Map

| Report figure | Notebook | Source |
| --- | --- | --- |
| Figure 1 | `fig01-overview.ipynb` | W&B tag `ICLR-Fig1`, selected run `i0a1de0a` |
| Figures 2-3 | `fig02-03-full-model.ipynb` | W&B tags `ICLR-full`, `ICLR-dataset`, run `dbrr8opu` |
| Figure 4 left | `fig04a-init-scales.ipynb` | W&B tag `ICLR-Fig1` |
| Figure 4 right | `fig04b-multiplicative-constant.ipynb` | W&B tag `ablation-mc` |
| Figure 5 | `fig05-dataset-size.ipynb` | W&B tag `ICLR-minimal-dataset` |
| Figures 6-7 | `fig06-07-infinite-data.ipynb` | W&B tag `ICLR-infinite-data`, run `p6mze2ux` |
| Figures 8-9 | `fig08-09-reverse-importance.ipynb` | W&B tag `ICLR-reversed`, run `seo4z7uz` |
| Figure 10 | `fig10-gradient-flow-simulation.ipynb` | Local gradient-flow simulation from Equation (4) |
| Figures 11-12 | `fig11-12-two-layer-transformers.ipynb` | W&B tags `2-layer`, `2-layer-full` |
| Figures 13-14 | `fig13-14-non-uniform-alpha.ipynb` | W&B tag `non-uniform-span-weights`, run `mgj8u2o3` |
| Figures 15-17 | `fig15-17-overlapping-intervals.ipynb` | W&B tag `overlapping-spans`, runs `c9pd8yft`, `d5w941d1` |
| Figures 18-19 | `fig18-19-sgd.ipynb` | W&B tag `sgd-full-1layer`, run `m0qjhj8n` |
| ICML rebuttal Figure B | `figB-rebuttal-head-dynamics.ipynb` | W&B tag `ICML-minimal-transformer-small-init-orthogonal`, run `05c1tr5d` |

## Utilities

- `utils.py` centralizes W&B fetching and plotting helpers.
- `fetch_runs()` and `fetch_run_data()` use the default project
  `r-alvarezlucendo16/incremental-learning`.
- `REPORT_HEAD_COLORS` and `report_head_color_map()` keep colored heads in the
  report order: blue, yellow, green. Pass raw W&B head indices to
  `report_head_color_map()` in that order whenever a run needs a permutation.
- `style_report_axis()` applies the report's clean curve-plot style: large axis
  labels and ticks, framed legends, and no grid unless a specific figure calls
  for one.
- `plot_combined_heads()` accepts either a bare run ID or the full W&B artifact
  path `run-<id>-val_attention_weights`.
- `plot_separated_heads()` is used for the overlapping-interval appendix
  attention grids.

## Usage

Run notebooks from the repository root so imports resolve cleanly:

```python
from analysis.utils import fetch_runs, plot_kl_divergence_simple
```

Each notebook defines `SAVE_FIGURES = False` by default. Set it to `True` to
write PDFs into `analysis/figures/`, which is ignored by git.
