import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib.colors as mcolors

# Default constants
DEFAULT_ENTITY = "r-alvarezlucendo16"
DEFAULT_PROJECT = "incremental-learning"
FIGURES_DIR = Path(__file__).parent / "results"

# Strategy colors used across plots
STRATEGY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    FIGURES_DIR.mkdir(exist_ok=True)


def get_table(
    artifact_path: str,
    step: int,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> pd.DataFrame:
    """Get attention table from wandb artifact.

    Args:
        artifact_path: Path to the wandb artifact
        step: Version step of the artifact
        split: Dataset split - either "val" or "train"
        entity: Wandb entity (default: r-alvarezlucendo16)
        project: Wandb project (default: incremental-learning)

    Returns:
        DataFrame with attention weights
    """
    api = wandb.Api()
    project_path = f"{entity}/{project}"
    artifact = api.artifact(f"{project_path}/{artifact_path}:v{step}")

    # Use the correct key based on split
    table_key = f"{split}_attention_weights"
    table = artifact.get(table_key)

    if table is None:
        raise ValueError(
            f"No attention table found with key '{table_key}' in artifact {artifact_path}:v{step}"
        )

    return pd.DataFrame(data=table.data, columns=table.columns)


def plot_combined_heads_individual(
    artifact_path: str,
    step: int,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    gamma: float = 0.75,  # <1 boosts small weights (e.g., 0.6–0.8)
    per_row_norm: bool = False,  # normalize each query row per head
    colors=("#2ca02c", "#1f77b4", "#ff7f0e"),  # saturated G,R,B
) -> None:
    """Plot combined attention heads with color blending."""
    ensure_figures_dir()

    _, ax = plt.subplots(1, 1, figsize=(12, 12))
    df = get_table(artifact_path, step, split=split, entity=entity, project=project)

    head_indices = sorted(df["head"].unique())
    head_colors = {
        h: np.array(mcolors.to_rgb(colors[i % len(colors)]), dtype=float)
        for i, h in enumerate(head_indices)
    }

    query_indices = sorted(df["query_idx"].unique())
    key_indices = sorted(df["key_idx"].unique())
    nq, nk = len(query_indices), len(key_indices)

    # Accumulate products for screen and coverage
    prod_color = np.ones((nq, nk, 3), dtype=float)  # ∏(1 - w * color)
    prod_alpha = np.ones((nq, nk), dtype=float)  # ∏(1 - w)

    for h in head_indices:
        df_h = df[df["head"] == h]
        A = (
            df_h.pivot(index="query_idx", columns="key_idx", values="weight")
            .reindex(index=query_indices, columns=key_indices)
            .fillna(0.0)
            .to_numpy()
            .astype(float)
        )

        if per_row_norm:
            row_max = A.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            A = A / row_max

        if gamma != 1.0:
            A = np.power(A, gamma)

        W = np.clip(A, 0.0, 1.0)
        C = head_colors[h]  # (3,)

        prod_color *= 1.0 - W[..., None] * C[None, None, :]
        prod_alpha *= 1.0 - W

    # Screen color per channel
    S = 1.0 - prod_color  # (nq, nk, 3)
    # Coverage scalar (how far from white to move)
    t = 1.0 - prod_alpha  # (nq, nk)

    # Lift towards white so zero-weight pixels stay white
    combined_rgb = (1.0 - t)[..., None] + S
    combined_rgb = np.clip(combined_rgb, 0.0, 1.0)

    ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")

    # Draw staircase on diagonal
    offsets_by_step = {150: [0, -2], 500: [0, -2, -4], 1000: [0, -2, -4, -6]}
    diagonal_offsets = offsets_by_step.get(step, [])
    x_edges = np.arange(-0.5, nk + 0.5, 1)

    for off in diagonal_offsets:
        y_centers = np.clip(np.arange(nk) - off, 0, nq)
        y_post = np.r_[y_centers, y_centers[-1]] - 0.5
        ax.step(x_edges, y_post, where="post", color="black", linewidth=3)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"combined_heads_{step}.png", bbox_inches="tight")
    plt.close()


def fetch_run_data(
    run_id: str,
    metrics: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> Dict[str, Any]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Always include _step in metrics
    all_metrics = list(set(metrics + ["_step"]))

    # Get history data
    history_df = pd.DataFrame(run.scan_history(keys=all_metrics))

    return {
        "df": history_df,
        "config": dict(run.config),
        "name": run.name,
        "id": run.id,
    }


def plot_kl_divergence_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    learnable: bool = False,
    shift_steps: bool = True,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot KL divergence metrics over training steps."""
    ensure_figures_dir()

    # Define metrics and labels based on learnable flag
    if learnable:
        kl_metrics = {
            "kl_div_unigram_learned_val": r"$f_\theta(x_{t-2:t})$",
            "kl_div_bigram_learned_val": r"$f_\theta(x_{t-4:t-2})$",
            "kl_div_teacher_val": r"$f_\theta(x_{t-6:t-4})$",
        }
    else:
        kl_metrics = {
            "kl_div_prefix_1_teacher_val": r"$\mathbf{A^*_0}\,x_{t-13:t-12}$",
            "kl_div_prefix_2_teacher_val": r"$\mathbf{A^*_0}\,x_{t-13:t-12} + \mathbf{A^*_1}\,x_{t-11:t-8}$",
            "kl_div_prefix_3_teacher_val": r"$\mathbf{A^*_0}\,x_{t-13:t-12} + \mathbf{A^*_1}\,x_{t-11:t-8} + \mathbf{A^*_2}\,x_{t-7:t}$",
        }

    # Fetch run data
    data = fetch_run_data(run_id, list(kl_metrics.keys()), entity=entity, project=project)
    df = data["df"]

    if df.empty:
        print(f"No data found for run {run_id}")
        return

    # Limit steps and optionally shift
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps].copy()
    if shift_steps:
        plot_df["_step"] = plot_df["_step"] - 2000

    # Create the plot
    plt.figure(figsize=figsize)

    for i, (metric, label) in enumerate(kl_metrics.items()):
        if metric in plot_df.columns:
            plt.plot(
                plot_df["_step"],
                plot_df[metric],
                label=label,
                linewidth=5,
                color=STRATEGY_COLORS[i],
            )

    # Add vertical lines at divergence steps
    plt.axvline(x=330, color=STRATEGY_COLORS[0], linestyle=":", linewidth=5)
    plt.axvline(x=575, color=STRATEGY_COLORS[1], linestyle=":", linewidth=5)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kl_divergence_fig1.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_val_loss_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    shift_steps: bool = True,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot validation loss over training steps."""
    ensure_figures_dir()

    data = fetch_run_data(run_id, ["val_loss"], entity=entity, project=project)
    df = data["df"]

    if df.empty:
        print(f"No data found for run {run_id}")
        return

    # Limit steps and optionally shift
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps].copy()
    if shift_steps:
        plot_df["_step"] = plot_df["_step"] - 2000

    # Create the plot
    plt.figure(figsize=figsize)

    # Add vertical lines at divergence steps
    if divergence_steps and len(divergence_steps) >= 2:
        plt.axvline(x=330, color=STRATEGY_COLORS[0], linestyle=":", linewidth=5)
        plt.axvline(x=575, color=STRATEGY_COLORS[1], linestyle=":", linewidth=5)

    # Plot validation loss
    if "val_loss" in plot_df.columns:
        plt.plot(plot_df["_step"], plot_df["val_loss"], linewidth=5, color="black")

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "val_loss_fig1.png", bbox_inches="tight", dpi=300)
    plt.close()
