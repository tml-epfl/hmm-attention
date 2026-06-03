import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Union, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from pathlib import Path
import hashlib
import pickle

# Default constants
DEFAULT_ENTITY = "r-alvarezlucendo16"
DEFAULT_PROJECT = "incremental-learning"


def get_table(
    artifact_path: str,
    step: int,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Get attention table from wandb artifact with pickle caching.

    Args:
        artifact_path: Path to the wandb artifact
        step: Version step of the artifact
        split: Dataset split - either "val" or "train"
        entity: Wandb entity (default: r-alvarezlucendo16)
        project: Wandb project (default: incremental-learning)
        use_cache: Whether to use cached pickle files (default: True)

    Returns:
        DataFrame with attention weights
    """
    # Create cache directory
    cache_dir = Path("figures/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename based on parameters
    cache_key = f"{entity}_{project}_{artifact_path}_{step}_{split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        return pd.read_pickle(cache_file)

    # Download from wandb
    print(f"Downloading from wandb: {artifact_path}:v{step} ({split})")
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

    df = pd.DataFrame(data=table.data, columns=table.columns)

    # Save to cache
    if use_cache:
        df.to_pickle(cache_file)
        print(f"Saved to cache: {cache_file.name}")

    return df


def plot_attention(
    artifact_path: str,
    steps: List[int],
    save_name: Optional[str] = None,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot attention heatmaps for given steps.

    Args:
        artifact_path: Path to the wandb artifact
        steps: List of steps to plot
        save_name: Name for saved figure (optional)
        split: Dataset split - either "val" or "train"
        entity: Wandb entity (default: r-alvarezlucendo16)
        project: Wandb project (default: incremental-learning)
    """
    steps = list(steps)
    head_indices = [0, 1, 2]
    n_rows, n_cols = len(steps), len(head_indices)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10 * n_cols, 8 * n_rows),
        sharey=False,
    )

    # Ensure axes is always 2D for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, step in enumerate(steps):
        df = get_table(artifact_path, step, split=split, entity=entity, project=project)

        for col_idx, head_idx in enumerate(head_indices):
            ax = axes[row_idx][col_idx]

            df_head = df[df["head"] == head_idx]

            # --- pivot & silence the names so they don't show up as labels ----
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")
            attn.index.name = None  # no "query_idx"
            attn.columns.name = None  # no "key_idx"
            # ------------------------------------------------------------------

            heatmap = sns.heatmap(
                attn,
                cmap="Blues",
                xticklabels=attn.columns,  # numeric positions
                yticklabels=attn.index,  # numeric positions
                ax=ax,
                vmin=0.0,
                vmax=1.0,
                cbar=(col_idx == n_cols - 1),  # one colour-bar per row
            )

            if col_idx == n_cols - 1:
                cbar = heatmap.collections[0].colorbar
                cbar.ax.tick_params(labelsize=24)

            if row_idx == 0:
                ax.set_title(f"$\\mathbf{{Head~{head_idx}}}$", fontsize=32, pad=15)

            # show y-label only on first column
            if col_idx == 0:
                ax.set_ylabel("Query Positions", fontsize=28, labelpad=15)
            else:
                ax.set_ylabel("")

            # show x-label only on last row
            if row_idx == n_rows - 1:
                ax.set_xlabel("Key Positions", fontsize=28, labelpad=15)
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="x", labelrotation=90, labelsize=24)
            ax.tick_params(axis="y", labelsize=24, labelrotation=0)

    fig.suptitle(
        f"Attention Heads at $\\mathbf{{Step~{step}}}$ ({split.title()})",
        fontsize=36,
        y=0.98,
    )

    # Tighter layout
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.90, bottom=0.08, left=0.06, right=0.95, wspace=0.15, hspace=0.25
    )

    if save_name:
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_name}.pdf")
    else:
        plt.show()


def plot_combined_heads(
    run_id: str,
    steps: Union[int, List[int]],
    frequency: int = 1,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    save_name: Optional[str] = None,
    head_colors: Optional[Dict[int, str]] = None,
    staircases: Optional[Dict[int, List[int]]] = None,
) -> None:
    artifact_path = f"run-{run_id}-val_attention_weights"

    if isinstance(steps, int):
        steps = [steps]

    steps = [step // frequency for step in steps]
    n_cols = len(steps)

    fig, axes = plt.subplots(1, n_cols, figsize=(12 * n_cols, 12), sharey=False)

    # Ensure axes is always iterable for consistent indexing
    if n_cols == 1:
        axes = [axes]

    # Define colors for each head - using distinct colors
    default_colors = [
        "#2ca02c",  # Green
        "#ff7f0e",  # Orange
        "#1f77b4",  # Blue
    ]

    for col_idx, step in enumerate(steps):
        ax = axes[col_idx]

        df = get_table(artifact_path, step, split=split, entity=entity, project=project)

        # Get unique head indices
        head_indices = sorted(df["head"].unique())

        # Use provided head_colors or fall back to defaults
        if head_colors is None:
            colors_map = {
                head: default_colors[i % len(default_colors)] for i, head in enumerate(head_indices)
            }
        else:
            colors_map = head_colors

        # Get the dimensions
        first_head_df = df[df["head"] == head_indices[0]]
        query_indices = sorted(first_head_df["query_idx"].unique())
        key_indices = sorted(first_head_df["key_idx"].unique())

        # Start with white background
        combined_rgb = np.ones(
            (len(query_indices), len(key_indices), 3)
        )  # Start with white

        # For each head, subtract from white based on attention weights
        for head_idx in head_indices:
            df_head = df[df["head"] == head_idx]
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")

            # Convert color name to RGB
            color_rgb = np.array(plt.cm.colors.to_rgb(colors_map[head_idx]))

            # For each position, blend the head's attention with its color
            for i, query_idx in enumerate(query_indices):
                for j, key_idx in enumerate(key_indices):
                    weight = attn.loc[query_idx, key_idx]
                    # Blend: white background becomes more colored based on attention weight
                    combined_rgb[i, j] = (
                        combined_rgb[i, j] * (1 - weight) + color_rgb * weight
                    )

        # Display the combined attention matrix
        ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")

        # ---- Draw staircase on a chosen diagonal ----
        if staircases is not None:
            offsets_for_this_step = staircases.get(col_idx)
        else:
            offsets_for_this_step = None

        if offsets_for_this_step is not None:
            nq = len(query_indices)
            nk = len(key_indices)
            x_edges = np.arange(-0.5, nk + 0.5, 1)
            for off in offsets_for_this_step:
                # diagonal row indices (centers), clipped to matrix bounds
                y_centers = np.clip(np.arange(nk) - off, 0, nq)
                # convert to edge coords and append last value to make length nk+1
                y_post = np.r_[y_centers, y_centers[-1]] - 0.5
                ax.step(x_edges, y_post, where="post", color="black", linewidth=3)

        # Set ticks and labels to match seaborn heatmap (origin='upper' makes 0 at top)
        ax.set_xticks(range(len(key_indices)))
        ax.set_xticklabels(key_indices, fontsize=32, rotation=90)
        ax.set_yticks(range(len(query_indices)))
        ax.set_yticklabels(query_indices, fontsize=32)
        if col_idx == 0:
            ax.set_ylabel("Query Positions", fontsize=55)

        ax.set_xlabel("Key Positions", fontsize=55)
        ax.set_title(f"$\\mathbf{{Step~{step*frequency}}}$", fontsize=70, pad=20)

    plt.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_name}.pdf")
    else:
        plt.show()


def plot_separated_heads(
    run_id: str,
    steps: Union[int, List[int]],
    frequency: int = 1,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    save_name: Optional[str] = None,
    head_colors: Optional[Dict[int, str]] = None,
    staircases: Optional[Dict[tuple, List[int]]] = None,
) -> None:
    
    artifact_path = f"run-{run_id}-val_attention_weights"

    if isinstance(steps, int):
        steps = [steps]

    steps = [step // frequency for step in steps]
    n_rows = len(steps)

    # Get first dataframe to determine number of heads
    df_first = get_table(artifact_path, steps[0], split=split, entity=entity, project=project)
    head_indices = sorted(df_first["head"].unique())
    n_cols = len(head_indices)

    # Define colors for each head - using distinct colors
    default_colors = [
        "#2ca02c",  # Green
        "#ff7f0e",  # Orange
        "#1f77b4",  # Blue
    ]

    # Use provided head_colors or fall back to defaults
    if head_colors is None:
        colors_map = {
            head: default_colors[i % len(default_colors)] for i, head in enumerate(head_indices)
        }
    else:
        colors_map = head_colors

    # Create figure with grid layout
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 8 * n_rows),
        sharey=False,
    )

    # Ensure axes is always 2D for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, step in enumerate(steps):
        df = get_table(artifact_path, step, split=split, entity=entity, project=project)

        # Get dimensions from first head
        first_head_df = df[df["head"] == head_indices[0]]
        query_indices = sorted(first_head_df["query_idx"].unique())
        key_indices = sorted(first_head_df["key_idx"].unique())

        for col_idx, head_idx in enumerate(head_indices):
            ax = axes[row_idx][col_idx]

            # Get attention data for this head
            df_head = df[df["head"] == head_idx]
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")

            # Create custom colormap from white to head color
            head_color = colors_map[head_idx]
            
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("custom", ["white", head_color])

            # Create single-color heatmap
            im = ax.imshow(
                attn.values,
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )

            # Draw staircases if provided for this subplot
            if staircases is not None:
                offsets_for_this_subplot = staircases.get((row_idx, col_idx))

                if offsets_for_this_subplot is not None:
                    nq = len(query_indices)
                    nk = len(key_indices)
                    x_edges = np.arange(-0.5, nk + 0.5, 1)

                    for off in offsets_for_this_subplot:
                        # diagonal row indices (centers), clipped to matrix bounds
                        y_centers = np.clip(np.arange(nk) - off, 0, nq)
                        # convert to edge coords and append last value to make length nk+1
                        y_post = np.r_[y_centers, y_centers[-1]] - 0.5
                        ax.step(x_edges, y_post, where="post", color="black", linewidth=3)

            # Set ticks and labels
            ax.set_xticks(range(len(key_indices)))
            ax.set_xticklabels(key_indices, fontsize=24, rotation=90)
            ax.set_yticks(range(len(query_indices)))
            ax.set_yticklabels(query_indices, fontsize=24)

            # Y-label only on first column
            if col_idx == 0:
                ax.set_ylabel("Query Positions", fontsize=28, labelpad=15)
            else:
                ax.set_ylabel("")

            # X-label only on last row
            if row_idx == n_rows - 1:
                ax.set_xlabel("Key Positions", fontsize=28, labelpad=10)
            else:
                ax.set_xlabel("")

            # Titles: head index on top row, step number on left column
            if row_idx == 0:
                ax.set_title(f"$\\mathbf{{Head~{head_idx + 1}}}$", fontsize=32, pad=15)

            # Add step label on the left side
            if col_idx == 0:
                # Add text label outside the plot
                ax.text(
                    -0.25, 0.5,
                    f"$\\mathbf{{Step~{step*frequency}}}$",
                    transform=ax.transAxes,
                    fontsize=32,
                    va='center',
                    ha='right',
                    rotation=90,
                )

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08, wspace=0.2, hspace=0.25)

    if save_name:
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_name}.pdf")
    else:
        plt.show()


def fetch_run_data(
    run_id: str,
    metrics: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> Dict[str, Any]:
    """
    Fetch data for a specific run by run_id.

    Args:
        run_id: The wandb run ID
        metrics: List of metric names to fetch
        entity: The wandb entity (default: r-alvarezlucendo16)
        project: The wandb project name (default: incremental-learning)

    Returns:
        Dictionary containing run data and config
    """
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
    shift_steps: int = 0,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    save_name: Optional[str] = None,
) -> None:
    if learnable:
        metrics = [
            "kl_div_unigram_learned_val",
            "kl_div_bigram_learned_val",
            "kl_div_teacher_val",
        ]
    else:
        metrics = [
            "kl_div_prefix_1_teacher_val",
            "kl_div_prefix_2_teacher_val",
            "kl_div_prefix_3_teacher_val",
        ]

    # Fetch run data
    data = fetch_run_data(run_id, metrics, entity=entity, project=project)
    df = data["df"]

    if df.empty:
        print(f"No data found for run {run_id}")
        return

    # Define the KL divergence metrics and their labels based on learnable flag
    if learnable:
        # Use learned model metrics and labels
        kl_metrics = {
            "kl_div_unigram_learned_val": "4-gram",
            "kl_div_bigram_learned_val": "8-gram",
            "kl_div_teacher_val": "12-gram",
        }
    else:
        # Use teacher metrics and labels with bold matrices and asterisks
        kl_metrics = {
            "kl_div_prefix_1_teacher_val": r"${A^{*}_{1}}$",
            "kl_div_prefix_2_teacher_val": r"${A^{*}_{1:2}}$",
            "kl_div_prefix_3_teacher_val": r"${A^{*}_{1:3}}$",
        }

    # Get data, limiting steps if specified
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]

    # Optionally adjust step numbering
    plot_df = plot_df.copy()
    if shift_steps != 0:
        plot_df["_step"] = plot_df["_step"] - shift_steps

    # Create the plot
    plt.figure(figsize=figsize)
    x_min = plot_df["_step"].min()
    x_max = plot_df["_step"].max()
    plt.margins(x=0)  # remove automatic x padding
    plt.xlim(x_min, x_max)  # clamp to data range

    # Add background coloring for strategy regions if divergence_steps provided
    if divergence_steps and len(divergence_steps) >= 2:
        # Define strategy colors (matching line colors)
        strategy_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
        ]

        # Adjust divergence steps to match the new numbering
        adjusted_divergence_steps = (
            [step - shift_steps for step in divergence_steps]
            if shift_steps != 0
            else divergence_steps
        )

        # Get the x-axis limits with some padding to ensure full coverage
        x_min = plot_df["_step"].min()
        x_max = plot_df["_step"].max()

        # Extend the range slightly to avoid gaps
        x_range = x_max - x_min
        x_min_extended = x_min - 0.01 * x_range
        x_max_extended = x_max + 0.01 * x_range

        # Region 1: Unigram strategy (blue background)
        plt.axvspan(
            x_min_extended,
            adjusted_divergence_steps[0],
            alpha=0.2,
            color=strategy_colors[0],
        )

        # Region 2: Bigram strategy (orange background)
        plt.axvspan(
            adjusted_divergence_steps[0],
            adjusted_divergence_steps[1],
            alpha=0.2,
            color=strategy_colors[1],
        )

        # Region 3: Trigram strategy (green background)
        plt.axvspan(
            adjusted_divergence_steps[1],
            x_max_extended,
            alpha=0.2,
            color=strategy_colors[2],
        )

    # Plot each KL divergence metric
    for i, (metric, label) in enumerate(kl_metrics.items()):
        if metric in df.columns:
            plt.plot(plot_df["_step"], plot_df[metric], label=label, linewidth=4)

    # Formatting
    plt.xlabel("Training Step", fontsize=30)
    plt.ylabel("KL Divergence", fontsize=30)
    plt.legend(fontsize=32, loc="upper right", framealpha=1)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(f"{save_name}.pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_name}.pdf")
    else:
        plt.show()


def plot_val_loss_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    shift_steps: int = 0,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    data = fetch_run_data(run_id, ["val_loss"], entity=entity, project=project)
    df = data["df"]
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]
    plot_df = plot_df.copy()
    
    if shift_steps != 0:
        plot_df["_step"] = plot_df["_step"] - shift_steps

    # Create the plot
    plt.figure(figsize=figsize)
    plt.margins(x=0)  # remove automatic x padding

    # Add background coloring for strategy regions if divergence_steps provided
    if divergence_steps and len(divergence_steps) >= 2:
        # Define strategy colors (matching line colors)
        strategy_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

        # Adjust divergence steps to match the new numbering
        adjusted_divergence_steps = (
            [step - shift_steps for step in divergence_steps]
            if shift_steps != 0
            else divergence_steps
        )

        # Get the x-axis limits with some padding to ensure full coverage
        x_min = plot_df["_step"].min()
        x_max = plot_df["_step"].max()

        # Extend the range slightly to avoid gaps
        x_range = x_max - x_min
        x_min_extended = x_min - 0.01 * x_range
        x_max_extended = x_max + 0.01 * x_range

        # Region 1: Unigram strategy (blue background)
        plt.axvspan(
            x_min_extended,
            adjusted_divergence_steps[0],
            alpha=0.2,
            color=strategy_colors[0],
        )

        # Region 2: Bigram strategy (orange background)
        plt.axvspan(
            adjusted_divergence_steps[0],
            adjusted_divergence_steps[1],
            alpha=0.2,
            color=strategy_colors[1],
        )

        # Region 3: Trigram strategy (green background)
        plt.axvspan(
            adjusted_divergence_steps[1],
            x_max_extended,
            alpha=0.2,
            color=strategy_colors[2],
        )

    # Plot validation loss (without label for legend)
    if "val_loss" in plot_df.columns:
        plt.plot(plot_df["_step"], plot_df["val_loss"], linewidth=2, color="black")

    # Formatting
    plt.xlabel("Training Step", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Add custom legend for strategy regions if divergence_steps provided
    if divergence_steps and len(divergence_steps) >= 2:
        from matplotlib.patches import Rectangle

        strategy_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        strategy_labels = [
            r"$A^*_1$",
            r"$A^*_{1:2}$",
            r"$A^*_{1:3}$",
        ]

        # Create legend patches
        legend_patches = [
            Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2)
            for color in strategy_colors
        ]

        plt.legend(
            legend_patches,
            strategy_labels,
            fontsize=32,
            loc="upper right",
            framealpha=1,
        )

    plt.tight_layout()
    if save_name:
        plt.savefig(f"{save_name}.pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {save_name}.pdf")
    else:
        plt.show()
