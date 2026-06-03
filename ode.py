"""Simulation of a low-rank attention-like dynamical system.

This script:
1. Builds a block-structured ground-truth matrix `Delta` representing groups.
2. Initializes attention scores `s` and direction weights `d` in several ways.
3. Defines and integrates an ODE system that tries to factorize `Delta`.
4. Optionally applies a softmax-like geometry and weight decay.
5. Produces interactive heatmaps and paper-style summary plots.
"""

import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn_image as isns
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

sns.set_context(context="notebook")

# ---------------------------------------------------------------------------
# Argument parsing and global configuration
# ---------------------------------------------------------------------------

# Parse command line arguments once at import time (kept for backward compat).
parser = argparse.ArgumentParser(
    description="Simulation script for a low-rank attention-style dynamical system."
)
parser.add_argument("--seed", type=int, default=4, help="Random seed.")
parser.add_argument(
    "--steps",
    type=int,
    default=int(1e5),
    help="Number of time points recorded over the integration window.",
)
parser.add_argument(
    "--solver",
    type=str,
    default="RK45",
    choices=["RK45", "DOP853", "Radau", "BDF", "LSODA"],
    help='ODE solver method. If stiff, try "Radau", "BDF", or "LSODA".',
)
parser.add_argument(
    "--fixed",
    action="store_true",
    help="If set, keep D fixed during the dynamics (only S evolves).",
)
parser.add_argument(
    "--init",
    type=str,
    default="random",
    choices=["random", "uniform", "bad", "adversarial", "waterfall"],
    help=(
        "Initialization of attention scores S. "
        '"random": near-uniform random, '
        '"uniform": structured near-uniform, '
        '"bad": adversarial hard case, '
        '"adversarial": all mass on a single position initially, '
        '"waterfall": hierarchical cumulative structure.'
    ),
)
parser.add_argument(
    "--scaled",
    action="store_true",
    help="If set, recompute the optimal D (least-squares) at all times.",
)
parser.add_argument(
    "--squared",
    action="store_true",
    default=True,
    help="Use squared softmax Jacobian (i.e. apply the metric twice).",
)
parser.add_argument(
    "--no-squared",
    dest="squared",
    action="store_false",
    help="Disable squared softmax Jacobian; use it only once.",
)
parser.add_argument(
    "--log",
    action="store_true",
    default=True,
    help="Plot the loss vs time on a log-scaled x-axis.",
)
parser.add_argument(
    "--no-log",
    dest="log",
    action="store_false",
    help="Disable log scaling of the time axis in the loss plot.",
)
parser.add_argument(
    "--softmax",
    action="store_true",
    default=True,
    help="If set, apply the softmax geometry (Jacobian metric) to S updates.",
)
parser.add_argument(
    "--no-softmax",
    dest="softmax",
    action="store_false",
    help="Disable the softmax metric; treat S as Euclidean coordinates.",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0,
    help="L2-style weight decay strength applied to S and D.",
)
parser.add_argument(
    "--directions",
    type=str,
    default="sparse",
    choices=["sparse", "orthogonal"],
    help=(
        '"sparse": Delta block structure in the standard basis. '
        '"orthogonal": rotate Delta by a random orthogonal matrix.'
    ),
)
parser.add_argument(
    "--color", type=str, default="viridis", help="Matplotlib colormap name."
)
parser.add_argument(
    "--T", type=int, default=int(1e5), help="Final time of the integration window."
)
parser.add_argument(
    "--B",
    type=float,
    default=1,
    help="Base value used to construct the geometric progression of group magnitudes.",
)
parser.add_argument(
    "--M",
    type=float,
    default=2.5,
    help="Multiplicative ratio between consecutive group magnitudes.",
)
parser.add_argument(
    "--H", type=int, default=3, help="Number of attention heads (columns of S and D)."
)
parser.add_argument(
    "--G",
    type=int,
    default=3,
    help="Number of groups (blocks) in the ground-truth matrix Delta.",
)
parser.add_argument(
    "--D",
    type=int,
    default=50,
    help="Number of directions / rows of D (rank upper bound).",
)
parser.add_argument(
    "--R",
    type=int,
    default=None,
    help="Effective rank of the problem (defaults to D if not provided).",
)
parser.add_argument(
    "--N", type=int, default=40, help="Sequence length (number of positions)."
)
parser.add_argument(
    "--P1",
    type=int,
    default=1,
    help="Per-group row multiplicity used to build the list I.",
)
parser.add_argument(
    "--P2",
    type=int,
    default=1,
    help="Per-group column multiplicity used to build the list J.",
)
parser.add_argument(
    "--eps",
    type=float,
    default=1e-3,
    help="Small perturbation scale used in the different S initializations.",
)
parser.add_argument(
    "--scale",
    type=float,
    default=0,
    help="Scale of features used when initializing D (for 'uniform' init).",
)
parser.add_argument(
    "--output",
    type=str,
    default="simulations.pdf",
    help="Output filename for the paper-style summary figure.",
)
parser.add_argument(
    "--tol",
    type=float,
    default=1e-12,
    help="Relative and absolute tolerance for the ODE solver.",
)

args = parser.parse_args()

# Extract parameters from arguments into short, convenient variable names.
seed = args.seed
steps = args.steps
fixed = args.fixed
init = args.init
scaled = args.scaled
squared = args.squared
log = args.log
softmax = args.softmax
weight_decay = args.weight_decay
directions = args.directions
solver = args.solver
color = args.color
T = args.T
B = args.B
M = args.M
H = args.H
G = args.G
D = args.D
R = args.R if args.R is not None else D
N = args.N
P1, P2 = args.P1, args.P2

# I and J encode how many rows/columns belong to each group of Delta.
I = [P1 for _ in range(G)]
J = [P2 for _ in range(G)]

eps = args.eps
scale = args.scale
output = args.output
tol = args.tol

# Set random seed so experiments are reproducible.
np.random.seed(seed)

# ---------------------------------------------------------------------------
# Ground-truth matrix construction (Delta)
# ---------------------------------------------------------------------------

# Construct the list of group magnitudes M. If a scalar is provided, turn it
# into a geometric progression with ratio M and base B, one value per group.
if isinstance(M, (int, float)):
    cur, base = B, M
    M = []
    for _ in range(len(J)):
        M.append(cur)
        cur *= base
    # Larger groups (later in the sequence) get larger values by default.
    M = list(reversed(M))

# Delta has shape (R, N). The top sum(I) rows are filled with block-diagonal
# M[j] values, the remaining rows (if any) stay zero.
Delta_base = np.zeros((R, N))

cur1 = 0  # Row pointer
cur2 = 0  # Column pointer
for j in range(len(J)):
    # Fill an I[j] x J[j] block with scalar M[j]
    Delta_base[cur1 : cur1 + I[j], cur2 : cur2 + J[j]] = M[j]
    cur1 += I[j]
    cur2 += J[j]

# By default, we work in the axis-aligned basis.
q = np.eye(R)
Delta = Delta_base.copy()

if directions == "orthogonal":
    # Optionally rotate Delta by a random orthogonal matrix so that the
    # block-structure is no longer axis-aligned.
    q, _ = np.linalg.qr(np.random.randn(R, R))
    Delta = q @ Delta_base


def solve_d(p, Delta):
    """Given the current state vector p, compute the optimal D (least squares).

    We treat p as the concatenation of S (first N rows) and D (last R rows):
        p.shape  = ((N + R) * H,)
        S.shape  = (N, H)
        D.shape  = (R, H)

    This function ignores the current D and recomputes it as:
        D = Delta @ pinv(S^T)
    which is the least-squares solution of Delta ≈ D S^T for fixed S.
    """
    p = p.reshape((N + R, H))
    s = p[:N, :]
    d = Delta @ np.linalg.pinv(s.T)
    return np.concatenate((s, d), axis=0).flatten()


def ode_system(t, p, Delta, scaled=False, fixed=False):
    """Right-hand side of the ODE for (S, D).

    The dynamics performs (roughly) gradient flow on the reconstruction loss
        L(S, D) = ||Delta - D S^T||_F^2
    optionally modified by:
      - a softmax-induced Riemannian metric on S (if `softmax=True`)
      - weight decay on S and D (if `weight_decay > 0`)
      - perfect re-optimization of D at each step (if `scaled=True`)
      - freezing D so only S changes (if `fixed=True`)
    """
    p = p.reshape((N + R, H))
    s = p[:N, :]
    d = p[N:, :] if not scaled else solve_d(p, Delta).reshape((N + R, H))[N:, :]

    dd = (Delta - d @ s.T) @ s if not scaled else np.zeros_like(d)
    ds = (Delta - d @ s.T).T @ d

    # Optionally project the S-update through the softmax geometry.
    # For each head i, we compute the softmax Jacobian H(s_i):
    #   H(s_i) = diag(s_i) - s_i s_i^T
    # and left-multiply the gradient ds by H (or H^2 if `squared` is set).
    if softmax:
        for i in range(H):
            h = np.diag(s[:, i]) - np.outer(s[:, i], s[:, i])
            if squared:
                h = h @ h
            ds[:, i] = h @ ds[:, i]

    # Optional L2-style weight decay on both S and D.
    if weight_decay > 0:
        if softmax:
            for i in range(H):
                h = np.diag(s[:, i]) - np.outer(s[:, i], s[:, i])
                ds[:, i] -= weight_decay * (h @ s[:, i])
        else:
            ds -= weight_decay * s
        dd -= weight_decay * d

    # If D is fixed, zero out its update.
    if fixed:
        dd = np.zeros_like(d)

    return np.concatenate((ds, dd), axis=0).flatten()


def compute_residuals(y, Delta):
    """Compute squared Frobenius reconstruction error for each time snapshot.

    Parameters
    ----------
    y : ndarray, shape ((N + R) * H, T)
        Trajectory returned by `solve_ivp`; each column is a flattened (S, D).
    Delta : ndarray, shape (R, N)
        Ground-truth matrix we are trying to factorize.
    """
    residuals = []
    for i in range(y.shape[1]):
        p = y[:, i].reshape((N + R, H))
        s = p[:N, :]
        d = p[N:, :]
        res = np.linalg.norm(Delta - d @ s.T, ord="fro") ** 2
        residuals.append(res)
    return residuals


# ---------------------------------------------------------------------------
# Initialization of S (attention scores) and D (directions)
# ---------------------------------------------------------------------------

if init == "random":
    # Start close to uniform over positions, then add a small random bump and
    # renormalize to keep columns of S on the simplex.
    s = (1.0 / N) * np.ones((N, H)) + eps * np.random.rand(N, H)
    s = np.abs(s)
    s = s / s.sum(axis=0, keepdims=True)
elif init == "uniform":
    # Start from exactly uniform and then shift mass within each group
    # to create a mild bias while staying near the simplex.
    cur = 0
    s = (1.0 / N) * np.ones((N, H))

    if eps > 0:
        for j in range(min(H, G)):
            s[:, j] -= J[j] * (eps / N)
            s[cur : cur + J[j], j] += eps
            cur += J[j]
elif init == "bad":
    # "Bad" initialization: for each head j (up to N), place almost all mass
    # on position j, with only eps elsewhere.
    s = np.zeros((N, H)) + eps
    for j in range(min(H, N)):
        s[:, j] = eps
        s[j, j] = 1.0 - eps * (N - 1)
elif init == "adversarial":
    # Strongly adversarial: put almost all mass on a single position (index 1)
    # for all heads, with only eps everywhere else.
    s = np.zeros((N, H)) + eps
    s[1, :] = 1.0 - (N - 1) * eps
elif init == "waterfall":
    # Waterfall initialization: head j puts increasing mass on the first
    # (cur + J[j]) positions, forming a cumulative "waterfall" pattern.
    cur = 0
    s = (1.0 / N) * np.ones((N, H))
    for j in range(min(H, G)):
        s[:, j] -= (cur + J[j]) * (eps / N)
        s[: cur + J[j], j] += eps * N
        cur += J[j]

# Initialize D. By default we start at zero, but for the "uniform" S init we
# can also seed D with small random features.
d = np.zeros((R, H))
if init == "uniform":
    d = scale * np.random.rand(R, H)

# Pack (S, D) into a single vector state for the ODE solver.
p = np.concatenate((s, d), axis=0).flatten()

if scaled or init == "bad":
    # If we are in "scaled" mode, or for the "bad" initialization, re-solve
    # for D exactly at time t=0 so that Delta ≈ D S^T.
    p = solve_d(p, Delta)
    s = p.reshape((N + R, H))[:N, :]
    d = p.reshape((N + R, H))[N:, :]
elif fixed:
    # If D is fixed, overwrite it with a block-structured pattern aligned with
    # the groups and keep it constant during the dynamics.
    cur = 0
    for i in range(min(H, G)):
        d[cur : cur + I[i], i] = M[i] * J[i]
        cur += I[i]
    p = np.concatenate((s, d), axis=0).flatten()    

# ---------------------------------------------------------------------------
# Solve the ODE system
# ---------------------------------------------------------------------------

# Integration window [0, T].
t_span = (0.0, T)

# Solve the ODE. We pass Delta through a lambda so that the solver only
# depends on the time t and state y, as required by `solve_ivp`.
sol = solve_ivp(
    fun=lambda t, y: ode_system(
        t, y, Delta.copy(), scaled=scaled, fixed=fixed
    ),
    t_span=t_span,
    y0=p,
    t_eval=np.linspace(t_span[0], t_span[1], steps),  # points to record
    method=solver,
    rtol=tol,
    atol=tol,  # tighter tolerances if needed
)

# If we use the "scaled" dynamics, we project every state back onto the
# manifold where D is the optimal least-squares solution for the current S.
if scaled:
    for i in range(len(sol.t)):
        p = sol.y[:, i]
        p = solve_d(p, Delta)
        sol.y[:, i] = p

# ---------------------------------------------------------------------------
# Interactive heatmaps of S, D, and Delta - D S^T at a single time point
# ---------------------------------------------------------------------------

p = p.reshape((N + R, H))
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

axs1 = fig.add_subplot(gs[0, 0])
axs2 = fig.add_subplot(gs[0, 1])
axs3 = fig.add_subplot(gs[1, 0])
axs4 = fig.add_subplot(gs[1, 1])

# First panel: heatmap of the attentions S (rows of S across heads).
im1 = axs1.imshow(p[:N, :].T, cmap=color, origin="upper", aspect="auto")
axs1.set_title("Attention scores")
axs1.set_xlabel("Position")
axs1.set_ylabel("Head")
axs1.xaxis.set_major_locator(MaxNLocator(integer=True))
axs1.yaxis.set_major_locator(MaxNLocator(integer=True))
cbar1 = fig.colorbar(im1, ax=axs1, orientation="vertical")
im1.set_clim(0, 1)  # Set color limits for the imshow object

# Second panel: heatmap of the directions D (rows of D across heads).
# For the interactive figure, we show D in the coordinates used by the ODE.
D_plot = p[N:, :]
im2 = axs2.imshow(D_plot.T, cmap=color, origin="upper", aspect="auto")
axs2.set_title("Diagonal outputs")
axs2.set_xlabel("Direction")
axs2.set_ylabel("Head")
axs2.xaxis.set_major_locator(MaxNLocator(integer=True))
axs2.yaxis.set_major_locator(MaxNLocator(integer=True))
cbar2 = fig.colorbar(im2, ax=axs2, orientation="vertical")

# Third panel: residual Delta - D S^T to show reconstruction error structure.
im3 = axs3.imshow(
    Delta - p[N:, :] @ p[:N, :].T, cmap=color, origin="upper", aspect="auto"
)
axs3.set_title("Delta - DS^T")
axs3.set_xlabel("Position")
axs3.set_ylabel("Direction")
axs3.xaxis.set_major_locator(MaxNLocator(integer=True))
axs3.yaxis.set_major_locator(MaxNLocator(integer=True))
cbar3 = fig.colorbar(im3, ax=axs3, orientation="vertical")

# --- Slider to move along the trajectory in time ---
ax_slider = fig.add_axes([0.15, 0.1, 0.7, 0.03])

# Create the slider object.
# 'valinit' is the initial value, 'valstep' is the step size (integer index).
time_slider = Slider(
    ax_slider, "Time Step", 0, len(sol.t) - 1, valinit=0, valstep=1, valfmt="%0.0f"
)

# Compute per-time-step reconstruction losses for plotting in the original basis.
residuals = compute_residuals(sol.y, Delta)
axs4.plot(sol.t, residuals)
axs4.set_xlabel("Timestep")
axs4.set_ylabel("Loss")
if log:
    axs4.set_xscale("log")
# axs4.xaxis.set_major_locator(MaxNLocator(integer=True))
axs4.yaxis.set_major_locator(MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Slider callback: update all heatmaps when the time index changes
# ---------------------------------------------------------------------------
def update(val):
    """Update the three heatmaps when the slider's value changes.

    This reads out the current time index from the slider, reshapes the state
    into (S, D), and updates:
      - the attention scores heatmap (top-left),
      - the directions heatmap (top-right),
      - the residual Delta - D S^T (bottom-left),
    including re-scaling the color limits and colorbars.
    """
    time_idx = int(time_slider.val)
    p = sol.y[:, time_idx].reshape((N + R, H))

    # Update the data for each heatmap
    data1 = p[:N, :].T
    im1.set_data(data1)

    # Update D in the current ODE coordinates.
    data2 = p[N:, :].T
    im2.set_data(data2)
    current_min2, current_max2 = data2.min(), data2.max()
    im2.set_clim(current_min2, current_max2)  # Set color limits for the imshow object
    cbar2.update_normal(im2)  # Crucially updates the colorbar ticks and display

    data3 = Delta - p[N:, :] @ p[:N, :].T
    im3.set_data(data3)
    current_min3, current_max3 = data3.min(), data3.max()
    im3.set_clim(current_min3, current_max3)  # Set color limits for the imshow object
    cbar3.update_normal(im3)  # Crucially updates the color

    # Redraw the canvas to show the changes
    fig.canvas.draw_idle()


# Register the slider callback.
time_slider.on_changed(update)

# ---------------------------------------------------------------------------
# Plots: evolution of S, D, and losses over time
# ---------------------------------------------------------------------------

# For each (group i, head j) pair we track:
#   - attns[(i, j)]: S[i, j] over time,
#   - vals[(i, j)]:  D[row_start(i), j] over time for a representative row
#     belonging to group i.
# Additionally, we track a per-group loss that aggregates the reconstruction
# error over the entire (row, column) block corresponding to each group,
# so these losses depend only on the group structure (I, J, Delta)
attns = {(i, j): [] for i in range(G) for j in range(H)}
vals = {(i, j): [] for i in range(G) for j in range(H)}
group_loss = {i: [] for i in range(G)}

# Permutation of heads; by default we simply keep the natural ordering.
perm = [i for i in range(H)]

# Precompute the row/column ranges for each group block in Delta_base.
group_row_ranges = []
group_col_ranges = []
cur_r, cur_c = 0, 0
for g in range(G):
    group_row_ranges.append((cur_r, cur_r + I[g]))
    group_col_ranges.append((cur_c, cur_c + J[g]))
    cur_r += I[g]
    cur_c += J[g]

for t_idx in range(len(sol.t)):
    p = sol.y[:, t_idx]
    p = p.reshape((N + R, H))
    s = p[:N, :]
    d = p[N:, :]

    # Record per-(group, head) S and D values using averages over the
    # corresponding column/row blocks so that P1 and P2 are taken into account.
    for i in range(G):
        r0, r1 = group_row_ranges[i]
        c0, c1 = group_col_ranges[i]
        for j in range(H):
            attns[(i, j)].append(s[c0:c1, perm[j]].sum())
            vals[(i, j)].append(d[r0:r1, perm[j]].sum())

    # Record per-group block losses in the original basis (independent of H).
    recon = d @ s.T
    recon_base = q.T @ recon
    for g in range(G):
        r0, r1 = group_row_ranges[g]
        c0, c1 = group_col_ranges[g]
        block_err = Delta_base[r0:r1, c0:c1] - recon_base[r0:r1, c0:c1]
        group_loss[g].append(np.linalg.norm(block_err, ord="fro") ** 2)

# Figure 2 shows:
#   - first row: S values for each group and head over time,
#   - second row: D values for each group and head over time,
#   - third row: losses (total and per head) over time.
fig2 = plt.figure(figsize=(11, 11))
gs = gridspec.GridSpec(nrows=3, ncols=G, figure=fig2)

axs = [[], [], []]
for i in range(2):
    for j in range(G):
        axs[i].append(fig2.add_subplot(gs[i, j]))
axs[-1].append(fig2.add_subplot(gs[-1, :]))

titles = [f"$\\mathbf{{Position}}$", f"$\\mathbf{{Feature}}$"]
for i in range(3):
    for j in range(len(axs[i])):
        axs[i][j].set_xlabel("Time")
        axs[i][j].set_xscale("log")
        axs[i][j].grid(True)
        axs[i][j].sharey(axs[i][0])
        axs[i][j].sharex(axs[i][0])
        
        if i != 1:
            axs[i][j].set_yscale("log")

        if i < len(titles):
            base_title = titles[i] + f" ($\\mathbf{{{j+1}}}$)"

            # For the first row (positions), optionally append the position range
            # [start, end] (1-indexed) only if the interval length is > 1.
            if i == 0:
                c0, c1 = group_col_ranges[j]
                if c1 - c0 > 1:
                    base_title += f" [{c0 + 1}-{c1}]"

            # For the second row (features), optionally append the feature range
            # [start, end] (1-indexed) only if the interval length is > 1.
            elif i == 1:
                r0, r1 = group_row_ranges[j]
                if r1 - r0 > 1:
                    base_title += f" [{r0 + 1}-{r1}]"

            axs[i][j].set_title(base_title)


colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "gold"
]
plot_data = [attns, vals]
for i in range(2):
    for j in range(G):
        for k in range(H):
            axs[i][j].plot(
                sol.t,
                plot_data[i][(j, k)],
                color=colors[k % len(colors)],
                label=f"$\\mathbf{{Head \ {k+1}}}$",
            )

handles, labels = axs[0][0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=H)

axs[-1][0].plot(sol.t, residuals, color="black", label="Sum", linestyle="--")
for k in range(G):
    axs[-1][0].plot(
        sol.t,
        group_loss[k],
        color=colors[k % len(colors)],
        label=f"($\\mathbf{{{k+1}, {k+1}}}$)",
    )
axs[-1][0].set_ylabel("Loss")
axs[-1][0].legend()

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(output, bbox_inches="tight")
plt.show()

