"""Hidden-state belief probe for HierarchicalTeacher + TransformerDecoder.

For each residual position `t` (post any of the transformer's layers), fits a
linear probe that decodes the true hidden id of chunk `c(t) + k`, where
`c(t) = t // chunk_size` and `k` is a configurable chunk offset. Three regimes
emerge from one target family:

    k <  0 : retention        (target chunk fully observed, Bayes = 100%)
    k == 0 : belief refinement (sharpens with within-chunk slot)
    k == +1: planning/lookahead (Bayes = base-teacher next-hidden accuracy)

Two fitting modes are supported:

    "warm_start" : LBFGS re-fit each eval step from previous weights.
    "sgd"        : Adam step per training step, measured at eval time.

Both share the same measurement path (accuracy + NLL on a val holdout split).
See LoggingConfig.probe_* fields for cfg.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.model import TransformerDecoder
from src.teachers import HierarchicalTeacher
from src.trainer.config import LoggingConfig

VALID_MODES = ("off", "warm_start", "sgd")


class ProbeLogger:
    """Owns residual capture, probe fitting, and per-eval logging.

    No-op unless `teacher` is a HierarchicalTeacher and `student` is a
    TransformerDecoder. `mode == 'off'` also disables the logger entirely.

    Lifecycle expected from the trainer:

        # in _init_loop
        probe_logger = ProbeLogger(writer, teacher, student, cfg)

        # around each student forward — training and val alike
        probe_logger.before_forward()
        (student forward runs — hooks capture residuals)
        probe_logger.after_forward(data)

        # after each training step (only meaningful in sgd mode)
        probe_logger.sgd_step()

        # per val batch, after after_forward
        probe_logger.collect_val_batch()

        # once per eval, after all val batches
        probe_logger.log(step, split='val')
    """

    def __init__(
        self,
        writer: Optional["wandb.run"],
        teacher: nn.Module,
        student: nn.Module,
        cfg: LoggingConfig,
    ) -> None:
        self.writer = writer
        self.teacher = teacher
        self.student = student
        self.cfg = cfg
        self.mode = cfg.probe_mode
        self.logger = logging.getLogger()

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"probe_mode must be one of {VALID_MODES}; got {self.mode!r}"
            )

        self.enabled = (
            self.mode != "off"
            and isinstance(teacher, HierarchicalTeacher)
            and isinstance(student, TransformerDecoder)
        )
        if not self.enabled:
            return

        # Adaptive default: cover the base teacher's AR context window on the
        # retention side, plus one step of lookahead. `burn_in` == context_length
        # for bounded bases; for an adaptive (unbounded) base there is no finite
        # window, so burn_in is the finite fallback — override via probe_offsets.
        if cfg.probe_offsets is None:
            base_ctx = teacher.base_teacher.burn_in
            self.offsets: List[int] = list(range(-base_ctx, 2))
        else:
            self.offsets = list(cfg.probe_offsets)

        self.chunk_size = teacher.chunk_size
        self.hidden_dim = teacher.hidden_dim
        self.num_blocks = student.num_blocks

        # Layer 0 = post pos-encoder; layers 1..N = post each DecoderBlock.
        self.num_layers = self.num_blocks + 1

        # Persistent probe modules + (SGD only) their optimizers. Lazily built
        # on the first residual arrival — we need to know feature dim first
        # (student.hidden_dim after the encoder, dim before at layer 0 when
        # pe_type=='one_hot' since encoder is nn.Identity).
        self.probes: Dict[Tuple[int, int, int], nn.Linear] = {}
        self.optimizers: Dict[Tuple[int, int, int], torch.optim.Optimizer] = {}

        # Hook state.
        self._capture_enabled = False
        self._captures: List[Tuple[int, torch.Tensor]] = []
        self._current_residuals: Tuple[Tuple[int, torch.Tensor], ...] = ()
        self._current_data: Optional[torch.Tensor] = None
        self._val_residuals: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_layers)
        }
        self._val_data: List[torch.Tensor] = []

        self._handles = self._install_hooks()

    # ------------------------------------------------------------------ hooks
    def _install_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        handles: List[torch.utils.hooks.RemovableHandle] = []
        # Layer 0 residual: post encoder + positional-encoding.
        handles.append(
            self.student.pos_encoder.register_forward_hook(self._hook_factory(0))
        )
        # Layers 1..N: post each transformer block.
        for i, block in enumerate(self.student.transformer_blocks):
            handles.append(block.register_forward_hook(self._hook_factory(i + 1)))
        return handles

    def _hook_factory(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self._capture_enabled:
                return
            residual = output[0] if isinstance(output, tuple) else output
            self._captures.append((layer_idx, residual.detach()))

        return hook

    def remove_hooks(self) -> None:
        """Optional teardown — mostly for tests. Trainer does not call this."""
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------- trainer-facing methods
    def before_forward(self, context: str = "train") -> None:
        """Enable residual capture for the upcoming forward.

        `context="val"` always captures. `context="train"` only captures in
        sgd mode — warm_start doesn't need training residuals and skipping
        avoids per-step detach cost.
        """
        if not self.enabled:
            return
        self._captures = []
        self._capture_enabled = context == "val" or self.mode == "sgd"

    def after_forward(self, data: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._capture_enabled = False
        self._current_residuals = tuple(self._captures)
        self._current_data = data

    def sgd_step(self) -> None:
        """One Adam step per probe on the current training batch. No-op unless
        `mode == 'sgd'`."""
        if not self.enabled or self.mode != "sgd":
            return
        if self._current_data is None or not self._current_residuals:
            return
        by_layer = _residuals_by_layer(self._current_residuals, self.num_layers)
        labels = self._decode_hidden(self._current_data)  # (B, L_h)
        for layer_idx, residual in enumerate(by_layer):
            if residual is None:
                continue
            for slot in range(self.chunk_size):
                for offset in self.offsets:
                    X, y = self._gather(residual, labels, slot, offset)
                    if X is None or X.shape[0] == 0:
                        continue
                    probe, opt = self._ensure_probe(
                        layer_idx, slot, offset, X.shape[-1], X.device, need_opt=True
                    )
                    opt.zero_grad()
                    logits = probe(X)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()

    def collect_val_batch(self) -> None:
        if not self.enabled or self._current_data is None:
            return
        by_layer = _residuals_by_layer(self._current_residuals, self.num_layers)
        for i, r in enumerate(by_layer):
            if r is not None:
                self._val_residuals[i].append(r)
        self._val_data.append(self._current_data)

    def log(self, step: int, split: str) -> None:
        if not self.enabled or self.writer is None:
            self._clear_val_buffers()
            return
        if step % self.cfg.probe_frequency != 0:
            self._clear_val_buffers()
            return
        if not self._val_data:
            return

        data = torch.cat(self._val_data, dim=0)  # (N, L, dim)
        labels = self._decode_hidden(data)  # (N, L_h)
        residuals = {
            i: torch.cat(rs, dim=0) for i, rs in self._val_residuals.items() if rs
        }

        train_frac = self.cfg.probe_train_frac

        metrics: Dict[str, float] = {}
        heatmaps: Dict[int, np.ndarray] = {
            k: np.full((self.num_layers, self.chunk_size), np.nan) for k in self.offsets
        }

        for layer_idx, R in residuals.items():
            for slot in range(self.chunk_size):
                for offset in self.offsets:
                    X, y = self._gather(R, labels, slot, offset)
                    if X is None or X.shape[0] < 2:
                        continue

                    n_train = max(1, int(X.shape[0] * train_frac))
                    X_train, X_eval = X[:n_train], X[n_train:]
                    y_train, y_eval = y[:n_train], y[n_train:]
                    if X_eval.shape[0] == 0:
                        X_eval, y_eval = X_train, y_train

                    probe, _ = self._ensure_probe(
                        layer_idx, slot, offset, X.shape[-1], X.device, need_opt=False
                    )

                    if self.mode == "warm_start":
                        self._lbfgs_fit(probe, X_train, y_train)

                    with torch.no_grad():
                        logits = probe(X_eval)
                        nll = F.cross_entropy(logits, y_eval).item()
                        acc = (logits.argmax(dim=-1) == y_eval).float().mean().item()

                    offset_name = _offset_name(offset)
                    key_stub = f"probe/L{layer_idx}/slot{slot}/{offset_name}"
                    metrics[f"{key_stub}/acc/{split}"] = acc
                    metrics[f"{key_stub}/nll/{split}"] = nll
                    metrics[f"{key_stub}/n/{split}"] = X_eval.shape[0]
                    heatmaps[offset][layer_idx, slot] = acc

        # Emit per-offset heatmaps.
        for offset, mat in heatmaps.items():
            if np.isnan(mat).all():
                continue
            fig = _accuracy_heatmap(mat, offset, split)
            offset_name = _offset_name(offset)
            metrics[f"probe/summary/{offset_name}/acc_heatmap/{split}"] = wandb.Image(
                fig
            )
            plt.close(fig)

        self.writer.log(metrics, step=step)
        self._clear_val_buffers()

    # ---------------------------------------------------------------- helpers
    def _clear_val_buffers(self) -> None:
        self._val_residuals = {i: [] for i in range(self.num_layers)}
        self._val_data = []

    def _decode_hidden(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            one_hot = self.teacher.decode_chunk_aligned(data)  # (N, L_h, hidden_dim)
        return one_hot.argmax(dim=-1)  # (N, L_h)

    def _ensure_probe(
        self,
        layer: int,
        slot: int,
        offset: int,
        in_dim: int,
        device: torch.device,
        need_opt: bool,
    ) -> Tuple[nn.Linear, Optional[torch.optim.Optimizer]]:
        key = (layer, slot, offset)
        probe = self.probes.get(key)
        if probe is None:
            probe = nn.Linear(in_dim, self.hidden_dim).to(device)
            nn.init.xavier_uniform_(probe.weight)
            nn.init.zeros_(probe.bias)
            self.probes[key] = probe
        opt = self.optimizers.get(key)
        if need_opt and opt is None:
            opt = torch.optim.Adam(probe.parameters(), lr=self.cfg.probe_lr)
            self.optimizers[key] = opt
        return probe, opt

    def _gather(
        self,
        residual: torch.Tensor,  # (N, T, D)
        labels: torch.Tensor,  # (N, L_h)
        slot: int,
        offset: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract flat (X, y) pairs at positions with the given within-chunk
        slot and a valid target chunk `c(t) + offset`.
        """
        _, T, D = residual.shape
        L_h = labels.shape[1]

        pos = torch.arange(slot, T, self.chunk_size, device=residual.device)
        if pos.numel() == 0:
            return None, None
        c = pos // self.chunk_size  # (num_pos,)
        target_c = c + offset
        valid = (target_c >= 0) & (target_c < L_h)
        if not valid.any():
            return None, None
        pos = pos[valid]
        target_c = target_c[valid]

        X = residual[:, pos, :]  # (N, num_valid, D)
        y = labels[:, target_c]  # (N, num_valid)
        return X.reshape(-1, D), y.reshape(-1)

    def _lbfgs_fit(self, probe: nn.Linear, X: torch.Tensor, y: torch.Tensor) -> None:
        opt = torch.optim.LBFGS(
            probe.parameters(),
            max_iter=self.cfg.probe_max_iters,
            tolerance_grad=1e-4,
            tolerance_change=1e-6,
            line_search_fn="strong_wolfe",
        )
        l2 = self.cfg.probe_l2

        def closure() -> torch.Tensor:
            opt.zero_grad()
            logits = probe(X)
            loss = F.cross_entropy(logits, y) + l2 * probe.weight.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)


def _residuals_by_layer(
    captures: Tuple[Tuple[int, torch.Tensor], ...], num_layers: int
) -> List[Optional[torch.Tensor]]:
    """Reorder captures into a `[num_layers]`-length list, one tensor per layer.

    Hooks fire in a deterministic layer order per forward pass, but this
    normalizes the shape and tolerates missing layers (returns None entries).
    """
    out: List[Optional[torch.Tensor]] = [None] * num_layers
    for layer_idx, tensor in captures:
        out[layer_idx] = tensor
    return out


def _accuracy_heatmap(mat: np.ndarray, offset: int, split: str) -> plt.Figure:
    """Layer × slot heatmap of probe accuracy for a single offset."""
    n_layers, n_slots = mat.shape
    fig, ax = plt.subplots(figsize=(1.5 + 0.6 * n_slots, 1.0 + 0.5 * n_layers))
    sns.heatmap(
        mat,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        xticklabels=[f"s{s}" for s in range(n_slots)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"probe acc — {split}, k={offset:+d}")
    ax.set_xlabel("within-chunk slot")
    ax.set_ylabel("layer")
    fig.tight_layout()
    return fig


def _offset_name(offset: int) -> str:
    """Return an explicit signed relative-offset label."""
    if offset > 0:
        return f"k+{offset}"
    return f"k{offset}"
