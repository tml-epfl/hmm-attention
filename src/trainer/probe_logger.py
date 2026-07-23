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
from src.teachers import HierarchicalTeacher, MultiLevelHierarchicalTeacher
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

        self.multilevel = isinstance(teacher, MultiLevelHierarchicalTeacher)
        self.enabled = (
            self.mode != "off"
            and (isinstance(teacher, HierarchicalTeacher) or self.multilevel)
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

        if self.multilevel:
            # One latent per level: the open path from surface up to the base.
            # `span[l]` surface tokens per level-`l` unit; slot = mixed-radix
            # digit; classes = that level's input alphabet.
            self.num_levels = teacher.num_levels
            self.level_spans = list(teacher._span)  # length num_levels + 1
            self.level_arity = [teacher.levels[l].size for l in range(self.num_levels)]
            self.level_alphabet = [
                teacher.levels[l].in_dim for l in range(self.num_levels)
            ]
            self.burn_in = teacher.burn_in
        else:
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
        if self.multilevel:
            return self._sgd_step_multilevel()
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
        if not self.enabled:
            # Disabled: __init__ returned early, so num_layers/buffers were never
            # set — and nothing was collected (collect_* guard on `enabled`).
            return
        if self.writer is None or step % self.cfg.probe_frequency != 0:
            self._clear_val_buffers()
            return
        if not self._val_data:
            return
        if self.multilevel:
            return self._log_multilevel(step, split)

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
        level: Optional[int] = None,
    ) -> Tuple[nn.Linear, Optional[torch.optim.Optimizer]]:
        key = (layer, slot, offset) if level is None else (layer, level, slot, offset)
        num_classes = self.hidden_dim if level is None else self.level_alphabet[level]
        probe = self.probes.get(key)
        if probe is None:
            probe = nn.Linear(in_dim, num_classes).to(device)
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

    # ------------------------------------------------------ multi-level path
    def _decode_level(self, data: torch.Tensor, level: int) -> torch.Tensor:
        """Decode the level-`level` unit id covering each span. (N, L_level)."""
        with torch.no_grad():
            one_hot = self.teacher._decode_levels(data, stop_after=level)
        return one_hot.argmax(dim=-1)

    def _gather_level(
        self,
        residual: torch.Tensor,  # (N, T, D)
        labels: torch.Tensor,  # (N, L_level)
        belief_level: Optional[torch.Tensor],  # (N, L - burn_in, C) or None
        level: int,
        slot: int,
        offset: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Like `_gather`, but for level `level`: positions whose mixed-radix
        digit at that level equals `slot`, targeting unit `chunk + offset`.

        Returns (X, y, belief, belief_valid). For offset 0, `belief` holds the
        Bayes-optimal log-belief over the current unit at each row (with
        `belief_valid` masking rows before burn-in); otherwise both are None.
        """
        N, T, D = residual.shape
        L_level = labels.shape[1]
        span_l = self.level_spans[level]
        span_child = self.level_spans[level + 1]

        t = torch.arange(T, device=residual.device)
        tsel = t[(t % span_l) // span_child == slot]
        if tsel.numel() == 0:
            return None, None, None, None
        target_c = tsel // span_l + offset
        valid = (target_c >= 0) & (target_c < L_level)
        if not valid.any():
            return None, None, None, None
        tsel = tsel[valid]
        target_c = target_c[valid]

        X = residual[:, tsel, :].reshape(-1, D)
        y = labels[:, target_c].reshape(-1)

        belief = belief_valid = None
        if offset == 0 and belief_level is not None:
            idx = tsel - self.burn_in
            vb = idx >= 0
            b = belief_level[:, idx.clamp(min=0), :]  # (N, P, C)
            belief = b.reshape(-1, b.shape[-1])
            belief_valid = vb.unsqueeze(0).expand(N, -1).reshape(-1)
        return X, y, belief, belief_valid

    def _bayes_ceiling(
        self,
        offset: int,
        belief: Optional[torch.Tensor],
        belief_valid: Optional[torch.Tensor],
        eval_slice: slice,
        logits: torch.Tensor,
        y_eval: torch.Tensor,
    ) -> Optional[Tuple[float, float, float]]:
        """Bayes-optimal (acc, nll, excess_nll) for the eval rows, or None.

        Retention (offset<0): the target unit is already complete → optimum is
        deterministic (acc 1.0, nll 0). Refinement (offset 0): the teacher's
        current-unit belief. Planning (offset>0): deferred (returns None).
        """
        if offset < 0:
            probe_nll = F.cross_entropy(logits, y_eval).item()
            return 1.0, 0.0, probe_nll
        if offset == 0 and belief is not None:
            bv = belief_valid[eval_slice]
            if not bool(bv.any()):
                return None
            blp = belief[eval_slice][bv]  # (m, C) log-probs
            yb = y_eval[bv]
            with torch.no_grad():
                bayes_acc = (blp.argmax(dim=-1) == yb).float().mean().item()
                bayes_nll = F.nll_loss(blp, yb).item()
                probe_nll = F.cross_entropy(logits[bv], yb).item()
            return bayes_acc, bayes_nll, probe_nll - bayes_nll
        return None

    def _nan_mat(self, level: int) -> np.ndarray:
        return np.full((self.num_layers, self.level_arity[level]), np.nan)

    def _sgd_step_multilevel(self) -> None:
        by_layer = _residuals_by_layer(self._current_residuals, self.num_layers)
        labels_per_level = [
            self._decode_level(self._current_data, l) for l in range(self.num_levels)
        ]
        for layer_idx, residual in enumerate(by_layer):
            if residual is None:
                continue
            for level in range(self.num_levels):
                labels = labels_per_level[level]
                for slot in range(self.level_arity[level]):
                    for offset in self.offsets:
                        X, y, _, _ = self._gather_level(
                            residual, labels, None, level, slot, offset
                        )
                        if X is None or X.shape[0] == 0:
                            continue
                        probe, opt = self._ensure_probe(
                            layer_idx, slot, offset, X.shape[-1], X.device,
                            need_opt=True, level=level,
                        )
                        opt.zero_grad()
                        loss = F.cross_entropy(probe(X), y)
                        loss.backward()
                        opt.step()

    def _log_multilevel(self, step: int, split: str) -> None:
        data = torch.cat(self._val_data, dim=0)  # (N, L, dim)
        beliefs = self.teacher.latent_beliefs(data)  # per level (N, L-burn_in, C)
        labels_per_level = [self._decode_level(data, l) for l in range(self.num_levels)]
        residuals = {
            i: torch.cat(rs, dim=0) for i, rs in self._val_residuals.items() if rs
        }
        train_frac = self.cfg.probe_train_frac

        metrics: Dict[str, float] = {}
        heat_acc: Dict[Tuple[int, int], np.ndarray] = {}
        heat_excess: Dict[Tuple[int, int], np.ndarray] = {}

        for layer_idx, R in residuals.items():
            for level in range(self.num_levels):
                labels = labels_per_level[level]
                for slot in range(self.level_arity[level]):
                    for offset in self.offsets:
                        X, y, belief, bvalid = self._gather_level(
                            R, labels, beliefs[level], level, slot, offset
                        )
                        if X is None or X.shape[0] < 2:
                            continue

                        n_train = max(1, int(X.shape[0] * train_frac))
                        X_train, X_eval = X[:n_train], X[n_train:]
                        y_train, y_eval = y[:n_train], y[n_train:]
                        if X_eval.shape[0] == 0:
                            X_eval, y_eval = X_train, y_train
                            eval_slice = slice(0, n_train)
                        else:
                            eval_slice = slice(n_train, X.shape[0])

                        probe, _ = self._ensure_probe(
                            layer_idx, slot, offset, X.shape[-1], X.device,
                            need_opt=False, level=level,
                        )
                        if self.mode == "warm_start":
                            self._lbfgs_fit(probe, X_train, y_train)

                        with torch.no_grad():
                            logits = probe(X_eval)
                            nll = F.cross_entropy(logits, y_eval).item()
                            acc = (logits.argmax(dim=-1) == y_eval).float().mean().item()

                        offset_name = _offset_name(offset)
                        stub = f"probe/L{layer_idx}/level{level}/slot{slot}/{offset_name}"
                        metrics[f"{stub}/acc/{split}"] = acc
                        metrics[f"{stub}/nll/{split}"] = nll
                        metrics[f"{stub}/n/{split}"] = X_eval.shape[0]
                        heat_acc.setdefault((level, offset), self._nan_mat(level))[
                            layer_idx, slot
                        ] = acc

                        bayes = self._bayes_ceiling(
                            offset, belief, bvalid, eval_slice, logits, y_eval
                        )
                        if bayes is not None:
                            bayes_acc, bayes_nll, excess = bayes
                            metrics[f"{stub}/bayes_acc/{split}"] = bayes_acc
                            metrics[f"{stub}/bayes_nll/{split}"] = bayes_nll
                            metrics[f"{stub}/excess_nll/{split}"] = excess
                            heat_excess.setdefault(
                                (level, offset), self._nan_mat(level)
                            )[layer_idx, slot] = excess

        for (level, offset), mat in heat_acc.items():
            if np.isnan(mat).all():
                continue
            fig = _grid_heatmap(
                mat, f"probe acc — level{level}, {split}, k={offset:+d}", 0.0, 1.0, "viridis"
            )
            key = f"probe/summary/level{level}/{_offset_name(offset)}/acc_heatmap/{split}"
            metrics[key] = wandb.Image(fig)
            plt.close(fig)
        for (level, offset), mat in heat_excess.items():
            if np.isnan(mat).all():
                continue
            vmax = max(float(np.nanmax(mat)), 1e-6)
            fig = _grid_heatmap(
                mat, f"excess nll — level{level}, {split}, k={offset:+d}", 0.0, vmax, "magma"
            )
            key = f"probe/summary/level{level}/{_offset_name(offset)}/excess_heatmap/{split}"
            metrics[key] = wandb.Image(fig)
            plt.close(fig)

        self.writer.log(metrics, step=step)
        self._clear_val_buffers()

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


def _grid_heatmap(
    mat: np.ndarray, title: str, vmin: float, vmax: float, cmap: str
) -> plt.Figure:
    """Layer × slot heatmap of an arbitrary per-cell value (multi-level path)."""
    n_layers, n_slots = mat.shape
    fig, ax = plt.subplots(figsize=(1.5 + 0.6 * n_slots, 1.0 + 0.5 * n_layers))
    sns.heatmap(
        mat,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        xticklabels=[f"s{s}" for s in range(n_slots)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        cbar=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("within-unit slot")
    ax.set_ylabel("layer")
    fig.tight_layout()
    return fig


def _offset_name(offset: int) -> str:
    """Return an explicit signed relative-offset label."""
    if offset > 0:
        return f"k+{offset}"
    return f"k{offset}"
