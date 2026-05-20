"""Render an MLP ``model.pt`` as a node-and-edge diagram.

* Asks for the path to a ``model.pt`` (a ``state_dict`` produced by
  ``clean_directory/nn``).
* Shows every layer (input, hidden, output) as a column of nodes.
* Edge color encodes the weight value, node fill color encodes the bias.
* Includes a "Save" button matching the other helper visualizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- ask for model path --------------------------------------------
def _ask_for_model_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Select model.pt",
            filetypes=[("PyTorch state_dict", "*.pt *.pth"), ("All files", "*")],
        )
        root.destroy()
        if chosen:
            return Path(chosen).expanduser().resolve()
    except Exception:
        pass
    raw = input("Path to model.pt: ").strip()
    return Path(raw).expanduser().resolve()


MODEL_PATH = _ask_for_model_path()
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)


# ---- load state_dict & extract Linear layers -----------------------
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
if not isinstance(state, dict):
    raise TypeError(
        f"Expected a state_dict (dict) in {MODEL_PATH}, got {type(state)}"
    )

# Pair every "*.weight" with its corresponding "*.bias", in insertion order.
linears: list[tuple[np.ndarray, np.ndarray | None]] = []
seen_prefixes: list[str] = []
for key, tensor in state.items():
    if not key.endswith(".weight"):
        continue
    if tensor.ndim != 2:  # skip non-Linear weights (e.g. BatchNorm)
        continue
    prefix = key[: -len(".weight")]
    bias = state.get(f"{prefix}.bias")
    linears.append(
        (tensor.detach().cpu().numpy(),
         bias.detach().cpu().numpy() if bias is not None else None)
    )
    seen_prefixes.append(prefix)

if not linears:
    raise RuntimeError(f"No Linear (2-D weight) layers found in {MODEL_PATH}")

# Layer sizes: input dim from first weight, then each Linear's out_features.
layer_sizes = [linears[0][0].shape[1]] + [W.shape[0] for W, _ in linears]
n_layers = len(layer_sizes)


# ---- input / output names (from sibling config.yaml if present) ----
DEFAULT_OUTPUT_NAMES = [r"$x$", r"$y$", r"$z$"]

import re as _re


def _prettify(name: str) -> str:
    """Convert a raw config feature name to a matplotlib math label."""
    _PRETTY = {
        "c_u":       r"$c_u$",
        "c_v":       r"$c_v$",
        "n_visible": r"$n_{\mathrm{visible}}$",
    }
    if name in _PRETTY:
        return _PRETTY[name]
    m = _re.fullmatch(r"d([uv])(\d+)", name)
    if m:
        return fr"$\Delta {m.group(1)}_{{{m.group(2)}}}$"
    m = _re.fullmatch(r"([uv])(\d+)", name)
    if m:
        return fr"${m.group(1)}_{{{m.group(2)}}}$"
    m = _re.fullmatch(r"m(\d+)", name)
    if m:
        return fr"$m_{{{m.group(1)}}}$"
    return name


def _load_io_names(model_path: Path,
                   in_dim: int, out_dim: int) -> tuple[list[str], list[str]]:
    cfg_path = model_path.with_name("config.yaml")
    in_names: list[str] = []
    out_names: list[str] = []
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            meta = cfg.get("_meta", {}) or {}
            in_names = [_prettify(n) for n in (meta.get("feature_names") or [])]
            out_names = list(meta.get("target_names", []) or [])
        except Exception:
            pass
    if len(in_names) != in_dim:
        in_names = [_prettify(f"u{i+1}") if i % 3 == 0
                    else _prettify(f"v{i+1//3*3+2}") if i % 3 == 1
                    else f"in{i + 1}" for i in range(in_dim)]
        in_names = [f"in{i + 1}" for i in range(in_dim)]
    if len(out_names) != out_dim:
        out_names = (DEFAULT_OUTPUT_NAMES[:out_dim]
                     if out_dim <= len(DEFAULT_OUTPUT_NAMES)
                     else [f"out{i + 1}" for i in range(out_dim)])
    return in_names, out_names


input_names, output_names = _load_io_names(
    MODEL_PATH, layer_sizes[0], layer_sizes[-1]
)


# ---- color scaling -------------------------------------------------
def _sym_norm(values: np.ndarray) -> mcolors.TwoSlopeNorm:
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


all_weights = np.concatenate([W.ravel() for W, _ in linears])
all_biases = np.concatenate(
    [b.ravel() for _, b in linears if b is not None]
) if any(b is not None for _, b in linears) else np.array([0.0])

w_norm = _sym_norm(all_weights)
b_norm = _sym_norm(all_biases)
w_cmap = plt.get_cmap("RdBu_r")
b_cmap = plt.get_cmap("PuOr_r")


# ---- node positions (vertical layout: layers top->bottom) ----------
def _row_positions(n: int, span: float = 1.0) -> np.ndarray:
    if n == 1:
        return np.array([0.0])
    return np.linspace(-span / 2.0, span / 2.0, n)


# y_layer[k] = vertical position of layer k (top = input, bottom = output)
# Increase LAYER_SPACING to push layers further apart.
LAYER_SPACING = 0.45
y_layers = np.array([1.0 - k * LAYER_SPACING for k in range(n_layers)])
x_positions = [_row_positions(n) for n in layer_sizes]


# ---- figure --------------------------------------------------------
# Height grows with depth, width grows (mildly) with the widest layer.
max_layer = max(layer_sizes)
fig_h = max(7.0, 2.2 * n_layers + 2.0)
fig_w = max(5.0, min(11.0, 0.06 * max_layer + 4.0))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))


# Node sizes scale down for very wide layers.
def _node_size(n: int) -> float:
    return float(np.clip(2000.0 / max(n, 1), 20.0, 300.0))


# ---- draw edges (weights) ------------------------------------------
for k, (W, _) in enumerate(linears):
    src_y = y_layers[k]
    dst_y = y_layers[k + 1]
    src_x = x_positions[k]
    dst_x = x_positions[k + 1]

    out_dim, in_dim = W.shape  # rows = next layer, cols = current layer
    segments = np.empty((in_dim * out_dim, 2, 2), dtype=float)
    colors = np.empty((in_dim * out_dim, 4), dtype=float)
    idx = 0
    for j in range(out_dim):
        for i in range(in_dim):
            segments[idx, 0] = (src_x[i], src_y)
            segments[idx, 1] = (dst_x[j], dst_y)
            colors[idx] = w_cmap(w_norm(W[j, i]))
            idx += 1

    # Make low-magnitude weights faint to reduce visual clutter.
    mags = np.abs(W).ravel()
    if mags.max() > 0:
        alphas = 0.15 + 0.85 * (mags / mags.max())
    else:
        alphas = np.full_like(mags, 0.3)
    colors[:, 3] = alphas

    lc = LineCollection(segments, colors=colors, linewidths=0.6, zorder=1)
    ax.add_collection(lc)


# ---- draw nodes (biases) -------------------------------------------
# Input layer has no bias -> draw as hollow grey circles.
input_size = _node_size(layer_sizes[0])
ax.scatter(
    x_positions[0],
    np.full(layer_sizes[0], y_layers[0]),
    s=input_size, facecolors="white", edgecolors="0.3",
    linewidths=0.8, zorder=3,
)

for k, (_, b) in enumerate(linears, start=1):
    sz = _node_size(layer_sizes[k])
    xs = x_positions[k]
    ys = np.full(layer_sizes[k], y_layers[k])
    if b is None:
        face = "white"
        ax.scatter(xs, ys, s=sz, facecolors=face, edgecolors="0.3",
                   linewidths=0.8, zorder=3)
    else:
        face = b_cmap(b_norm(b))
        ax.scatter(xs, ys, s=sz, c=face, edgecolors="0.3",
                   linewidths=0.6, zorder=3)


# ---- labels --------------------------------------------------------
labels = ["input"] + [
    f"hidden {i + 1}" for i in range(n_layers - 2)
] + ["output"]
if n_layers == 2:
    labels = ["input", "output"]

# Layer name labels on the left side.
for y, n, lbl in zip(y_layers, layer_sizes, labels):
    ax.text(-0.6, y, f"{lbl}\n({n})", ha="right", va="center")

# Per-node labels: inputs above the top row, outputs below the bottom row.
for name, x in zip(input_names, x_positions[0]):
    ax.text(x, y_layers[0] + 0.04, name,
            ha="center", va="bottom", rotation=0)
for name, x in zip(output_names, x_positions[-1]):
    ax.text(x, y_layers[-1] - 0.04, name,
            ha="center", va="top", rotation=0)

ax.set_xlim(-0.65, 0.65)
y_top = y_layers[0]
y_bot = y_layers[-1]
pad_y = 0.18
ax.set_ylim(y_bot - pad_y, y_top + pad_y)
ax.set_axis_off()


# ---- colorbars -----------------------------------------------------
cb_w = fig.add_axes([0.92, 0.55, 0.015, 0.35])
fig.colorbar(plt.cm.ScalarMappable(norm=w_norm, cmap=w_cmap),
             cax=cb_w, label="weight")
cb_b = fig.add_axes([0.92, 0.10, 0.015, 0.35])
fig.colorbar(plt.cm.ScalarMappable(norm=b_norm, cmap=b_cmap),
             cax=cb_b, label="bias")

fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.05)


# ---- save button ---------------------------------------------------
SAVE_BASENAME = str(Path(__file__).resolve().parent.parent
                    / f"model_visualization_{MODEL_PATH.stem}")

save_ax = fig.add_axes([0.88, 0.02, 0.10, 0.05])
_button = Button(save_ax, "Save")


def _save(event):
    save_ax.set_visible(False)
    fig.savefig(f"{SAVE_BASENAME}.pgf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{SAVE_BASENAME}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{SAVE_BASENAME}.svg", bbox_inches="tight", pad_inches=0.02)
    save_ax.set_visible(True)
    fig.canvas.draw_idle()
    print(f"Saved {SAVE_BASENAME}.{{pgf,pdf,svg}}")


_button.on_clicked(_save)
fig._save_button = _button
fig._save_ax = save_ax


if __name__ == "__main__":
    plt.show()
