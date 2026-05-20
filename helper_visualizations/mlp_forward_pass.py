"""Vector-style diagram of a simple MLP forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- colors --------------------------------------------------------
POS_COLOR = "tab:blue"
NEG_COLOR = "tab:red"
ZERO_NODE = "0.88"
ACTIVE_NODE = "#d9ecff"
EDGE_NEUTRAL = "0.35"
TEXT_COLOR = "0.12"

NODE_R = 0.18


# ---- network specification ----------------------------------------
nodes = {
    "x":    {"pos": (0.0, 0.0), "label": r"$x=2$",              "active": True},
    "h11":  {"pos": (1.8, 0.55), "label": r"$h_{1,1}=1$",        "active": True},
    "h12":  {"pos": (1.8, -0.55), "label": r"$h_{1,2}=0$",       "active": False},
    "h21":  {"pos": (3.6, 0.55), "label": r"$h_{2,1}=0.5$",      "active": True},
    "h22":  {"pos": (3.6, -0.55), "label": r"$h_{2,2}=0$",       "active": False},
    "yhat": {"pos": (5.4, 0.0), "label": r"$\hat{y}=0.5$",       "active": True},
}

edges = [
    ("x", "h11", 1.0, r"$w=1$"),
    ("x", "h12", -0.5, r"$w=-0.5$"),
    ("h11", "h21", 0.5, r"$0.5$"),
    ("h12", "h21", 1.0, r"$1$"),
    ("h11", "h22", -1.0, r"$-1$"),
    ("h12", "h22", 0.5, r"$0.5$"),
    ("h21", "yhat", 1.0, r"$1$"),
    ("h22", "yhat", -1.0, r"$-1$"),
]

bias_labels = {
    "h11": r"$b=-1$",
    "h12": r"$b=0$",
    "h21": r"$b=0$",
    "h22": r"$b=1$",
    "yhat": r"$b=0$",
}

layer_labels = [
    (0.0, "Input"),
    (1.8, "Hidden layer 1"),
    (3.6, "Hidden layer 2"),
    (5.4, "Output"),
]

calculations = [
    (1.8, -1.05, r"$\mathrm{ReLU}(1\cdot2-1)=\mathrm{ReLU}(1)=1$"),
    (1.8, -1.30, r"$\mathrm{ReLU}(-0.5\cdot2+0)=\mathrm{ReLU}(-1)=0$"),
    (3.6, -1.05, r"$\mathrm{ReLU}(0.5\cdot1+1\cdot0+0)=\mathrm{ReLU}(0.5)=0.5$"),
    (3.6, -1.30, r"$\mathrm{ReLU}(-1\cdot1+0.5\cdot0+1)=\mathrm{ReLU}(0)=0$"),
    (5.4, -1.05, r"$\hat{y}=1\cdot0.5+(-1)\cdot0+0=0.5$"),
]


# ---- drawing helpers ----------------------------------------------
def _edge_color(weight: float) -> str:
    if weight > 0.0:
        return POS_COLOR
    if weight < 0.0:
        return NEG_COLOR
    return EDGE_NEUTRAL


def _draw_edge(ax, src: str, dst: str, weight: float, label: str) -> None:
    x0, y0 = nodes[src]["pos"]
    x1, y1 = nodes[dst]["pos"]
    arrow = FancyArrowPatch(
        (x0 + NODE_R, y0), (x1 - NODE_R, y1),
        arrowstyle="->", mutation_scale=11,
        linewidth=1.2, color=_edge_color(weight), alpha=0.85,
        shrinkA=0.0, shrinkB=0.0,
        zorder=1,
    )
    ax.add_patch(arrow)

    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    dy = 0.08 if y1 >= y0 else -0.08
    ax.text(mx, my + dy, label, ha="center", va="center",
            color=_edge_color(weight), fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            zorder=4)


def _draw_node(ax, name: str) -> None:
    info = nodes[name]
    x, y = info["pos"]
    face = ACTIVE_NODE if info["active"] else ZERO_NODE
    circ = Circle((x, y), NODE_R, facecolor=face, edgecolor="0.2",
                  linewidth=1.0, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, info["label"], ha="center", va="center",
            color=TEXT_COLOR, zorder=4)

    if name in bias_labels:
        ax.text(x, y + NODE_R + 0.10, bias_labels[name],
                ha="center", va="bottom", color=TEXT_COLOR)


# ---- figure --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for src, dst, weight, label in edges:
    _draw_edge(ax, src, dst, weight, label)

for name in nodes:
    _draw_node(ax, name)

for x, label in layer_labels:
    ax.text(x, 1.12, label, ha="center", va="bottom")

for x, y, text in calculations:
    ax.text(x, y, text, ha="center", va="center", fontsize=10)

# Compact legend for sign and activation convention.
ax.plot([], [], color=POS_COLOR, linewidth=1.2, label="positive weight")
ax.plot([], [], color=NEG_COLOR, linewidth=1.2, label="negative weight")
ax.scatter([], [], s=80, facecolor=ACTIVE_NODE, edgecolor="0.2",
           label="active neuron")
ax.scatter([], [], s=80, facecolor=ZERO_NODE, edgecolor="0.2",
           label="zero activation")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4,
          frameon=False)

ax.set_xlim(-0.45, 5.85)
ax.set_ylim(-1.55, 1.35)
ax.set_axis_off()
fig.tight_layout()


# ---- save button ---------------------------------------------------
SAVE_BASENAME = str(Path(__file__).resolve().parent.parent / "mlp_forward_pass")

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
