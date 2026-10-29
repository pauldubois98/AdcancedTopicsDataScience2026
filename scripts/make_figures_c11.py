#!/usr/bin/env python3
"""Render the illustration figures used by the Course11 lecture slides.

Session 11 has two halves: feeding our own data to PyTorch (Dataset, DataLoader,
transforms), then convolutions in 1D, on ECG heartbeats. The heartbeats are the
ECG200 training set, downloaded by Course11/temporal_signal.ipynb into data/ECG200;
the other concept figures are synthetic and seeded; `results.png` plots the numbers
measured in Course11/temporal_signal-solutions.ipynb.

Usage: make_figures_c11.py [outdir]      (default: Course11/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#1b1b1b"
GREY = "#9aa0a6"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#00ab0e"
AMBER = "#b7791f"
PURPLE = "#6b46c1"
TEAL = "#0f766e"
FAINT = "#e8eaed"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)

# the .pptx content placeholder is 9.00in x 3.71in (aspect 2.42); figures that
# match it fill the slide instead of being scaled down to fit its height
WIDE = (12.6, 5.2)

SLIDE_ASPECT = 9.00 / 3.71

ECG200 = Path(__file__).resolve().parents[1] / "data" / "ECG200" / "ECG200_TRAIN.txt"

# measured in Course11/temporal_signal-solutions.ipynb, 5 seeds:
# name -> (test mean, test std, shifted test mean, shifted test std), in %
RESULTS = {
    "logistic\nregression": (81.4, 0.8, 62.6, 5.1),
    "MLP": (87.4, 0.5, 63.8, 4.0),
    "CNN\nk=3": (79.0, 1.9, 77.4, 0.5),
    "CNN\nk=7": (84.4, 2.2, 77.6, 2.4),
    "CNN\nk=15": (85.8, 1.6, 78.6, 3.0),
    "MLP\n+ shifts": (82.4, 0.8, 83.4, 1.6),
    "CNN k=15\n+ shifts": (88.6, 1.2, 85.4, 2.2),
}


def pad_to_aspect(path: Path, aspect: float = SLIDE_ASPECT) -> None:
    """Pad a saved figure with white so it exactly fills the slide placeholder."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w / h < aspect:
        tw, th = int(round(h * aspect)), h
    else:
        tw, th = w, int(round(w / aspect))
    if (tw, th) == (w, h):
        return
    canvas = Image.new("RGB", (tw, th), "white")
    canvas.paste(im, ((tw - w) // 2, (th - h) // 2))
    canvas.save(path)


def save(fig, out: Path, name: str) -> None:
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    pad_to_aspect(path)


def box(ax, xy, w, h, text, fc="white", ec=BLUE, fs=11, lw=1.8, tc=INK, ls="-"):
    """A rounded label box, used by the flow diagrams."""
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
        )
    )
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.4)


def arrow(ax, a, b, color=GREY, lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
            shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}",
        )
    )


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def cells(ax, x0, y0, values, w, h, cmap=None, fc="white", ec=GREY, fs=12,
          fmt="{:g}", highlight=()):
    """A row of square cells holding numbers; `highlight` indices get a thick border."""
    for i, v in enumerate(values):
        face = cmap(v) if cmap is not None else fc
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, facecolor=face, edgecolor=ec,
                               lw=1.2))
        if fmt:
            ax.text(x0 + (i + 0.5) * w, y0 + h / 2, fmt.format(v), ha="center",
                    va="center", fontsize=fs)
    if highlight:
        a, b = min(highlight), max(highlight)
        ax.add_patch(Rectangle((x0 + a * w, y0), (b - a + 1) * w, h, fill=False,
                               edgecolor=RED, lw=3))


def load_ecg():
    """ECG200 training set: signals [100, 96], labels in {-1 (infarction), 1}."""
    if not ECG200.exists():
        sys.exit(f"{ECG200} is missing: run Course11/temporal_signal.ipynb first")
    data = np.loadtxt(ECG200)
    return data[:, 1:], data[:, 0]


# ================================================================ I. datasets


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.set_xlim(-0.01, 1)
    xs = (0.1, 0.32, 0.545, 0.755, 0.93)
    texts = (
        "files on disk\n\nECG200_TRAIN.txt\n(one beat per line)",
        "Dataset\n\nds[i] → (x, y)\nx: [1, 96]   y: 0 or 1",
        "DataLoader\n\nshuffles the indices,\nstacks 16 examples",
        "batch\n\nx: [16, 1, 96]\ny: [16]",
        "model",
    )
    widths = (0.18, 0.2, 0.19, 0.15, 0.1)
    cols = (GREY, BLUE, TEAL, AMBER, PURPLE)
    for x, t, w, c in zip(xs, texts, widths, cols):
        box(ax, (x, 0.6), w, 0.42, t, ec=c, fs=12)
    for a, b, wa, wb in zip(xs, xs[1:], widths, widths[1:]):
        arrow(ax, (a + wa / 2 + 0.005, 0.6), (b - wb / 2 - 0.005, 0.6), lw=2.2)
    box(ax, (0.32, 0.16), 0.2, 0.16, "transform (optional)\napplied to each x",
        ec=BLUE, fs=11.5, ls="--")
    arrow(ax, (0.32, 0.25), (0.32, 0.38), color=BLUE)
    ax.text(0.32, 0.88, "you write this", ha="center", fontsize=12.5, color=BLUE,
            weight="bold")
    ax.text(0.55, 0.88, "PyTorch does this", ha="center", fontsize=12.5, color=TEAL,
            weight="bold")
    save(fig, out, "pipeline")


def fig_random_shift(out: Path) -> None:
    signals, _ = load_ecg()
    x = signals[0]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    ax = axes[0]
    for i in range(6):
        ax.plot(signals[i], lw=1.6, alpha=0.8)
    ax.set_title("ECG200: the beats are aligned (R peak at the start)")
    ax.set_xlabel("time step")
    ax = axes[1]
    ax.plot(x, color=INK, lw=2.4, label="original")
    for s, c in zip(rng.integers(-10, 11, 3), (BLUE, RED, GREEN)):
        ax.plot(np.roll(x, s), color=c, lw=1.5, alpha=0.85, label=f"roll by {s:+d}")
    ax.set_title("RandomShift(10): a new shift every time ds[i] is read")
    ax.set_xlabel("time step")
    ax.legend(frameon=False, fontsize=11)
    save(fig, out, "random_shift")


# ================================================================ II. 1D convolutions


def fig_signals(out: Path) -> None:
    signals, labels = load_ecg()
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    for ax, lab, name, col in zip(axes, (1, -1), ("normal", "infarction"), (BLUE, RED)):
        sel = signals[labels == lab][:8]
        for s in sel:
            ax.plot(s, color=col, lw=1, alpha=0.35)
        ax.plot(signals[labels == lab].mean(0), color=col, lw=2.8)
        ax.set_title(f"ECG heartbeat: {name}")
        ax.set_xlabel("time step (96)")
        ax.set_ylim(-3, 4)
    ax = axes[2]
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 200)
    for k, (c, name) in enumerate(zip((BLUE, RED, GREEN), "xyz")):
        sig = np.sin(2 * np.pi * (1.1 + 0.2 * k) * t + k) + 0.3 * rng.normal(size=t.size)
        ax.plot(t, sig + 3 * (1 - k), color=c, lw=1.2)
        ax.text(-0.15, 3 * (1 - k), name, color=c, ha="right", va="center", fontsize=13)
    ax.set_title("accelerometer: 3 channels")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    save(fig, out, "signals")


def fig_conv2d_vs_1d(out: Path) -> None:
    rng = np.random.default_rng(1)
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1.6, 6.8)
    ax.set_aspect("equal")
    ax.axis("off")
    img = rng.uniform(0.1, 0.7, (6, 6))
    for r in range(6):
        cells(ax, 0, 5 - r, img[r], 1, 1, cmap=cmap, fmt=None)
    ax.add_patch(Rectangle((1, 2), 3, 3, fill=False, edgecolor=RED, lw=3.5))
    arrow(ax, (4.2, 3.5), (6.6, 3.5), color=RED, lw=2.4)
    arrow(ax, (2.5, 1.8), (2.5, -0.5), color=RED, lw=2.4)
    ax.text(3, 6.3, "image  [C, H, W]", ha="center", fontsize=13)
    ax.text(3, -1.3, "kernel [C, 3, 3] slides in 2 directions", ha="center",
            fontsize=12.5, color=RED)

    ax = axes[1]
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-3.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    sig = 0.4 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 16))
    cells(ax, 0, 0, sig, 1, 1, cmap=cmap, fmt=None, highlight=(4, 5, 6))
    tt = np.arange(16) + 0.5
    ax.plot(tt, 1.6 + 2 * (sig - 0.1), color=BLUE, lw=2, marker="o", ms=4)
    arrow(ax, (7.2, -0.6), (11, -0.6), color=RED, lw=2.4)
    ax.text(8, -1.5, "signal [C, L]: kernel [C, k] slides along time only",
            ha="center", fontsize=12.5, color=RED)
    ax.text(8, -2.6, "same ideas: shared weights, local, stride, padding, pooling",
            ha="center", fontsize=12.5)
    save(fig, out, "conv2d_vs_1d")


def fig_conv1d_sliding(out: Path) -> None:
    x = np.array([1, 3, 2, 0, -1, 2, 4])
    w = np.array([1, 0, -1])
    y = np.correlate(x, w, mode="valid")
    fig, ax = plt.subplots(figsize=WIDE)
    ax.set_xlim(-2.8, 10.5)
    ax.set_ylim(-3.2, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    i = 1
    cells(ax, 0, 2, x, 1, 1, fs=15, highlight=(i, i + 1, i + 2))
    cells(ax, i, 0.4, w, 1, 1, fc="#fde8e8", ec=RED, fs=15)
    cells(ax, 1, -1.8, y, 1, 1, fc="#e6f0fa", fs=15, highlight=(i,))
    for name, yy in (("input x", 2.5), ("kernel w", 0.9), ("output y", -1.3)):
        ax.text(-0.4, yy, name, ha="right", va="center", fontsize=14)
    arrow(ax, (i + 1.5, 0.3), (i + 1.5, -0.7), color=RED, lw=2.2)
    ax.text(5.2, 0.9, "3·1 + 2·0 + 0·(−1) = 3", fontsize=15, color=RED, va="center")
    ax.text(4, -2.7, "y[t] = Σ_j w[j] · x[t + j]   (no flip: a cross-correlation, "
            "as in 2D)", ha="center", fontsize=13)
    ax.text(4, 3.3, "7 inputs, kernel 3, no padding → 7 − 3 + 1 = 5 outputs",
            ha="center", fontsize=13)
    save(fig, out, "conv1d_sliding")


def fig_conv1d_channels(out: Path) -> None:
    rng = np.random.default_rng(2)
    cmaps = [plt.get_cmap(n) for n in ("Blues", "Reds", "Greens")]
    fig, ax = plt.subplots(figsize=WIDE)
    ax.set_xlim(-1.2, 22)
    ax.set_ylim(-3.6, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    L, k = 10, 3
    for c in range(3):
        vals = rng.uniform(0.15, 0.7, L)
        cells(ax, 0, 3 - c, vals, 1, 1, cmap=cmaps[c], fmt=None)
        ax.text(-0.3, 3.5 - c, "xyz"[c], ha="right", va="center", fontsize=13)
    ax.add_patch(Rectangle((3, 1), k, 3, fill=False, edgecolor=RED, lw=3.5))
    ax.text(5, 4.5, "input [C_in = 3, L = 10]", ha="center", fontsize=13)
    ax.text(4.5, 0.3, "one kernel: [C_in, k] = [3, 3]\n9 weights + 1 bias",
            ha="center", va="top", fontsize=12.5, color=RED)
    arrow(ax, (10.2, 2.5), (12.8, 2.5), color=RED, lw=2.2)
    ax.text(11.5, 2.1, "sum over\nchannels\nand window", ha="center", va="top",
            fontsize=11.5, color=RED)
    for o, col in enumerate((PURPLE, TEAL, AMBER, BLUE)):
        vals = rng.uniform(0.15, 0.7, L - k + 1)
        cells(ax, 13, 3.4 - 1.2 * o, vals, 1, 1, fc="white", ec=col, fmt=None)
        ax.add_patch(Rectangle((13 + 3, 3.4 - 1.2 * o), 1, 1, facecolor=col,
                               alpha=0.25 if o else 0.6, edgecolor=col))
    ax.text(17.5, 4.65, "output [C_out = 4, 8]", ha="center", fontsize=13)
    ax.text(17.5, -1.7, "one kernel per output channel\nweight: [C_out, C_in, k] = "
            "[4, 3, 3]", ha="center", va="top", fontsize=12.5)
    save(fig, out, "conv1d_channels")


def fig_derivative(out: Path) -> None:
    signals, _ = load_ecg()
    x = signals[0]
    # nn.Conv1d is a cross-correlation: y[t] = 0.5 * (x[t+1] - x[t-1])
    d = np.correlate(np.pad(x, 1), [-0.5, 0, 0.5], mode="valid")
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharex=True)
    for ax, s, title, col in zip(
        axes, (x, d, np.maximum(d, 0)),
        ("input: one heartbeat", "conv with [−0.5, 0, 0.5]: slope",
         "ReLU: only the rising parts"),
        (INK, BLUE, RED),
    ):
        ax.plot(s, color=col, lw=2)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("time step")
    save(fig, out, "derivative")


def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    ax.set_xlim(0, 3.7 * 8)
    ax.set_ylim(-4.2, 6.5)
    ax.axis("off")
    # (channels, length, label, colour); length 96 is drawn 6 units wide
    stages = [
        (1, 96, "input\n[1, 96]", INK),
        (16, 96, "conv k + ReLU\n[16, 96]", BLUE),
        (16, 48, "max pool 2\n[16, 48]", GREY),
        (32, 48, "conv k + ReLU\n[32, 48]", BLUE),
        (32, 24, "max pool 2\n[32, 24]", GREY),
        (32, 24, "conv k + ReLU\n[32, 24]", BLUE),
        (32, 1, "global avg\npool [32]", AMBER),
        (2, 1, "linear\n[2]", PURPLE),
    ]
    slot = 3.7
    for i, (c, length, label, col) in enumerate(stages):
        cx = slot * (i + 0.5)
        w = max(0.3, 2.6 * length / 96)
        h = 0.13 * c + 0.2
        ax.add_patch(Rectangle((cx - w / 2, 1 - h / 2), w, h, facecolor=col,
                               alpha=0.25, edgecolor=col, lw=1.6))
        ax.text(cx, -2.0, label, ha="center", va="top", fontsize=11.5)
        if i < len(stages) - 1:
            arrow(ax, (cx + 1.45, 1), (cx + slot - 1.45, 1))
    ax.text(slot * 4, 5.8, "height = channels, width = length:  more channels, shorter "
            "signals", ha="center", fontsize=13)
    save(fig, out, "architecture")


def fig_receptive_field(out: Path) -> None:
    signals, _ = load_ecg()
    x = signals[0]
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.plot(x, color=INK, lw=2)
    centre = 40
    for row, (k, col) in enumerate(((3, GREEN), (7, BLUE), (15, RED))):
        rf = 7 * k - 3
        a, b = max(centre - rf / 2, 0), min(centre + rf / 2, 95)
        y = -2.4 - 0.55 * row
        ax.plot([a, b], [y, y], color=col, lw=6, solid_capstyle="butt")
        ax.text(b + 2, y, f"k = {k}: {rf} steps" + (" > 96" if rf > 96 else ""),
                va="center", fontsize=12, color=col)
    ax.set_xlim(-2, 130)
    ax.set_ylim(-4, 4)
    ax.set_title("what one output of the last conv layer sees")
    ax.set_xlabel("time step")
    ax = axes[1]
    blank(ax)
    rows = [("layer", "receptive field", "step"), ("conv k", "k", "1"),
            ("pool 2", "k + 1", "2"), ("conv k", "3k − 1", "2"),
            ("pool 2", "3k + 1", "4"), ("conv k", "7k − 3", "4")]
    for r, (a, b, c) in enumerate(rows):
        y = 0.88 - 0.13 * r
        kw = {"weight": "bold"} if r == 0 else {}
        ax.text(0.05, y, a, fontsize=13, **kw)
        ax.text(0.32, y, b, fontsize=13, color=RED if r == 5 else INK, **kw)
        ax.text(0.9, y, c, fontsize=13, **kw)
    ax.text(0.05, 0.02, "each conv adds (k − 1) × step;\neach pool doubles the step",
            fontsize=12, color=BLUE)
    save(fig, out, "receptive_field")


def fig_equivariance(out: Path) -> None:
    signals, _ = load_ecg()
    x = signals[0]
    s = 20
    xs = np.roll(x, s)
    kern = [-0.5, 0, 0.5]

    def feat(v):  # circular padding, so the roll is exact
        return np.maximum(np.correlate(np.r_[v[-1], v, v[0]], kern, mode="valid"), 0)

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    ax = axes[0]
    ax.plot(x, color=BLUE, lw=2, label="x")
    ax.plot(xs, color=RED, lw=2, alpha=0.8, label=f"x shifted by {s}")
    ax.set_title("input")
    ax.legend(frameon=False, fontsize=11)
    ax = axes[1]
    ax.plot(feat(x), color=BLUE, lw=2)
    ax.plot(feat(xs), color=RED, lw=2, alpha=0.8)
    ax.set_title("feature map: shifted the same way\n(equivariance)")
    ax = axes[2]
    vals = [feat(x).mean(), feat(xs).mean()]
    ax.bar(["x", "shifted"], vals, color=(BLUE, RED), width=0.6)
    ax.set_title("global average pool:\nidentical (invariance)")
    ax.set_ylim(0, max(vals) * 1.3)
    for k, v in enumerate(vals):
        ax.text(k, v * 1.03, f"{v:.3f}", ha="center", fontsize=12)
    save(fig, out, "equivariance")


def fig_results(out: Path) -> None:
    names = list(RESULTS)
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=WIDE)
    for off, j, col, lab in ((-0.2, 0, BLUE, "test"), (0.2, 2, RED, "shifted test")):
        m = [RESULTS[n][j] for n in names]
        sd = [RESULTS[n][j + 1] for n in names]
        ax.bar(pos + off, m, 0.38, yerr=sd, color=col, alpha=0.85, label=lab,
               capsize=3)
        for p, v, e in zip(pos + off, m, sd):
            ax.text(p, v + e + 0.8, f"{v:.0f}", ha="center", fontsize=10.5)
    ax.axhline(64, color=GREY, ls="--", lw=1.2, label="always 'normal' (64%)")
    ax.set_xticks(pos, names, fontsize=11)
    ax.set_ylim(50, 100)
    ax.set_ylabel("accuracy (%), mean ± std over 5 seeds")
    ax.legend(frameon=False, loc="upper left", ncols=3)
    save(fig, out, "results")


def fig_dilation(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    n = 16
    for ax, dils, title in zip(axes, ((1, 1, 1), (1, 2, 4)),
                               ("causal conv, k = 2: sees 4 steps",
                                "dilations 1, 2, 4: sees 8 steps")):
        ax.set_xlim(-0.8, n - 0.2)
        ax.set_ylim(-0.7, 3.7)
        ax.axis("off")
        ax.set_title(title)
        used = {(3, n - 1)}
        for layer in range(3, 0, -1):
            d = dils[layer - 1]
            for (l, t) in list(used):
                if l != layer:
                    continue
                for src in (t, t - d):
                    if src >= 0:
                        used.add((layer - 1, src))
                        ax.plot([src, t], [layer - 1, layer], color=RED, lw=1.8,
                                zorder=1)
        for layer in range(4):
            for t in range(n):
                on = (layer, t) in used
                ax.plot(t, layer, "o", ms=9, zorder=2,
                        color=RED if on else "white",
                        markeredgecolor=RED if on else GREY)
            ax.text(-0.7, layer, ("input", "layer 1", "layer 2", "layer 3")[layer],
                    ha="right", va="center", fontsize=11)
        ax.text(n / 2, -0.6, "time →   (output at t only uses inputs ≤ t)",
                ha="center", fontsize=11, color=GREY)
    save(fig, out, "dilation")


FIGURES = (
    fig_pipeline, fig_random_shift, fig_signals, fig_conv2d_vs_1d, fig_conv1d_sliding,
    fig_conv1d_channels, fig_derivative, fig_architecture, fig_receptive_field,
    fig_equivariance, fig_results, fig_dilation,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course11/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
