#!/usr/bin/env python3
"""Render the normalization figures used by the Course09 lecture slides.

Session 9 builds dense and convolutional networks, then adds batch normalization to
both. The concept figures are synthetic and seeded; `bn_results.png` plots the numbers
measured in Course09/linear_vs_conv-solutions.ipynb (without batch norm) and
Course09/linear_vs_conv-norm.ipynb (with batch norm).

Usage: make_figures_c09.py [outdir]      (default: Course09/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

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

# the picture frame below the title of the .pptx is 11.95in x 5.5in (aspect 2.17)
WIDE = (12.6, 5.8)

SLIDE_ASPECT = 11.95 / 5.5

# training accuracy per epoch (%) and test accuracy (%), from the notebooks
RESULTS = {
    "dense": (
        [27.06, 33.50, 37.38, 37.25, 41.12, 44.69, 47.44, 46.38, 46.31, 48.62], 49.75,
    ),
    "conv": (
        [26.19, 37.12, 38.25, 41.44, 42.81, 43.94, 45.06, 44.44, 46.50, 45.75], 44.75,
    ),
    "dense + BN": (
        [52.88, 57.88, 61.62, 62.25, 63.81, 65.69, 65.69, 66.38, 66.81, 66.94], 58.25,
    ),
    "conv + BN": (
        [48.50, 58.69, 61.81, 62.44, 64.06, 65.81, 67.50, 69.94, 69.81, 72.06], 60.25,
    ),
}


def pad_to_aspect(path: Path, aspect: float = SLIDE_ASPECT) -> None:
    """Pad a saved figure with white so it exactly fills the slide picture frame."""
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


def box(ax, xy, w, h, text, fc="white", ec=BLUE, fs=11, lw=1.8, tc=INK):
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


# ---------------------------------------------------------------------------------
# why normalize: gradient descent on badly vs well scaled features


def _descent(scales, lr, start, steps=25):
    """Gradient descent on the quadratic loss sum_i (scale_i * w_i)^2."""
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * 2 * scales**2 * w
        path.append(w.copy())
    return np.array(path)


def fig_norm_why(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    cases = (
        (np.array([1.0, 6.0]), 0.0262, "raw features (very different scales)", RED),
        (np.array([1.0, 1.0]), 0.3, r"standardized features: $x' = (x - \mu)\,/\,\sigma$",
         GREEN),
    )
    g = np.linspace(-3.3, 3.3, 300)
    W1, W2 = np.meshgrid(g, g)
    for ax, (scales, lr, title, col) in zip(axes, cases):
        loss = (scales[0] * W1) ** 2 + (scales[1] * W2) ** 2
        ax.contour(W1, W2, loss, levels=np.geomspace(0.3, loss.max(), 12),
                   colors=GREY, linewidths=0.9)
        path = _descent(scales, lr, start=(-3.0, 0.45), steps=25 if col == RED else 8)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=4, lw=1.8)
        ax.plot(0, 0, "*", color=INK, ms=14)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"weight $w_1$")
        ax.set_ylabel(r"weight $w_2$")
        ax.set_title(title, color=col)
    axes[0].text(0, -3.05, "zig-zags: the learning rate must stay small\n"
                 "for the steep direction", ha="center", fontsize=12, color=RED)
    axes[1].text(0, -3.05, "round loss: the gradient points\nto the minimum",
                 ha="center", fontsize=12, color=GREEN)
    save(fig, out, "norm_why")


# ---------------------------------------------------------------------------------
# inside the network: the input distribution of each layer keeps moving


def fig_covariate_shift(out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    bins = np.linspace(-6, 12, 70)
    drift = ((-1.0, 1.0), (2.0, 2.0), (5.0, 3.0))
    cols = (BLUE, PURPLE, RED)
    for (mu, sd), col, epoch in zip(drift, cols, (1, 5, 10)):
        z = rng.normal(mu, sd, 4000)
        axes[0].hist(z, bins=bins, density=True, alpha=0.45, color=col,
                     label=f"epoch {epoch}")
        zn = (z - z.mean()) / z.std()
        axes[1].hist(zn, bins=bins, density=True, alpha=0.45, color=col,
                     label=f"epoch {epoch}")
    axes[0].set_title("without normalization: layer $l+1$ sees a moving target",
                      color=RED)
    axes[1].set_title("with batch norm: always mean 0, std 1", color=GREEN)
    for ax in axes:
        ax.set_xlabel("activation of one neuron of layer $l$")
        ax.set_yticks([])
        ax.legend(frameon=False)
        ax.spines["left"].set_visible(False)
    save(fig, out, "covariate_shift")


# ---------------------------------------------------------------------------------
# the batch norm equations


BN_EQS = (
    ("batch mean", BLUE,
     r"$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$"),
    ("batch variance", BLUE,
     r"$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$"),
    ("normalize", PURPLE,
     r"$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$"),
    ("scale and shift", GREEN,
     r"$y_i = \gamma\,\hat{x}_i + \beta$"),
)


def fig_bn_equations(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    for k, (name, col, eq) in enumerate(BN_EQS):
        y = 0.90 - 0.26 * k
        ax.text(0.0, y, name, fontsize=18, color=col, va="center")
        ax.text(0.22, y, eq, fontsize=28, color=col, va="center")
    notes = (
        r"$x_1, \dots, x_m$: the values of ONE neuron" "\n(or ONE channel) over the batch",
        r"$\epsilon \approx 10^{-5}$: avoids dividing by zero",
        r"$\gamma, \beta$: learned, like weights",
        r"($\gamma = \sigma_B$, $\beta = \mu_B$ undoes the normalization:"
        "\n the network can choose the scale it needs)",
    )
    for k, t in enumerate(notes):
        ax.text(0.66, 0.92 - 0.2 * k, t, fontsize=17, color=INK if k < 2 else GREEN,
                va="top")
    save(fig, out, "bn_equations")


# ---------------------------------------------------------------------------------
# a worked example on a batch of 4


def _column(ax, x, values, col, title, fmt="{:g}"):
    h = 0.13
    ax.text(x, 0.86, title, ha="center", va="bottom", fontsize=13, color=col)
    for k, v in enumerate(values):
        y = 0.76 - k * (h + 0.025)
        ax.add_patch(plt.Rectangle((x - 0.07, y - h / 2), 0.14, h, facecolor=col,
                                   edgecolor="white", alpha=0.9))
        ax.text(x, y, fmt.format(v), ha="center", va="center", fontsize=15,
                color="white")


def fig_bn_example(out: Path) -> None:
    x = np.array([2.0, 4.0, 6.0, 12.0])
    mu, sd = x.mean(), x.std()
    xh = (x - mu) / sd
    gamma, beta = 2.0, 1.0
    y = gamma * xh + beta
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    _column(ax, 0.10, x, GREY, "$x$ (batch of 4)")
    _column(ax, 0.52, xh, PURPLE, r"$\hat{x}$", fmt="{:+.2f}")
    _column(ax, 0.90, y, GREEN, "$y$", fmt="{:+.2f}")
    arrow(ax, (0.19, 0.53), (0.43, 0.53), color=PURPLE, lw=2.2)
    ax.text(0.31, 0.62, rf"$\mu_B = {mu:g}$" "\n" rf"$\sigma_B = {sd:.2f}$",
            ha="center", fontsize=15, color=BLUE)
    ax.text(0.31, 0.44, "subtract the mean,\ndivide by the std", ha="center",
            va="top", fontsize=12, color=PURPLE)
    arrow(ax, (0.61, 0.53), (0.81, 0.53), color=GREEN, lw=2.2)
    ax.text(0.71, 0.62, rf"$\gamma = {gamma:g}$" "\n" rf"$\beta = {beta:g}$",
            ha="center", fontsize=15, color=GREEN)
    ax.text(0.71, 0.44, "learned\nscale & shift", ha="center", va="top", fontsize=12,
            color=GREEN)
    ax.text(0.52, 0.12, "mean 0, std 1", ha="center", fontsize=12, color=PURPLE)
    ax.text(0.90, 0.12, rf"mean {beta:g}, std {gamma:g}", ha="center", fontsize=12,
            color=GREEN)
    save(fig, out, "bn_example")


# ---------------------------------------------------------------------------------
# train vs eval


def fig_bn_train_eval(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    box(ax, (0.25, 0.80), 0.42, 0.14, "model.train()", ec=BLUE, fs=17, tc=BLUE)
    box(ax, (0.75, 0.80), 0.42, 0.14, "model.eval()", ec=GREEN, fs=17, tc=GREEN)
    ax.text(0.05, 0.62, "normalize with the statistics\nof the current batch:",
            fontsize=14, va="top")
    ax.text(0.07, 0.42, r"$\hat{x} = (x - \mu_B)\,/\,\sigma_B$", fontsize=20,
            color=BLUE, va="center")
    ax.text(0.05, 0.30, "and update the running averages:", fontsize=14, va="top")
    ax.text(0.07, 0.14, r"$\mu_{run} \leftarrow 0.9\,\mu_{run} + 0.1\,\mu_B$",
            fontsize=18, color=BLUE, va="center")
    ax.text(0.07, 0.03, r"$\sigma^2_{run} \leftarrow 0.9\,\sigma^2_{run} + 0.1\,\sigma^2_B$",
            fontsize=18, color=BLUE, va="center")
    ax.text(0.55, 0.62, "normalize with the running averages\n(fixed numbers):",
            fontsize=14, va="top")
    ax.text(0.57, 0.42, r"$\hat{x} = (x - \mu_{run})\,/\,\sigma_{run}$", fontsize=20,
            color=GREEN, va="center")
    ax.text(0.55, 0.30, "the prediction for one image does not\ndepend on the other "
            "images of the batch\n(works with a batch of size 1)", fontsize=14,
            va="top", color=GREEN)
    ax.plot([0.5, 0.5], [0.0, 0.66], color=FAINT, lw=2)
    save(fig, out, "bn_train_eval")


# ---------------------------------------------------------------------------------
# which values are normalized together: batch, layer, instance and group norm


def _cube(ax, x0, shaded, title, note, n=5, c=6, hw=5, s=0.09, d=0.045):
    """An (N, C, H*W) block drawn in oblique projection, shaded cells in BLUE.

    `shaded(ni, ci)` says whether the cells of sample ni, channel ci are normalized
    together (always over all of H*W).
    """
    def pt(ci, hi, ni):
        return (x0 + ci * s + ni * d, 0.30 + hi * s * 0.9 + ni * d)

    def quad(pts, on):
        ax.add_patch(Polygon(pts, closed=True, facecolor=BLUE if on else "white",
                             edgecolor=INK, lw=0.7, alpha=0.85 if on else 1))

    for ci in range(c):  # front face: sample 0
        for hi in range(hw):
            quad([pt(ci, hi, 0), pt(ci + 1, hi, 0), pt(ci + 1, hi + 1, 0),
                  pt(ci, hi + 1, 0)], shaded(0, ci))
    for ci in range(c):  # top face
        for ni in range(n):
            quad([pt(ci, hw, ni), pt(ci + 1, hw, ni), pt(ci + 1, hw, ni + 1),
                  pt(ci, hw, ni + 1)], shaded(ni, ci))
    for ni in range(n):  # right face: last channel
        for hi in range(hw):
            quad([pt(c, hi, ni), pt(c, hi, ni + 1), pt(c, hi + 1, ni + 1),
                  pt(c, hi + 1, ni)], shaded(ni, c - 1))
    mid = x0 + (c * s + n * d) / 2
    ax.text(mid, 1.0, title, ha="center", fontsize=15, fontweight="bold")
    ax.text(x0 + c * s / 2, 0.23, "C (channels)", ha="center", fontsize=11)
    ax.text(mid, 0.17, note, ha="center", va="top", fontsize=12, color=BLUE,
            linespacing=1.4)
    ax.text(x0 - 0.03, 0.30 + hw * s * 0.45, "H, W", ha="right", va="center",
            fontsize=11)
    a, b = pt(c, 0, 0), pt(c, 0, n)
    ax.text((a[0] + b[0]) / 2 + 0.04, (a[1] + b[1]) / 2 - 0.04, "N (batch)",
            fontsize=11, rotation=40, ha="center", va="center")


def fig_norm_axes(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.04, 1.07)
    ax.axis("off")
    _cube(ax, 0.15, lambda ni, ci: ci == 0, "Batch Norm",
          "per channel,\nover the batch\n(CNNs)")
    _cube(ax, 1.15, lambda ni, ci: ni == 0, "Layer Norm",
          "per sample,\nover all channels\n(RNNs, Transformers)")
    _cube(ax, 2.15, lambda ni, ci: ni == 0 and ci == 0, "Instance Norm",
          "per sample\nand per channel\n(style transfer)")
    _cube(ax, 3.15, lambda ni, ci: ni == 0 and ci < 3, "Group Norm",
          "per sample,\nper group of channels\n(small batches)")
    save(fig, out, "norm_axes")


# ---------------------------------------------------------------------------------
# results of the notebook


def fig_bn_results(out: Path) -> None:
    cols = {"dense": AMBER, "conv": RED, "dense + BN": TEAL, "conv + BN": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": (3, 2)})
    ax = axes[0]
    epochs = np.arange(1, 11)
    for name, (train, _) in RESULTS.items():
        ls = "-" if "BN" in name else "--"
        ax.plot(epochs, train, ls, marker="o", ms=4, color=cols[name], lw=2, label=name)
    ax.axhline(25, color=GREY, lw=1, ls=":")
    ax.text(10, 26, "chance", color=GREY, ha="right", fontsize=11)
    ax.set_ylim(20, 90)
    ax.set_xticks(epochs)
    ax.set_xlabel("epoch")
    ax.set_title("training accuracy (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax = axes[1]
    names = list(RESULTS)
    test = [RESULTS[n][1] for n in names]
    ax.bar(names, test, color=[cols[n] for n in names])
    for k, v in enumerate(test):
        ax.text(k, v + 1, f"{v:.1f}", ha="center", fontsize=12)
    ax.axhline(25, color=GREY, lw=1, ls=":")
    ax.set_ylim(0, 80)
    ax.set_title("test accuracy (%) after 10 epochs")
    ax.set_xticks(range(len(names)), [n.replace(" + ", "\n+ ") for n in names])
    ax.tick_params(axis="x", labelsize=11)
    save(fig, out, "bn_results")


FIGURES = (
    fig_norm_why, fig_covariate_shift, fig_bn_equations, fig_bn_example,
    fig_bn_train_eval, fig_bn_results, fig_norm_axes,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course09/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
