#!/usr/bin/env python3
"""Render the illustration figures used by the Course06 lecture slides.

Session 6 is about what happens after `loss.backward()`: which optimizer turns the
gradient into an update, how the learning rate should move during training, and the
layers and losses that make a network train well. Every figure is synthetic and
seeded, so `make figures` reproduces the same output anywhere.

Usage: make_figures_c06.py [outdir]      (default: Course06/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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


def arrow(ax, a, b, color=GREY, lw=1.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
            shrinkA=2, shrinkB=2,
        )
    )


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ================================================================ I. optimizers


def fig_optimizer_role(out: Path) -> None:
    """Backprop gives the gradient; the optimizer decides what to do with it."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    y = 0.68
    steps = ((0.11, 0.18, "forward pass\n" r"$\hat y = f_\theta(x)$", GREY),
             (0.34, 0.16, "loss\n" r"$L(\hat y, y)$", GREY),
             (0.58, 0.20, "backpropagation\n" r"$\nabla_\theta L$", BLUE),
             (0.86, 0.24, "optimizer\n" r"$\theta \leftarrow \mathrm{update}"
              r"(\theta, \nabla_\theta L)$", RED))
    for x, w, txt, col in steps:
        box(ax, (x, y), w, 0.22, txt, ec=col, fs=12.5)
    for (a, wa, _, _), (b, wb, _, _) in zip(steps, steps[1:]):
        arrow(ax, (a + wa / 2 + 0.01, y), (b - wb / 2 - 0.01, y))
    ax.annotate("", (0.11, 0.80), (0.86, 0.80),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, ls="--",
                                connectionstyle="arc3,rad=0.18"))
    ax.text(0.485, 0.985, "next mini-batch", color=GREY, fontsize=11, ha="center")
    ax.text(0.58, 0.30, "computes the gradient\n(autograd, last session)",
            color=BLUE, ha="center", fontsize=12.5, linespacing=1.5)
    ax.text(0.86, 0.26, "turns it into a step:\nSGD, momentum, Adam, ...\n"
            "a hyperparameter, like\nits learning rate",
            color=RED, ha="center", fontsize=12.5, linespacing=1.5)
    save(fig, out, "optimizer_role")


def fig_crowded_valley(out: Path) -> None:
    """The three findings of Schmidt, Schneider & Hennig (2021)."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.5, 0.95, "Descending through a Crowded Valley  ·  Schmidt, Schneider, "
            "Hennig  ·  ICML 2021", ha="center", fontsize=12, color=GREY)
    ax.text(0.5, 0.86, "15 optimizers  ×  8 deep learning problems  →  "
            "more than 50,000 training runs", ha="center", fontsize=13)
    cards = (("1", "No universal winner",
              "optimizer performance\nvaries greatly\nacross tasks", BLUE),
             ("2", "Changing optimizer\nbeats tuning one",
              "several optimizers with\ndefault settings work about\nas well as "
              "tuning one", AMBER),
             ("3", "Adam is a good\nstarting point",
              "newer, more complicated\nmethods do not consistently\n"
              "outperform it", GREEN))
    for k, (num, head, body, col) in enumerate(cards):
        x = 0.18 + 0.32 * k
        ax.add_patch(FancyBboxPatch((x - 0.14, 0.06), 0.28, 0.68,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="white", edgecolor=col, lw=2.0))
        ax.text(x, 0.64, num, ha="center", va="center", fontsize=22, color=col,
                fontweight="bold")
        ax.text(x, 0.48, head, ha="center", va="center", fontsize=14, color=col,
                linespacing=1.3)
        ax.text(x, 0.23, body, ha="center", va="center", fontsize=12, color=INK,
                linespacing=1.5)
    save(fig, out, "crowded_valley")


def _bowl(x, y):
    """An elongated quadratic: steep across, shallow along."""
    return 0.5 * (x ** 2 / 9.0 + 4.0 * y ** 2)


def _bowl_grad(p):
    return np.array([p[0] / 9.0, 4.0 * p[1]])


def _contours(ax, lim=(-10, 2, -2.2, 2.2)):
    xs = np.linspace(lim[0], lim[1], 300)
    ys = np.linspace(lim[2], lim[3], 300)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, _bowl(X, Y), levels=np.geomspace(0.05, 30, 12),
               colors=GREY, linewidths=0.7, alpha=0.7)
    ax.plot(0, 0, marker="*", color=INK, ms=14, zorder=5)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"parameter $\theta_1$")
    ax.set_ylabel(r"parameter $\theta_2$")


def fig_sgd_paths(out: Path) -> None:
    """Full-batch gradient descent against the same update on noisy mini-batches."""
    rng = np.random.default_rng(6)
    start = np.array([-9.0, 1.6])
    lr = 0.22
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, noise, title, col in ((axes[0], 0.0, "full dataset per step: gradient descent",
                                   BLUE),
                                  (axes[1], 1.1, "shuffled mini-batch per step: SGD", RED)):
        _contours(ax)
        p = start.copy()
        path = [p.copy()]
        for _ in range(60):
            g = _bowl_grad(p) + noise * rng.normal(size=2) * np.array([0.25, 1.0])
            p = p - lr * g
            path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=3, lw=1.4)
        ax.set_title(title)
    save(fig, out, "sgd_paths")


def fig_batch_noise(out: Path) -> None:
    """The mini-batch gradient is an average: its noise shrinks like 1/sqrt(B)."""
    rng = np.random.default_rng(0)
    # per-example gradients of a 1D least-squares problem at a fixed theta
    n = 20_000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=1.0, size=n)
    theta = 0.5
    g = (theta * x - y) * x
    full = g.mean()

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.3})
    ax = axes[0]
    for B, col in ((1, RED), (32, AMBER), (512, GREEN)):
        est = np.array([g[rng.integers(0, n, B)].mean() for _ in range(4000)])
        ax.hist(est, bins=np.linspace(-8, 2, 120), density=True, histtype="stepfilled",
                alpha=0.35, color=col, label=f"batch size {B}")
    ax.axvline(full, color=INK, lw=1.6, ls="--")
    ax.text(full + 0.15, ax.get_ylim()[1] * 0.55, "full-data\ngradient", fontsize=10.5)
    ax.set_xlabel("gradient estimate from one mini-batch")
    ax.set_yticks([])
    ax.set_title("each step sees a noisy estimate of the gradient")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    Bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    sd = g.std() / np.sqrt(Bs)
    ax.loglog(Bs, sd, "-o", color=BLUE, lw=2.4)
    ax.set_xlabel("batch size B")
    ax.set_ylabel("noise (std of the estimate)")
    ax.set_title(r"noise $\propto 1/\sqrt{B}$  →  room for a larger step")
    ax.spines["top"].set_visible(False)
    save(fig, out, "batch_noise")


def fig_momentum(out: Path) -> None:
    """Momentum as a velocity: the update and what it does in a ravine."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [0.95, 1.2], "wspace": 0.12})
    ax = axes[0]
    blank(ax)
    ax.text(0.02, 0.86, "velocity: a running sum of past gradients", fontsize=12,
            color=GREY)
    ax.text(0.06, 0.70, r"$v_t = \gamma\, v_{t-1} + \nabla_\theta L(\theta_t)$",
            fontsize=19, color=BLUE)
    ax.text(0.02, 0.50, "step along the velocity, not the raw gradient", fontsize=12,
            color=GREY)
    ax.text(0.06, 0.34, r"$\theta_{t+1} = \theta_t - \mathrm{lr}\cdot v_t$",
            fontsize=19, color=RED)
    ax.text(0.02, 0.10, r"Momentum $\gamma$ is a hyperparameter (typically 0.9);"
            "\n" r"Plain SGD if $\gamma = 0$", fontsize=12, linespacing=1.6)

    ax = axes[1]
    _contours(ax)
    start = np.array([-9.0, 1.6])
    for gamma, lr, col, lab in ((0.0, 0.3, RED, "SGD"),
                                (0.8, 0.1, GREEN, "SGD + momentum (γ = 0.8)")):
        p, v = start.copy(), np.zeros(2)
        path = [p.copy()]
        for _ in range(40):
            v = gamma * v + _bowl_grad(p)
            p = p - lr * v
            path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=3, lw=1.4, label=lab)
    ax.set_title("40 steps in a ravine")
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    save(fig, out, "momentum")


# ============================================================= II. learning rate


def _parabola_steps(ax, lr, n=8, x0=-2.6, title="", col=BLUE):
    xs = np.linspace(-3.2, 3.2, 200)
    ax.plot(xs, xs ** 2, color=GREY, lw=1.6)
    x = x0
    pts = [x]
    for _ in range(n):
        x = x - lr * 2 * x
        pts.append(x)
    pts = np.clip(np.array(pts), -3.2, 3.2)
    ax.plot(pts, pts ** 2, "-o", color=col, ms=4.5, lw=1.3)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-0.4, 10.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=col, fontsize=12)
    ax.set_xlabel(r"parameter $\theta$")


def fig_lr_effect(out: Path) -> None:
    """Too small, right, too large, divergent — on a bowl and on the loss curve."""
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.18)
    cases = ((0.04, "too low", BLUE), (0.3, "good", GREEN),
             (0.85, "too high", AMBER), (1.05, "much too high", RED))
    for k, (lr, lab, col) in enumerate(cases):
        ax = fig.add_subplot(gs[0, k])
        _parabola_steps(ax, lr, title=f"lr = {lr}: {lab}", col=col)
        if k == 0:
            ax.set_ylabel(r"loss $L(\theta)$")
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(40)
    for lr, lab, col in cases:
        x = -2.6 * (1 - 2 * lr) ** t
        ax.plot(t, np.minimum(x ** 2, 12), color=col, lw=2.4, label=lab)
    ax.set_ylim(0, 12)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=10.5)
    save(fig, out, "lr_effect")


def fig_lr_decay(out: Path) -> None:
    """A constant step bounces around the minimum; a decaying one settles in it."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    x0, n = -2.6, 14
    for ax, decay, title, col in ((axes[0], False, "constant learning rate", RED),
                                  (axes[1], True, "decaying learning rate", GREEN)):
        xs = np.linspace(-3.2, 3.2, 200)
        ax.plot(xs, xs ** 2, color=GREY, lw=1.6)
        x, pts = x0, [x0]
        for t in range(n):
            # on L = theta^2 a step multiplies theta by (1 - 2 lr): close to -1
            # means it jumps to the other wall almost as high as it started
            lr = 0.97 * 0.8 ** t if decay else 0.97
            x = x - lr * 2 * x
            pts.append(x)
        pts = np.array(pts)
        ax.plot(pts, pts ** 2, "-o", color=col, ms=4.5, lw=1.2)
        ax.set_title(title, color=col)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"parameter $\theta$")
    axes[0].set_ylabel(r"loss $L(\theta)$")
    axes[0].text(0, 8.5, "keeps oscillating\naround the minimum", ha="center",
                 color=RED, fontsize=12)
    axes[1].text(0, 8.5, "big steps early,\nsmall steps late", ha="center",
                 color=GREEN, fontsize=12)
    save(fig, out, "lr_decay")


def fig_lr_schedules(out: Path) -> None:
    """The schedules torch.optim.lr_scheduler gives you."""
    ep = np.linspace(0, 100, 1001)
    lr0 = 1e-2
    scheds = (
        ("StepLR", lr0 * 0.3 ** np.floor(ep / 30), BLUE),
        ("ExponentialLR", lr0 * 0.95 ** ep, AMBER),
        ("CosineAnnealingLR", lr0 * 0.5 * (1 + np.cos(np.pi * ep / 100)), GREEN),
        ("CyclicLR (triangular, decaying)",
         lr0 * 0.5 ** np.floor(ep / 20) * (1 - np.abs((ep % 20) / 10 - 1)), RED),
    )
    fig, axes = plt.subplots(1, 4, figsize=WIDE, sharey=True)
    for ax, (name, lr, col) in zip(axes, scheds):
        ax.plot(ep, lr, color=col, lw=2.4)
        ax.set_title(name, fontsize=11.5, color=col)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, lr0 * 1.08)
    axes[0].set_ylabel("learning rate")
    fig.text(0.5, -0.04, "the schedule is one more hyperparameter", ha="center",
             color=GREY, fontsize=12)
    save(fig, out, "lr_schedules")


# ================================================= III. stabilising and regularising


def _cliff(x):
    """A loss with a high plateau that drops through a steep cliff into a valley."""
    return 1.2 + 0.03 * (x - 5) ** 2 - 1.0 / (1 + np.exp(-40 * (x - 2.6)))


def _cliff_grad(x, h=1e-4):
    return (_cliff(x + h) - _cliff(x - h)) / (2 * h)


def fig_grad_clipping(out: Path) -> None:
    """At a cliff the gradient explodes; clipping its norm keeps the step sane."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1, 1, 0.85], "wspace": 0.25})
    xs = np.linspace(0, 11.5, 800)
    # the start is chosen so that the sixth step lands right on the cliff
    lr, thr, x0 = 0.8, 0.5, 1.72
    for ax, clip, title, col in ((axes[0], False, "without clipping", RED),
                                 (axes[1], True, "with clipping", GREEN)):
        ax.plot(xs, _cliff(xs), color=GREY, lw=1.8)
        x, pts = x0, [x0]
        for _ in range(7 if not clip else 16):
            g = _cliff_grad(x)
            if clip and abs(g) > thr:
                g = thr * np.sign(g)
            x = x - lr * g
            pts.append(x)
        pts = np.array(pts)
        ax.plot(pts, _cliff(pts), "-o", color=col, ms=5, lw=1.4)
        ax.set_title(title, color=col)
        ax.set_xlabel(r"parameter $\theta$")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel(r"loss $L(\theta)$")
    axes[0].text(5.6, 1.85, "on the cliff: huge gradient,\nthrown far past the valley",
                 color=RED, fontsize=11, ha="center", linespacing=1.4)
    axes[1].text(5.6, 1.85, "same direction, bounded step:\nwalks down into the valley",
                 color=GREEN, fontsize=11, ha="center", linespacing=1.4)
    for ax in axes[:2]:
        ax.set_ylim(0.1, 2.2)

    ax = axes[2]
    blank(ax)
    ax.text(0.0, 0.92, "clip by norm", fontsize=13, color=BLUE)
    ax.text(0.0, 0.72, r"$g \leftarrow \nabla_\theta L$", fontsize=15)
    ax.text(0.0, 0.54, r"if $\|g\| > \tau$:", fontsize=15)
    ax.text(0.10, 0.36, r"$g \leftarrow \dfrac{\tau}{\|g\|}\, g$", fontsize=15)
    ax.text(0.0, 0.08, "same direction,\nbounded length", fontsize=12, color=GREY,
            linespacing=1.5)
    save(fig, out, "grad_clipping")


def fig_batchnorm(out: Path) -> None:
    """Normalise each feature over the mini-batch, then let the layer rescale it."""
    rng = np.random.default_rng(3)
    a = rng.normal(6.0, 3.0, 64)
    b = rng.normal(-40.0, 15.0, 64)
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.32})
    ax = axes[0]
    ax.hist(a, bins=15, color=BLUE, alpha=0.6, label="feature 1")
    ax.hist(b, bins=15, color=AMBER, alpha=0.6, label="feature 2")
    ax.set_title("activations entering a layer")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    for v, col in ((a, BLUE), (b, AMBER)):
        ax.hist((v - v.mean()) / v.std(), bins=15, color=col, alpha=0.6)
    ax.set_xlim(-4, 4)
    ax.set_title("after normalisation: mean 0, std 1")
    ax.set_yticks([])
    ax = axes[2]
    blank(ax)
    ax.text(0.0, 0.90, "for each feature, over the mini-batch:", fontsize=11.5,
            color=GREY)
    ax.text(0.04, 0.70, r"$\hat x = \dfrac{x - \mu_{\mathrm{batch}}}"
            r"{\sqrt{\sigma^2_{\mathrm{batch}} + \epsilon}}$", fontsize=16, color=BLUE)
    ax.text(0.0, 0.44, "then a learned scale and shift:", fontsize=11.5, color=GREY)
    ax.text(0.04, 0.30, r"$y = \gamma\, \hat x + \beta$", fontsize=16, color=RED)
    ax.text(0.0, 0.06, "nn.BatchNorm1d / nn.BatchNorm2d", fontsize=11.5,
            family="monospace")
    save(fig, out, "batchnorm")


def fig_batchnorm_batch(out: Path) -> None:
    """Why 'batch': the statistics come from the mini-batch, not the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.15})
    ax = axes[0]
    blank(ax)
    rows, cols = 6, 4
    x0, y0, w, h = 0.10, 0.18, 0.12, 0.09
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(plt.Rectangle((x0 + j * w, y0 + i * h), w * 0.94, h * 0.9,
                                       facecolor=FAINT if j != 1 else "#d6e4f5",
                                       edgecolor="white"))
    ax.annotate("", (x0 + 1.5 * w, y0 - 0.03), (x0 + 1.5 * w, y0 + rows * h + 0.01),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(x0 + 1.5 * w, y0 - 0.09, r"$\mu,\ \sigma$ of one feature", ha="center",
            color=BLUE, fontsize=12)
    ax.text(x0 + cols * w + 0.03, y0 + rows * h / 2, "one row\nper example\nin the batch",
            va="center", fontsize=11.5, color=GREY, linespacing=1.4)
    ax.text(x0 + cols * w / 2, y0 + rows * h + 0.05, "features", ha="center",
            fontsize=11.5, color=GREY)
    ax.set_title("training: statistics of the current mini-batch")

    ax = axes[1]
    blank(ax)
    ax.set_title("evaluation: running averages")
    box(ax, (0.5, 0.74), 0.84, 0.22,
        "model.train()\nnormalise with this batch's mean and std,\n"
        "update a running average of both", ec=BLUE, fs=11.5)
    box(ax, (0.5, 0.36), 0.84, 0.22,
        "model.eval()\nnormalise with the running averages:\n"
        "the prediction no longer depends on the batch", ec=GREEN, fs=11.5)
    arrow(ax, (0.5, 0.62), (0.5, 0.48))
    ax.text(0.5, 0.08, "forgetting model.eval() is a classic bug", ha="center",
            color=RED, fontsize=12)
    save(fig, out, "batchnorm_batch")


def _network(ax, dropped, title, p_text=None):
    layers = (3, 5, 5, 2)
    xs = np.linspace(0.08, 0.92, len(layers))
    pos = []
    for x, n in zip(xs, layers):
        ys = np.linspace(0.5 - 0.15 * (n - 1) / 2, 0.5 + 0.15 * (n - 1) / 2, n)
        pos.append([(x, y) for y in ys])
    for l in range(len(layers) - 1):
        for i, a in enumerate(pos[l]):
            for j, b in enumerate(pos[l + 1]):
                dead = (l, i) in dropped or (l + 1, j) in dropped
                ax.plot([a[0], b[0]], [a[1], b[1]], color=FAINT if dead else GREY,
                        lw=0.8 if dead else 1.1, zorder=1)
    for l, layer in enumerate(pos):
        for i, (x, y) in enumerate(layer):
            dead = (l, i) in dropped
            ax.add_patch(Circle((x, y), 0.035, facecolor="white",
                                edgecolor=FAINT if dead else INK, lw=1.6, zorder=2))
            if dead:
                ax.plot([x - 0.022, x + 0.022], [y - 0.022, y + 0.022], color=RED, lw=2,
                        zorder=3)
                ax.plot([x - 0.022, x + 0.022], [y + 0.022, y - 0.022], color=RED, lw=2,
                        zorder=3)
    ax.set_title(title, fontsize=12)


def fig_dropout(out: Path) -> None:
    """Each forward pass trains a different thinned network."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE, gridspec_kw={"wspace": 0.08})
    for ax in axes:
        blank(ax)
        ax.set_aspect("equal")
    _network(axes[0], set(), "evaluation: every neuron active")
    _network(axes[1], {(1, 0), (1, 3), (2, 2)}, "training step t")
    _network(axes[2], {(1, 2), (2, 0), (2, 4), (0, 1)}, "training step t'")
    fig.text(0.5, 0.02, "each neuron is switched off with probability p (hyperparameter); survivors are scaled by 1/(1 − p) during training",
             ha="center", fontsize=12, color=GREY)
    save(fig, out, "dropout")


# ==================================================================== IV. losses


def fig_regression_losses(out: Path) -> None:
    """L1 and MSE as functions of the residual."""
    r = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, f, name, eq, col, note in (
            (axes[0], np.abs(r), "torch.nn.L1Loss", r"$\ell(x, y) = |x - y|$", BLUE,
             "every error costs in proportion:\nrobust to outliers"),
            (axes[1], r ** 2, "torch.nn.MSELoss", r"$\ell(x, y) = (x - y)^2$", RED,
             "large errors cost much more:\nsmooth gradient near 0")):
        ax.plot(r, f, color=col, lw=2.8)
        ax.set_title(name, family="monospace", color=col)
        ax.text(0, 7.6, eq, ha="center", fontsize=17)
        ax.text(0, 5.4, note, ha="center", fontsize=11.5, color=GREY, linespacing=1.5)
        ax.set_xlabel("prediction − target")
        ax.set_ylim(0, 9)
    axes[0].set_ylabel("loss")
    fig.text(0.5, -0.04, "averaged over the batch by default (reduction='mean')",
             ha="center", color=GREY, fontsize=12)
    save(fig, out, "regression_losses")


CLASSES = ("dog", "cat", "horse", "human")


def fig_cross_entropy(out: Path) -> None:
    """Cross entropy on one example: only the true class counts."""
    p_true = np.array([1, 0, 0, 0])
    p_hat = np.array([0.6, 0.25, 0.1, 0.05])
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.25})
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.80, r"$H(P^*, P) = -\sum_{i} P^*(i)\, \log P(i)$", ha="center",
            fontsize=20)
    ax.text(0.02, 0.52, r"$i$ : a class", fontsize=13)
    ax.text(0.02, 0.40, r"$P^*(i)$ : true probability, here 0 or 1", fontsize=13)
    ax.text(0.02, 0.28, r"$P(i)$ : predicted probability", fontsize=13)
    ax.text(0.02, 0.08, "one-hot target  →  the sum keeps one term:\n"
            r"$H = -\log P(\mathrm{true\ class})$", fontsize=13, color=BLUE,
            linespacing=1.6)

    ax = axes[1]
    xs = np.arange(len(CLASSES))
    ax.bar(xs - 0.2, p_true, width=0.38, color=GREY, label=r"true $P^*$")
    ax.bar(xs + 0.2, p_hat, width=0.38, color=BLUE, label=r"predicted $P$")
    ax.set_xticks(xs, CLASSES)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("probability")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(r"a dog picture:  $H = -\log 0.6 = 0.51$")
    ax.text(1.5, 0.75, r"if $P(\mathrm{dog}) = 0.05$:  $H = 3.0$", ha="center",
            color=RED, fontsize=12)
    save(fig, out, "cross_entropy")


def fig_nll_loss(out: Path) -> None:
    """torch.nn.NLLLoss: pick the log-probability of the target, weight, reduce."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.02, 0.92, "torch.nn.NLLLoss  —  input: log-probabilities, one row per "
            "example", fontsize=13, family="monospace", color=BLUE)
    ax.text(0.04, 0.72, r"$\ell_n = -\, w_{y_n}\; x_{n,\,y_n}$", fontsize=21)
    ax.text(0.40, 0.78, r"$n$ : index of the example in the batch", fontsize=12.5)
    ax.text(0.40, 0.70, r"$y_n$ : its true class", fontsize=12.5)
    ax.text(0.40, 0.62, r"$x_{n,c}$ : predicted log-probability of class $c$",
            fontsize=12.5)
    ax.text(0.40, 0.54, r"$w_c$ : optional class weight (unbalanced data)",
            fontsize=12.5)
    ax.text(0.04, 0.36, "then over the batch:", fontsize=12.5, color=GREY)
    ax.text(0.04, 0.20, r"mean:  $\ell = \dfrac{\sum_n \ell_n}{\sum_n w_{y_n}}$",
            fontsize=18)
    ax.text(0.40, 0.20, r"sum:  $\ell = \sum_n \ell_n$", fontsize=18)
    ax.text(0.68, 0.30, "the mean divides by the\nsum of weights, not by N",
            fontsize=12, color=RED, linespacing=1.5)
    ax.text(0.68, 0.10, "no log inside: the last layer\nmust be nn.LogSoftmax",
            fontsize=12, color=RED, linespacing=1.5)
    save(fig, out, "nll_loss")


def fig_softmax_why(out: Path) -> None:
    """Raw network outputs are not probabilities; softmax makes them so."""
    logits = np.array([2.1, 1.2, 0.3, -0.8])
    p = softmax(logits)
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1, 0.55, 1], "wspace": 0.25})
    xs = np.arange(len(CLASSES))
    ax = axes[0]
    ax.bar(xs, logits, color=[RED if v < 0 or v > 1 else GREY for v in logits])
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(1, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(xs, CLASSES)
    ax.set_title("raw outputs (logits)")
    ax.text(1.5, -1.4, f"sum = {logits.sum():.1f}; some > 1, some < 0", ha="center", color=RED, fontsize=11)
    ax.set_ylim(-1.9, 2.6)

    ax = axes[1]
    blank(ax)
    arrow(ax, (0.05, 0.55), (0.95, 0.55), color=BLUE, lw=2.4)
    ax.text(0.5, 0.70, "softmax", ha="center", fontsize=14, color=BLUE)
    ax.text(0.5, 0.33, r"$P(i) = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$", ha="center",
            fontsize=16)

    ax = axes[2]
    ax.bar(xs, p, color=BLUE)
    ax.set_xticks(xs, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_title("probabilities")
    for x, v in zip(xs, p):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.text(1.5, 0.82, "all in (0, 1), sum = 1", ha="center", color=GREEN, fontsize=11.5)
    save(fig, out, "softmax_why")


def fig_ce_pipeline(out: Path) -> None:
    """CrossEntropyLoss is LogSoftmax followed by NLLLoss."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    y = 0.70
    box(ax, (0.10, y), 0.15, 0.18, "network\noutputs\n(logits)", ec=GREY, fs=12)
    box(ax, (0.38, y), 0.20, 0.18, "nn.LogSoftmax", ec=BLUE, fs=13)
    box(ax, (0.66, y), 0.20, 0.18, "nn.NLLLoss", ec=AMBER, fs=13)
    box(ax, (0.90, y), 0.12, 0.18, "loss", ec=INK, fs=13)
    arrow(ax, (0.18, y), (0.28, y))
    arrow(ax, (0.48, y), (0.56, y))
    arrow(ax, (0.76, y), (0.84, y))
    ax.add_patch(FancyBboxPatch((0.265, 0.53), 0.51, 0.34,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="none", edgecolor=RED, lw=2.0, ls="--"))
    ax.text(0.52, 0.46, "nn.CrossEntropyLoss  (takes the logits directly)", ha="center",
            color=RED, fontsize=13)
    ax.text(0.5, 0.24, r"$\ell_n = -\, w_{y_n} \log \dfrac{\exp(x_{n,\,y_n})}"
            r"{\sum_c \exp(x_{n,c})}$", ha="center", fontsize=19)
    ax.text(0.5, 0.04, "log: from the cross-entropy formula  ·  softmax: to get "
            "probabilities", ha="center", fontsize=12, color=GREY)
    save(fig, out, "ce_pipeline")


FIGURES = (
    fig_optimizer_role, fig_crowded_valley, fig_sgd_paths, fig_batch_noise,
    fig_momentum, fig_lr_effect, fig_lr_decay, fig_lr_schedules,
    fig_grad_clipping, fig_batchnorm, fig_batchnorm_batch, fig_dropout,
    fig_regression_losses, fig_cross_entropy, fig_nll_loss, fig_softmax_why,
    fig_ce_pipeline,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course06/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
