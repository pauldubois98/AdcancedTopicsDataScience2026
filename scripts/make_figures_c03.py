#!/usr/bin/env python3
"""Render the illustration figures used by the Course03 lecture slides.

Every figure is synthetic and seeded, so `make figures` reproduces the same output
anywhere without touching `data/`. Nothing here needs xgboost/lightgbm/optuna: the
boosting figures use sklearn's gradient boosting and the Bayesian-optimization and
Optuna figures simulate the search, which is what the slide is about anyway.

Usage: make_figures_c03.py [outdir]      (default: Course03/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#1b1b1b"
GREY = "#9aa0a6"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#00ab0e"
AMBER = "#b7791f"
PURPLE = "#6b46c1"
FAINT = "#e8eaed"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        # the .png is placed 9in wide on the slide, so 300 dpi of a 12.6in figure
        # is ~420 ppi there: sharp full-screen instead of soft on a projector
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
HALF = (10.8, 4.45)


# The .pptx content placeholder is 9.00in x 3.71in. Pandoc scales an image to fit
# inside it, so anything narrower than this aspect is shrunk and leaves the slide
# two-thirds empty. `savefig(bbox_inches="tight")` crops by a margin we cannot
# predict, so pad the saved file to the target aspect instead of guessing figsizes.
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
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=tc)


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


# --------------------------------------------------------------- true regression
def _sine_data(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise, n)
    return x, y


def _polyfit(x, y, d):
    return np.polynomial.Polynomial.fit(x, y, d)


# =============================================================== I. generalization
def fig_train_test(out: Path) -> None:
    """The classical U: training error falls forever, test error turns back up."""
    c = np.linspace(0.4, 10, 300)
    train = 0.9 * np.exp(-0.45 * c) + 0.04
    test = train + 0.030 * (c ** 2.3) / 10 + 0.06

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(c, train, color=BLUE, lw=2.6)
    ax.plot(c, test, color=RED, lw=2.6)
    # label the curves inline: a legend box lands on top of them here
    # curve labels far right of the minimum, so the "sweet spot" callout is clear
    ax.text(c[215], train[215] + 0.06, "training error", color=BLUE, fontsize=12.5,
            va="bottom", ha="center")
    ax.text(c[215], test[215] + 0.05, "test error", color=RED, fontsize=12.5,
            ha="center")

    k = int(np.argmin(test))
    ax.axvline(c[k], color=GREY, ls="--", lw=1.2)
    ax.plot([c[k]], [test[k]], "o", color=RED, ms=8, zorder=4)
    ax.annotate(
        "sweet spot",
        (c[k], test[k]),
        (c[k] - 1.7, test[k] + 0.30),
        color=INK,
        fontsize=11.5,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREY),
    )
    ax.annotate(
        "",
        (c[-6], train[-6]),
        (c[-6], test[-6]),
        arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0),
    )
    ax.text(c[-6] - 0.22, (train[-6] + test[-6]) / 2, "generalization\ngap",
            ha="right", va="center", color=AMBER, fontsize=11.5)

    ax.text(1.0, 0.86, "underfitting", color=GREY, fontsize=11, ha="center")
    ax.text(8.6, 0.86, "overfitting", color=GREY, fontsize=11, ha="center")
    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, out, "train_test")


def fig_polyfit(out: Path) -> None:
    """One dataset, three capacities: the picture everyone remembers."""
    x, y = _sine_data(18, 0.22, 0)
    grid = np.linspace(0, 1, 400)
    truth = np.sin(2 * np.pi * grid)

    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, d, title in zip(
        axes, (1, 3, 15), ("d = 1  underfit", "d = 3  good fit", "d = 15  overfit")
    ):
        ax.plot(grid, truth, color=GREY, lw=2, ls="--", label="truth")
        ax.plot(grid, _polyfit(x, y, d)(grid), color=RED, lw=2.6, label="fit")
        ax.scatter(x, y, s=34, color=BLUE, zorder=3, label="training data")
        ax.set_title(title, fontsize=12.5)
        ax.set_ylim(-2.1, 2.1)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    save(fig, out, "polyfit")


def fig_polyfit_curve(out: Path) -> None:
    """The same experiment, scored: train MSE collapses, test MSE explodes."""
    degrees = np.arange(1, 16)
    xtr, ytr = _sine_data(18, 0.22, 0)
    xte, yte = _sine_data(400, 0.22, 1)

    tr, te = [], []
    for d in degrees:
        p = _polyfit(xtr, ytr, d)
        tr.append(np.mean((ytr - p(xtr)) ** 2))
        te.append(np.mean((yte - p(xte)) ** 2))

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(degrees, tr, "o-", color=BLUE, lw=2.4, ms=5, label="training MSE")
    ax.plot(degrees, te, "o-", color=RED, lw=2.4, ms=5, label="test MSE")
    ax.axhline(0.22 ** 2, color=GREY, ls=":", lw=1.6)
    ax.text(15, 0.22 ** 2 * 1.35, r"irreducible noise $\sigma^2$",
            ha="right", color=GREY, fontsize=10.5)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("mean squared error (log)")
    ax.set_xticks(degrees[::2])
    ax.legend(frameon=False)
    save(fig, out, "polyfit_curve")


def fig_bv_targets(out: Path) -> None:
    """Dartboards: bias is where the cluster sits, variance is how wide it is."""
    rng = np.random.default_rng(3)
    cases = (
        ("low bias\nlow variance", (0, 0), 0.16),
        ("low bias\nhigh variance", (0, 0), 0.52),
        ("high bias\nlow variance", (0.62, 0.42), 0.16),
        ("high bias\nhigh variance", (0.62, 0.42), 0.52),
    )

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7))
    for ax, (title, centre, spread) in zip(axes, cases):
        for r, c in ((1.0, FAINT), (0.66, "#f3f4f6"), (0.33, "#ffffff")):
            ax.add_patch(Circle((0, 0), r, facecolor=c, edgecolor=GREY, lw=1.0))
        ax.plot([0], [0], "+", color=INK, ms=10, mew=1.6)
        pts = rng.normal(centre, spread, (26, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=26, color=RED, alpha=0.8,
                   edgecolors="none", zorder=3)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.02, "each dot = one model fitted on one resampled dataset;"
             "  + = the truth", ha="center", fontsize=11, color=GREY)
    save(fig, out, "bv_targets")


def fig_bv_fits(out: Path) -> None:
    """40 datasets, 40 fits, three capacities. Bias and variance, drawn."""
    grid = np.linspace(0, 1, 300)
    truth = np.sin(2 * np.pi * grid)

    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, d in zip(axes, (1, 3, 9)):
        preds = []
        for s in range(40):
            x, y = _sine_data(25, 0.22, 100 + s)
            preds.append(_polyfit(x, y, d)(grid))
            ax.plot(grid, preds[-1], color=RED, lw=0.7, alpha=0.25)
        mean = np.mean(preds, axis=0)
        ax.plot(grid, mean, color=RED, lw=2.8, label="average fit")
        ax.plot(grid, truth, color=GREY, lw=2.4, ls="--", label="truth")
        bias2 = np.mean((mean - truth) ** 2)
        var = np.mean(np.var(preds, axis=0))
        ax.set_title(f"d = {d}\nbias² = {bias2:.2f}    var = {var:.2f}", fontsize=12)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    save(fig, out, "bv_fits")


def fig_bv_decomposition(out: Path) -> None:
    """The decomposition, measured rather than asserted."""
    degrees = np.arange(1, 13)
    grid = np.linspace(0.02, 0.98, 200)
    truth = np.sin(2 * np.pi * grid)
    sigma = 0.22

    bias2, var = [], []
    for d in degrees:
        preds = np.array(
            [_polyfit(*_sine_data(25, sigma, 100 + s), d)(grid) for s in range(120)]
        )
        bias2.append(np.mean((preds.mean(axis=0) - truth) ** 2))
        var.append(np.mean(preds.var(axis=0)))
    bias2, var = np.array(bias2), np.array(var)
    total = bias2 + var + sigma ** 2

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(degrees, bias2, "o-", color=BLUE, lw=2.4, ms=5, label=r"bias$^2$")
    ax.plot(degrees, var, "o-", color=AMBER, lw=2.4, ms=5, label="variance")
    ax.axhline(sigma ** 2, color=GREY, ls=":", lw=1.8, label=r"noise $\sigma^2$")
    ax.plot(degrees, total, "o-", color=RED, lw=2.8, ms=5, label="total = test error")
    k = int(np.argmin(total))
    ax.axvline(degrees[k], color=GREY, ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 20)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("expected squared error (log)")
    ax.legend(frameon=False, ncol=2, fontsize=10.5)
    save(fig, out, "bv_decomposition")


def fig_double_descent(out: Path) -> None:
    """Classical U on the left of the threshold, second descent on the right."""
    c1 = np.linspace(0.05, 1.0, 200)
    u = 0.55 * np.exp(-3.4 * c1) + 0.20 + 0.62 * c1 ** 5
    peak = 0.20 + 0.62
    c2 = np.linspace(1.0, 3.4, 200)
    d2 = peak * np.exp(-2.3 * (c2 - 1)) + 0.155

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(c1, u, color=RED, lw=2.8)
    ax.plot(c2, d2, color=RED, lw=2.8)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.4)
    ax.text(1.06, 0.99, "interpolation threshold\n(parameters ≈ observations)",
            color=GREY, fontsize=10.5, va="top", ha="left")
    ax.axhspan(0, 1, xmin=0, xmax=0.295, color=FAINT, zorder=0)
    ax.text(0.5, 0.05, "classical regime", ha="center", color=GREY, fontsize=11)
    ax.text(2.2, 0.05, "modern / overparameterized regime",
            ha="center", color=GREY, fontsize=11)
    ax.set_xlabel("model capacity")
    ax.set_ylabel("test error")
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, out, "double_descent")


# Double descent, measured rather than asserted: ridgeless (minimum-norm) least
# squares on p random ReLU features, sweeping p through the number of training
# points n. The spike sits exactly at p = n, and the coefficient norm explains it.
DD_N, DD_D, DD_TEST, DD_SEEDS = 50, 30, 2000, 20


def _dd_sweep(ps, lam=0.0):
    """Return (mean test MSE, mean ||theta||) over seeds for each p."""
    err = np.zeros(len(ps))
    nrm = np.zeros(len(ps))
    for seed in range(DD_SEEDS):
        rng = np.random.default_rng(seed)
        beta = rng.normal(size=DD_D) / np.sqrt(DD_D)
        X = rng.normal(size=(DD_N, DD_D))
        y = X @ beta + rng.normal(0, 0.5, DD_N)
        Xt = rng.normal(size=(DD_TEST, DD_D))
        yt = Xt @ beta
        W = rng.normal(size=(ps.max(), DD_D)) / np.sqrt(DD_D)
        for i, p in enumerate(ps):
            F = np.maximum(X @ W[:p].T, 0)
            Ft = np.maximum(Xt @ W[:p].T, 0)
            if lam == 0:
                theta = np.linalg.pinv(F) @ y          # minimum-norm solution
            elif p <= DD_N:
                theta = np.linalg.solve(F.T @ F + lam * np.eye(p), F.T @ y)
            else:                                       # kernel form, cheaper
                theta = F.T @ np.linalg.solve(F @ F.T + lam * np.eye(DD_N), y)
            err[i] += np.mean((yt - Ft @ theta) ** 2)
            nrm[i] += np.linalg.norm(theta)
    return err / DD_SEEDS, nrm / DD_SEEDS


def _dd_grid():
    return np.unique(np.round(np.concatenate([
        np.arange(2, 46, 2), np.arange(46, 56), np.arange(56, 100, 4),
        np.geomspace(100, 400, 12)])).astype(int))


def _dd_phases(ax, top=0.965):
    """Badge the three phases of the sweep, identically on every double-descent axis."""
    for xfrac, num in ((0.24, "1"), (0.605, "2"), (0.86, "3")):
        ax.text(xfrac, top, num, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, color="white", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.28", fc=GREY, ec="none"))


def fig_double_descent_measured(out: Path) -> None:
    """The real thing: 50 training points, p random features, no regularization."""
    ps = _dd_grid()
    err, nrm = _dd_sweep(ps)
    r = ps / DD_N

    best_classical = err[r < 0.9].min()
    peak = err.max()
    cross = r[(r > 1.2) & (err < best_classical)].min()   # where descent 2 wins

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    ax0, ax1 = axes

    # ---------------------------------------------------------------- the symptom
    ax0.plot(r, err, color=RED, lw=2.8, zorder=4)
    ax0.axhline(best_classical, color=GREEN, ls=":", lw=1.8, zorder=2)
    ax0.scatter([cross], [err[r == cross]], s=70, color=GREEN, zorder=6)
    ax0.set_ylim(0.055, peak * 4)
    ax0.annotate(f"p/n \u2248 {cross:.1f}",
                 (cross, best_classical * 0.9), (8.5, 0.085),
                 fontsize=10.5, color=GREEN, ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax0.annotate(f"{peak / best_classical:.0f}\u00d7 worse than\neither side of it",
                 (1.0, peak), (0.19, peak * 0.9), fontsize=11.5, color=RED,
                 ha="center", va="top",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax0.set_ylabel("test MSE   (log scale)")
    ax0.set_title("the symptom: test error", fontsize=13, color=RED)

    # ------------------------------------------------------------------ the cause
    ax1.plot(r, nrm, color=PURPLE, lw=2.8, zorder=4)
    ax1.set_ylim(nrm.min() * 0.55, nrm.max() * 4)
    ax1.annotate(f"{nrm.max() / nrm.min():.0f}\u00d7 larger than the\n"
                 "smallest fit on the sweep",
                 (1.0, nrm.max()), (0.19, nrm.max() * 0.9), fontsize=11.5,
                 color=PURPLE, ha="center", va="top",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))
    ax1.annotate("more features to spread the fit over,\nso the smallest exact "
                 "fit keeps shrinking",
                 (5.5, nrm[np.argmin(np.abs(r - 5.5))] * 0.93), (9.0, 0.235),
                 fontsize=10.5, color=PURPLE, ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))
    ax1.set_ylabel(r"size of the fitted weights  $\|\theta\|$   (log scale)")
    ax1.set_title("the cause: how big the weights have to get", fontsize=13,
                  color=PURPLE)

    # ------------------------------------------------- shared framing on both axes
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvspan(r.min(), 1.0, color=FAINT, zorder=0)
        ax.axvline(1.0, color=GREY, ls="--", lw=1.4, zorder=1)
        ax.set_xlabel("parameters / training points   p / n     (log scale)")
        _dd_phases(ax)

    fig.text(0.5, -0.045,
             "\u2460  fewer parameters than data points        "
             "\u2461  the interpolation threshold: exactly one exact fit exists     "
             "\u2462  many exact fits: least squares takes the smallest",
             ha="center", fontsize=11, color=INK)
    fig.text(0.5, -0.115,
             "50 training points, p random ReLU features, least squares with no "
             "penalty at all \u2014 averaged over 20 draws of the data",
             ha="center", fontsize=10.5, color=GREY)
    save(fig, out, "double_descent_measured")


def fig_dd_ridge(out: Path) -> None:
    """The practical punchline: regularize, and the peak is simply not there."""
    ps = _dd_grid()
    r = ps / DD_N

    settings = ((0.0, RED, "-", 3.0, r"$\lambda = 0$  (no penalty)"),
                (1e-3, AMBER, (0, (5, 2)), 2.6, r"$\lambda = 10^{-3}$"),
                (1e-1, BLUE, (0, (1.5, 1.6)), 2.6, r"$\lambda = 10^{-1}$"),
                (1.0, GREEN, "-", 3.0, r"$\lambda = 1$  (tuned)"))
    curves = [(lam, colour, ls, lw, lab, _dd_sweep(ps, lam)[0])
              for lam, colour, ls, lw, lab in settings]

    fig, (ax, bar) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                  gridspec_kw=dict(width_ratios=(2.5, 1), wspace=0.28))

    for _, colour, ls, lw, lab, err in curves:
        ax.plot(r, err, color=colour, lw=lw, ls=ls, label=lab, zorder=4)
    ax.axvspan(r.min(), 1.0, color=FAINT, zorder=0)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.4, zorder=1)
    ax.text(1.0, 1.055, "interpolation threshold at p = n", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, color=GREY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters / training points   p / n     (log scale)")
    ax.set_ylabel("test MSE   (log scale)")
    ax.legend(frameon=False, fontsize=11, loc="upper left", handlelength=2.4,
              labelspacing=0.55, borderaxespad=0.9)

    peak = curves[0][-1].max()
    ax.set_ylim(0.17, peak * 5)
    
    # ------------------------------------------ the same four numbers, side by side
    heights = [c[-1].max() for c in curves]
    bar.bar(range(4), heights, color=[c[1] for c in curves], width=0.66)
    for i, h in enumerate(heights):
        bar.text(i, h * 1.18, f"{h:.0f}" if h >= 10 else f"{h:.1f}", ha="center",
                 fontsize=11, color=curves[i][1])
    bar.set_xlim(-0.7, 4.6)
    bar.set_yscale("log")
    bar.set_ylim(0.5, peak * 6)
    bar.set_xticks(range(4))
    bar.set_xticklabels([r"$0$", r"$10^{-3}$", r"$10^{-1}$", r"$1$"], fontsize=11)
    bar.set_xlabel(r"ridge penalty  $\lambda$")
    bar.set_ylabel("worst test MSE on the sweep   (log scale)")
    bar.set_title("how tall the spike gets", fontsize=12.5)

    fig.text(0.5, -0.055, "The spike only exists when nothing holds the weights down",
             ha="center", fontsize=11, color=GREY)
    save(fig, out, "dd_ridge")


def fig_interpolation(out: Path) -> None:
    """Two interpolants of the same points: the loss cannot tell them apart."""
    n = 12
    rng = np.random.default_rng(5)
    x = np.linspace(0.03, 0.97, n) + rng.uniform(-0.018, 0.018, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.18, n)
    grid = np.linspace(0, 1, 1500)
    truth = np.sin(2 * np.pi * grid)

    def interpolant(g):
        """Exact Gaussian-kernel interpolant of (x, y); g sets its smoothness."""
        K = np.exp(-g * (x[:, None] - x[None, :]) ** 2)
        a = np.linalg.solve(K + 1e-11 * np.eye(n), y)
        return np.exp(-g * (grid[:, None] - x[None, :]) ** 2) @ a

    smooth = interpolant(40.0)        # the minimum-norm / smooth solution
    spiky = interpolant(3000.0)       # memorizes each point, flat in between
    mse = lambda c: np.mean((c - truth) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    panels = (
        (axes[0], spiky, RED, "the fit that memorizes the points"),
        (axes[1], smooth, GREEN, "the fit gradient descent actually finds"),
    )
    for ax, curve, color, title in panels:
        # any blend of two interpolants is another interpolant: infinitely many exist
        for t in (0.2, 0.4, 0.6, 0.8):
            ax.plot(grid, (1 - t) * smooth + t * spiky, color=GREY, lw=0.9, alpha=0.35)
        ax.plot(grid, truth, color=GREY, lw=2, ls="--", label="true function (unknown)")
        ax.plot(grid, curve, color=color, lw=2.8, zorder=4, label="the fit")
        ax.scatter(x, y, s=44, color=BLUE, zorder=5, label="training data")
        ax.set_title(title, fontsize=12.5, color=color)
        ax.set_ylim(-1.75, 1.95)
        ax.set_xlabel("x")
        ax.text(0.5, 1.82,
                f"training MSE = 0.000        test MSE = {mse(curve):.2f}",
                ha="center", va="top", fontsize=11.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.4))

    axes[0].text(0.5, -1.62, "thin grey curves: other exact interpolants,\nequally perfect on the training set",
                 ha="center", va="bottom", fontsize=10.5, color=GREY)
    axes[1].legend(loc="lower center", frameon=False, fontsize=10.5, ncol=3,
                   handlelength=1.6, columnspacing=1.2)
    fig.text(0.5, -0.03,
             "identical training loss (zero); what picks the right curve is the optimizer's preference for small, smooth solutions",
             ha="center", fontsize=11.5, color=GREY)
    save(fig, out, "interpolation")


# ================================================================ II. regularization
def _corr_design(n, p, seed, rho=0.6):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    X = rho * base + np.sqrt(1 - rho ** 2) * rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:4] = (2.5, -2.0, 1.4, 0.9)
    y = X @ beta + rng.normal(0, 1.4, n)
    return X, y, beta


def fig_reg_path(out: Path) -> None:
    """Ridge shrinks everything; lasso switches coefficients off one by one."""
    from sklearn.linear_model import Lasso, Ridge

    X, y, beta = _corr_design(60, 10, 7)
    lams = np.logspace(-2, 3, 60)

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, model, title, ls in (
        (axes[0], Ridge, r"Ridge (L2): shrunk, never zero", lams),
        (axes[1], Lasso, r"Lasso (L1): exactly zero, one by one", np.logspace(-3, 0.5, 60)),
    ):
        coefs = np.array([model(alpha=a).fit(X, y).coef_ for a in ls])
        for j in range(coefs.shape[1]):
            colour = RED if beta[j] != 0 else GREY
            ax.plot(ls, coefs[:, j], color=colour, lw=2.2 if beta[j] else 1.1,
                    alpha=1.0 if beta[j] else 0.55)
        ax.set_xscale("log")
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel(r"$\lambda$  (log scale)")
    axes[0].set_ylabel("coefficient value")
    fig.text(0.5, -0.02, "red = truly nonzero coefficient    grey = truly zero",
             ha="center", fontsize=11.5, color=GREY)
    save(fig, out, "reg_path")


def fig_l1_l2_geometry(out: Path) -> None:
    """Why sparsity is geometry: the L1 ball has corners on the axes."""
    b = np.array([2.4, 1.5])
    A = np.array([[1.0, 0.72], [0.72, 1.0]])
    g1, g2 = np.meshgrid(np.linspace(-1.2, 3.6, 300), np.linspace(-1.6, 3.0, 300))
    d1, d2 = g1 - b[0], g2 - b[1]
    rss = A[0, 0] * d1 ** 2 + 2 * A[0, 1] * d1 * d2 + A[1, 1] * d2 ** 2

    def touch(norm, t):
        """The constrained optimum, found by brute force on a fine grid."""
        ok = (np.abs(g1) ** norm + np.abs(g2) ** norm) ** (1 / norm) <= t
        k = np.argmin(np.where(ok, rss, np.inf))
        return g1.flat[k], g2.flat[k]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)
    for ax, norm, t, title in (
        (axes[0], 2, 1.15, "Ridge: a smooth ball\n→ contact off the axes"),
        (axes[1], 1, 1.35, "Lasso: corners on the axes\n→ contact at a corner, $\\beta_2 = 0$"),
    ):
        ax.contour(g1, g2, rss, levels=np.linspace(0.3, 14, 9), colors=GREY,
                   linewidths=1.0)
        th = np.linspace(0, 2 * np.pi, 600)
        if norm == 2:
            bx, by = t * np.cos(th), t * np.sin(th)
        else:
            bx = t * np.sign(np.cos(th)) * np.abs(np.cos(th))
            by = t * np.sign(np.sin(th)) * np.abs(np.sin(th))
            bx, by = t * np.array([1, 0, -1, 0, 1]), t * np.array([0, 1, 0, -1, 0])
        ax.fill(bx, by, color=BLUE, alpha=0.16)
        ax.plot(bx, by, color=BLUE, lw=2.2)

        px, py = touch(norm, t)
        ax.plot([px], [py], "o", color=RED, ms=10, zorder=5)
        ax.plot([b[0]], [b[1]], "+", color=INK, ms=12, mew=2)
        # contours cover the whole panel, so give the label an opaque backing
        ax.text(b[0] + 0.12, b[1] + 0.16, r"$\hat\beta^{\,\mathrm{OLS}}$", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.axhline(0, color=INK, lw=0.9)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel(r"$\beta_1$")
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 3.4)
        ax.set_ylim(-1.6, 2.8)
    axes[0].set_ylabel(r"$\beta_2$")
    save(fig, out, "l1_l2_geometry")


def fig_lambda_curve(out: Path) -> None:
    """The validation curve: the bias-variance dial, with lambda as the knob."""
    from sklearn.linear_model import Ridge

    X, y, _ = _corr_design(45, 30, 11)
    Xte, yte, _ = _corr_design(2000, 30, 12)
    lams = np.logspace(-3, 4, 60)
    tr = [np.mean((y - Ridge(alpha=a).fit(X, y).predict(X)) ** 2) for a in lams]
    te = [np.mean((yte - Ridge(alpha=a).fit(X, y).predict(Xte)) ** 2) for a in lams]

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(lams, tr, color=BLUE, lw=2.6, label="training MSE")
    ax.plot(lams, te, color=RED, lw=2.6, label="test MSE")
    k = int(np.argmin(te))
    ax.plot([lams[k]], [te[k]], "o", color=RED, ms=9, zorder=4)
    ax.axvline(lams[k], color=GREY, ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$  (log scale)")
    ax.set_ylabel("mean squared error")
    # headroom above the curves so neither regime label lands on one
    ax.set_ylim(top=max(max(te), max(tr)) * 1.32)
    top = ax.get_ylim()[1]
    ax.text(lams[2], top * 0.93, "high variance\nlow bias", fontsize=11, color=GREY,
            va="top")
    ax.text(lams[-3], top * 0.93, "low variance\nhigh bias", fontsize=11,
            color=GREY, ha="right", va="top")
    ax.legend(frameon=False, loc="lower left")
    save(fig, out, "lambda_curve")


def fig_tree_depth(out: Path) -> None:
    """The same dial on a tree: depth is a regularization knob."""
    from sklearn.datasets import make_moons
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_moons(n_samples=180, noise=0.32, random_state=0)
    gx, gy = np.meshgrid(np.linspace(-2, 3, 400), np.linspace(-1.8, 2.3, 400))
    grid = np.c_[gx.ravel(), gy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), sharey=True)
    for ax, depth, title in zip(
        axes, (1, 4, None), ("max_depth = 1\nunderfit", "max_depth = 4\nabout right",
                             "max_depth = None\noverfit")
    ):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
        z = clf.predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, RED], alpha=0.16)
        ax.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.4)
        for cls, colour in ((0, BLUE), (1, RED)):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=22, color=colour, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, out, "tree_depth")


def fig_early_stopping(out: Path) -> None:
    """Regularization without a penalty term: just stop."""
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = make_friedman1(n_samples=200, noise=3.0, random_state=0)
    Xte, yte = make_friedman1(n_samples=2000, noise=3.0, random_state=1)
    gb = GradientBoostingRegressor(n_estimators=600, learning_rate=0.1,
                                   max_depth=5, random_state=0).fit(X, y)
    tr = np.array([np.mean((y - p) ** 2) for p in gb.staged_predict(X)])
    te = np.array([np.mean((yte - p) ** 2) for p in gb.staged_predict(Xte)])
    m = np.arange(1, len(te) + 1)

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(m, tr, color=BLUE, lw=2.4, label="training MSE")
    ax.plot(m, te, color=RED, lw=2.4, label="test MSE")
    k = int(np.argmin(te))
    ax.axvline(m[k], color=GREEN, ls="--", lw=2)
    ax.plot([m[k]], [te[k]], "o", color=GREEN, ms=9, zorder=4)
    ax.text(m[k] + 18, te[k] * 1.9, f"stop here\n({m[k]} trees)", color=GREEN, fontsize=11)
    ax.set_yscale("log")
    ax.set_xlabel("boosting rounds")
    ax.set_ylabel("mean squared error (log)")
    ax.legend(frameon=False)
    save(fig, out, "early_stopping")


def fig_reg_zoo(out: Path) -> None:
    """One idea, many surfaces: every model has a bias-variance dial."""
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    blank(ax)
    box(ax, (0.5, 0.90), 0.44, 0.15, "REGULARIZATION\nconstrain the fit: trade bias for variance",
        ec=INK, fs=12.5)
    cols = (
        ("Linear models", (r"L2 (ridge)", "L1 (lasso)", "elastic net"), BLUE),
        ("Trees", ("max_depth", "min_samples_leaf", "pruning (ccp_alpha)"), GREEN),
        ("Ensembles", ("learning rate η", "subsample", "colsample"), AMBER),
        ("Neural nets", ("weight decay", "dropout", "early stopping"), PURPLE),
    )
    for i, (name, items, colour) in enumerate(cols):
        cx = 0.125 + i * 0.25
        arrow(ax, (0.5, 0.82), (cx, 0.68), color=GREY, lw=1.4)
        box(ax, (cx, 0.60), 0.21, 0.13, name, ec=colour, fs=12)
        for j, it in enumerate(items):
            ax.text(cx, 0.44 - j * 0.115, it, ha="center", fontsize=11.5, color=INK)
    save(fig, out, "reg_zoo")


# ============================================================ III/IV. the landscape
def _val_surface(le, dep):
    """A plausible validation-AUC surface over (log learning rate, depth)."""
    return (
        0.86
        - 0.055 * (le + 1.5) ** 2
        - 0.0022 * (dep - 6.0) ** 2
        - 0.010 * (le + 1.5) * (dep - 6.0)
    )


def fig_landscape_1d(out: Path) -> None:
    """Smooth on a log scale, flat near the optimum, noisy when measured."""
    rng = np.random.default_rng(2)
    lam = np.logspace(-5, 2, 400)
    loss = 0.30 + 0.052 * (np.log10(lam) + 1.6) ** 2

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(lam, loss, color=BLUE, lw=2.6, label="true validation loss")
    probe = np.logspace(-5, 2, 22)
    ax.plot(probe, 0.30 + 0.052 * (np.log10(probe) + 1.6) ** 2
            + rng.normal(0, 0.012, len(probe)), "o", color=RED, ms=6,
            label="validation measured (5-fold CV)")
    k = int(np.argmin(loss))
    ax.axvline(lam[k], color=GREY, ls="--", lw=1.2)
    ax.axhspan(loss[k], loss[k] + 0.02, color=GREEN, alpha=0.16)
    ax.set_ylim(bottom=0.14)      # room under the band for the caption
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$  (log scale)")
    ax.set_ylabel("validation loss")
    ax.legend(frameon=False, loc="upper center", fontsize=10.5)
    save(fig, out, "landscape_1d")


def fig_landscape_2d(out: Path) -> None:
    """Good regions are ridges, not points — so one knob at a time is unreliable."""
    le = np.linspace(-4, 0, 300)
    dep = np.linspace(2, 14, 300)
    LE, DEP = np.meshgrid(le, dep)
    auc = _val_surface(LE, DEP)

    fig, ax = plt.subplots(figsize=HALF)
    im = ax.contourf(10 ** LE, DEP, auc, levels=22, cmap="viridis")
    ax.contour(10 ** LE, DEP, auc, levels=10, colors="white", linewidths=0.6, alpha=0.5)
    j, i = np.unravel_index(np.argmax(auc), auc.shape)
    ax.plot([10 ** LE[j, i]], [DEP[j, i]], "*", color="white", ms=20, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate η  (log scale)")
    ax.set_ylabel("max_depth")
    fig.colorbar(im, ax=ax, label="validation AUC")
    ax.set_title("the good region is a diagonal ridge: η and depth interact",
                 fontsize=12.5)
    save(fig, out, "landscape_2d")


def fig_log_scale(out: Path) -> None:
    """Sample a learning rate linearly and you never see a small one."""
    rng = np.random.default_rng(4)
    n = 40
    lin = rng.uniform(1e-4, 1.0, n)
    log = 10 ** rng.uniform(-4, 0, n)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 3.9), sharex=True)
    for ax, s, name, colour in ((axes[0], lin, "uniform on [1e-4, 1]", RED),
                                (axes[1], log, "log-uniform on [1e-4, 1]", GREEN)):
        ax.plot(s, np.zeros_like(s), "|", color=colour, ms=26, mew=2)
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.text(1.2e-4, 0.45, name, fontsize=12, color=colour)
        ax.text(1.2e-4, -0.75,
                f"{np.sum(s < 0.01)} of {n} draws below 0.01", fontsize=10.5, color=GREY)
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_xlabel("learning rate")
    save(fig, out, "log_scale")


def _gp(xs, ys, grid, ell=0.15, sig=0.09, amp=0.55):
    k = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * ell ** 2))
    K = k(xs, xs) + sig ** 2 * np.eye(len(xs))
    a = np.linalg.solve(K, ys)
    mu = k(grid, xs) @ a
    v = 1.0 - np.einsum("ij,ij->i", k(grid, xs), np.linalg.solve(K, k(grid, xs).T).T)
    return mu, amp * np.sqrt(np.clip(v, 1e-6, None))


def _ei(mu, sd, best):
    from scipy.stats import norm
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def _bo_target(x):
    return np.sin(3 * np.pi * x) * 0.45 + (x - 0.55) ** 2 * 3.2


def fig_bo_surrogate(out: Path) -> None:
    """What you actually measured, and what the surrogate makes of it."""
    grid = np.linspace(0, 1, 400)
    xs = np.array([0.08, 0.42, 0.88])
    ys = _bo_target(xs)
    mu, sd = _gp(xs, ys, grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax in axes:
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-1.5, 2.3)
        ax.plot(xs, ys, "o", color=INK, ms=8, zorder=5)

    axes[0].set_title("what you have: 3 numbers, 3 full fits", fontsize=12.5)
    axes[0].set_ylabel("validation loss")
    for x, y in zip(xs, ys):
        axes[0].plot([x, x], [-1.5, y], color=GREY, lw=1, ls=":")
    axes[0].text(0.5, -1.15, "the curve between them is unknown,\nand every extra point costs a full CV run",
                 ha="center", fontsize=11, color=GREY, linespacing=1.3)

    axes[1].set_title("what the surrogate believes", fontsize=12.5)
    axes[1].plot(grid, mu, color=BLUE, lw=2.4, label="mean: the best guess")
    axes[1].fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16,
                         label="uncertainty: how wrong that guess could be")
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper left",
                   bbox_to_anchor=(0, 1.0))
    axes[1].annotate("no uncertainty where\nyou have measured", (xs[1], ys[1]),
                     (0.30, -1.15), fontsize=10.5, color=AMBER, ha="center",
                     arrowprops=dict(arrowstyle="->", color=AMBER))
    axes[1].annotate("wide where you have not", (0.70, (mu - 2 * sd)[280]),
                     (0.76, -1.15), fontsize=10.5, color=PURPLE, ha="center",
                     arrowprops=dict(arrowstyle="->", color=PURPLE))
    save(fig, out, "bo_surrogate")


def fig_bo_uncertainty(out: Path) -> None:
    """Every evaluation buys a narrower band — that is all the surrogate learns."""
    grid = np.linspace(0, 1, 400)
    truth = _bo_target(grid)
    # a fixed order that keeps every panel spread over the whole range
    pool = np.array([0.12, 0.86, 0.45, 0.66, 0.28, 0.04, 0.55, 0.95,
                     0.36, 0.75, 0.20, 0.60])
    counts = (2, 5, 12)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharey=True)
    for ax, n in zip(axes, counts):
        xs = pool[:n]
        ys = _bo_target(xs)
        mu, sd = _gp(xs, ys, grid)
        ax.plot(grid, truth, color=GREY, lw=1.6, ls="--", label="true (unknown) $f$")
        ax.plot(grid, mu, color=BLUE, lw=2.2, label="surrogate mean")
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16,
                        label="uncertainty")
        ax.plot(xs, ys, "o", color=INK, ms=7, zorder=5, label="evaluated")
        ax.set_title(f"{n} evaluations", fontsize=12.5)
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-1.6, 2.4)
        ax.text(0.5, 2.15, f"average band width  {(4 * sd).mean():.2f}",
                ha="center", fontsize=11, color=GREY)
    axes[0].set_ylabel("validation loss")
    axes[0].legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
                   bbox_to_anchor=(0, 1.12))
    save(fig, out, "bo_uncertainty")


def _gp_draws(grid, n, rng, xs=None, ys=None, ell=0.15, sig=0.09, amp=0.55):
    """Sample curves from the prior (no xs) or from the posterior given (xs, ys)."""
    k = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * ell ** 2))
    if xs is None:
        mu = np.zeros_like(grid)
        cov = k(grid, grid)
    else:
        K = k(xs, xs) + sig ** 2 * np.eye(len(xs))
        Kx = k(grid, xs)
        mu = Kx @ np.linalg.solve(K, ys)
        cov = k(grid, grid) - Kx @ np.linalg.solve(K, Kx.T)
    L = np.linalg.cholesky(amp ** 2 * cov + 1e-9 * np.eye(len(grid)))
    return mu, (mu[:, None] + L @ rng.standard_normal((len(grid), n))).T


def fig_bo_prior_posterior(out: Path) -> None:
    """A surrogate is a distribution over curves, narrowed by the data."""
    grid = np.linspace(0, 1, 300)
    xs = np.array([0.08, 0.42, 0.88])
    ys = _bo_target(xs)
    rng = np.random.default_rng(11)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    panels = ((None, None, "before any data: the prior",
               "every curve is a guess the surrogate\nconsiders possible"),
              (xs, ys, "after 3 evaluations: the posterior",
               "only the curves that pass through\nthe measurements survive"))
    for ax, (px, py, title, caption) in zip(axes, panels):
        mu, draws = _gp_draws(grid, 6, np.random.default_rng(11), px, py)
        for d in draws:
            ax.plot(grid, d, color=BLUE, lw=1.1, alpha=0.55)
        ax.plot(grid, mu, color=BLUE, lw=2.6, label="mean of the draws")
        ax.fill_between(grid, draws.min(0), draws.max(0), color=BLUE, alpha=0.10)
        if px is not None:
            ax.plot(px, py, "o", color=INK, ms=8, zorder=5, label="evaluated")
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-2.2, 2.6)
        ax.text(0.5, -2.05, caption, ha="center", fontsize=10.5, color=GREY,
                linespacing=1.3)
    axes[0].set_ylabel("validation loss")
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper center", ncol=2)
    save(fig, out, "bo_prior_posterior")


def fig_bo_kernel(out: Path) -> None:
    """The kernel is the whole modelling choice: what counts as 'nearby'."""
    grid = np.linspace(0, 1, 400)
    xs = np.array([0.08, 0.30, 0.42, 0.88])
    ys = _bo_target(xs)
    lengths = (0.03, 0.15, 0.60)
    verdicts = ("too short: no generalization\nbetween trials",
                "about right: smooth, and honest\nwhere nothing was measured",
                "too long: the structure is\nsmoothed away")

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1], "wspace": 0.22})
    d = np.linspace(0, 1, 300)
    for ell, colour in zip(lengths, (AMBER, BLUE, PURPLE)):
        axes[0].plot(d, np.exp(-d ** 2 / (2 * ell ** 2)), color=colour, lw=2.2,
                     label=rf"$\ell={ell}$")
    axes[0].set_title("the kernel", fontsize=12.5)
    axes[0].set_xlabel("distance $|x-x'|$")
    axes[0].set_ylabel("similarity $k(x,x')$")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].text(0.5, 1.18, r"$k(x,x')=\exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)$",
                 transform=axes[0].transAxes, ha="center", fontsize=12, color=INK)

    for ax, ell, verdict, colour in zip(axes[1:], lengths, verdicts,
                                        (AMBER, BLUE, PURPLE)):
        mu, sd = _gp(xs, ys, grid, ell=ell)
        ax.plot(grid, _bo_target(grid), color=GREY, lw=1.5, ls="--")
        ax.plot(grid, mu, color=colour, lw=2.2)
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=colour, alpha=0.16)
        ax.plot(xs, ys, "o", color=INK, ms=6.5, zorder=5)
        ax.set_title(rf"$\ell={ell}$", fontsize=12.5, color=colour)
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-2.3, 2.5)
        ax.set_yticks([])
        ax.text(0.5, -2.15, verdict, ha="center", fontsize=10, color=GREY,
                linespacing=1.3)
    save(fig, out, "bo_kernel")


def _k(a, b, ell=0.15):
    return np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * ell ** 2))


def fig_bo_kernel_basis(out: Path) -> None:
    """The fitted mean is one kernel bump per trial, added up."""
    grid = np.linspace(0, 1, 400)
    xs = np.array([0.08, 0.30, 0.42, 0.62, 0.88])
    ys = _bo_target(xs)
    alpha = np.linalg.solve(_k(xs, xs) + 0.09 ** 2 * np.eye(len(xs)), ys)
    bumps = alpha[:, None] * _k(xs, grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    palette = (BLUE, AMBER, PURPLE, GREEN, RED)
    for ax in axes:
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-1.6, 2.4)

    for b, x, colour in zip(bumps, xs, palette):
        axes[0].plot(grid, b, color=colour, lw=1.8)
        axes[0].plot([x], [0], "v", color=colour, ms=8, zorder=6)
    axes[0].set_ylabel("contribution")
    axes[0].set_title("one bump per trial, centred on it", fontsize=12.5)
    axes[0].text(0.5, -1.45, r"height $\alpha_i$ fitted so the sum hits every measurement",
                 ha="center", fontsize=10.5, color=GREY)

    for b, colour in zip(bumps, palette):
        axes[1].plot(grid, b, color=colour, lw=1.2, alpha=0.45)
    axes[1].plot(grid, bumps.sum(0), color=INK, lw=2.8,
                 label=r"$\mu(x)=\sum_i\alpha_i k(x,x_i)$")
    axes[1].plot(xs, ys, "o", color=INK, ms=8, zorder=5, label="evaluated")
    axes[1].set_title("their sum is the surrogate mean", fontsize=12.5)
    axes[1].legend(frameon=False, fontsize=11, loc="lower center")
    save(fig, out, "bo_kernel_basis")


def fig_bo_noise(out: Path) -> None:
    """The second knob: how much of the data the model is allowed to disbelieve."""
    grid = np.linspace(0, 1, 400)
    rng = np.random.default_rng(2)
    # pairs of nearby trials: the only way to see a model chase measurement noise
    xs = np.array([0.05, 0.11, 0.24, 0.29, 0.42, 0.47, 0.61, 0.67, 0.82, 0.88])
    ys = _bo_target(xs) + rng.normal(0, 0.18, len(xs))
    settings = ((0.01, "trusts every score exactly:\nchases the CV noise"),
                (0.12, "matches the real noise:\nsmooth, and honest between trials"),
                (0.80, "distrusts the data:\nignores structure, band stays wide"))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
    for ax, (sig, caption) in zip(axes, settings):
        mu, sd = _gp(xs, ys, grid, sig=sig)
        ax.plot(grid, _bo_target(grid), color=GREY, lw=1.5, ls="--")
        ax.plot(grid, mu, color=BLUE, lw=2.2)
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16)
        ax.plot(xs, ys, "o", color=INK, ms=6.5, zorder=5)
        ax.set_title(rf"$\sigma={sig}$", fontsize=12.5)
        ax.set_xlabel("hyperparameter $x$")
        ax.set_ylim(-1.9, 2.5)
        ax.text(0.5, -1.75, caption, ha="center", fontsize=10, color=GREY,
                linespacing=1.3)
    axes[0].set_ylabel("validation loss")
    save(fig, out, "bo_noise")


def fig_bayesopt(out: Path) -> None:
    """Surrogate + acquisition, three iterations: the whole algorithm in one image."""
    grid = np.linspace(0, 1, 400)
    truth = _bo_target(grid)
    xs = np.array([0.08, 0.42, 0.88])

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    for col in range(3):
        ys = _bo_target(xs)
        mu, sd = _gp(xs, ys, grid)
        top, bot = axes[0, col], axes[1, col]

        top.plot(grid, truth, color=GREY, lw=1.8, ls="--", label="true (unknown) f")
        top.plot(grid, mu, color=BLUE, lw=2.4, label="surrogate mean")
        top.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16,
                         label="uncertainty")
        top.plot(xs, ys, "o", color=INK, ms=7, zorder=5, label="evaluated")
        top.set_title(f"iteration {col + 1}", fontsize=12.5)
        top.set_ylim(-1.5, 2.3)

        ei = _ei(mu, sd, ys.min())
        nxt = grid[int(np.argmax(ei))]
        bot.fill_between(grid, 0, ei, color=GREEN, alpha=0.35)
        bot.plot(grid, ei, color=GREEN, lw=2)
        bot.axvline(nxt, color=RED, lw=2)
        top.axvline(nxt, color=RED, lw=1.6, ls=":")
        bot.set_yticks([])
        bot.set_xlabel("hyperparameter x")
        if col == 0:
            top.set_ylabel("validation loss")
            bot.set_ylabel("expected\nimprovement", fontsize=10.5)
            top.legend(frameon=False, fontsize=9, ncol=2,
                       loc="lower left", bbox_to_anchor=(0, 1.14))
        xs = np.append(xs, nxt)
    save(fig, out, "bayesopt")


def fig_acquisition(out: Path) -> None:
    """Exploit where the mean is good, explore where the band is wide."""
    grid = np.linspace(0, 1, 400)
    xs = np.array([0.08, 0.30, 0.42, 0.86])
    ys = _bo_target(xs)
    mu, sd = _gp(xs, ys, grid)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.15]})
    axes[0].plot(grid, mu, color=BLUE, lw=2.4)
    axes[0].fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.16)
    axes[0].plot(xs, ys, "o", color=INK, ms=7, zorder=5)
    axes[0].axhline(ys.min(), color=GREY, ls=":", lw=1.4)
    axes[0].text(0.005, ys.min() + 0.06, r"best so far $f^\star$", fontsize=10.5, color=GREY)
    axes[0].set_ylabel("validation loss")
    axes[0].annotate("exploit\n(mean is low)", (0.42, -0.34), (0.20, -0.88),
                     fontsize=10.5, color=AMBER, ha="center",
                     arrowprops=dict(arrowstyle="->", color=AMBER))
    axes[0].annotate("explore\n(band is wide)", (0.64, 0.55), (0.62, 1.35),
                     fontsize=10.5, color=PURPLE,
                     arrowprops=dict(arrowstyle="->", color=PURPLE))

    ei = _ei(mu, sd, ys.min())
    ucb = -(mu - 2.0 * sd)
    for s, lab, colour in ((ei / ei.max(), "Expected Improvement", GREEN),
                           ((ucb - ucb.min()) / np.ptp(ucb), r"UCB  ($\kappa=2$)", PURPLE)):
        axes[1].plot(grid, s, color=colour, lw=2.4, label=lab)
        axes[1].axvline(grid[int(np.argmax(s))], color=colour, ls="--", lw=1.4)
    axes[1].set_yticks([])
    axes[1].set_xlabel("hyperparameter x")
    axes[1].set_ylabel("acquisition\n(rescaled)", fontsize=10.5)
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper left")
    save(fig, out, "acquisition")


def fig_search_compare(out: Path) -> None:
    """Where the three strategies actually put their trials, and what they find."""
    rng = np.random.default_rng(0)
    le = np.linspace(-4, 0, 200)
    dep = np.linspace(2, 14, 200)
    LE, DEP = np.meshgrid(le, dep)
    auc = _val_surface(LE, DEP)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
    picks = {}
    g = np.meshgrid(np.linspace(-3.5, -0.5, 5), np.linspace(3, 13, 5))
    picks["grid search"] = (g[0].ravel(), g[1].ravel())
    picks["random search"] = (rng.uniform(-4, 0, 25), rng.uniform(2, 14, 25))

    # a greedy Bayesian-ish search: 6 random starts, then exploit the best region
    bx, by = list(rng.uniform(-4, 0, 6)), list(rng.uniform(2, 14, 6))
    for _ in range(19):
        k = int(np.argmax([_val_surface(a, b) for a, b in zip(bx, by)]))
        bx.append(np.clip(bx[k] + rng.normal(0, 0.45), -4, 0))
        by.append(np.clip(by[k] + rng.normal(0, 1.5), 2, 14))
    picks["Bayesian optimization"] = (np.array(bx), np.array(by))

    for ax, (name, (px, py)) in zip(axes, picks.items()):
        ax.contourf(LE, DEP, auc, levels=18, cmap="viridis", alpha=0.9)
        ax.scatter(px, py, s=34, color="white", edgecolors=INK, lw=0.8, zorder=4)
        best = max(_val_surface(a, b) for a, b in zip(px, py))
        ax.set_title(f"{name}\n25 trials — best AUC {best:.4f}", fontsize=12)
        ax.set_xlabel(r"$\log_{10}\eta$")
    axes[0].set_ylabel("max_depth")
    save(fig, out, "search_compare")


def fig_optuna_history(out: Path) -> None:
    """What `plot_optimization_history` shows, and how to read it."""
    rng = np.random.default_rng(6)
    n = 120
    trials = np.arange(1, n + 1)
    vals = 0.735 + 0.055 * (1 - np.exp(-trials / 22)) + rng.normal(0, 0.014, n)
    vals[:10] = 0.70 + rng.normal(0, 0.03, 10)
    best = np.maximum.accumulate(vals)

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(trials, vals, "o", color=GREY, ms=5, label="trial")
    ax.plot(trials, best, color=RED, lw=2.6, label="best so far")
    ax.axvspan(0, 10, color=FAINT)
    ax.text(5, 0.665, "random\nstart-up", ha="center", fontsize=10.5, color=GREY)
    ax.axvspan(70, n, color=GREEN, alpha=0.10)
    ax.text(90, 0.665, "flat: more budget is not\nbuying anything", ha="center",
            fontsize=10.5, color=GREEN)
    ax.set_xlabel("trial")
    ax.set_ylabel("validation AUC")
    ax.set_ylim(0.64, 0.82)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "optuna_history")


def fig_pruning(out: Path) -> None:
    """Kill hopeless trials early: the same budget buys several times more trials."""
    rng = np.random.default_rng(8)
    steps = np.arange(1, 101)

    fig, ax = plt.subplots(figsize=HALF)
    kept = 0
    for i in range(24):
        ceiling = rng.uniform(0.70, 0.81)
        curve = ceiling * (1 - np.exp(-steps / rng.uniform(8, 26))) + rng.normal(0, 0.004, 100)
        good = ceiling > 0.775
        if good:
            ax.plot(steps, curve, color=GREEN, lw=1.9, alpha=0.9)
            kept += 1
        else:
            stop = int(rng.integers(14, 34))
            ax.plot(steps[:stop], curve[:stop], color=GREY, lw=1.2, alpha=0.7)
            ax.plot([steps[stop - 1]], [curve[stop - 1]], "x", color=RED, ms=7, mew=1.8)
    ax.set_xlabel("boosting rounds within a trial")
    ax.set_ylabel("validation AUC")
    ax.set_title(f"× = pruned.  {24 - kept} of 24 trials stopped early; "
                 "the budget goes to the survivors", fontsize=12)
    save(fig, out, "pruning")


def fig_nested_cv(out: Path) -> None:
    """Outer loop measures, inner loop selects."""
    fig, ax = plt.subplots(figsize=(11.6, 4.2))
    blank(ax)
    ax.text(0.06, 0.95, "outer loop: estimates performance", fontsize=12, color=INK)
    for k in range(4):
        y = 0.80 - k * 0.10
        for j in range(4):
            x = 0.06 + j * 0.115
            test = j == k
            ax.add_patch(Rectangle((x, y - 0.035), 0.105, 0.07,
                                   facecolor=RED if test else FAINT,
                                   edgecolor=GREY, lw=1.0))
        ax.text(0.06 + 4 * 0.115 + 0.02, y, "→  score fold "
                f"{k + 1}", fontsize=10.5, color=INK, va="center")
    ax.text(0.06, 0.36, "inner loop: selects hyper-parameters",
            fontsize=12, color=INK)
    for j in range(3):
        x = 0.06 + j * 0.115
        for i in range(3):
            ax.add_patch(Rectangle((x, 0.24 - i * 0.06), 0.105, 0.05,
                                   facecolor=AMBER if i == j else "white",
                                   edgecolor=GREY, lw=1.0, alpha=0.75))
    ax.text(0.42, 0.16, "the search is part of the model,\nso it belongs inside the outer split",
            fontsize=11.5, color=GREY)
    ax.add_patch(Rectangle((0.60, 0.86), 0.03, 0.05, facecolor=RED, edgecolor=GREY))
    ax.text(0.645, 0.885, "held out", fontsize=10.5, va="center")
    ax.add_patch(Rectangle((0.76, 0.86), 0.03, 0.05, facecolor=FAINT, edgecolor=GREY))
    ax.text(0.805, 0.885, "used for fitting + search", fontsize=10.5, va="center")
    save(fig, out, "nested_cv")


# ================================================== V-VII. ensembles and boosting
def fig_ensemble_avg_demo(out: Path) -> None:
    """Averaging, made concrete: many wrong curves, one good average."""
    rng = np.random.default_rng(3)
    xs = np.linspace(0, 1, 300)
    truth = np.sin(2 * np.pi * xs)
    M = 15
    x0 = 0.34

    def wiggle():
        a = rng.normal(0, 0.30, 6)
        b = rng.normal(0, 0.30, 6)
        return sum(a[k] * np.sin((k + 1) * 3.1 * xs) + b[k] * np.cos((k + 1) * 2.7 * xs)
                   for k in range(6))

    curves = np.array([truth + wiggle() for _ in range(M)])
    avg = curves.mean(axis=0)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.7),
                                 gridspec_kw={"width_ratios": [2.2, 1.0]})

    for c in curves:
        ax.plot(xs, c, color=GREY, lw=1.0, alpha=0.7)
    ax.plot(xs, truth, color=INK, lw=2.6, ls="--", label="truth")
    ax.plot(xs, avg, color=GREEN, lw=3.2, label=f"average of the {M}")
    ax.axvline(x0, color=BLUE, lw=1.0, ls=":")
    ax.text(x0, -2.35, "$x_0$", color=BLUE, ha="center", fontsize=12)
    ax.set_title(f"{M} unbiased, high-variance models (grey)", fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-2.5, 2.5)
    ax.legend(frameon=False, loc="upper right", fontsize=11.5)

    # the same story at one input, as a dot strip: the spread is what collapses
    i0 = int(x0 * (len(xs) - 1))
    preds = curves[:, i0]
    t0 = truth[i0]
    jitter = rng.uniform(-0.05, 0.05, M)
    bx.axhline(t0, color=INK, lw=2.2, ls="--")
    bx.text(1.85, t0 + 0.16, "truth at $x_0$", ha="right", fontsize=11.5, color=INK)
    bx.scatter(0.55 + jitter, preds, s=46, color=GREY, zorder=3)
    bx.text(0.55, -1.75, "each dot is a model", ha="center", fontsize=11.5,
            color=GREY)
    bx.scatter([1.45], [preds.mean()], s=170, color=GREEN, marker="D", zorder=4)
    bx.text(1.45, preds.mean() - 0.75, "average", ha="center", fontsize=11.5,
            color=GREEN)
    # the spread of the individual models, as a bracket on the left
    bx.annotate("", xy=(0.12, preds.min()), xytext=(0.12, preds.max()),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    bx.text(0.02, t0 + 1, "spread\n$\\sigma$", fontsize=12, color=BLUE,
            va="center", ha="center")
    bx.set_title("at that one input", fontsize=12.5)
    bx.set_xlim(-0.25, 1.9)
    bx.set_ylim(-2.5, 2.5)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.text(0.85, -2.3, r"spread of the average $\approx \sigma/\sqrt{M}$",
            ha="center", fontsize=12, color=GREEN)
    for sp in list(ax.spines.values()) + list(bx.spines.values()):
        sp.set_visible(False)
    save(fig, out, "ensemble_avg_demo")


def fig_ensemble_correlation(out: Path) -> None:
    """Only mistakes that differ cancel: the same M, three values of rho."""
    rng = np.random.default_rng(11)
    M = 15
    # one shared bias and one private error per model, reused across the panels so
    # the three columns differ only by how much of the error is shared
    shared = 1.15
    private = rng.normal(0, 1.15, M)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=True)
    cases = (
        (1.0, "identical models\n" + r"$\rho = 1$", RED),
        (0.5, "somewhat similar\n" + r"$\rho = 0.5$", AMBER),
        (0.0, "diverse models\n" + r"$\rho = 0$", GREEN),
    )
    for ax, (rho, title, colour) in zip(axes, cases):
        err = np.sqrt(rho) * shared + np.sqrt(1 - rho) * private
        y = np.arange(M)[::-1]
        for yi, e in zip(y, err):
            ax.plot([0, e], [yi, yi], color=GREY, lw=1.4, zorder=2)
            ax.scatter([e], [yi], s=36, color=colour, zorder=3)
        ax.axvline(0, color=INK, lw=1.6)
        m = err.mean()
        ax.plot([m, m], [-2.6, M - 1], color=colour, lw=1.2, ls=":", zorder=1)
        ax.scatter([m], [-2.6], s=190, color=colour, marker="D", zorder=4)
        ax.text(0.0, -7.5, f"error of the average:  {m:+.2f}", ha="center",
                fontsize=12.5, color=colour, backgroundcolor='#FFFFFF')
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-6.6, M + 1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].text(-5, M / 2, "one row represents\none model's error", fontsize=11,
                 color=GREY, va="center", ha="left")
    save(fig, out, "ensemble_correlation")


def fig_bagging_boundary(out: Path) -> None:
    """Averaging deep trees: same bias, far less variance."""
    from sklearn.datasets import make_moons
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_moons(n_samples=220, noise=0.26, random_state=0)
    # keep the grid on the data: far from it every tree just extrapolates its last
    # split, which shows up as streaks and hides the point of the figure
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 350), np.linspace(-1.2, 1.7, 350))
    grid = np.c_[gx.ravel(), gy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), sharey=True)
    models = (
        ("one deep tree", DecisionTreeClassifier(random_state=0)),
        ("100 trees, bagged", RandomForestClassifier(100, max_features=None,
                                                     random_state=0)),
        ("100 trees, random forest\n(max_features='sqrt')",
         RandomForestClassifier(100, max_features="sqrt", random_state=0)),
    )
    for ax, (title, m) in zip(axes, models):
        m.fit(X, y)
        z = m.predict_proba(grid)[:, 1].reshape(gx.shape)
        # show the probability field, not the 0.5 contour: the contour is jagged for
        # all three, while the field is exactly where averaging shows up
        ax.imshow(z, extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin="lower",
                  cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75, aspect="auto",
                  interpolation="bilinear")
        for cls, colour in ((0, BLUE), (1, RED)):
            m_ = y == cls
            ax.scatter(X[m_, 0], X[m_, 1], s=18, color=colour, edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, out, "bagging_boundary")


def fig_bagging_vs_boosting(out: Path) -> None:
    """Two ways to spend many trees, and the two errors they attack."""
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    blank(ax)
    ax.text(0.25, 0.95, "BAGGING — parallel", ha="center", fontsize=13, color=GREEN)
    ax.text(0.75, 0.95, "BOOSTING — sequential", ha="center", fontsize=13, color=AMBER)
    ax.plot([0.5, 0.5], [0.05, 0.9], color=GREY, lw=1, ls="--")

    box(ax, (0.25, 0.80), 0.16, 0.09, "training data", ec=INK, fs=11)
    for i in range(4):
        x = 0.08 + i * 0.113
        arrow(ax, (0.25, 0.755), (x, 0.63), color=GREY, lw=1.2)
        box(ax, (x, 0.56), 0.10, 0.11, f"deep\ntree {i + 1}", ec=GREEN, fs=10.5)
        arrow(ax, (x, 0.50), (0.25, 0.34), color=GREY, lw=1.2)
    box(ax, (0.25, 0.27), 0.20, 0.10, "average / vote", ec=GREEN, fs=11.5)
    ax.text(0.25, 0.06, "each tree sees a bootstrap sample\n→ attacks VARIANCE",
            ha="center", fontsize=11.5, color=INK)

    for i in range(4):
        x = 0.575 + i * 0.115
        box(ax, (x, 0.56), 0.10, 0.11, f"shallow\ntree {i + 1}", ec=AMBER, fs=10.5)
        if i:
            arrow(ax, (x - 0.058, 0.56), (x - 0.052, 0.56), color=GREY, lw=1.6)
    ax.text(0.75, 0.74, "each tree fits the errors of the sum so far",
            ha="center", fontsize=11, color=GREY)
    for i in range(4):
        arrow(ax, (0.575 + i * 0.115, 0.50), (0.75, 0.34), color=GREY, lw=1.2)
    box(ax, (0.75, 0.27), 0.20, 0.10, r"weighted sum  $\sum \eta\, h_m$", ec=AMBER, fs=11.5)
    ax.text(0.75, 0.06, "each tree corrects the running model\n→ attacks BIAS",
            ha="center", fontsize=11.5, color=INK)
    save(fig, out, "bagging_vs_boosting")


def fig_boosting_stages(out: Path) -> None:
    """Watch the residuals flatten: three rounds of boosting, drawn."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 90))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.18, 90)
    grid = np.linspace(0, 1, 400)
    eta = 0.8

    F = np.zeros_like(y)
    Fg = np.zeros_like(grid)
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 5.8), sharex=True)
    for m in range(4):
        r = y - F
        axes[0, m].plot(grid, np.sin(2 * np.pi * grid), color=GREY, lw=1.6, ls="--")
        axes[0, m].scatter(x, y, s=14, color=BLUE, alpha=0.6, edgecolors="none")
        axes[0, m].plot(grid, Fg, color=RED, lw=2.6)
        axes[0, m].set_title(f"$F_{m}$  (after {m} trees)", fontsize=12)
        axes[0, m].set_ylim(-1.9, 1.9)

        axes[1, m].axhline(0, color=INK, lw=0.9)
        axes[1, m].scatter(x, r, s=14, color=AMBER, edgecolors="none")
        axes[1, m].set_ylim(-1.9, 1.9)
        axes[1, m].set_xlabel("x")
        if m < 3:
            h = DecisionTreeRegressor(max_depth=2).fit(x[:, None], r)
            axes[1, m].plot(grid, h.predict(grid[:, None]), color=GREEN, lw=2.2)
            axes[1, m].set_title(f"residuals + tree $h_{m + 1}$ fitted to them", fontsize=11)
            F = F + eta * h.predict(x[:, None])
            Fg = Fg + eta * h.predict(grid[:, None])
        else:
            axes[1, m].set_title("residuals: nearly flat", fontsize=11)
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("residual")
    save(fig, out, "boosting_stages")


def fig_additive_model(out: Path) -> None:
    """What F_m IS: a starting constant plus a list of trees, added up."""
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    blank(ax)

    box(ax, (0.09, 0.62), 0.13, 0.14, "one patient\n$x$", ec=INK, fs=12)

    rows = [
        (0.91, r"$F_0$", "the base rate\n(one constant)", "$+0.30$", GREY),
        (0.745, r"$h_1$", "fitted to what\n$F_0$ got wrong", r"$+\eta\,(+1.4)$", BLUE),
        (0.58, r"$h_2$", "fitted to what\n$F_1$ got wrong", r"$+\eta\,(-0.6)$", BLUE),
        (0.415, r"$h_3$", "fitted to what\n$F_2$ got wrong", r"$+\eta\,(+0.9)$", BLUE),
    ]
    for y, name, caption, contrib, colour in rows:
        box(ax, (0.34, y), 0.11, 0.13, name, ec=colour, fs=14)
        ax.text(0.42, y, caption, fontsize=10.5, color=GREY, va="center", ha="left")
        arrow(ax, (0.155, 0.62), (0.283, y), color=GREY, lw=1.2)
        ax.text(0.70, y, contrib, fontsize=13, color=colour, va="center", ha="center")
        arrow(ax, (0.60, y), (0.655, y), color=GREY, lw=1.2)
    ax.text(0.34, 0.30, r"$\vdots$", fontsize=20, ha="center", color=GREY)
    ax.text(0.70, 0.30, r"$\vdots$", fontsize=20, ha="center", color=GREY)
    ax.text(0.42, 0.27, "M trees in total", fontsize=11, color=GREY, ha="left")

    # the sum, and what it is called
    ax.plot([0.79, 0.79], [0.28, 0.95], color=GREY, lw=1.2)
    arrow(ax, (0.79, 0.62), (0.845, 0.62), color=GREY, lw=1.4)
    box(ax, (0.91, 0.62), 0.15, 0.18,
        r"$F_M(x)$" + "\nthe prediction", ec=GREEN, fs=13)
    ax.text(0.83, 0.985, "add them all up", fontsize=11.5, color=GREY, ha="center")

    ax.text(0.5, 0.09,
            r"$F_M(x)\;=\;F_0\;+\;\eta\,h_1(x)\;+\;\eta\,h_2(x)\;+\;\cdots\;"
            r"+\;\eta\,h_M(x)$",
            fontsize=17, color=INK, ha="center")
    save(fig, out, "additive_model")


def fig_shrinkage(out: Path) -> None:
    """What eta actually does: take a fraction of each tree's correction."""
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 70))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.22, x.size)
    xg = np.linspace(0, 1, 400)
    tg = np.sin(2 * np.pi * xg)

    # the same tree counts on both sides: the comparison is what M trees buy you
    stages = (1, 3, 10, 60)
    shades = ("#cfe0f0", "#8fb8de", "#4a86c6", BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)
    panels = ((1.0, r"$\eta = 1$ — add each tree whole", RED,
               "the fit lurches after every tree\nand ends up chasing the noise"),
              (0.1, r"$\eta = 0.1$ — add a tenth of it", GREEN,
               "each tree only nudges the fit;\nno single tree dominates"))
    for ax, (eta, title, colour, caption) in zip(axes, panels):
        gb = GradientBoostingRegressor(n_estimators=max(stages), learning_rate=eta,
                                       max_depth=2, random_state=0)
        gb.fit(x[:, None], y)
        preds = list(gb.staged_predict(xg[:, None]))
        ax.scatter(x, y, s=18, color=GREY, zorder=1)
        ax.plot(xg, tg, color=INK, lw=2.2, ls="--", zorder=2, label="truth")
        for m, c in zip(stages, shades):
            ax.plot(xg, preds[m - 1], color=c, lw=2.4, zorder=3,
                    label=f"after {m} tree" + ("s" if m > 1 else ""))
        ax.set_title(title, fontsize=13.5)
        ax.text(0.02, 2.28, caption, fontsize=12, color=colour, va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-1.9, 2.45)
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=11.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(r"$F_m = F_{m-1} + \eta\, h_m$   —   $\eta$ shrinks every correction",
                 fontsize=14.5, y=1.02)
    save(fig, out, "shrinkage")


def fig_learning_rate(out: Path) -> None:
    """η and M are one knob in two halves."""
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = make_friedman1(n_samples=250, noise=3.0, random_state=0)
    Xte, yte = make_friedman1(n_samples=3000, noise=3.0, random_state=1)

    fig, ax = plt.subplots(figsize=HALF)
    for eta, colour in ((0.5, RED), (0.2, AMBER), (0.05, BLUE), (0.01, GREEN)):
        gb = GradientBoostingRegressor(n_estimators=800, learning_rate=eta,
                                       max_depth=4, random_state=0).fit(X, y)
        te = np.array([np.mean((yte - p) ** 2) for p in gb.staged_predict(Xte)])
        ax.plot(np.arange(1, 801), te, color=colour, lw=2.2, label=rf"$\eta$ = {eta}")
        k = int(np.argmin(te))
        ax.plot([k + 1], [te[k]], "o", color=colour, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("number of trees  (log scale)")
    ax.set_ylabel("test MSE")
    ax.set_ylim(4, 22)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("small η reaches a lower minimum — and needs far more trees",
                 fontsize=12.5)
    save(fig, out, "learning_rate")


def fig_stacking(out: Path) -> None:
    """Base learners with different inductive biases, then a small meta-model."""
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    blank(ax)
    box(ax, (0.10, 0.55), 0.14, 0.13, "features\n$x$", ec=INK, fs=12)
    bases = (("logistic\nregression", BLUE, "smooth, linear"),
             ("random\nforest", GREEN, "axis-aligned"),
             ("LightGBM", AMBER, "sharp interactions"),
             ("k-NN", PURPLE, "local"))
    for i, (name, colour, note) in enumerate(bases):
        y = 0.88 - i * 0.22
        arrow(ax, (0.175, 0.55), (0.30, y), color=GREY, lw=1.3)
        box(ax, (0.38, y), 0.16, 0.15, name, ec=colour, fs=11)
        ax.text(0.478, y, note, fontsize=10, color=GREY, va="center")
        arrow(ax, (0.615, y), (0.70, 0.55), color=GREY, lw=1.3)
        # clear of the connector: above it for the descending pair, below for the
        # ascending pair
        off = 0.055 if y > 0.55 else -0.055
        ax.text(0.607, y + off, f"$f_{i + 1}(x)$", fontsize=11, color=colour,
                ha="right", va="center")
    box(ax, (0.79, 0.55), 0.17, 0.16, "meta-learner\n$g(f_1,\\ldots,f_4)$", ec=RED, fs=12)
    arrow(ax, (0.875, 0.55), (0.95, 0.55), color=GREY)
    ax.text(0.965, 0.55, r"$\hat y$", fontsize=13, va="center")
    ax.text(0.5, 0, "The gain comes from errors that DIFFER.\n"
            r"Using different families lower $\rho$ far more than using different seeds.",
            ha="center", fontsize=11.5, color=GREY, style="italic")
    save(fig, out, "stacking")


def fig_oof(out: Path) -> None:
    """The one slide that decides whether your stack is real."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for ax, ok in zip(axes, (False, True)):
        blank(ax)
        colour = GREEN if ok else RED
        ax.text(0.5, 0.95, "CORRECT: out-of-fold" if ok else "WRONG: in-fold",
                ha="center", fontsize=13.5, color=colour)
        for k in range(3):
            y = 0.74 - k * 0.19
            for j in range(3):
                x = 0.10 + j * 0.15
                held = ok and j == k
                ax.add_patch(Rectangle((x, y - 0.055), 0.14, 0.11,
                                       facecolor=RED if held else FAINT,
                                       edgecolor=GREY, lw=1.1))
                ax.text(x + 0.07, y, f"fold {j + 1}", ha="center", va="center",
                        fontsize=10, color="white" if held else INK)
            if ok:
                ax.text(0.58, y, f"train on the grey folds\n→ predict fold {k + 1}",
                        fontsize=10.5, va="center", color=INK)
            else:
                ax.text(0.58, y, "train on everything\n→ predict the SAME rows",
                        fontsize=10.5, va="center", color=INK)
        ax.text(0.5, 0.13,
                "every row has a prediction from a model\nthat never saw that row"
                if ok else
                "the base models have already seen these rows;\n"
                "the meta-model learns to trust whoever memorized best",
                ha="center", fontsize=11.5, color=colour)
    save(fig, out, "oof")


def fig_cv_schemes(out: Path) -> None:
    """Getting the split wrong is the most common leak in clinical data."""
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 5.0))
    n = 24
    patient = np.repeat(np.arange(8), 3)
    # a left gutter so the "patient" row label has somewhere to live
    x0, step, w = 0.095, 0.0378, 0.034

    schemes = (
        ("K-fold: random rows", np.tile(np.arange(4), 6), None),
        ("Group K-fold: whole patients", np.repeat(np.arange(4), 6), patient),
        ("Time-series split: the past only", np.repeat(np.arange(4), 6), None),
    )
    test_fold = 2
    for ax, (title, folds, group) in zip(axes, schemes):
        blank(ax)
        ax.text(0.0, 0.80, title, fontsize=12, color=INK)
        for i in range(n):
            x = x0 + i * step
            if "Time" in title:
                is_test, unused = i // 6 == test_fold, i // 6 > test_fold
            else:
                is_test, unused = folds[i] == test_fold, False
            colour = "white" if unused else (RED if is_test else FAINT)
            ax.add_patch(Rectangle((x, 0.22), w, 0.34, facecolor=colour,
                                   edgecolor=GREY, lw=0.9))
            if group is not None:
                ax.text(x + w / 2, 0.10, str(patient[i]), ha="center", fontsize=8,
                        color=GREY)
        if group is not None:
            ax.text(x0 - 0.012, 0.10, "patient", fontsize=9, color=GREY,
                    ha="right", va="center")
        if "Time" in title:
            # above the boxes, not on them
            ax.text(x0 + 20.5 * step, 0.68, "future: unused", fontsize=10,
                    color=GREY, ha="center", va="center")
    fig.text(0.5, -0.02, "red are the held-out folds",
             ha="center", fontsize=11.5, color=GREY)
    save(fig, out, "cv_schemes")


def fig_concept_map(out: Path) -> None:
    """The whole session on one slide."""
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    blank(ax)
    box(ax, (0.5, 0.945), 0.26, 0.08, "GENERALIZATION", ec=INK, fs=13.5)
    box(ax, (0.30, 0.80), 0.24, 0.07, "model complexity", ec=INK, fs=12)
    box(ax, (0.76, 0.80), 0.16, 0.07, "data", ec=INK, fs=12)
    arrow(ax, (0.45, 0.905), (0.34, 0.836))
    arrow(ax, (0.55, 0.905), (0.72, 0.836))

    box(ax, (0.13, 0.645), 0.20, 0.07, "regularization", ec=BLUE, fs=12)
    box(ax, (0.47, 0.645), 0.17, 0.07, "ensembles", ec=GREEN, fs=12)
    arrow(ax, (0.25, 0.765), (0.16, 0.681))
    arrow(ax, (0.35, 0.765), (0.44, 0.681))

    box(ax, (0.34, 0.485), 0.15, 0.068, "bagging", ec=GREEN, fs=11.5)
    box(ax, (0.66, 0.485), 0.15, 0.068, "boosting", ec=AMBER, fs=11.5)
    arrow(ax, (0.44, 0.610), (0.37, 0.520))
    arrow(ax, (0.51, 0.610), (0.63, 0.520))
    ax.text(0.34, 0.398, "random forest", ha="center", fontsize=11, color=GREEN)

    for i, name in enumerate(("XGBoost", "LightGBM", "CatBoost")):
        cx = 0.50 + i * 0.19
        arrow(ax, (0.66, 0.451), (cx, 0.352), color=GREY, lw=1.3)
        box(ax, (cx, 0.315), 0.16, 0.068, name, ec=AMBER, fs=11)
        arrow(ax, (cx, 0.281), (0.60, 0.152), color=GREY, lw=1.2)

    box(ax, (0.13, 0.485), 0.22, 0.075, "hyperparameter\noptimization",
        ec=PURPLE, fs=11.5)
    arrow(ax, (0.13, 0.610), (0.13, 0.523), color=PURPLE)
    ax.text(0.13, 0.395, "Bayesian search", ha="center", fontsize=11, color=PURPLE)
    ax.text(0.13, 0.335, "Optuna", ha="center", fontsize=11, color=PURPLE)

    box(ax, (0.60, 0.115), 0.20, 0.072, "stacking", ec=RED, fs=12)
    arrow(ax, (0.19, 0.315), (0.50, 0.130), color=GREY, lw=1.3)
    save(fig, out, "concept_map")


def fig_recap(out: Path) -> None:
    """Problem → technique, the table the whole session was built on."""
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    blank(ax)
    rows = (
        ("the model overfits", "bias–variance", BLUE),
        ("control complexity", "regularization", BLUE),
        ("too many configurations", "hyperparameter optimization", PURPLE),
        ("search is expensive", "Bayesian optimization", PURPLE),
        ("doing it in practice", "Optuna", PURPLE),
        ("one model is not enough", "ensembles", GREEN),
        ("strong tabular baseline", "gradient boosting", AMBER),
        ("state of the art on tables", "XGBoost / LightGBM / CatBoost", AMBER),
        ("complementary errors", "stacking", RED),
        ("not cheating", "cross-validation, out-of-fold", RED),
    )
    ax.text(0.30, 0.95, "PROBLEM", ha="right", fontsize=12.5, color=GREY)
    ax.text(0.40, 0.95, "TECHNIQUE", fontsize=12.5, color=GREY)
    for i, (p, t, colour) in enumerate(rows):
        y = 0.86 - i * 0.088
        ax.text(0.30, y, p, ha="right", fontsize=12, color=INK, va="center")
        arrow(ax, (0.32, y), (0.38, y), color=colour, lw=1.4)
        ax.text(0.40, y, t, fontsize=12, color=colour, va="center")
    save(fig, out, "recap")


# ------------------------------------------------------------------- equations
# Pandoc writes display math as OOXML (a14:m) with no mc:Fallback, so LibreOffice
# and Impress drop the whole shape and the slide renders blank. Rendering the six
# display equations as images makes them look identical in every viewer.
def equation(out: Path, name: str, lines, fs=34, annots=()) -> None:
    """Render display math as an image, auto-sized to fill the slide width.

    Built at the slide aspect rather than padded up to it, and the font size is
    fitted by measuring the drawn text: a hand-picked size either overflows or
    leaves the equation tiny on the slide. `annots` are (x, text, colour) callouts
    at explicit figure coordinates — space-padding a caption does not line up with
    the terms it names.
    """
    fig = plt.figure(figsize=(12.0, 12.0 / SLIDE_ASPECT))
    n = len(lines)
    top, bot = 0.86, (0.42 if annots else 0.10)
    handles = []
    for i, line in enumerate(lines):
        y = top - (i + 0.5) * (top - bot) / n
        handles.append(fig.text(0.5, y, line, ha="center", va="center",
                                fontsize=fs, color=INK))

    # scale up (or down) until the widest line spans ~92% of the figure
    fig.canvas.draw()
    widest = max(h.get_window_extent(fig.canvas.get_renderer()).width
                 for h in handles) / fig.get_size_inches()[0] / fig.dpi
    for h in handles:
        h.set_fontsize(min(fs * 0.92 / widest, 44))

    for x, text, colour in annots:
        fig.text(x, 0.24, text, ha="center", va="center", fontsize=19, color=colour)
        fig.text(x, 0.33, "↑", ha="center", va="center", fontsize=17, color=colour)
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / f"{name}.png")
    plt.close(fig)


def equation_parts(out: Path, name: str, parts, annots=(), fs=34, max_fs=44,
                   gaps=None) -> None:
    r"""Render an equation laid out from measured fragments, with aligned callouts.

    `parts` are mathtext fragments placed left to right on a common baseline; the
    layout measures each one, so a callout can be anchored to the exact centre of
    the term it names instead of at a guessed x. `annots` are
    (part index, label, colour). Labels are pushed apart if they would collide, and
    the arrow keeps pointing at the true centre.

    The space *between* fragments must be given here, as `gaps` (one width per
    boundary, in em of the final font size), never as `\;` or `\qquad` inside a
    fragment: mathtext drops trailing space from a text's bounding box, so a
    fragment that ends in a space measures as if it did not and the next fragment
    is laid down flush against it.
    """
    fig = plt.figure(figsize=(12.0, 12.0 / SLIDE_ASPECT))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    figw = fig.get_size_inches()[0] * fig.dpi

    y_eq = 0.64
    texts = [fig.text(0.0, y_eq, frag, fontsize=fs, ha="left", va="baseline")
             for frag in parts]

    gaps = [0.5] * (len(parts) - 1) if gaps is None else list(gaps)
    if len(gaps) != len(parts) - 1:
        raise ValueError(f"{name}: {len(parts)} parts need {len(parts) - 1} gaps")

    def widths():
        fig.canvas.draw()
        return [t.get_window_extent(r).width for t in texts]

    def em(size):
        return size * fig.dpi / 72.0

    # scale the whole equation to ~90% of the width; the gaps scale with the font,
    # so the fitted size follows from the same linear relation as the fragments
    w = widths()
    fs_final = min(fs * 0.90 * figw / (sum(w) + em(fs) * sum(gaps)), max_fs)
    for t in texts:
        t.set_fontsize(fs_final)
    w = widths()
    gap_px = [g * em(fs_final) for g in gaps] + [0.0]

    x = (figw - sum(w) - sum(gap_px)) / 2
    centres = []
    for t, wi, gi in zip(texts, w, gap_px):
        t.set_position((x / figw, y_eq))
        centres.append((x + wi / 2) / figw)
        x += wi + gi

    if not annots:
        with matplotlib.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(out / f"{name}.png")
        plt.close(fig)
        return

    # place the labels, then separate any that would run into each other
    labels = [fig.text(centres[i], 0.24, txt, fontsize=19, color=col,
                       ha="center", va="center") for i, txt, col in annots]
    fig.canvas.draw()
    half = [t.get_window_extent(r).width / figw / 2 for t in labels]
    xs = [centres[i] for i, _, _ in annots]
    gap = 0.03
    for j in range(1, len(xs)):                       # push right
        need = xs[j - 1] + half[j - 1] + gap + half[j]
        xs[j] = max(xs[j], need)
    over = xs[-1] + half[-1] - 0.98                   # then back inside the page
    if over > 0:
        xs[-1] -= over
        for j in range(len(xs) - 2, -1, -1):
            xs[j] = min(xs[j], xs[j + 1] - half[j + 1] - gap - half[j])
    for t, x_ in zip(labels, xs):
        t.set_position((x_, 0.24))
        t.set_va("center")

    for (i, _, col), x_ in zip(annots, xs):
        fig.add_artist(FancyArrowPatch(
            (x_, 0.32), (centres[i], 0.56), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=13, color=col, lw=1.6,
            shrinkA=2, shrinkB=2))

    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_eq_bias_variance(out: Path) -> None:
    equation(out, "eq_bias_variance", [
        r"$\mathbb{E}\left[(Y-\hat{f}(x_0))^2\right]"
        r"\;=\;\sigma^2\;+\;\mathrm{Bias}[\hat{f}(x_0)]^2"
        r"\;+\;\mathrm{Var}[\hat{f}(x_0)]$",
        r"$\mathrm{Bias}[\hat{f}(x_0)]=\mathbb{E}[\hat{f}(x_0)]-f(x_0)$",
    ])


def fig_eq_erm(out: Path) -> None:
    equation_parts(out, "eq_erm", [
        r"$\hat{f}\;=\;\mathrm{arg\,min}_{f}$",
        r"$\frac{1}{n}\sum_{i=1}^{n}L(y_i,f(x_i))$",
        r"$+$",
        r"$\lambda\,\Omega(f)$",
    ], gaps=(0.55, 0.5, 0.5),
       annots=((1, "fit the data", BLUE), (3, "complexity penalty", RED)))


def fig_eq_ridge_lasso(out: Path) -> None:
    equation(out, "eq_ridge_lasso", [
        r"Ridge (L2)$\qquad\min_{\beta}\;\|y-X\beta\|^2"
        r"+\lambda\|\beta\|_2^2$",
        r"Lasso (L1)$\qquad\min_{\beta}\;\|y-X\beta\|^2"
        r"+\lambda\|\beta\|_1$",
    ], fs=32)


def fig_eq_params(out: Path) -> None:
    equation_parts(out, "eq_params", [
        r"$\theta\;=\;\mathrm{arg\,min}_{\theta}\;\hat{R}(\theta;\lambda)$",
        r"$\lambda\;=\;?$",
    ], gaps=(3.0,),
       annots=((0, "learned by fitting the data", BLUE),
               (1, "chosen before fitting", RED)))


def fig_eq_boosting(out: Path) -> None:
    equation(out, "eq_boosting", [
        r"$r_{im}\;=\;-\left[\frac{\partial L(y_i,F(x_i))}"
        r"{\partial F(x_i)}\right]_{F=F_{m-1}}$"
        r"$\qquad\qquad F_m\;=\;F_{m-1}+\eta\,h_m$",
    ], fs=27)


def fig_eq_xgboost(out: Path) -> None:
    """The XGBoost objective with every term named on the slide.

    Written out rather than routed through `equation`: the point of the slide is
    that each term is a thing you can point at, so the callouts have to be
    anchored to the exact summand they name. What this objective collapses to
    for a single leaf is the next slide, `eq_leaf_objective`.
    """
    fig = plt.figure(figsize=(12.0, 12.0 / SLIDE_ASPECT))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    figw = fig.get_size_inches()[0] * fig.dpi

    parts = [
        r"$\sum_{i}\,\ell\left(y_i,\;F_{m-1}(x_i)+h(x_i)\right)$",
        r"$+\;\gamma\,T$",
        r"$+\;\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$",
    ]
    fs, y_eq = 26, 0.76
    texts = [fig.text(0.0, y_eq, frag, fontsize=fs, ha="left", va="center")
             for frag in parts]
    fig.text(0.0, y_eq, r"$\mathcal{L}^{(m)}\;=\;$", fontsize=fs,
             ha="left", va="center")

    def widths():
        fig.canvas.draw()
        return [t.get_window_extent(r).width for t in texts]

    # one measured scale-to-fit pass, gaps included, exactly as `equation_parts`
    lead = fig.texts[-1]
    w = widths()
    lead_w = lead.get_window_extent(r).width
    gaps = [0.6, 0.6]  # em, between the three summands
    fs_final = min(fs * 0.88 * figw / (sum(w) + lead_w + fs * fig.dpi / 72.0
                                       * sum(gaps)), 30)
    for t in texts + [lead]:
        t.set_fontsize(fs_final)
    w = widths()
    lead_w = lead.get_window_extent(r).width
    gap_px = [g * fs_final * fig.dpi / 72.0 for g in gaps] + [0.0]

    x = (figw - sum(w) - lead_w - sum(gap_px)) / 2
    lead.set_position((x / figw, y_eq))
    x += lead_w + 0.3 * fs_final * fig.dpi / 72.0
    centres = []
    for t, wi, gi in zip(texts, w, gap_px):
        t.set_position((x / figw, y_eq))
        centres.append((x + wi / 2) / figw)
        x += wi + gi

    annots = [
        (0, "how wrong the tree still\nleaves us on the data", BLUE),
        (1, r"$\gamma$ = price of one leaf:" "\n" "charges for complexity, so it prunes", RED),
        (2, r"$\lambda$ = shrinks every leaf value" "\n" "toward 0: no leaf shouts", PURPLE),
    ]
    labels = [fig.text(centres[i], 0.46, txt, fontsize=14, color=col,
                       ha="center", va="center", linespacing=1.4)
              for i, txt, col in annots]

    # the three callouts are wider than the terms they name, so separate them:
    # push each one clear of its left neighbour, then slide the row back on-page
    fig.canvas.draw()
    half = [t.get_window_extent(r).width / figw / 2 for t in labels]
    xs = [centres[i] for i, _, _ in annots]
    gap = 0.025
    for j in range(1, len(xs)):
        xs[j] = max(xs[j], xs[j - 1] + half[j - 1] + gap + half[j])
    over = xs[-1] + half[-1] - 0.99
    if over > 0:
        xs[-1] -= over
        for j in range(len(xs) - 2, -1, -1):
            xs[j] = min(xs[j], xs[j + 1] - half[j + 1] - gap - half[j])
    xs[0] = max(xs[0], 0.01 + half[0])

    for (i, _, col), x_, t in zip(annots, xs, labels):
        t.set_position((x_, 0.46))
        fig.add_artist(FancyArrowPatch(
            (x_, 0.55), (centres[i], 0.68), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=12, color=col, lw=1.5,
            shrinkA=2, shrinkB=2))

    fig.text(0.5, 0.19,
             r"$T$ = number of leaves,   $w_j$ = the value the tree outputs "
             r"in leaf $j$",
             fontsize=14, ha="center", va="center", color=GREY)

    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / "eq_xgboost.png")
    plt.close(fig)


def fig_eq_leaf_objective(out: Path) -> None:
    """From the whole objective to one leaf's quadratic, in three moves.

    Written as stacked figure text rather than through `equation` because the
    point is the sequence: each line is one manipulation, with the reason for it
    beside it, and the last line is the one the rest of the section uses.

    The legend is not decoration: h_i and h(x_i) are different objects sharing a
    letter, and every year somebody reads the second-order term as the tree
    squared. Naming all three symbols before the derivation is what stops that.
    """
    fig = plt.figure(figsize=(12.6, 12.6 / SLIDE_ASPECT))

    # ---------------------------------------------------------- what the symbols are
    legend = ((0.17, BLUE, r"$g_i$",
               "one number per row:\n" r"the slope $\partial\ell/\partial F$ at row $i$"),
              (0.50, PURPLE, r"$h_i$",
               "one number per row:\n" r"the curvature $\partial^2\ell/\partial F^2$"),
              (0.83, AMBER, r"$h(x_i)$",
               "the new tree evaluated at row $i$"))
    for x, col, sym, gloss in legend:
        fig.text(x, 0.955, sym, fontsize=19, color=col, ha="center", va="center")
        fig.text(x, 0.875, gloss, fontsize=11, color=INK, ha="center",
                 va="center", linespacing=1.45)
    fig.text(0.5, 0.795,
             r"careful: $h_i$ carries a SUBSCRIPT and is a number; "
             r"$h(x_i)$ carries an ARGUMENT and is the tree",
             fontsize=11.5, color=RED, ha="center", va="center")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.755, 0.755], color=FAINT, lw=1.6,
                              transform=fig.transFigure))

    # ------------------------------------------------------------- the three moves
    steps = (
        (0.715, "the objective for the tree we are adding",
         r"$\mathcal{L}^{(m)}=\sum_i \ell\left(y_i,\,F_{m-1}(x_i)+h(x_i)\right)"
         r"\;+\;\gamma T\;+\;\frac{1}{2}\lambda\sum_j w_j^2$", INK),
        (0.520, "expand to 2nd order in the new tree, and drop "
                r"$\ell(y_i,F_{m-1})$ (constant)",
         r"$\simeq\;\sum_i\left[\,g_i\,h(x_i)+\frac{1}{2}h_i\,h(x_i)^2\right]"
         r"\;+\;\gamma T\;+\;\frac{1}{2}\lambda\sum_j w_j^2$", INK),
        (0.325, "a tree is constant on each leaf: $h(x_i)=w_j$ for every row "
                r"$i\in I_j$, so add the rows up leaf by leaf",
         r"$=\;\sum_{j=1}^{T}\left[\,G_j\,w_j+\frac{1}{2}"
         r"\left(H_j+\lambda\right)w_j^{2}\,\right]\;+\;\gamma T$"
         r"       with $G_j=\sum_{i\in I_j} g_i$, $H_j=\sum_{i\in I_j} h_i$",
         BLUE),
    )
    for y, why, maths, col in steps:
        fig.text(0.5, y, why, fontsize=11.5, color=GREY, ha="center", va="center")
        fig.text(0.5, y - 0.075, maths, fontsize=15, color=col, ha="center",
                 va="center")

    # -------------------------------------------------------------- the payoff
    fig.patches.append(FancyBboxPatch(
        (0.13, 0.02), 0.74, 0.135, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="white", edgecolor=GREEN, linewidth=2.0,
        transform=fig.transFigure))
    fig.text(0.5, 0.125, "every leaf sits in its own term, so the sum splits into "
                         r"$T$ independent problems:",
             fontsize=12, color=GREEN, ha="center", va="center")
    fig.text(0.5, 0.062, r"$\mathrm{obj}_j(w)\;=\;G_j\,w\;+\;\frac{1}{2}"
                         r"\left(H_j+\lambda\right)w^{2}$"
                         "        one quadratic, one unknown",
             fontsize=15, color=INK, ha="center", va="center")

    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / "eq_leaf_objective.png")
    plt.close(fig)


def _logloss_curve(F):
    """Log loss of a positive row as a function of its predicted log-odds."""
    return np.log1p(np.exp(-F))


def fig_xgb_gh(out: Path) -> None:
    """Where g and h come from, and what they are for the two usual losses."""
    F = np.linspace(-3.4, 3.4, 400)
    F0 = 0.0
    l0, p0 = _logloss_curve(F0), 1 / (1 + np.exp(-F0))
    g0, h0 = p0 - 1.0, p0 * (1 - p0)

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    ax.plot(F, _logloss_curve(F), color=INK, lw=2.4, label=r"the row's loss $\ell$")
    t = np.array([F0 - 1.9, F0 + 1.9])
    ax.plot(t, l0 + g0 * (t - F0), color=BLUE, lw=2.0, ls="--",
            label=r"slope at this point $\;\to\;g_i$")
    d = np.linspace(-2.1, 2.1, 200)
    ax.plot(F0 + d, l0 + g0 * d + 0.5 * h0 * d ** 2, color=PURPLE, lw=2.0,
            label=r"curvature at this point $\;\to\;h_i$")
    ax.plot([F0], [l0], "o", color=INK, ms=8, zorder=5)
    ax.annotate("this row's current\nprediction", (F0, l0), (F0 - 3.2, 0.55),
                fontsize=11.5, color=GREY, linespacing=1.5, va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
    ax.set_xlabel(r"prediction for row $i$,  $F_{m-1}(x_i)$")
    ax.set_ylabel(r"that row's loss $\ell$")
    ax.set_ylim(-0.35, 3.4)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    ax.set_title("both are read off the loss at one row", fontsize=13)

    ax = axes[1]
    blank(ax)
    ax.set_title("and both have a closed form for the losses we use",
                 fontsize=13)
    cols = (0.30, 0.60, 0.85)
    heads = (r"loss  $\ell(y_i, F)$", r"$g_i=\partial\ell/\partial F$",
             r"$h_i=\partial^2\ell/\partial F^2$")
    rows = ((r"squared  $\frac{1}{2}(y_i-F)^2$", r"$F-y_i$", r"$1$",
             "the plain residual"),
            (r"log loss", r"$p_i-y_i$", r"$p_i(1-p_i)$",
             r"$p_i=\sigma(F)$"))
    for x, head, col in zip(cols, heads, (INK, BLUE, PURPLE)):
        ax.text(x, 0.86, head, fontsize=13, color=col, ha="center", va="center")
    ax.plot([0.02, 0.98], [0.78, 0.78], color=GREY, lw=1.2)
    for k, (name, g, h, note) in enumerate(rows):
        y = 0.62 - 0.30 * k
        ax.text(cols[0], y, name, fontsize=13.5, ha="center", va="center")
        ax.text(cols[1], y, g, fontsize=14, color=BLUE, ha="center", va="center")
        ax.text(cols[2], y, h, fontsize=14, color=PURPLE, ha="center", va="center")
        ax.text(cols[0], y - 0.115, note, fontsize=11, color=GREY,
                ha="center", va="center")
    ax.text(0.5, 0.05,
            "one $g$ and one $h$ per row are recomputed once for each tree",
            fontsize=12, color=GREY, ha="center", va="center", linespacing=1.5)
    save(fig, out, "xgb_gh")


def fig_xgb_leaf_predict(out: Path) -> None:
    """Use 1: a row lands in one leaf per tree, and collects eta times its value."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    blank(ax)
    ax.set_title(r"one tree: the row is routed to exactly one leaf", fontsize=13)
    r_, a_ = (0.50, 0.86), (0.30, 0.56)
    leaves = {"l1": ((0.13, 0.26), "+0.4", False),
              "l2": ((0.47, 0.26), "-1.2", True),
              "l3": ((0.80, 0.30), "+0.9", False)}
    for a, b_, on in ((r_, a_, True), (r_, leaves["l3"][0], False),
                      (a_, leaves["l1"][0], False), (a_, leaves["l2"][0], True)):
        ax.plot(*zip(a, b_), color=GREEN if on else GREY,
                lw=2.8 if on else 1.6, zorder=1)
    for xy in (r_, a_):
        ax.add_patch(Circle(xy, 0.035, facecolor="white", edgecolor=GREY,
                            lw=1.8, zorder=2))
    for xy, val, on in leaves.values():
        box(ax, xy, 0.21, 0.15, rf"$w^{{\star}}={val}$",
            ec=GREEN if on else GREY, fs=12, lw=2.4 if on else 1.5)
    for xy, txt in (((0.40, 0.71), "$x_3<0.4$"), ((0.34, 0.41), "$x_7<2$")):
        ax.text(*xy, txt, fontsize=11, color=GREY, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax.text(0.47, 0.09, "the splits only decide WHICH leaf;\n"
                        "the leaf value is the answer",
            fontsize=11.5, color=GREEN, ha="center", va="center", linespacing=1.5)

    ax = axes[1]
    blank(ax)
    ax.set_title(r"the ensemble: one leaf value per tree, scaled by $\eta$",
                 fontsize=13)
    eta = 0.3
    ledger = ((r"$F_0$", "base score", "", 0.20),
              ("tree 1", r"$w^{\star}=-1.2$", r"$\eta\,w^{\star}=-0.36$", -0.16),
              ("tree 2", r"$w^{\star}=+0.8$", r"$\eta\,w^{\star}=+0.24$", 0.08),
              ("tree 3", r"$w^{\star}=-0.5$", r"$\eta\,w^{\star}=-0.15$", -0.07))
    cols = (0.11, 0.38, 0.68, 0.93)
    for x, head in zip(cols, ("", "leaf it lands in", rf"$\eta={eta}$ of it",
                              "running $F$")):
        ax.text(x, 0.84, head, fontsize=11.5, color=GREY, ha="center", va="center")
    ax.plot([0.02, 0.99], [0.77, 0.77], color=GREY, lw=1.1)
    for k, (name, leaf, step, tot) in enumerate(ledger):
        y = 0.66 - 0.145 * k
        ax.text(cols[0], y, name, fontsize=12.5, ha="center", va="center")
        ax.text(cols[1], y, leaf, fontsize=12.5, ha="center", va="center",
                color=GREEN if k else GREY)
        ax.text(cols[2], y, step, fontsize=12.5, ha="center", va="center",
                color=BLUE)
        ax.text(cols[3], y, f"{tot:+.2f}", fontsize=12.5, ha="center",
                va="center", color=INK)
    ax.text(cols[3], 0.10, r"$\vdots$", fontsize=14, ha="center", va="center")
    ax.text(0.5, 0.02,
            r"$F_M(x)=F_0+\eta\sum_{m}w^{\star}_{\mathrm{leaf}_m(x)}$"
            "\nfor classification, add $\\sigma(F_M)$ at the very end",
            fontsize=12.5, color=INK, ha="center", va="center")
    save(fig, out, "xgb_leaf_predict")


def fig_xgb_leaf_score(out: Path) -> None:
    """Use 2: substituted back, the same w* becomes the score a split is judged by."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.set_title("substitute $w_j^{\\star}$ back into the objective and the leaf "
                 "stops being an unknown", fontsize=13.5)

    steps = ((0.17, "the objective, restricted\nto one leaf $j$:",
              r"$G_jw+\frac{1}{2}(H_j+\lambda)w^2+\gamma$", GREY),
             (0.50, "a quadratic in $w$, so its\nminimum is closed-form:",
              r"$w_j^{\star}=-\dfrac{G_j}{H_j+\lambda}$", PURPLE),
             (0.83, "and its value THERE is a\nnumber, with no $w$ left:",
              r"$-\frac{1}{2}\dfrac{G_j^2}{H_j+\lambda}+\gamma$", BLUE))
    for x, cap, formula, col in steps:
        ax.text(x, 0.90, cap, fontsize=12, color=GREY, ha="center", va="center",
                linespacing=1.5)
        box(ax, (x, 0.68), 0.28, 0.20, formula, ec=col, fs=15)
    arrow(ax, (0.32, 0.68), (0.355, 0.68), color=GREY)
    arrow(ax, (0.65, 0.68), (0.685, 0.68), color=GREY)

    ax.text(0.5, 0.44, r"$s(G,H)=\dfrac{G^2}{H+\lambda}$ is the score of a set of rows",
            fontsize=13, color=INK, ha="center", va="center")
    box(ax, (0.5, 0.23), 0.92, 0.20,
        r"gain $=\frac{1}{2}\left[\,s(G_L,H_L)+s(G_R,H_R)-s(G,H)\,\right]"
        r"-\gamma$", ec=GREEN, fs=16)
    ax.text(0.5, 0.02, "a CANDIDATE SPLIT is one (feature, threshold) pair; the "
                       "node takes the one with the largest gain,\n"
                       "and is left as a leaf if no candidate gets the gain above 0",
            fontsize=12, color=GREY, ha="center", va="center", linespacing=1.5)
    save(fig, out, "xgb_leaf_score")

LAM, GAM = 1.0, 0.5          # the lambda and gamma used by the split figures


def _split_demo():
    """The 40-row toy node the split figures share, so the slides link up.

    Log loss at the very first tree: every prediction is 0.5, so g is +-0.5 and
    h is 0.25 for every row, and only feature 0 carries signal.
    """
    rng = np.random.default_rng(4)
    n = 40
    X = rng.uniform(0, 1, (n, 3))
    y = (X[:, 0] > 0.55).astype(float)
    flip = rng.choice(n, 5, replace=False)          # a little label noise
    y[flip] = 1 - y[flip]
    return X, y, 0.5 - y, np.full(n, 0.25)


def _gain_at(xs, gs, hs, cuts, lam=LAM, gam=GAM):
    """Gain of `xs < cut` for each cut — the formula from the XGBoost slides."""
    G, H = gs.sum(), hs.sum()
    s = lambda a, b: a ** 2 / (b + lam)
    out = []
    for c in cuts:
        m = xs < c
        GL, HL = gs[m].sum(), hs[m].sum()
        out.append(0.5 * (s(GL, HL) + s(G - GL, H - HL) - s(G, H)) - gam)
    return np.array(out)


def fig_xgb_split_search(out: Path) -> None:
    """Where candidate splits come from, and how one of them is chosen.

    Real numbers: 40 rows, log loss at the first tree (so every g is +-0.5 and
    every h is 0.25), the gain evaluated at every midpoint of every feature. The
    point of the figure is that nothing here is clever — it is an exhaustive scan
    scored by one formula.
    """
    X, y, g, h = _split_demo()
    n, lam, gam = len(y), LAM, GAM
    G, H = g.sum(), h.sum()
    rng = np.random.default_rng(11)           # vertical jitter of the row strip

    def scan(col):
        """Every midpoint of this feature, with the gain it would buy."""
        order = np.argsort(X[:, col])
        xs, gs, hs = X[order, col], g[order], h[order]
        cuts = (xs[:-1] + xs[1:]) / 2
        GL, HL = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
        s = lambda a, b: a ** 2 / (b + lam)
        return cuts, 0.5 * (s(GL, HL) + s(G - GL, H - HL) - s(G, H)) - gam

    scans = [scan(c) for c in range(3)]
    best_col = int(np.argmax([sc[1].max() for sc in scans]))
    best_cut = scans[best_col][0][int(np.argmax(scans[best_col][1]))]

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})

    ax = axes[0]
    jitter = rng.uniform(-0.35, 0.35, n)
    for lab, col in ((0.0, RED), (1.0, BLUE)):
        m = y == lab
        ax.scatter(X[m, best_col], jitter[m], s=34, color=col, zorder=3)
    for c in scans[best_col][0]:
        ax.plot([c, c], [-0.90, -0.72], color=GREY, lw=0.9, zorder=1)
    ax.axvline(best_cut, color=GREEN, lw=2.2, zorder=2)
    ax.text(0.005, -0.55, "one candidate per midpoint  \u2193", fontsize=10.5,
            color=GREY, va="center")
    ax.text(0.005, 0.62, r"$y=0$: $g=+0.5$", fontsize=11, color=RED, va="center")
    ax.text(0.155, 0.62, r"$y=1$: $g=-0.5$", fontsize=11, color=BLUE, va="center")
    ax.set_ylim(-1.0, 0.85)
    ax.set_yticks([])
    ax.set_title(rf"the $n-1$ candidate thresholds of one feature "
                 rf"(here $x_{best_col + 1}$), and the winner", fontsize=13)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    for c, (cuts, gains) in enumerate(scans):
        win = c == best_col
        ax.step(cuts, gains, where="mid", color=GREEN if win else GREY,
                lw=2.4 if win else 1.6, zorder=3 if win else 2,
                label=rf"$x_{c + 1}$" + ("  (best feature)" if win else ""))
    ax.axhline(0, color=INK, lw=1.1)
    ax.axvline(best_cut, color=GREEN, lw=2.2, ls=":", zorder=1)
    top = scans[best_col][1].max()
    ax.plot([best_cut], [top], "*", color=GREEN, ms=17, zorder=5)
    ax.annotate("largest gain over all features and all thresholds:\n"
                rf"split this node on  $x_{best_col + 1}<{best_cut:.2f}$",
                (best_cut, top), (best_cut + 0.045, top * 1.16), fontsize=12,
                color=GREEN, linespacing=1.5, va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))
    ax.set_ylim(-2.2, top * 1.35)
    ax.text(0.005, -1.6, r"a gain below 0 buys less than the leaf costs "
                         r"($\gamma$), so that split is not made",
            fontsize=11, color=GREY)
    ax.set_xlabel("threshold")
    ax.set_ylabel("gain of that split")
    ax.legend(frameon=False, fontsize=11.5, loc="upper left", ncol=3)
    save(fig, out, "xgb_split_search")




# --------------------------------------------------------------- 14. LightGBM
# Same objective, same gain, same leaf value as the XGBoost slides — these four
# figures are only about what LightGBM does to make the scan cheap enough to run
# on far more rows, so each one is drawn against the exact-scan baseline.

def _bin_demo(n_bins=8):
    """The shared node, binned: edges, per-row bin index, and the bin totals."""
    X, y, g, h = _split_demo()
    xs = X[:, 0]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, n_bins - 1)
    return xs, y, g, h, edges, idx


def _bin_bars(ax, edges, idx, g, rows=None, colour_by_sign=True, colour=None):
    """Draw the sum-of-g histogram LightGBM stores, one bar per bin."""
    n_bins = len(edges) - 1
    rows = np.ones(len(idx), bool) if rows is None else rows
    for b in range(n_bins):
        m = (idx == b) & rows
        c = (edges[b] + edges[b + 1]) / 2
        tot = g[m].sum()
        col = colour if colour else (BLUE if tot < 0 else RED)
        ax.bar(c, tot, width=(edges[1] - edges[0]) * 0.82, color=col, alpha=0.8)
    for e_ in edges:
        ax.axvline(e_, color=FAINT, lw=1.3, zorder=0)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlim(0, 1)
    ax.spines["left"].set_visible(False)


def fig_lgbm_bins(out: Path) -> None:
    """Step 1 of histogram split finding: bin the feature, once, up front."""
    xs, y, g, h, edges, idx = _bin_demo()
    n_bins = len(edges) - 1

    fig, axd = plt.subplot_mosaic([["strip", "store"], ["hist", "store"]],
                                  figsize=WIDE, height_ratios=[1.0, 1.9],
                                  width_ratios=[1.6, 1.0])

    ax = axd["strip"]
    rng = np.random.default_rng(11)
    ax.scatter(xs, rng.uniform(-0.4, 0.4, xs.size), s=30,
               color=[BLUE if v else RED for v in (y == 1)], zorder=3)
    for e_ in edges:
        ax.axvline(e_, color=FAINT, lw=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"the same 40 rows, sorted once and cut into {n_bins} bins",
                 fontsize=13)

    ax = axd["hist"]
    _bin_bars(ax, edges, idx, g)
    for b in range(n_bins):
        m = idx == b
        c = (edges[b] + edges[b + 1]) / 2
        ax.text(c, 0.25 if g[m].sum() < 0 else -0.25, f"n={m.sum()}",
                fontsize=10, color=GREY, ha="center", va="center")
    ax.set_xlabel("feature value")
    ax.set_ylabel(r"$\sum g$ in the bin")
    ax.set_title("one pass over the rows fills the histogram", fontsize=13)

    ax = axd["store"]
    blank(ax)
    ax.set_title("keep per bin what the scan will need:", fontsize=13)
    cols = (0.14, 0.42, 0.66, 0.88)
    for x, head in zip(cols, ("bin", r"$\sum g$", r"$\sum h$", r"$n$")):
        ax.text(x, 0.73, head, fontsize=12.5, color=INK, ha="center")
    ax.plot([0.04, 0.96], [0.68, 0.68], color=GREY, lw=1.1)
    for k, b in enumerate((0, 1, 2)):
        m = idx == b
        yy = 0.57 - 0.10 * k
        for x, v in zip(cols, (f"{b}", f"{g[m].sum():+.2f}",
                               f"{h[m].sum():.2f}", f"{m.sum()}")):
            ax.text(x, yy, v, fontsize=12, ha="center", va="center",
                    color=INK if x == cols[0] else GREY)
    ax.text(0.5, 0.24, r"$\vdots$", fontsize=14, ha="center")
    save(fig, out, "lgbm_bins")


def fig_lgbm_prefix(out: Path) -> None:
    """Why a histogram is enough: both running totals, and the gain they feed.

    Everything on this figure is computed from the same 40-row node the rest of
    the section uses, so the +6.05 at the bottom is literally the peak of the
    binned scan on the next slide.
    """
    xs, y, g, h, edges, idx = _bin_demo()
    n_bins = len(edges) - 1
    gb = np.array([g[idx == b].sum() for b in range(n_bins)])
    hb = np.array([h[idx == b].sum() for b in range(n_bins)])
    cg, ch = np.cumsum(gb), np.cumsum(hb)
    k = 4                                        # the cut the scan picks
    GL, HL, G, H = cg[k], ch[k], gb.sum(), hb.sum()
    GR, HR = G - GL, H - HL
    s = lambda a, b: a ** 2 / (b + LAM)
    sL, sR, sN = s(GL, HL), s(GR, HR), s(G, H)
    gain = 0.5 * (sL + sR - sN) - GAM

    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    blank(ax)
    cx = np.linspace(0.245, 0.94, n_bins)
    w = 0.072

    rows = ((0.90, r"$\sum g$", gb, "{:+.1f}", INK, False),
            (0.79, r"$\sum h$", hb, "{:.2f}", INK, False),
            (0.64, r"$\sum g$", cg, "{:+.1f}", BLUE, True),
            (0.53, r"$\sum h$", ch, "{:.2f}", PURPLE, True))
    for ypos, label, vals, fmt, col, cumulative in rows:
        ax.text(0, ypos, label, fontsize=11.5, color=col, ha="right",
                va="center")
        for b, x in enumerate(cx):
            on = cumulative and b == k
            box(ax, (x, ypos), w, 0.085, fmt.format(vals[b]),
                ec=GREEN if on else (GREY if cumulative else col),
                fs=10.5, lw=2.4 if on else 1.3)
    for b, x in enumerate(cx):
        ax.text(x, 0.96, f"{b}", fontsize=9.5, color=GREY, ha="center")
    ax.text(0.233, 0.96, "bin", fontsize=9.5, color=GREY, ha="right")

    # the cut: everything left of it is the left child
    xcut = (cx[k] + cx[k + 1]) / 2
    ax.plot([xcut, xcut], [0.46, 0.99], color=GREEN, lw=2.2, zorder=5)
    ax.text(xcut, 0.435, "cut here", fontsize=11, color=GREEN, ha="center",
            va="top")

    ax.text(0.32, 0.36, rf"$G_L={GL:+.1f}$,  $H_L={HL:.2f}$", fontsize=13,
            color=GREEN, ha="center", va="center")
    ax.text(0.75, 0.36, rf"$G_R=G-G_L={GR:+.1f}$,  $H_R=H-H_L={HR:.2f}$",
            fontsize=13, color=GREEN, ha="center", va="center")
    ax.text(0.5, 0.27, rf"whole node:  $G={G:+.1f}$,  $H={H:.2f}$   "
                       rf"(the last cell of each running row)",
            fontsize=11.5, color=GREY, ha="center", va="center")

    ax.plot([0.03, 0.97], [0.215, 0.215], color=FAINT, lw=1.4)
    ax.text(0.5, 0.145,
            r"gain $=\frac{1}{2}\left[\,s(G_L,H_L)+s(G_R,H_R)-s(G,H)\,\right]"
            rf"-\gamma$,      $s(G,H)=\frac{{G^2}}{{H+\lambda}}$,  "
            rf"$\lambda={LAM:g}$, $\gamma={GAM:g}$",
            fontsize=12.5, color=INK, ha="center", va="center")
    ax.text(0.5, 0.065,
            rf"$=\frac{{1}}{{2}}\left[\frac{{{GL:.1f}^2}}{{{HL:.2f}+1}}"
            rf"+\frac{{({GR:.1f})^2}}{{{HR:.2f}+1}}"
            rf"-\frac{{{G:.1f}^2}}{{{H:.1f}+1}}\right]-{GAM:g}"
            rf"=\frac{{1}}{{2}}\left[{sL:.2f}+{sR:.2f}-{sN:.2f}\right]-{GAM:g}"
            rf"={gain:+.2f}$",
            fontsize=13.5, color=INK, ha="center", va="center")
    ax.set_title("one sweep fills both running rows; every candidate is one "
                 "column of them", fontsize=13)
    save(fig, out, "lgbm_prefix")


def fig_lgbm_binned_scan(out: Path) -> None:
    """Step 2: the scan now walks bin edges, and lands next door to the exact one."""
    xs, y, g, h, edges, idx = _bin_demo()
    inner = edges[1:-1]
    srt = np.sort(xs)
    mids = (srt[:-1] + srt[1:]) / 2
    gain_exact, gain_bin = _gain_at(xs, g, h, mids), _gain_at(xs, g, h, inner)
    best_exact = mids[int(np.argmax(gain_exact))]
    best_bin = inner[int(np.argmax(gain_bin))]

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.step(mids, gain_exact, where="mid", color=GREY, lw=1.9,
            label=f"exact scan: {mids.size} candidates, one per midpoint")
    ax.step(inner, gain_bin, where="mid", color=GREEN, lw=2.8,
            label=f"binned scan: {inner.size} candidates, one per bin edge")
    ax.plot(inner, gain_bin, "o", color=GREEN, ms=7, zorder=4)
    # the two winners are one bin apart, so the labels lean away from each other
    for x_, col, lab, ha, dx in (
            (best_exact, GREY, f"exact winner  {best_exact:.2f}", "right", -0.012),
            (best_bin, GREEN, f"{best_bin:.2f}  binned winner", "left", 0.012)):
        ax.axvline(x_, color=col, lw=1.4, ls=":")
        ax.text(x_ + dx, -1.9, lab, fontsize=12, color=col, ha=ha, va="top")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("threshold")
    ax.set_ylabel("gain")
    ax.set_ylim(-3.0, gain_exact.max() * 1.18)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title("The exact winner is not on the grid, so the scan returns the edge beside it\n"
                 "Example drawn with 8 bins; at LightGBM's 255 the gap is invisible",
                 fontsize=13)
    save(fig, out, "lgbm_binned_scan")


def fig_lgbm_subtract(out: Path) -> None:
    """Step 3: the free half of the work — one child comes from a subtraction."""
    xs, y, g, h, edges, idx = _bin_demo()
    cut = edges[5]                                   # the split chosen above
    right = xs >= cut

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    panels = ((np.ones_like(right), "the parent's histogram\nchose this split", GREY),
              (right, "right child", BLUE),
              (~right, "left child", GREEN))
    for ax, (rows, title, col) in zip(axes, panels):
        _bin_bars(ax, edges, idx, g, rows=rows.astype(bool), colour=col)
        ax.axvline(cut, color=INK, lw=1.6, ls="--")
        ax.set_title(title, fontsize=12.5, color=col, linespacing=1.5)
        ax.set_xlabel("feature value")
    axes[0].set_ylabel(r"$\sum g$ per bin")
    for ax, sym in zip(axes[:2], ("$-$", "$=$")):
        ax.text(1.07, 0.5, sym, transform=ax.transAxes, fontsize=26,
                color=INK, ha="center", va="center")
    save(fig, out, "lgbm_subtract")


def fig_lgbm_leafwise(out: Path) -> None:
    """Level-wise vs leaf-wise: same candidates, same gain, different next node."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    def draw(ax, nodes, edges, leaves, title, caption):
        blank(ax)
        for u, v, gain in edges:
            # a split worth almost nothing is drawn in red, not green
            colour = GREY if not gain else (GREEN if gain > 1 else RED)
            ax.plot(*zip(nodes[u], nodes[v]), color=colour, lw=2.4, zorder=1)
            if gain:
                ax.text((nodes[u][0] + nodes[v][0]) / 2 - 0.055,
                        (nodes[u][1] + nodes[v][1]) / 2, f"{gain}",
                        fontsize=12.5, color=colour, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none"))
        for k, xy in nodes.items():
            ax.plot(*xy, "o", ms=15, mfc="white", zorder=3,
                    mec=BLUE if k in leaves else GREY, mew=1.9)
        ax.set_title(title, fontsize=13.5)
        ax.set_xlabel(caption, fontsize=12, linespacing=1.6, color=INK)

    draw(axes[0],
         {"r": (0.50, 0.90), "a": (0.26, 0.58), "b": (0.74, 0.58),
          "l1": (0.10, 0.26), "l2": (0.40, 0.26),
          "l3": (0.60, 0.26), "l4": (0.90, 0.26)},
         (("r", "a", 4.8), ("r", "b", None), ("a", "l1", 2.1),
          ("a", "l2", None), ("b", "l3", 0.3), ("b", "l4", None)),
         {"l1", "l2", "l3", "l4"},
         "level-wise  (XGBoost default)",
         "finishes a level before starting the next")

    draw(axes[1],
         {"r": (0.50, 0.90), "a": (0.26, 0.58), "b": (0.80, 0.58),
          "c": (0.08, 0.26), "d": (0.44, 0.26),
          "e": (0.28, -0.06), "f": (0.62, -0.06)},
         (("r", "a", 4.8), ("r", "b", None), ("a", "c", 4.1),
          ("a", "d", None), ("d", "e", 2.2), ("d", "f", None)),
         {"b", "c", "e", "f"},
         "leaf-wise / best-first  (LightGBM)",
         "keeps a queue of leaves and always splits the one\n"
         "offering the most, anywhere in the tree")
    for ax in axes:
        ax.set_ylim(-0.22, 1.02)
    save(fig, out, "lgbm_leafwise")


def fig_lgbm_goss(out: Path) -> None:
    """GOSS: keep the big gradients, subsample the rest, reweight, same gain."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.35, n)
    g, h = -y, np.ones(n)                     # squared loss, F = 0 so far
    a, b = 0.2, 0.1                           # top_rate, other_rate

    order = np.argsort(-np.abs(g))
    n_top = int(a * n)
    top, rest = order[:n_top], order[n_top:]
    samp = rng.choice(rest, int(b * rest.size), replace=False)
    keep = np.concatenate([top, samp])
    w = np.concatenate([np.ones(n_top), np.full(samp.size, (1 - a) / b)])

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    mag = np.abs(g)[order]
    ax.fill_between(np.arange(n_top), mag[:n_top], color=GREEN, alpha=0.30)
    ax.fill_between(np.arange(n_top, n), mag[n_top:], color=FAINT)
    ax.plot(np.arange(n), mag, color=GREY, lw=1.2)
    ax.axvline(n_top, color=INK, lw=1.2, ls=":")
    ax.text(n_top * 0.5, mag.max() * 0.30, f"keep all\ntop {a:.0%}", fontsize=12,
            color=GREEN, ha="center", va="center", linespacing=1.5)
    ax.text(n * 0.62, mag.max() * 0.62,
            f"sample {b:.0%} of the rest,\n"
            rf"and scale their $g,h$ by {int((1 - a) / b)}$",
            fontsize=12, color=GREY, ha="center", va="center", linespacing=1.5)
    ax.set_xlabel("rows, sorted by how badly they are still fitted")
    ax.set_ylabel(r"$|g_i|$")
    ax.set_title(f"Use {keep.size} rows instead of {n}", fontsize=13)

    ax = axes[1]
    cuts = np.linspace(0.03, 0.97, 60)
    ax.plot(cuts, _gain_at(x, g, h, cuts), color=GREY, lw=3.0,
            label="gain on all 2000 rows")
    ax.plot(cuts, _gain_at(x[keep], g[keep] * w, h[keep] * w, cuts),
            color=GREEN, lw=2.0, ls="--",
            label=f"gain on the {keep.size} GOSS rows")
    ax.set_xlabel("threshold")
    ax.set_ylabel("gain")
    ax.legend(frameon=False, fontsize=11.5, loc="lower center")
    ax.set_title("the reweighting is what keeps the two curves together",
                 fontsize=13)
    save(fig, out, "lgbm_goss")


# -------------------------------------------------------------- 15. CatBoost
# The categorical story, measured: a category that carries no signal at all, so
# anything the encoded feature seems to know about y is leakage and nothing else.

def _cat_demo(n=600, levels=200, seed=5):
    """High-cardinality category, independent of the label — pure noise by design."""
    rng = np.random.default_rng(seed)
    cat = np.repeat(np.arange(levels), n // levels)
    rng.shuffle(cat)
    y = rng.integers(0, 2, n).astype(float)
    cat_te = rng.integers(0, levels, n)
    y_te = rng.integers(0, 2, n).astype(float)
    return cat, y, cat_te, y_te


def _greedy_ts(cat, y, levels=200):
    """Mean target per category, computed on all rows — the row's own y included."""
    tot = np.bincount(cat, weights=y, minlength=levels)
    cnt = np.bincount(cat, minlength=levels)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), y.mean())


def _ordered_ts(cat, y, a=1.0, levels=200, seed=0):
    """CatBoost's ordered target statistic: only the rows before this one count."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    prior = y.mean()
    tot = np.zeros(levels)
    cnt = np.zeros(levels)
    enc = np.empty(len(y))
    for i in order:                      # one pass in permutation order
        c = cat[i]
        enc[i] = (tot[c] + a * prior) / (cnt[c] + a)
        tot[c] += y[i]                   # the row joins the history only after
        cnt[c] += 1
    return enc, order


def _enc_rate(ax, enc, y, title, note=None, colour=BLUE, counts=True,
              base_label=True):
    """Share of positives at each level of the encoded feature.

    A histogram of the encoded value hides the leak; what a tree would actually
    exploit is P(y=1 | encoding), so plot that. Flat at the base rate = the
    feature knows nothing.
    """
    edges = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(enc, edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        # the bar sits at the mean encoding in the bin, not the bin centre, so
        # it can be compared with the diagonal it is supposed to lie on
        c = enc[m].mean()
        ax.bar(c, y[m].mean(), width=0.13, color=colour, alpha=0.8)
        if counts:
            ax.text(c, y[m].mean() + 0.03, f"n={m.sum()}", fontsize=10,
                    color=GREY, ha="center")
        elif y[m].mean() == 0:      # a zero-height bar reads as a missing one
            ax.text(c, 0.02, "0%", fontsize=10, color=colour, ha="center")
    ax.axhline(y.mean(), color=INK, lw=1.4, ls="--")
    if base_label:
        ax.text(-0.06, y.mean() + 0.03, "base rate", fontsize=11, color=INK,
                ha="left")
    ax.set_xlim(-0.09, 1.09)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("value of the encoded feature")
    ax.set_ylabel(r"share of $y=1$ among those rows")
    ax.set_title(title, fontsize=13)
    if note:
        ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=12,
                ha="left", va="top", color=INK, linespacing=1.5)


# the five-row worked example the CatBoost slides open on: small enough that the
# leak can be computed in your head, and reused by the ordered-permutation slide
FIVE_ROWS = ((1, "France", 1), (2, "France", 1), (3, "France", 0),
             (4, "Germany", 0), (5, "Germany", 0))
FIVE_COLOUR = {"France": BLUE, "Germany": PURPLE}
FIVE_PRIOR = sum(r[2] for r in FIVE_ROWS) / len(FIVE_ROWS)      # 0.4


def fig_cat_example(out: Path) -> None:
    """Target encoding on five rows: every row's own answer is inside its feature."""
    naive = {"France": 2 / 3, "Germany": 0.0}

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    blank(ax)
    ax.set_title("five customers, one categorical column", fontsize=13)
    cols = (0.14, 0.42, 0.66, 0.90)
    for x, head in zip(cols, ("Customer", "Country", "Buy",
                              "target\nencoding")):
        ax.text(x, 0.86, head, fontsize=12, color=GREY, ha="center",
                va="center", linespacing=1.4)
    ax.plot([0.03, 0.99], [0.76, 0.76], color=GREY, lw=1.1)
    for k, (cid, country, buy) in enumerate(FIVE_ROWS):
        y = 0.66 - 0.125 * k
        leaks = country == "Germany"
        ax.text(cols[0], y, f"{cid}", fontsize=12.5, ha="center", color=INK)
        ax.text(cols[1], y, country, fontsize=12.5, ha="center",
                color=FIVE_COLOUR[country])
        ax.text(cols[2], y, f"{buy}", fontsize=12.5, ha="center",
                color=BLUE if buy else RED)
        ax.text(cols[3], y, f"{naive[country]:.2f}", fontsize=12.5, ha="center",
                color=RED if leaks else INK,
                bbox=dict(boxstyle="round,pad=0.18", ec="none",
                          fc="#fdecec" if leaks else "white"))
    ax.text(0.5, 0.03, "each row's own Buy is one of the values averaged into "
                       "its own encoding",
            fontsize=12, color=RED, ha="center", va="center")

    ax = axes[1]
    blank(ax)
    ax.set_title("the naive calculation", fontsize=13)
    ax.text(0.5, 0.84, r"$P(\mathrm{Buy}=1\mid\mathrm{France})=\dfrac{2}{3}$",
            fontsize=18, ha="center", va="center", color=BLUE)
    ax.text(0.5, 0.60, r"$P(\mathrm{Buy}=1\mid\mathrm{Germany})=\dfrac{0}{2}=0$",
            fontsize=18, ha="center", va="center", color=PURPLE)
    ax.plot([0.06, 0.94], [0.46, 0.46], color=FAINT, lw=1.4)
    ax.text(0.5, 0.34,
            "both German customers are handed 0.00\n"
            "which is exactly, their answer",
            fontsize=13, color=RED, ha="center", va="center", linespacing=1.5)
    save(fig, out, "cat_example")


def fig_cat_chain(out: Path) -> None:
    """The ordered permutation: encode each row from the prefix that precedes it."""
    # one random ordering of the five customers, walked left to right
    order = (3, 1, 4, 2, 5)
    by_id = {cid: (country, buy) for cid, country, buy in FIVE_ROWS}

    fig, axd = plt.subplot_mosaic([["chain"], ["table"]], figsize=(12.6, 5.4),
                                  height_ratios=[1.0, 1.5])

    ax = axd["chain"]
    blank(ax)
    ax.set_title("draw a random ordering, then walk it", fontsize=13)
    xs = np.linspace(0.10, 0.90, 5)
    for k, (x, cid) in enumerate(zip(xs, order)):
        country, buy = by_id[cid]
        box(ax, (x, 0.66), 0.115, 0.34, "ABCDE"[k], ec=FIVE_COLOUR[country],
            fs=15)
        ax.text(x, 0.32, f"customer {cid}\n{country}, Buy={buy}", fontsize=10.5,
                color=GREY, ha="center", va="center", linespacing=1.4)
        if k < 4:
            arrow(ax, (x + 0.062, 0.66), (xs[k + 1] - 0.062, 0.66), color=GREY)
    # the bracket sits under A-C, where there is room, rather than over the title
    ax.plot([xs[0] - 0.05, xs[2] + 0.05], [0.10, 0.10], color=GREEN, lw=1.8)
    for x_ in (xs[0] - 0.05, xs[2] + 0.05):
        ax.plot([x_, x_], [0.10, 0.14], color=GREEN, lw=1.8)
    ax.annotate("these three, and only these, are what D is encoded from",
                (xs[3], 0.14), ((xs[0] + xs[2]) / 2, -0.02), fontsize=12,
                color=GREEN, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))

    ax = axd["table"]
    blank(ax)
    cols = (0.07, 0.20, 0.34, 0.46, 0.68, 0.88)
    heads = ("", "customer", "country", "Buy",
             "earlier rows of the same country",
             "encoding")
    for x, head in zip(cols, heads):
        ax.text(x, 0.90, head, fontsize=11.5, color=GREY, ha="center")
    ax.plot([0.02, 0.98], [0.81, 0.81], color=GREY, lw=1.1)
    seen = {}
    for k, cid in enumerate(order):
        country, buy = by_id[cid]
        hist = seen.get(country, [])
        val = (sum(hist) + FIVE_PRIOR) / (len(hist) + 1)
        y = 0.70 - 0.15 * k
        ax.text(cols[0], y, "ABCDE"[k], fontsize=12.5, ha="center",
                color=FIVE_COLOUR[country])
        ax.text(cols[1], y, f"{cid}", fontsize=12, ha="center", color=INK)
        ax.text(cols[2], y, country, fontsize=12, ha="center",
                color=FIVE_COLOUR[country])
        ax.text(cols[3], y, f"{buy}", fontsize=12, ha="center",
                color=BLUE if buy else RED)
        ax.text(cols[4], y, "none yet — use the prior" if not hist else
                "Buy = " + ", ".join(str(v) for v in hist),
                fontsize=11.5, ha="center", color=GREY)
        num = f"{sum(hist)} + {FIVE_PRIOR:g}"
        den = f"{len(hist)} + 1"
        ax.text(cols[5], y, rf"$\frac{{{num}}}{{{den}}}={val:.2f}$",
                fontsize=13, ha="center", color=GREEN)
        seen.setdefault(country, []).append(buy)
    ax.text(0.5, -0.05,
            r"prior $p=0.4$ (the base rate), weight $a=1$, and the row's own Buy is never in its own numerator",
            fontsize=12, color=INK, ha="center", va="center")
    save(fig, out, "cat_chain")


def fig_cat_leak_why(out: Path) -> None:
    """Why the ramp is arithmetic, not luck: your own label is one of the three."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [0.95, 1.25]})

    ax = axes[0]
    blank(ax)
    ax.set_title("one country with three customers", fontsize=13)
    ys = (1, 0, 1)
    for k, buy in enumerate(ys):
        y = 0.80 - 0.13 * k
        ax.text(0.14, y, f"customer {k + 1}", fontsize=12, color=GREY,
                ha="left", va="center")
        ax.text(0.66, y, f"Buy = {buy}", fontsize=12.5, va="center",
                color=BLUE if buy else RED)
    ax.plot([0.10, 0.90], [0.46, 0.46], color=GREY, lw=1.2)
    ax.text(0.5, 0.34, r"encoding $=\dfrac{1+0+1}{3}=\dfrac{2}{3}$", fontsize=17,
            ha="center", va="center")
    ax.text(0.5, 0.13,
            "all three are given 0.67\n" \
            "each of them contributed to the average",
            fontsize=12, color=INK, ha="center", va="center", linespacing=1.5)

    ax = axes[1]
    blank(ax)
    ax.set_title("the encoding tells you about your own Buy", fontsize=13)
    ax.text(0.5, 0.93, r"encoding $=\dfrac{\mathbf{y_i}+y_j+y_k}{3}$"
                       "     with $y_i$ = MY OWN Buy",
            fontsize=13, ha="center", va="center", color=INK)
    cols = (0.11, 0.37, 0.62, 0.88)
    heads = ("encoding", "how many of\nthe 3 bought", "so my own\nBuy",
             r"$P(\mathrm{Buy}{=}1)$")
    for x, head in zip(cols, heads):
        ax.text(x, 0.72, head, fontsize=11, color=GREY, ha="center",
                va="center", linespacing=1.4)
    ax.plot([0.02, 0.98], [0.60, 0.60], color=GREY, lw=1.1)
    rows = ((0.0, 0, "must be 0", RED),
            (1 / 3, 1, "0 or 1", GREY),
            (2 / 3, 2, "0 or 1", GREY),
            (1.0, 3, "must be 1", BLUE))
    for k, (enc, m, verdict, col) in enumerate(rows):
        y = 0.50 - 0.115 * k
        ax.text(cols[0], y, f"{enc:.2f}", fontsize=12.5, ha="center", color=INK)
        ax.text(cols[1], y, f"{m}", fontsize=12, ha="center", color=GREY)
        ax.text(cols[2], y, verdict, fontsize=12, ha="center", color=col)
        ax.text(cols[3], y, f"{m / 3:.2f}", fontsize=12.5, ha="center",
                color=GREEN)
    ax.text(0.5, 0.03,
            "When the last two columns are the same number\n"
            r"$P(\mathrm{Buy}=1\mid \mathrm{encoding})=\mathrm{encoding}$,"
            "\nthe feature is a readout of the label it was built from",
            fontsize=12, color=INK, ha="center", va="center", linespacing=1.5)
    save(fig, out, "cat_leak_why")


def fig_cat_leak(out: Path) -> None:
    """The predicted ramp, measured — and gone the moment the rows are new."""
    cat, y, cat_te, y_te = _cat_demo()
    enc_map = _greedy_ts(cat, y)

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    _enc_rate(axes[0], enc_map[cat], y, "on the 600 rows it was computed from",
              "every bar lands on the diagonal:\nthe arithmetic was not a guess",
              RED)
    _enc_rate(axes[1], enc_map[cat_te], y_te, "on 600 rows it has never seen",
              "flat at the base rate:\nthe feature knows nothing", GREY)
    # the identity line is what the previous slide predicts, so draw it
    axes[0].plot([0, 1], [0, 1], color=INK, lw=1.6, ls=":", zorder=1)
    axes[0].text(0.31, 0.44, r"$P=$ encoding", fontsize=11.5, color=INK,
                 rotation=30, ha="center")
    fig.suptitle("200 countries, 3 customers each, and Buy decided by a coin "
                 "flip — there is nothing here to learn",
                 fontsize=13.5, y=1.02)
    save(fig, out, "cat_leak")


def fig_cat_ordered_ts(out: Path) -> None:
    """What "the leak is gone" means: a training row now looks like a fresh row.

    Three panels of the SAME statistic, so the only thing that varies is how the
    encoding was computed. The third panel is the target, not a result: at
    prediction time both approaches encode a new row from the whole training
    table, so that panel is what a leak-free feature has always looked like.
    """
    cat, y, cat_te, y_te = _cat_demo()
    greedy_map = _greedy_ts(cat, y)
    ordered, _ = _ordered_ts(cat, y)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9), sharey=True)
    panels = (
        (axes[0], greedy_map[cat], y, RED,
         "GREEDY encoding,\non the training rows",
         "each row helped build\nits own encoding"),
        (axes[1], ordered, y, GREEN,
         "ORDERED encoding,\non the same training rows",
         "no row is in its own\nencoding any more"),
        (axes[2], greedy_map[cat_te], y_te, GREY,
         "either encoding,\non fresh rows",
         "new rows"),
    )
    for ax, enc, yy, col, title, note in panels:
        _enc_rate(ax, enc, yy, "", colour=col, counts=False,
                  base_label=ax is axes[0])
        ax.set_title(title, fontsize=12.5, color=col, linespacing=1.5)
        r = np.corrcoef(enc, yy)[0, 1]
        ax.text(0.5, 1.05, rf"corr(encoding, $y$) $= {r:+.2f}$", fontsize=12.5,
                color=col, ha="center", va="center")
        ax.text(0.5, 0.90, note, fontsize=11, color=INK, ha="center",
                va="center", linespacing=1.45)
        ax.set_xlabel("value of the encoded feature", fontsize=11)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle("read one bar: of the rows whose encoding lands there, what "
                 "share actually bought?\n"
                 "a feature that knows nothing about a row keeps every bar on "
                 "the base rate",
                 fontsize=13, y=1.14, linespacing=1.5)
    fig.text(0.5, -0.06,
             "the fix makes a TRAINING row look like the fresh row",
             fontsize=12.5, color=INK, ha="center", va="top")
    save(fig, out, "cat_ordered_ts")


def _shift_demo(depth=8, n=300, seed=3):
    """Residual of every row, measured with and without that row in the fit.

    A true leave-one-out refit, not cross-validation: the claim is about ONE row
    being inside the fit, so the demonstration should remove exactly one row.
    """
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + rng.normal(0, 0.35, n)
    mk = lambda: DecisionTreeRegressor(max_depth=depth, random_state=0)
    fitted = mk().fit(x, y).predict(x)
    left_out = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        left_out[i] = mk().fit(x[m], y[m]).predict(x[i:i + 1])[0]
    return y, fitted, left_out


def fig_cat_shift(out: Path) -> None:
    """One row first, then all of them: the residual you measure is too small."""
    y, fitted, left_out = _shift_demo()
    r_in, r_out = np.abs(y - fitted), np.abs(y - left_out)
    ratio = r_out / np.maximum(r_in, 1e-9)
    i = int(np.argmin(np.abs(ratio - r_out.mean() / r_in.mean())))   # typical row

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    blank(ax)
    ax.set_title("one row, and the two models that could score it", fontsize=13)
    ax.text(0.5, 0.90, rf"row $i$:   its true value is $y_i={y[i]:+.2f}$",
            fontsize=13, ha="center", va="center", color=INK)
    panels = ((0.25, RED, "the model was fitted\nWITH row $i$",
               fitted[i], y[i] - fitted[i]),
              (0.77, GREEN, "the model never\nSAW row $i$",
               left_out[i], y[i] - left_out[i]))
    for x0, col, label, pred, res in panels:
        ax.text(x0, 0.72, label, fontsize=12, color=col, ha="center",
                va="center", linespacing=1.5)
        box(ax, (x0, 0.50), 0.38, 0.20,
            rf"predicts ${pred:+.2f}$" "\n" rf"residual $= {res:+.2f}$",
            ec=col, fs=12.5)
    ax.text(0.51, 0.50, "vs", fontsize=12.5, color=GREY, ha="center",
            va="center")
    ax.text(0.5, 0.27,
            rf"the same row asks for a correction {ratio[i]:.1f}$\times$ larger"
            "\nonce the model has not already memorised it",
            fontsize=12.5, color=INK, ha="center", va="center", linespacing=1.5)
    ax.text(0.5, 0.06, "and boosting fits the next tree to exactly that number",
            fontsize=12, color=GREY, ha="center", va="center")

    ax = axes[1]
    bins = np.linspace(0, 1.2, 30)
    ax.hist(r_in, bins=bins, color=RED, alpha=0.65,
            label=f"fitted with the row   (mean {r_in.mean():.2f})")
    ax.hist(r_out, bins=bins, color=GREEN, alpha=0.55,
            label=f"row left out            (mean {r_out.mean():.2f})")
    ax.axvline(r_in.mean(), color=RED, lw=2.0, ls="--")
    ax.axvline(r_out.mean(), color=GREEN, lw=2.0, ls="--")
    ax.set_xlabel(r"$|$residual$|$")
    ax.set_ylabel("rows")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title(f"all {len(y)} rows: every one is biased the same way\n"
                 f"({100 * (1 - r_in.mean() / r_out.mean()):.0f}% too small "
                 "on average)", fontsize=12.5)
    save(fig, out, "cat_shift")


def fig_cat_shift_why(out: Path) -> None:
    """Why a smaller gradient is a problem at all: the shrink is not uniform.

    The obvious objection to the previous slide is that a uniformly smaller
    gradient is just a smaller learning rate. So measure the shrink separately
    for rows the tree has memorised and rows it has not: it is 100% in leaves
    holding one row and 22% in well-populated ones, which is a distortion of the
    residual SHAPE that no learning rate can undo.
    """
    from sklearn.tree import DecisionTreeRegressor

    y, fitted, left_out = _shift_demo()
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, (len(y), 1))          # same draw as _shift_demo
    tree = DecisionTreeRegressor(max_depth=8, random_state=0).fit(x, y)
    leaf = tree.apply(x)
    size = np.array([np.sum(leaf == l) for l in leaf])
    r_in, r_out = np.abs(y - fitted), np.abs(y - left_out)

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    blank(ax)
    ax.set_title("first, the objection", fontsize=13)
    box(ax, (0.5, 0.80), 0.96, 0.24,
        "if every gradient were simply 56% smaller,\n"
        r"that would be harmless — it is a smaller $\eta$,"
        "\nand $\\eta$ is already a parameter we set", ec=GREY, fs=12.5)
    arrow(ax, (0.5, 0.66), (0.5, 0.57), color=GREY)
    box(ax, (0.5, 0.44), 0.96, 0.22,
        "but the shrink is NOT uniform:\nit is largest exactly where the model "
        "memorised", ec=RED, fs=12.5)
    arrow(ax, (0.5, 0.32), (0.5, 0.23), color=GREY)
    ax.text(0.5, 0.11,
            "so the next tree is fitted to a residual surface\n"
            "with the wrong SHAPE, not merely the wrong scale —\n"
            r"and no value of $\eta$ can undo a wrong direction",
            fontsize=12.5, color=INK, ha="center", va="center", linespacing=1.55)

    ax = axes[1]
    buckets = ((1, 1, "alone in\nits leaf"), (2, 2, "2 rows"),
               (3, 4, "3-4 rows"), (5, 10 ** 6, "5+ rows"))
    xs = np.arange(len(buckets))
    rep = [r_in[(size >= lo) & (size <= hi)].mean() for lo, hi, _ in buckets]
    hon = [r_out[(size >= lo) & (size <= hi)].mean() for lo, hi, _ in buckets]
    ax.bar(xs - 0.19, rep, width=0.36, color=RED, alpha=0.8,
           label="what the row reports")
    ax.bar(xs + 0.19, hon, width=0.36, color=GREEN, alpha=0.8,
           label="what it is really wrong by")
    for k, (r, h) in enumerate(zip(rep, hon)):
        ax.text(k - 0.19, r + 0.012, f"{r:.2f}", fontsize=10.5, color=RED,
                ha="center")
        ax.text(k + 0.19, h + 0.012, f"{h:.2f}", fontsize=10.5, color=GREEN,
                ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[2] for b in buckets], fontsize=11)
    ax.set_xlabel("how many training rows share this row's leaf")
    ax.set_ylabel(r"mean $|$residual$|$")
    ax.set_ylim(0, max(hon) * 1.45)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    ax.set_title("the green bars are flat; the red ones collapse where the "
                 "model memorised —\na row alone in its leaf reports 0.00 and "
                 "is wrong by 0.43", fontsize=12.5, linespacing=1.5)
    save(fig, out, "cat_shift_why")


def fig_cat_ordered_boosting(out: Path) -> None:
    """What ordered boosting IS: a prefix model per position in a permutation."""
    n = 8
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    blank(ax)
    ax.set_title("shuffle the rows once, then never let a row's own gradient "
                 "come from a model that has seen it", fontsize=13.5)

    cx = np.linspace(0.360, 0.958, n)
    w = 0.062

    ax.text(0.288, 0.86, "the permutation", fontsize=12, color=INK,
            ha="right", va="center")
    for k, x in enumerate(cx):
        box(ax, (x, 0.86), w, 0.10, f"{k + 1}", ec=GREY, fs=11)
    ax.text(0.65, 0.755,
            "green = the model was trained on these rows        "
            "red = the row being scored, never trained on",
            fontsize=11, color=GREY, ha="center", va="center")

    for r, i in enumerate((2, 4, 6, 8)):
        yy = 0.66 - 0.155 * r
        ax.text(0.288, yy, rf"row {i}:  use $M_{{{i - 1}}}$",
                fontsize=12, color=INK, ha="right", va="center")
        for k, x in enumerate(cx):
            if k < i - 1:                        # the prefix it was trained on
                box(ax, (x, yy), w, 0.10, "", fc="#e8f5e9", ec=GREEN, fs=10)
            elif k == i - 1:                     # the row being scored
                box(ax, (x, yy), w, 0.10, "?", ec=RED, fs=12, lw=2.2)
            else:
                box(ax, (x, yy), w, 0.10, "", ec=FAINT, fs=10, lw=1.0)
    ax.text(0.65, 0.02,
            r"$M_{i-1}$ is fitted on the first $i-1$ rows only, so "
            r"$g_i=\partial\ell\left(y_i,\,M_{i-1}(x_i)\right)$ is "
            "out-of-sample by construction",
            fontsize=12.5, color=INK, ha="center", va="center")
    save(fig, out, "cat_ordered_boosting")


def fig_cat_ordered_cost(out: Path) -> None:
    """Why n models per round is not what CatBoost actually runs."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.set_title("the literal version is unaffordable", fontsize=13, color=RED)
    for k in range(5):
        box(ax, (0.11 + 0.15 * k, 0.68), 0.105, 0.16,
            rf"$M_{{{k}}}$", ec=RED, fs=12)
    ax.text(0.79, 0.68, r"$\cdots$", fontsize=17, color=RED, ha="center",
            va="center")
    box(ax, (0.92, 0.68), 0.125, 0.16, r"$M_{n-1}$", ec=RED, fs=12)
    ax.text(0.5, 0.47, r"one model per position, refitted every round:",
            fontsize=12.5, color=INK, ha="center", va="center")
    ax.text(0.5, 0.34, r"$O(n^2)$  trees per boosting round", fontsize=16,
            color=RED, ha="center", va="center")
    ax.text(0.5, 0.14, "correct, and nobody could run it",
            fontsize=12, color=GREY, ha="center", va="center")

    ax = axes[1]
    blank(ax)
    ax.set_title("what CatBoost keeps instead", fontsize=13, color=GREEN)
    spans = ((0, 1, "$M_0$", "1 row"), (1, 2, "$M_1$", "2 rows"),
             (2, 4, "$M_2$", "4 rows"), (3, 8, "$M_3$", "8 rows"))
    for k, (r, size, name, sub) in enumerate(spans):
        x0 = 0.10 + 0.23 * k
        box(ax, (x0, 0.72), 0.17, 0.17, name, ec=GREEN, fs=13)
        ax.text(x0, 0.55, f"first {sub}", fontsize=10.5, color=GREY,
                ha="center", va="center")
    ax.text(0.5, 0.40,
            r"a row at position $i$ takes its gradient from "
            r"$M_{\lfloor \log_2 i\rfloor}$," "\n"
            r"whose prefix stops before $i$ — so still out-of-sample",
            fontsize=12, color=INK, ha="center", va="center", linespacing=1.5)
    ax.text(0.5, 0.20, r"$\log_2 n$  models instead of  $n$", fontsize=16,
            color=GREEN, ha="center", va="center")
    ax.text(0.5, 0.04,
            "same permutations as the categorical statistics, and several of "
            "them,\nrotated between rounds so no row is permanently early",
            fontsize=11.5, color=GREY, ha="center", va="center", linespacing=1.5)
    save(fig, out, "cat_ordered_cost")


def fig_cat_oblivious(out: Path) -> None:
    """Symmetric trees: one split per level turns the tree into a lookup table."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    blank(ax)
    ax.set_title("every node of a level uses the SAME split", fontsize=13)
    xs = {0: [0.5], 1: [0.26, 0.74], 2: [0.09, 0.38, 0.62, 0.91]}
    ys = {0: 0.88, 1: 0.62, 2: 0.36}
    for lvl in (0, 1):
        for x in xs[lvl]:
            for child in (0, 1):
                nxt = xs[lvl + 1][xs[lvl].index(x) * 2 + child]
                ax.plot([x, nxt], [ys[lvl], ys[lvl + 1]], color=GREY, lw=1.7,
                        zorder=1)
    for lvl in (0, 1, 2):
        for x in xs[lvl]:
            ax.plot(x, ys[lvl], "o", ms=13, mfc="white", mec=GREY, mew=1.7,
                    zorder=3)
    for lvl, test, col in ((0, r"$x_3<0.4$ ?", BLUE), (1, r"$x_7<2$ ?", PURPLE)):
        ax.text(0.01, ys[lvl] - 0.13, test, fontsize=12.5, color=col,
                va="center")
    ax.text(0.30, ys[0] - 0.06, "no  0", fontsize=10.5, color=GREY, ha="right")
    ax.text(0.70, ys[0] - 0.06, "1  yes", fontsize=10.5, color=GREY, ha="left")
    vals = ("+0.4", "-1.2", "+0.9", "-0.1")
    for k, x in enumerate(xs[2]):
        box(ax, (x, 0.20), 0.155, 0.13, f"[{k}]\n{vals[k]}", ec=GREEN, fs=10.5)
    ax.text(0.5, 0.03, "four leaf values stored as one array of four numbers",
            fontsize=11.5, color=GREEN, ha="center", va="center")

    ax = axes[1]
    blank(ax)
    ax.set_title("two customers, two lookups", fontsize=13)
    people = (("row A", "$x_3=0.2$", "yes $\\to 1$", "$x_7=5$", "no $\\to 0$",
               "$(1,0)_2=2$", "+0.9", 0.72),
              ("row B", "$x_3=0.9$", "no $\\to 0$", "$x_7=1$", "yes $\\to 1$",
               "$(0,1)_2=1$", "-1.2", 0.34))
    for name, t1, b1, t2, b2, idx, val, y0 in people:
        ax.text(0.02, y0, name, fontsize=12.5, color=INK, va="center")
        ax.text(0.17, y0 + 0.06, t1, fontsize=11.5, color=BLUE, va="center")
        ax.text(0.17, y0 - 0.06, t2, fontsize=11.5, color=PURPLE, va="center")
        ax.text(0.36, y0 + 0.06, b1, fontsize=11.5, color=BLUE, va="center")
        ax.text(0.36, y0 - 0.06, b2, fontsize=11.5, color=PURPLE, va="center")
        arrow(ax, (0.53, y0), (0.60, y0), color=GREY)
        ax.text(0.72, y0, idx, fontsize=12.5, color=INK, va="center",
                ha="center")
        arrow(ax, (0.82, y0), (0.88, y0), color=GREY)
        ax.text(0.95, y0, val, fontsize=13, color=GREEN, va="center",
                ha="center")
    ax.text(0.5, 0.14,
            "two comparisons with no branching\nthis is why CatBoost predicts so fast",
            fontsize=11.5, color=INK, ha="center", va="center", linespacing=1.5)
    save(fig, out, "cat_oblivious")


def fig_gbm_compare(out: Path) -> None:
    """The three libraries side by side, once the shared maths is set aside."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.set_title("same objective, same gain, same leaf value — everything below "
                 "is a different answer to \"how do we afford it\"", fontsize=13)

    cols = (0.025, 0.27, 0.53, 0.78)
    for x, name, col in ((cols[1], "XGBoost", BLUE), (cols[2], "LightGBM", GREEN),
                         (cols[3], "CatBoost", PURPLE)):
        ax.text(x, 0.90, name, fontsize=14, color=col, ha="left")
    ax.plot([0.01, 0.99], [0.85, 0.85], color=GREY, lw=1.2)

    rows = (
        ("finding a split", "every midpoint,\nor sketched quantiles",
         "~255 bins, filled\nin one pass", "bins"),
         ("growing the tree", "level-wise,\ndepth-limited",
          "leaf-wise,\n$\\mathtt{num\\_leaves}$-limited",
          "oblivious: ONE split\nper level"),
          ("which rows", "all", "GOSS",
           "a permutation\nof the row"),
           ("categoricals", "to encode yourself",
            "greedy\ntarget statistics",
            "ordered\ntarget statistics"),
            # ("the residual $g_i$", "from a model fitted\non row $i$ too",
            #  "model fitted on row $i$", "from a model that\nnever saw row $i$"),
            ("what it is best at", "small data, forgiving\ndefaults",
             "large tabular data,\nspeed",
             "many categoricals,\nfast inference")
    )
    for k, (label, a, b_, c) in enumerate(rows):
        y = 0.745 - 0.132 * k
        ax.text(cols[0], y, label, fontsize=12, color=INK, va="center")
        for x, txt, col in ((cols[1], a, BLUE), (cols[2], b_, GREEN),
                            (cols[3], c, PURPLE)):
            ax.text(x, y, txt, fontsize=11, color=col, va="center",
                    linespacing=1.4)
        if k < len(rows) - 1:
            ax.plot([0.01, 0.99], [y - 0.066, y - 0.066], color=FAINT, lw=1.0)
    save(fig, out, "gbm_compare")


FIGURES = (
    fig_train_test, fig_polyfit, fig_polyfit_curve, fig_bv_targets,
    fig_bv_fits, fig_bv_decomposition, fig_double_descent,
    fig_double_descent_measured, fig_dd_ridge, fig_interpolation,
    fig_reg_path, fig_l1_l2_geometry, fig_lambda_curve, fig_tree_depth,
    fig_early_stopping, fig_reg_zoo, fig_landscape_1d, fig_landscape_2d,
    fig_log_scale, fig_bo_surrogate, fig_bo_uncertainty,
    fig_bo_prior_posterior, fig_bo_kernel, fig_bo_kernel_basis, fig_bo_noise,
    fig_bayesopt, fig_acquisition, fig_search_compare, fig_optuna_history,
    fig_pruning, fig_nested_cv, fig_ensemble_avg_demo,
    fig_ensemble_correlation, fig_bagging_boundary, fig_bagging_vs_boosting,
    fig_boosting_stages, fig_additive_model, fig_shrinkage,
    fig_learning_rate, fig_stacking, fig_oof, fig_cv_schemes,
    fig_concept_map, fig_recap, fig_eq_bias_variance, fig_eq_erm,
    fig_eq_ridge_lasso, fig_eq_params, fig_eq_boosting, fig_eq_xgboost, fig_eq_leaf_objective, fig_xgb_gh, fig_xgb_leaf_predict, fig_xgb_leaf_score,
    fig_xgb_split_search,
    fig_lgbm_bins, fig_lgbm_prefix, fig_lgbm_binned_scan, fig_lgbm_subtract,
    fig_lgbm_leafwise, fig_lgbm_goss,
    fig_cat_example, fig_cat_chain, fig_cat_leak_why, fig_cat_leak, fig_cat_ordered_ts,
    fig_cat_shift, fig_cat_shift_why, fig_cat_ordered_boosting, fig_cat_ordered_cost, fig_cat_oblivious,
    fig_gbm_compare,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course03/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
