#!/usr/bin/env python3
"""Render the illustration figures used by the Course04 lecture slides.

Every figure is synthetic and seeded, so `make figures` reproduces the same output
anywhere without touching `data/`. One clinical scenario runs through the whole
deck — 30-day complication risk — and two models, A and B, share a ranking and
therefore an AUC while disagreeing completely about probabilities. That pair is
the spine of the argument, so it is generated once in `_two_models` and reused.

Usage: make_figures_c04.py [outdir]      (default: Course04/img)
"""

from __future__ import annotations

from cProfile import label
import sys
from pathlib import Path

import matplotlib
from pyparsing import alphas

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
TEAL = "#0f766e"
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


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ the scenario
def _shift_to_mean(z, target):
    """Shift logits so that the MEAN of sigmoid(z) equals `target`.

    Centring on logit(target) would centre the median, and at these spreads the
    median and the mean of a squashed normal are far apart — a cohort asked for
    1% prevalence would arrive at 3%.
    """
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return z + (lo + hi) / 2


def _risk_scores(n, seed, prevalence=0.30, spread=1.35):
    """True risks and realised outcomes for a cohort.

    `p` is the honest probability of the 30-day complication; `y` is drawn from
    it. Because `p` IS the data-generating probability, a model that reports `p`
    is perfectly calibrated by construction — which is what lets the deck show a
    miscalibrated twin with the identical ranking.
    """
    rng = np.random.default_rng(seed)
    p = sigmoid(_shift_to_mean(rng.normal(0, spread, n), prevalence))
    y = rng.binomial(1, p)
    return p, y


def _auc(p, y):
    """ROC-AUC by rank, ties averaged; avoids a sklearn import in hot paths."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _bins(p, y, k=10, strategy="quantile"):
    """Reliability-diagram bins: mean prediction, observed rate, count."""
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, k + 1)))
    else:
        edges = np.linspace(0, 1, k + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def _diagonal(ax, label=True):
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, zorder=1)
    if label:
        ax.text(0.72, 0.66, "perfect\ncalibration", color=GREY, fontsize=10.5,
                rotation=38, ha="center", va="center", rotation_mode="anchor")


def _calib_axes(ax, xlabel="predicted probability",
                ylabel="observed frequency"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))


# ------------------------------------------------------------------- equations
# Pandoc writes display math as OOXML (a14:m) with no mc:Fallback, so LibreOffice
# and Impress drop the whole shape and the slide renders blank. Rendering the
# display equations as images makes them look identical in every viewer.


# ========================================================= III. calibration metrics


def _slope_intercept(p, y):
    """Cox recalibration: logistic regression of y on logit(p̂), by IRLS."""
    X = np.c_[np.ones(len(p)), logit(p)]
    b = np.zeros(2)
    for _ in range(60):
        mu = sigmoid(X @ b)
        W = np.clip(mu * (1 - mu), 1e-6, None)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2),
                             X.T @ (y - mu))
    return b[0], b[1]


# ============================================== III.b where miscalibration comes from
def _tabular(n=6000, d=8, seed=13, prevalence=0.30):
    """A plain tabular classification problem, with the true risk kept.

    Returned alongside the labels so a figure can say how far a model is from
    the truth, not merely from the diagonal.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    beta = np.array([1.5, -1.1, 0.8, -0.6, 0.4, 0.0, 0.0, 0.0])[:d]
    z = _shift_to_mean(X @ beta + 0.6 * X[:, 0] * X[:, 1], prevalence)
    p = sigmoid(z)
    return X, rng.binomial(1, p), p


# ====================================== I.b why a model's output is not a probability
# The four-model comparison follows the scikit-learn "Probability calibration"
# user guide: make_classification with redundant features, a deliberately tiny
# training set, and LinearSVC's decision_function squashed by min-max rather than
# by anything principled.
ZOO_TRAIN = 100


def _zoo_data(n=60_000, seed=42):
    from sklearn.datasets import make_classification

    return make_classification(n_samples=n, n_features=20, n_informative=2,
                               n_redundant=2, random_state=seed)


def _zoo_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC

    return (("logistic regression", LogisticRegression(C=0.3, max_iter=5000), BLUE),
            ("naive Bayes", GaussianNB(), PURPLE),
            ("random forest",
             RandomForestClassifier(n_estimators=100, random_state=0), GREEN),
            ("linear SVC", LinearSVC(C=1.0), RED))


def _zoo_fit():
    """Fit the four models once and return their test-set probabilities."""
    X, y = _zoo_data()
    k = ZOO_TRAIN
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]
    out = []
    for name, m, col in _zoo_models():
        m.fit(Xtr, ytr)
        if hasattr(m, "predict_proba"):
            q = m.predict_proba(Xte)[:, 1]
        else:
            # the naive squash: a margin is not a probability, and rescaling it
            # to [0, 1] does not make it one
            d = m.decision_function(Xte)
            q = (d - d.min()) / (d.max() - d.min())
        out.append((name, q, col))
    return out, yte


def fig_model_zoo_calibration(out: Path) -> None:
    """Four standard classifiers, four different ways of not being a probability."""
    models, yte = _zoo_fit()

    fig, axd = plt.subplot_mosaic(
        [["cal", "h0", "h1"], ["cal", "h2", "h3"]], figsize=(12.6, 5.2),
        width_ratios=[1.35, 1.0, 1.0], gridspec_kw={"wspace": 0.30,
                                                    "hspace": 0.55})

    ax = axd["cal"]
    _diagonal(ax, label=False)
    for name, q, col in models:
        xs, ys, _ = _bins(q, yte, 12)
        ax.plot(xs, ys, "o-", color=col, lw=2.4, ms=5, label=name)
    _calib_axes(ax)
    ax.set_title("calibration curves", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    for k, (name, q, col) in enumerate(models):
        ax = axd[f"h{k}"]
        counts, _, _ = ax.hist(q, bins=np.linspace(0, 1, 26), color=col,
                               alpha=0.75)
        ax.set_title(name, fontsize=11, color=col)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, counts.max() * 1.35)
        ax.set_yticks([])
        ax.set_xticks((0, 0.5, 1))
        ax.tick_params(labelsize=9)
        b = _slope_intercept(q, yte)[1]
        ax.text(0.5, 0.88, f"β = {b:.2f}", transform=ax.transAxes, fontsize=10,
                color=col, ha="center", va="top")
    axd["h2"].set_xlabel("predicted probability", fontsize=10)
    axd["h3"].set_xlabel("predicted probability", fontsize=10)
    save(fig, out, "model_zoo_calibration")


def fig_glm(out: Path) -> None:
    """A GLM is three declarations, and they already know three of them."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.2],
                                          "wspace": 0.36})

    ax = axes[0]
    blank(ax)
    ax.set_title("three declarations", fontsize=12.5)
    stages = ((0.80, r"$\eta=x^{\top}\beta$", "the linear predictor",
               "any real number", BLUE),
              (0.48, r"$\mu=g^{-1}(\eta)$", "the link",
               r"maps $\mathbb{R}$ onto the allowed values", AMBER),
              (0.16, r"$y\sim\mathrm{Dist}(\mu)$", "the distribution",
               r"how $y$ scatters around $\mu$", GREEN))
    for yy, formula, name, note, col in stages:
        ax.text(0.16, yy, formula, fontsize=16, color=col, ha="center",
                va="center")
        ax.text(0.36, yy + 0.055, name, fontsize=12, color=col, va="center")
        ax.text(0.36, yy - 0.055, note, fontsize=10, color=GREY, va="center")
        if yy > 0.2:
            arrow(ax, (0.16, yy - 0.11), (0.16, yy - 0.21), color=GREY, lw=1.4)

    ax = axes[1]
    eta = np.linspace(-6, 6, 400)
    ax.plot(eta, sigmoid(eta), color=BLUE, lw=3.0)
    ax.axhline(0, color=GREY, lw=1.0, ls=":")
    ax.axhline(1, color=GREY, lw=1.0, ls=":")
    ax.set_xlabel(r"$\eta=x^{\top}\beta$")
    ax.set_ylabel(r"$\mu=P(y=1\mid x)$")
    ax.set_xlim(-7.2, 6.5)
    ax.set_ylim(-0.14, 1.22)
    ax.set_yticks((0, 0.5, 1))
    ax.set_title("why a link is needed at all", fontsize=12.5)
    ax.annotate("", (6.2, 1.10), (-6.2, 1.10),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    ax.text(0, 1.15, r"$\eta$ is free on all of $\mathbb{R}$", fontsize=10.5,
            color=BLUE, ha="center")
    ax.annotate("", (-6.6, 0), (-6.6, 1),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(-6.3, 0.5, r"$\mu$ is trapped" "\n" r"in $[0,1]$", fontsize=10.5,
            color=RED, ha="left", va="center", linespacing=1.5)

    ax = axes[2]
    blank(ax)
    ax.set_title("three classical models", fontsize=12.5)
    for h, x in (("model", 0.0), ("family", 0.32), ("link", 0.64),
                 (r"$\mu$ in", 0.92)):
        ax.text(x, 0.86, h, fontsize=11, color=GREY)
    ax.plot([0.0, 1.06], [0.79, 0.79], color=GREY, lw=1.1)
    rows = (("linear", "Gaussian", "identity", r"$\mathbb{R}$", BLUE),
            ("logistic", "Bernoulli", "logit", r"$(0,1)$", GREEN),
            ("Poisson", "Poisson", "log", r"$(0,\infty)$", PURPLE))
    for k, (m, d, li, rng, col) in enumerate(rows):
        yy = 0.64 - 0.17 * k
        ax.text(0.0, yy, m, fontsize=11.5, color=col, va="center")
        ax.text(0.32, yy, d, fontsize=11.5, color=INK, va="center")
        ax.text(0.64, yy, li, fontsize=11.5, color=INK, va="center")
        ax.text(0.92, yy, rng, fontsize=12, color=INK, va="center")
    ax.plot([0.0, 1.06], [0.16, 0.16], color=FAINT, lw=1.4)
    ax.text(0.0, 0.05, "same machinery, same fitting code —\n"
                       "only the last two columns change",
            fontsize=11, color=GREY, va="center", linespacing=1.7)
    save(fig, out, "glm")


def fig_log_loss_intro(out: Path) -> None:
    """What the log loss actually does to a single prediction."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.0],
                                          "wspace": 0.34})

    ax = axes[0]
    blank(ax)
    ax.set_title("the number minimised by the log loss", fontsize=12.5)
    ax.text(0.04, 0.84, r"$y_i=1$", fontsize=13, color=GREEN, va="center")
    ax.text(0.40, 0.84, r"$\ell_i=-\log \hat{p}_i$", fontsize=15, va="center")
    ax.text(0.04, 0.64, r"$y_i=0$", fontsize=13, color=RED, va="center")
    ax.text(0.40, 0.64, r"$\ell_i=-\log(1-\hat{p}_i)$", fontsize=15, va="center")
    ax.plot([0.02, 0.98], [0.50, 0.50], color=FAINT, lw=1.6)
    ax.text(0.5, 0.36, r"$\ell_i=-\log q_i$", fontsize=20, ha="center", va="center", color=BLUE)
    ax.text(0.5, 0.20, r"with $q_i$ the probability given" "\n to the outcome that happened",
            fontsize=11.5, color=GREY, ha="center", va="center", linespacing=1.6)

    ax = axes[1]
    q = np.linspace(0.012, 1.0, 500)
    ax.plot(q, -np.log(q), color=BLUE, lw=2.4)
    for qq, col, off, ha in ((0.9, GREEN, (-12, 20), "left"),
                             (0.5, GREY, (10, 8), "left"),
                             (0.1, RED, (12, 6), "left")):
        ax.plot([qq], [-np.log(qq)], "o", color=col, ms=7, zorder=3)
        ax.annotate(f"{qq:.1f} → {-np.log(qq):.2f}", (qq, -np.log(qq)),
                    textcoords="offset points", xytext=off, ha=ha,
                    fontsize=10.5, color=col)
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.15, 4.4)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("loss")
    ax.text(0.60, 0.74, r"as $q\to 0$,   $\ell\to\infty$", fontsize=12,
            color=RED, transform=ax.transAxes, linespacing=1.5, ha="center")

    ax = axes[2]
    true = 0.70
    qq = np.linspace(0.02, 0.98, 500)
    exp_loss = -(true * np.log(qq) + (1 - true) * np.log(1 - qq))
    floor = -(true * np.log(true) + (1 - true) * np.log(1 - true))
    ax.plot(qq, exp_loss, color=PURPLE, lw=2.4)
    ax.axvline(true, color=GREEN, ls="--", lw=1.6)
    ax.plot([true], [floor], "o", color=GREEN, ms=8, zorder=3)
    ax.set_ylim(floor - 0.15, floor + 1.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"declared probability $\hat{p}$")
    ax.set_ylabel("expected loss")
    ax.set_title("the loss' minimum is at the true probability", fontsize=12.5)
    ax.text(true + 0.03, floor + 1.15,
            "true risk\n0.70", fontsize=11, color=GREEN, linespacing=1.5)
    save(fig, out, "log_loss_intro")


def fig_log_loss_likelihood(out: Path) -> None:
    """Log loss is the negative log-likelihood of a Bernoulli, worked once."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.0],
                                          "wspace": 0.30})

    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.3, 0.6])
    contrib = np.where(y == 1, p, 1 - p)
    lik = float(np.prod(contrib))
    nll = float(-np.log(lik))

    ax = axes[0]
    blank(ax)
    ax.set_title("the probability of the observed data", fontsize=12.5)
    ax.text(0.5, 0.87, r"$L=\prod_i\;\hat{p}_i^{\,y_i}\,(1-\hat{p}_i)^{1-y_i}$",
            fontsize=15, ha="center", va="center")
    ax.text(0.06, 0.68, r"$y_i$", fontsize=12, color=GREY, ha="center")
    ax.text(0.30, 0.68, r"$\hat{p}_i$", fontsize=12, color=GREY, ha="center")
    ax.text(0.68, 0.68, "its contribution", fontsize=11, color=GREY,
            ha="center")
    for k in range(3):
        yy = 0.55 - 0.13 * k
        col = GREEN if y[k] == 1 else RED
        ax.text(0.06, yy, f"{y[k]}", fontsize=13, color=col, ha="center")
        ax.text(0.30, yy, f"{p[k]:.1f}", fontsize=13, ha="center")
        ax.text(0.68, yy, f"{contrib[k]:.1f}", fontsize=13, color=col,
                ha="center")
    ax.plot([0.55, 0.82], [0.12, 0.12], color=FAINT, lw=1.6)
    ax.text(0.68, 0.02, f"$L={lik:.3f}$", fontsize=14, ha="center",
            color=BLUE)
    ax.text(0.06, 0.02, "multiply\ndown", fontsize=10.5, color=GREY,
            ha="center", va="center", linespacing=1.5)

    ax = axes[1]
    blank(ax)
    ax.set_title("turn the product into a sum", fontsize=12.5)
    ax.text(0.5, 0.84, r"$-\log L=\sum_i -\left[\,y_i\log\hat{p}_i"
                       r"+(1-y_i)\log(1-\hat{p}_i)\,\right]$",
            fontsize=13, ha="center", va="center")
    ax.text(0.5, 0.62, rf"$=-\log {lik:.3f}={nll:.3f}$", fontsize=15,
            ha="center", va="center", color=BLUE)
    ax.text(0.5, 0.44, rf"$\div\,n:\quad {nll / 3:.3f}$", fontsize=15,
            ha="center", va="center", color=GREEN)
    ax.text(0.5, 0.30, "which is what log_loss returns", fontsize=11,
            color=GREY, ha="center")
    ax.plot([0.06, 0.94], [0.21, 0.21], color=FAINT, lw=1.6)
    ax.text(0.5, 0.09, "log because products of probabilities would underflow;\n"
                       "log is monotone, so the argmin does not move;\n"
                       "log turns the product into a sum, easy to differentiate",
            fontsize=10.5, color=GREY, ha="center", va="center",
            linespacing=1.7)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.78, "minimising log loss\n" r"$\Longleftrightarrow$" "\nmaximising likelihood",
            fontsize=15, ha="center", va="center", color=GREEN,
            linespacing=1.8)
    ax.plot([0.06, 0.94], [0.50, 0.50], color=FAINT, lw=1.6)
    ax.text(0.5, 0.38, "only assumption:",
            fontsize=11.5, color=GREY, ha="center")
    ax.text(0.5, 0.24, r"$P(y\mid p)=p^{\,y}(1-p)^{1-y}$", fontsize=15,
            ha="center", va="center")
    ax.text(0.5, 0.06, "changing that changes the loss",
            fontsize=11.5, color=RED, ha="center", va="center",
            linespacing=1.6)
    save(fig, out, "log_loss_likelihood")


def fig_bernoulli_b(out: Path) -> None:
    """b, b' and b'' for the Bernoulli, drawn."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE, gridspec_kw={"wspace": 0.30})
    th = np.linspace(-5, 5, 500)
    p = sigmoid(th)

    specs = ((np.logaddexp(0, th), BLUE,
              r"$b(\theta)=\log(1+e^{\theta})$", "the log-normaliser",
              "the loss is convex"),
             (p, GREEN, r"$b'(\theta)=\sigma(\theta)=p$", "the mean",
              "the sigmoid — the inverse logit"),
             (p * (1 - p), AMBER, r"$b''(\theta)=p(1-p)$",
              "the variance function", "widest where least sure"))
    for ax, (vals, col, formula, name, note) in zip(axes, specs):
        ax.plot(th, vals, color=col, lw=2.6)
        ax.set_title(formula, fontsize=13.5, color=col, pad=12)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(name)
        ax.text(0.5, -0.24, note, fontsize=11, color=GREY, ha="center",
                transform=ax.transAxes)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(0, 0.30)
    save(fig, out, "bernoulli_b")


def fig_expfam(out: Path) -> None:
    """The exponential family form, with every symbol named."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.05],
                                          "wspace": 0.32})

    ax = axes[0]
    blank(ax)
    ax.set_title("one shape for many distributions", fontsize=12.5)
    ax.text(0.5, 0.74,
            r"$f(y_i;\theta_i)=\exp\!\left("
            r"\dfrac{y_i\,\theta_i-b(\theta_i)}{\phi}+c(y_i,\phi)\right)$",
            fontsize=15, ha="center", va="center")
    ax.text(0.5, 0.50, "Gaussian, Bernoulli, Poisson, Gamma —\n"
                       "all of them can be written like this",
            fontsize=11.5, color=GREY, ha="center", va="center",
            linespacing=1.6)
    ax.plot([0.06, 0.94], [0.38, 0.38], color=FAINT, lw=1.6)
    ax.text(0.5, 0.26, "note the subscript: one distribution\nper observation",
            fontsize=12, color=RED, ha="center", va="center", linespacing=1.6)

    ax = axes[1]
    blank(ax)
    rows = ((r"$y_i$", "the observation", INK),
            (r"$\theta_i$", "the natural parameter",
             GREEN),
            (r"$b(\theta)$", "the log-normaliser",
             AMBER),
            (r"$\phi$", "the dispersion: a scale", PURPLE),
            (r"$c(y,\phi)$", "everything with no $\\theta$ in it", GREY))
    for k, (sym, meaning, col) in enumerate(rows):
        yy = 0.80 - 0.185 * k
        ax.text(0.05, yy, sym, fontsize=15, color=col, ha="center",
                va="center")
        ax.text(0.24, yy, meaning, fontsize=10.5, color=INK, va="center",
                linespacing=1.5)

    ax = axes[2]
    blank(ax)
    ax.set_title(r"why $b$ is worth knowing", fontsize=12.5)
    ax.text(0.5, 0.76, r"$b'(\theta)=\mu$", fontsize=21, ha="center",
            va="center", color=BLUE)
    ax.text(0.5, 0.60, "differentiate once gives the mean", fontsize=11,
            color=GREY, ha="center")
    ax.text(0.5, 0.40, r"$b''(\theta)=V(\mu)$", fontsize=21, ha="center",
            va="center", color=BLUE)
    ax.text(0.5, 0.24, "differentiate twice gives the variance",
            fontsize=11, color=GREY, ha="center")
    save(fig, out, "expfam")


def fig_bernoulli_expfam(out: Path) -> None:
    """The Bernoulli put into that shape, in three lines of algebra."""
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.30, 1.0],
                          width_ratios=[1.35, 0.95, 1.0],
                          hspace=0.34, wspace=0.30)

    top = fig.add_subplot(gs[0, :])
    blank(top)
    top.text(0.5, 0.74, r"$f(y;\theta)=\exp\!\left("
                        r"\dfrac{y\,\theta-b(\theta)}{\phi}"
                        r"+c(y,\phi)\right)$",
             fontsize=17, ha="center", va="center")
    top.text(0.5, 0.10, "the shape we are aiming at",
             fontsize=11.5, color=RED, ha="center", va="center")

    axes = [fig.add_subplot(gs[1, k]) for k in range(3)]

    ax = axes[0]
    blank(ax)
    lines = (r"$P(y)=p^{\,y}(1-p)^{1-y}$",
             r"$=\exp\!\left(y\log p+(1-y)\log(1-p)\right)$",
             r"$=\exp\!\left(y\log\dfrac{p}{1-p}+\log(1-p)\right)$")
    for k, ln in enumerate(lines):
        ax.text(0.5, 0.78 - 0.26 * k, ln, fontsize=14, ha="center",
                va="center")
    ax.text(0.5, 0.04, "take logs, then collect the terms in $y$",
            fontsize=11.5, color=GREY, ha="center")

    ax = axes[1]
    blank(ax)
    ax.set_title("fill the slots", fontsize=12.5)
    rows = ((r"$\theta$", r"$\log\dfrac{p}{1-p}$", "what multiplies $y$", GREEN),
            (r"$b(\theta)$", r"$\log\!\left(1+e^{\theta}\right)$",
             "what is left over", AMBER),
            (r"$\phi$", r"$1$", "no free scale here", PURPLE),
            (r"$c(y,\phi)$", r"$0$", "nothing else remains", GREY))
    for k, (sym, val, why, col) in enumerate(rows):
        yy = 0.88 - 0.25 * k
        ax.text(0.16, yy, sym, fontsize=15, color=col, ha="center",
                va="center")
        ax.text(0.36, yy, "=", fontsize=13, color=GREY, va="center")
        ax.text(0.47, yy, val, fontsize=15, color=col, va="center")
        ax.text(0.16, yy - 0.11, why, fontsize=10, color=GREY, va="center")

    ax = axes[2]
    blank(ax)
    ax.set_title("compute the loss", fontsize=12)
    ax.text(0.5, 0.80, r"$-\log P(y)=b(\theta)-y\,\theta$", fontsize=15,
            ha="center", va="center", color=BLUE)
    ax.text(0.5, 0.60, r"$=\log\!\left(1+e^{\theta}\right)-y\,\theta$",
            fontsize=15, ha="center", va="center", color=BLUE)
    ax.text(0.5, 0.42, r"log loss in parameter $\theta$",
            fontsize=11, color=GREY, ha="center", va="center",
            linespacing=1.6)
    ax.plot([0.06, 0.94], [0.12, 0.12], color=FAINT, lw=1.6)
    ax.text(0.5, 0.02, r"for Bernoulli, $\theta$ IS the log-odds",
            fontsize=12, color=GREEN, ha="center")
    save(fig, out, "bernoulli_expfam")


def fig_expfam_loss(out: Path) -> None:
    """The loss is the density read backwards: minus the log-likelihood."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.25],
                                          "wspace": 0.30})

    ax = axes[0]
    blank(ax)
    ax.set_title("loss", fontsize=12.5)
    ax.text(0.5, 0.82, r"$f(y_i;\theta_i)=\exp\!\left("
                       r"\dfrac{y_i\theta_i-b(\theta_i)}{\phi}"
                       r"+c(y_i,\phi)\right)$",
            fontsize=13.5, ha="center", va="center", color=GREY)
    arrow(ax, (0.5, 0.72), (0.5, 0.62), color=GREY, lw=1.5)
    ax.text(0.56, 0.67, r"$-\log$", fontsize=11, color=GREY, va="center")
    ax.text(0.5, 0.50, r"$\ell_i=\dfrac{b(\theta_i)-y_i\theta_i}{\phi}"
                       r"\;-\;c(y_i,\phi)$",
            fontsize=15, ha="center", va="center")
    ax.text(0.5, 0.30, r"$c$ has no $\theta$ in it: it shifts the loss,"
                       "\nit never moves the minimum",
            fontsize=11, color=GREY, ha="center", va="center", linespacing=1.6)
    ax.plot([0.06, 0.94], [0.18, 0.18], color=FAINT, lw=1.6)
    ax.text(0.5, 0.06, "pick the distribution and\nthe loss is already decided",
            fontsize=12, color=RED, ha="center", va="center", linespacing=1.6)

    ax = axes[1]
    blank(ax)
    ax.set_title("gradient", fontsize=12.5)
    ax.text(0.5, 0.80, r"$\dfrac{\partial\ell_i}{\partial\theta_i}"
                       r"=\dfrac{b'(\theta_i)-y_i}{\phi}$",
            fontsize=17, ha="center", va="center")
    arrow(ax, (0.5, 0.64), (0.5, 0.55), color=GREY, lw=1.5)
    ax.text(0.56, 0.60, r"$b'=\mu$", fontsize=11, color=BLUE, va="center")
    ax.text(0.5, 0.43, r"$=\dfrac{\mu_i-y_i}{\phi}$", fontsize=20,
            ha="center", va="center", color=BLUE)
    ax.plot([0.06, 0.94], [0.24, 0.24], color=FAINT, lw=1.6)
    ax.text(0.5, 0.11, "predicted minus observed\n"
                       "(for all in exponential format)",
            fontsize=11.5, color=RED, ha="center", va="center",
            linespacing=1.6)

    ax = axes[2]
    blank(ax)
    rows = ((r"Gaussian", r"$\frac{1}{2}(y-\mu)^2$", "squared error", BLUE),
            (r"Bernoulli", r"$-[\,y\log p+(1-y)\log(1-p)\,]$", "log loss",
             GREEN),
            (r"Poisson", r"$\mu-y\log\mu$", "Poisson deviance", PURPLE))
    for k, (dist, expr, name, col) in enumerate(rows):
        yy = 0.82 - 0.23 * k
        ax.text(0.02, yy, dist, fontsize=12.5, color=col, va="center")
        ax.text(0.30, yy, expr, fontsize=12, color=INK, va="center")
        ax.text(0.30, yy - 0.09, name, fontsize=10.5, color=GREY, va="center")
    ax.plot([0.0, 1.0], [0.20, 0.20], color=FAINT, lw=1.6)
    ax.text(0.0, 0.08, "hinge loss and Gini impurity are not on this list:\n"
                       "no distribution behind them, no probability out",
            fontsize=11, color=RED, va="center", linespacing=1.7)
    save(fig, out, "expfam_loss")


def fig_naive_bayes_bias(out: Path) -> None:
    """Conditional independence: the same evidence counted once per copy."""
    from sklearn.naive_bayes import GaussianNB

    rng = np.random.default_rng(4)
    n = 40_000
    y = rng.binomial(1, 0.5, n)
    base = rng.normal(y * 1.4, 1.0)          # one genuinely informative feature
    truth = 1.4 * base - 0.5 * 1.4 ** 2      # the true log-odds, by Bayes

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    blank(ax)
    ax.set_title("naive Bayes assumes", fontsize=12.5)
    ax.text(0.5, 0.80, r"$\log\frac{P(y=1\mid x)}{P(y=0\mid x)}"
                       r"=\log\frac{\pi_1}{\pi_0}"
                       r"+\sum_{j=1}^{d}\log\frac{p(x_j\mid 1)}{p(x_j\mid 0)}$",
            fontsize=15, ha="center", va="center")
    ax.text(0.5, 0.54, "one term per feature, added up as if\n"
                       "every feature were fresh evidence",
            fontsize=11.5, color=GREY, ha="center", va="center",
            linespacing=1.6)
    
    ax = axes[1]
    grid = np.linspace(-6, 6, 200)
    for k, col in ((1, BLUE), (2, AMBER), (4, RED)):
        Xk = np.column_stack([base + rng.normal(0, 0.02, n) for _ in range(k)])
        nb = GaussianNB().fit(Xk[:2000], y[:2000])
        lp = nb.predict_log_proba(Xk[2000:])
        ax.plot(truth[2000:][::40], (lp[:, 1] - lp[:, 0])[::40], ".",
                color=col, ms=2.5, alpha=0.5)
        ax.plot([], [], "-", color=col, lw=2.6,
                label="1 copy" if k == 1 else f"{k} copies")
    ax.plot(grid, grid, ls="--", color=GREY, lw=1.6)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-18, 18)
    ax.set_xlabel("true log-odds")
    ax.set_ylabel("naive Bayes log-odds")
    ax.set_title("each copy multiplies the evidence", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")

    ax = axes[2]
    for k, col, alpha in ((1, BLUE, 0.75), (4, RED, 0.6)):
        Xk = np.column_stack([base + rng.normal(0, 0.02, n) for _ in range(k)])
        nb = GaussianNB().fit(Xk[:2000], y[:2000])
        q = nb.predict_proba(Xk[2000:])[:, 1]
        ax.hist(q, bins=np.linspace(0, 1, 31), color=col, alpha=alpha,
                label=("1 copy" if k == 1 else f"{k} copies")
                      + f",  β = {_slope_intercept(q, y[2000:])[1]:.2f}")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("cases")
    ax.set_title("and pushes everything to the ends", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    save(fig, out, "naive_bayes_bias")


def fig_forest_boundary_bias(out: Path) -> None:
    """Averaging plus a hard [0, 1] boundary makes the errors one-sided."""
    from sklearn.ensemble import RandomForestClassifier

    X, y, p_true = _tabular(12_000, seed=31, prevalence=0.50)
    m = 3000
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X[:m],
                                                                     y[:m])
    votes = np.array([t.predict(X[m:]) for t in rf.estimators_])
    q = votes.mean(0)
    p_te = p_true[m:]

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    k = int(np.argmin(p_te))                      # the lowest-risk patient there is
    v = votes[:, k]
    ax.bar((0, 1), (np.sum(v == 0), np.sum(v == 1)), color=(FAINT, RED),
           width=0.55, edgecolor=GREY)
    ax.set_xticks((0, 1))
    ax.set_xticklabels(("tree says 0", "tree says 1"))
    ax.set_ylabel("trees")
    ax.set_title(f"one patient, true risk {p_te[k]:.3f}", fontsize=12.5)
    ax.text(0.5, 0.72, f"{int(np.sum(v == 1))} of 200 trees are wrong\n"
                       f"→ the forest says {q[k]:.3f}, not 0",
            transform=ax.transAxes, fontsize=12, color=RED, ha="center",
            va="center", linespacing=1.7)

    ax = axes[1]
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(p_te, edges[1:-1]), 0, 19)
    xs = np.array([p_te[idx == b].mean() for b in range(20)])
    ys = np.array([q[idx == b].mean() for b in range(20)])
    lo = np.array([np.percentile(q[idx == b], 10) for b in range(20)])
    hi = np.array([np.percentile(q[idx == b], 90) for b in range(20)])
    ax.fill_between(xs, lo, hi, color=GREEN, alpha=0.15, lw=0)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.6)
    ax.plot(xs, ys, "o-", color=GREEN, lw=2.6, ms=5)
    ax.annotate("predictions lower cap", (xs[0], ys[0]), (0.18, 0.30),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("predictions upper cap", (xs[-1], ys[-1]), (0.42, 0.80),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("true probability")
    ax.set_ylabel("mean forest prediction")
    ax.set_title("the ends are unreachable", fontsize=12.5)

    # the consequence, on the four-model example: with only 100 training rows the
    # trees are high-variance enough that the two interior peaks are unmistakable
    ax = axes[2]
    models, yte = _zoo_fit()
    _, q_zoo, _ = models[2]
    ax.hist(q_zoo, bins=np.linspace(0, 1, 41), color=GREEN, alpha=0.75)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("cases")
    ax.set_title(f"so the mass sits inside  "
                 f"(β = {_slope_intercept(q_zoo, yte)[1]:.2f})", fontsize=12,
                 color=GREEN)
    ax.set_xlim(0, 1)
    save(fig, out, "forest_boundary_bias")


def fig_margin_bias(out: Path) -> None:
    """Hinge loss stops caring once a point is safely classified."""
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(6)
    n = 260
    yy = rng.binomial(1, 0.5, n)
    P = np.column_stack([rng.normal(yy * 3.5, 1.0), rng.normal(0, 1.3, n)])
    sv = LinearSVC(C=1.0).fit(P, yy)
    w, b = sv.coef_[0], sv.intercept_[0]
    margin = yy * 2 - 1
    f = P @ w + b

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    inside = np.abs(f) <= 1.0
    ax.scatter(P[~inside, 0], P[~inside, 1], s=16,
               c=[BLUE if v == 0 else RED for v in yy[~inside]], alpha=0.35,
               edgecolors="none")
    ax.scatter(P[inside, 0], P[inside, 1], s=42,
               c=[BLUE if v == 0 else RED for v in yy[inside]],
               edgecolors=INK, linewidths=1.1, zorder=3)
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 50)
    for c, ls, lw in ((0, "-", 2.4), (1, "--", 1.4), (-1, "--", 1.4)):
        ax.plot(xs, -(w[0] * xs + b - c) / w[1], color=INK, ls=ls, lw=lw)
    ax.set_xlim(P[:, 0].min() - 0.3, P[:, 0].max() + 0.3)
    ax.set_ylim(P[:, 1].min() - 0.4, P[:, 1].max() + 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"only these {inside.sum()} of {n} points set the fit",
                 fontsize=12.5)

    ax = axes[1]
    z = np.linspace(-3, 3, 300)
    ax.plot(z, np.maximum(0, 1 - z), color=RED, lw=2.8, label="hinge")
    ax.plot(z, np.log(1 + np.exp(-z)), color=BLUE, lw=2.8, label="log loss")
    ax.axvline(1, color=GREY, ls=":", lw=1.4)
    ax.text(1.25, 0.95, "beyond the\nmargin hinge\nis exactly 0", fontsize=10.5,
            color=RED, va="top", linespacing=1.5)
    ax.set_xlabel(r"margin  $y\,f(x)$")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 3.2)
    ax.set_title("a confident point contributes nothing", fontsize=12.5)
    ax.legend(frameon=False, fontsize=11)

    ax = axes[2]
    models, yte = _zoo_fit()
    name, q, col = models[3]
    _diagonal(ax, label=False)
    xs_, ys_, _ = _bins(q, yte, 12)
    ax.plot(xs_, ys_, "o-", color=RED, lw=2.6, ms=6)
    _calib_axes(ax)
    ax.set_title(f"the squashed margin, β = {_slope_intercept(q, yte)[1]:.2f}",
                 fontsize=12, color=RED)
    save(fig, out, "margin_bias")


# ========================================================== IV. recalibration
def _platt(s_tr, y_tr, s_new):
    """Logistic regression of the label on one score, by IRLS."""
    X = np.c_[np.ones(len(s_tr)), s_tr]
    b = np.zeros(2)
    for _ in range(80):
        mu = sigmoid(X @ b)
        W = np.clip(mu * (1 - mu), 1e-6, None)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2),
                             X.T @ (y_tr - mu))
    return sigmoid(b[0] + b[1] * s_new), b


def _pava(s, y):
    """Isotonic regression by pool-adjacent-violators; returns (knots, values)."""
    o = np.argsort(s, kind="mergesort")
    xs, ys = s[o], y[o].astype(float)
    val = list(ys)
    wgt = [1.0] * len(ys)
    pos = list(range(len(ys)))
    i = 0
    while i < len(val) - 1:
        if val[i] <= val[i + 1]:
            i += 1
            continue
        w = wgt[i] + wgt[i + 1]
        v = (val[i] * wgt[i] + val[i + 1] * wgt[i + 1]) / w
        val[i:i + 2] = [v]
        wgt[i:i + 2] = [w]
        pos[i:i + 2] = [pos[i]]
        i = max(i - 1, 0)
    step_x, step_y = [], []
    k = 0
    for v, w in zip(val, wgt):
        step_x.append(xs[k])
        step_y.append(v)
        k += int(w)
    return np.array(step_x), np.array(step_y)


def _isotonic_predict(step_x, step_y, s_new):
    idx = np.clip(np.searchsorted(step_x, s_new, side="right") - 1, 0,
                  len(step_y) - 1)
    return step_y[idx]


def _uncalibrated(n=6000, seed=2, shape="sigmoid"):
    """A miscalibrated scorer and the honest labels it was scored on.

    `shape="sigmoid"` distorts the true logit affinely, so the correct calibration
    map lies exactly inside Platt's family — the easy case the Platt slides use.
    `shape="wiggly"` bends it monotonically but non-logistically, which is the
    case where the extra flexibility of isotonic is worth paying for.
    """
    p, y = _risk_scores(n, seed)
    z = logit(p)
    if shape == "sigmoid":
        s = 2.0 * z + 0.6
    else:
        s = z + 0.5 * np.sin(1.6 * z)   # monotone: 1 + 0.8 cos > 0
    return s, sigmoid(s), y


def fig_platt_flow(out: Path) -> None:
    """The recalibration pipeline, and the split it depends on."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ys = 0.72
    stages = (("original\nmodel", BLUE), ("raw\nscores  s(x)", BLUE),
              ("logistic\ncalibration", RED),
              ("calibrated\nprobabilities", GREEN))
    xs = np.linspace(0.13, 0.87, 4)
    for (lab, col), x in zip(stages, xs):
        box(ax, (x, ys), 0.19, 0.22, lab, ec=col, fs=13, tc=col)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 0.098, ys), (b - 0.098, ys), color=GREY)

    ax.text(xs[0], 0.46, "fitted on\nTRAINING data", fontsize=11.5, color=BLUE,
            ha="center", va="center", linespacing=1.5)
    ax.text(xs[2], 0.46, "fitted on\nVALIDATION data", fontsize=11.5, color=RED,
            ha="center", va="center", linespacing=1.5)
    ax.text(xs[3], 0.46, "measured on\nTEST data", fontsize=11.5, color=GREEN,
            ha="center", va="center", linespacing=1.5)
    save(fig, out, "platt_flow")


def fig_platt_fit(out: Path) -> None:
    """The sigmoid fitted through the 0/1 labels, and what it repairs."""
    s, praw, y = _uncalibrated()
    n = len(s) // 2
    cal, (b0, b1) = _platt(s[:n], y[:n], s[n:])
    raw_te, y_te = praw[n:], y[n:]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(n, 350, replace=False)
    ax.scatter(s[sub], y[sub] + rng.normal(0, 0.025, len(sub)), s=14,
               color=GREY, alpha=0.55, edgecolors="none")
    grid = np.linspace(s.min(), s.max(), 300)
    ax.plot(grid, sigmoid(b0 + b1 * grid), color=RED, lw=3.0)
    ax.plot(grid, sigmoid(grid), color=BLUE, lw=2.0, ls="--")
    ax.text(0.02, 0.96, f"fitted  σ({b1:.2f}·s {b0:+.2f})", transform=ax.transAxes,
            color=RED, fontsize=12, va="top")
    ax.text(0.02, 0.86, "raw  σ(s)", transform=ax.transAxes, color=BLUE,
            fontsize=12, va="top")
    ax.set_xlabel("model score  s(x)")
    ax.set_ylabel("outcome  y")
    ax.set_title("fit a logistic curve to the labels", fontsize=12.5)

    ax = axes[1]
    _diagonal(ax, label=False)
    for p, col, lab in ((raw_te, RED, "before"), (cal, GREEN, "after")):
        xs_, ys_, _ = _bins(p, y_te, 10)
        ax.plot(xs_, ys_, "o-", color=col, lw=2.6, ms=6, label=lab)
    _calib_axes(ax)
    ax.set_title("on the held-out test set", fontsize=12.5)
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    ax = axes[2]
    blank(ax)
    rows = (("ROC-AUC", _auc(raw_te, y_te), _auc(cal, y_te), "{:.3f}"),
            ("Brier", np.mean((raw_te - y_te) ** 2), np.mean((cal - y_te) ** 2),
             "{:.3f}"),
            ("intercept α", _slope_intercept(raw_te, y_te)[0],
             _slope_intercept(cal, y_te)[0], "{:+.2f}"),
            ("slope β", _slope_intercept(raw_te, y_te)[1],
             _slope_intercept(cal, y_te)[1], "{:.2f}"))
    for x, head, col in ((0.55, "before", RED), (0.85, "after", GREEN)):
        ax.text(x, 0.80, head, fontsize=12.5, color=col, ha="center")
    ax.plot([0.0, 1.0], [0.73, 0.73], color=GREY, lw=1.1)
    for k, (lab, a, b, fmt) in enumerate(rows):
        yy = 0.62 - 0.145 * k
        ax.text(0.0, yy, lab, fontsize=12.5, color=INK, va="center")
        ax.text(0.55, yy, fmt.format(a), fontsize=12.5, color=RED, ha="center",
                va="center")
        ax.text(0.85, yy, fmt.format(b), fontsize=12.5, color=GREEN,
                ha="center", va="center")
    ax.text(0.0, 0.03, "AUC identical to 3 decimals",
            fontsize=11, color=GREY)
    save(fig, out, "platt_fit")


def fig_calibration_leak(out: Path) -> None:
    """Calibrating on the training data buys a beautiful, meaningless curve."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    n = 1500
    X = rng.normal(0, 1, (2 * n, 6))
    p = sigmoid(1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * X[:, 2] - 0.6)
    y = rng.binomial(1, p)
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]

    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                random_state=0).fit(Xtr, ytr)
    s_tr = rf.predict_proba(Xtr)[:, 1]
    s_te = rf.predict_proba(Xte)[:, 1]
    # calibrated the wrong way: Platt fitted on the rows the forest memorised
    wrong, _ = _platt(s_tr, ytr, s_te)
    wrong_tr, _ = _platt(s_tr, ytr, s_tr)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    _diagonal(ax, label=False)
    xs_, ys_, _ = _bins(wrong_tr, ytr, 10)
    ax.plot(xs_, ys_, "o-", color=GREEN, lw=2.6, ms=6)
    ax.set_title("calibrated on training data,\nchecked on training data",
                 fontsize=11.5, color=GREEN, linespacing=1.5)
    _calib_axes(ax)

    ax = axes[1]
    _diagonal(ax, label=False)
    xs_, ys_, _ = _bins(wrong, yte, 10)
    ax.plot(xs_, ys_, "o-", color=RED, lw=2.6, ms=6)
    ax.set_title("the same calibrator,\nchecked on new rows",
                 fontsize=11.5, color=RED, linespacing=1.5)
    _calib_axes(ax, ylabel="")

    ax = axes[2]
    m = n // 2
    right, _ = _platt(s_tr[m:], ytr[m:], s_te)     # honest: a held-out slice
    rf2 = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 random_state=0).fit(Xtr[:m], ytr[:m])
    right, _ = _platt(rf2.predict_proba(Xtr[m:])[:, 1], ytr[m:], s_te)
    _diagonal(ax, label=False)
    xs_, ys_, _ = _bins(right, yte, 10)
    ax.plot(xs_, ys_, "o-", color=BLUE, lw=2.6, ms=6)
    ax.set_title("calibrated on held-out data,\nchecked on new rows",
                 fontsize=11.5, color=BLUE, linespacing=1.5)
    _calib_axes(ax, ylabel="")
    save(fig, out, "calibration_leak")


def fig_isotonic_fit(out: Path) -> None:
    """The monotone step function, next to the sigmoid it replaces."""
    s, praw, y = _uncalibrated(n=3000, seed=6)
    n = len(s) // 2
    step_x, step_y = _pava(s[:n], y[:n])
    iso_te = _isotonic_predict(step_x, step_y, s[n:])
    platt_te, (b0, b1) = _platt(s[:n], y[:n], s[n:])

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    rng = np.random.default_rng(1)
    sub = rng.choice(n, 300, replace=False)
    ax.scatter(s[sub], y[sub] + rng.normal(0, 0.02, len(sub)), s=13,
               color=GREY, alpha=0.5, edgecolors="none")
    ax.step(step_x, step_y, where="post", color=AMBER, lw=2.6)
    ax.set_xlabel("model score  s(x)")
    ax.set_ylabel("outcome  y")
    ax.set_title("isotonic: a monotone step function", fontsize=12.5,
                 color=AMBER)

    ax = axes[1]
    grid = np.linspace(s.min(), s.max(), 400)
    ax.step(step_x, step_y, where="post", color=AMBER, lw=2.4, label="isotonic")
    ax.plot(grid, sigmoid(b0 + b1 * grid), color=RED, lw=2.6, label="Platt")
    ax.set_xlabel("model score  s(x)")
    ax.set_ylabel("calibrated probability")
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    ax = axes[2]
    _diagonal(ax, label=False)
    for p, col, lab in ((praw[n:], GREY, "raw"), (platt_te, RED, "Platt"),
                        (iso_te, AMBER, "isotonic")):
        xs_, ys_, _ = _bins(p, y[n:], 10)
        ax.plot(xs_, ys_, "o-", color=col, lw=2.4, ms=5, label=lab)
    _calib_axes(ax)
    ax.set_title("on held-out rows", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    save(fig, out, "isotonic_fit")


def fig_platt_vs_isotonic(out: Path) -> None:
    """The comparison table, and the sample size where the trade flips."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    blank(ax)
    rows = (("assumption", "sigmoid in s", "monotone only"),
            ("free parameters", "2", "up to n"),
            ("flexibility", "low", "high"),
            ("data needed", "hundreds", "thousands"),
            ("risk of overfitting", "low", "high"),
            ("fixes a non-sigmoid shape", "no", "yes"))
    ax.text(0.55, 0.93, "Platt", fontsize=14, color=RED, ha="center")
    ax.text(0.83, 0.93, "isotonic", fontsize=14, color=AMBER, ha="center")
    ax.plot([0.0, 0.97], [0.86, 0.86], color=GREY, lw=1.1)
    for k, (lab, a, b) in enumerate(rows):
        yy = 0.75 - 0.13 * k
        ax.text(0.0, yy, lab, fontsize=11.5, color=GREY, va="center")
        ax.text(0.55, yy, a, fontsize=11.5, color=INK, ha="center", va="center")
        ax.text(0.83, yy, b, fontsize=11.5, color=INK, ha="center", va="center")

    ax = axes[1]
    sizes = (100, 200, 500, 1000, 2000, 5000, 10000)
    pl, iso = [], []
    for m in sizes:
        pls, isos = [], []
        for seed in range(12):
            s, _, y = _uncalibrated(n=m + 4000, seed=100 + seed,
                                    shape="wiggly")
            s_tr, y_tr, s_te, y_te = s[:m], y[:m], s[m:], y[m:]
            p1, _ = _platt(s_tr, y_tr, s_te)
            sx, sy = _pava(s_tr, y_tr)
            p2 = _isotonic_predict(sx, sy, s_te)
            pls.append(np.mean((p1 - y_te) ** 2))
            isos.append(np.mean((p2 - y_te) ** 2))
        pl.append(np.mean(pls))
        iso.append(np.mean(isos))
    ax.plot(sizes, pl, "o-", color=RED, lw=2.6, ms=6, label="Platt")
    ax.plot(sizes, iso, "o-", color=AMBER, lw=2.6, ms=6, label="isotonic")
    ax.set_xscale("log")
    ax.set_xlabel("rows available for calibration")
    ax.set_ylabel("Brier score on new rows")
    ax.set_title("a distortion Platt cannot represent", fontsize=12.5)
    ax.legend(frameon=False, fontsize=11)
    k = int(np.argmin(np.abs(np.array(pl) - np.array(iso))))
    ax.axvline(sizes[k], color=GREY, ls="--", lw=1.2)
    ax.text(sizes[k] * 1.15, max(max(pl), max(iso)) * 0.99,
            "isotonic overtakes\nonce it has data", fontsize=11, color=GREY,
            va="top", linespacing=1.5)
    save(fig, out, "platt_vs_isotonic")


# the ten rows of fig_brier_score's middle panel, reused so the arithmetic
# slide and the concept slide are visibly the same cohort
BRIER_DEMO_P = np.array([0.05, 0.12, 0.20, 0.31, 0.40, 0.55, 0.62, 0.70, 0.85,
                         0.93])
BRIER_DEMO_Y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1])


def _confidence(p):
    """Guo's framing: how sure the model is, whichever side it came down on."""
    return np.maximum(p, 1 - p)


def _ece_bins(p, y, m=10, strategy="width"):
    """Per-bin count, mean confidence and accuracy — the ingredients of ECE.

    `strategy="width"` is the equal-width binning of the original paper;
    "mass" puts the same number of rows in each bin instead.
    """
    conf, correct = _confidence(p), (p > 0.5) == (y == 1)
    if strategy == "width":
        edges = np.linspace(0.5, 1.0, m + 1)
    else:
        edges = np.quantile(conf, np.linspace(0, 1, m + 1))
        edges[0], edges[-1] = 0.5, 1.0
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, m - 1)
    n, cf, ac = [], [], []
    for b in range(m):
        sel = idx == b
        n.append(int(sel.sum()))
        cf.append(float(conf[sel].mean()) if sel.any() else np.nan)
        ac.append(float(correct[sel].mean()) if sel.any() else np.nan)
    return edges, np.array(n), np.array(cf), np.array(ac)


def _ece(p, y, m=10, strategy="width"):
    _, n, cf, ac = _ece_bins(p, y, m, strategy)
    ok = n > 0
    return float((n[ok] / n.sum() * np.abs(ac[ok] - cf[ok])).sum())


def fig_ece_arithmetic(out: Path) -> None:
    """Bin the predictions, compare accuracy with confidence, weight by size."""
    models, yte = _zoo_fit()
    name, q, col = models[1]                       # naive Bayes: the confident one
    m = 10
    edges, n, cf, ac = _ece_bins(q, yte, m)
    mid = (edges[:-1] + edges[1:]) / 2
    w = n / n.sum()
    gap = np.abs(ac - cf)

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.1],
                                          "wspace": 0.33})

    # -------------------------------------------------- step 1: drop into bins
    ax = axes[0]
    ax.bar(mid, n, width=(edges[1] - edges[0]) * 0.92, color=col, alpha=0.75)
    for e in edges:
        ax.axvline(e, color=GREY, lw=0.8, alpha=0.5, zorder=0)
    ax.set_xlabel(r"confidence  $\max(\hat{p},\,1-\hat{p})$")
    ax.set_ylabel("rows in the bin")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("1.  sort the predictions into bins", fontsize=12.5)
    ax.text(0.03, 0.96, f"{m} equal-width bins\n{n.sum():,} rows",
            transform=ax.transAxes, fontsize=11, color=GREY, va="top",
            linespacing=1.5)

    # ------------------------------ step 2: accuracy against confidence per bin
    ax = axes[1]
    ok = n > 0
    width = (edges[1] - edges[0]) * 0.80
    ax.bar(mid[ok], ac[ok], width=width, color=col, alpha=0.85,
           label="accuracy in the bin")
    ax.bar(mid[ok], (cf - ac)[ok], bottom=ac[ok], width=width, color=RED,
           alpha=0.45, hatch="///", edgecolor=RED, linewidth=0,
           label="the gap")
    ax.plot([0.5, 1.0], [0.5, 1.0], ls="--", color=GREY, lw=1.6)
    ax.set_xlabel("confidence bin")
    ax.set_ylabel("fraction actually correct")
    ax.set_xlim(0.47, 1.03)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("2.  how sure  vs  how often right", fontsize=12.5)

    # ------------------------------------------ step 3: weight and add them up
    ax = axes[2]
    blank(ax)
    ax.set_title("3.  weight each gap by the bin's size", fontsize=12.5)
    ax.text(0.5, 0.90,
            r"$ECE = \sum_b \dfrac{n_b}{N}\,\left|\,\mathrm{acc}_b -"
            r"\ \mathrm{conf}_b\,\right|$",
            fontsize=16, color=INK, ha="center", va="center")
    cols = (0.05, 0.30, 0.52, 0.74, 0.98)
    for x, h in zip(cols, ("bin", r"$n_b/N$", r"$\mathrm{conf}_b$",
                           r"$\mathrm{acc}_b$", "share")):
        ax.text(x, 0.72, h, fontsize=11, color=GREY, ha="center")
    ax.plot([0.0, 1.0], [0.675, 0.675], color=GREY, lw=1.1)
    show = [b for b in range(m) if n[b] > 0][-5:]
    ax.text(cols[0], 0.625, "⋮", fontsize=11, color=GREY, ha="center",
            va="center")
    for k, b in enumerate(show):
        yy = 0.575 - 0.072 * k
        ax.text(cols[0], yy, f"{edges[b]:.2f}", fontsize=10.5, color=GREY,
                ha="center", va="center")
        ax.text(cols[1], yy, f"{w[b]:.2f}", fontsize=10.5, color=INK,
                ha="center", va="center")
        ax.text(cols[2], yy, f"{cf[b]:.2f}", fontsize=10.5, color=RED,
                ha="center", va="center")
        ax.text(cols[3], yy, f"{ac[b]:.2f}", fontsize=10.5, color=col,
                ha="center", va="center")
        ax.text(cols[4], yy, f"{w[b] * gap[b]:.3f}", fontsize=10.5, color=INK,
                ha="center", va="center")
    ax.plot([0.60, 1.0], [0.175, 0.175], color=GREY, lw=1.2)
    ax.text(cols[3], 0.1, "ECE", fontsize=13, color=INK, ha="center")
    ax.text(cols[4], 0.1, f"{_ece(q, yte, m):.3f}", fontsize=13, color=INK, ha="center")
    save(fig, out, "ece_arithmetic")


def fig_ece_caveats(out: Path) -> None:
    """Three reasons not to report ECE on its own."""
    models, yte = _zoo_fit()
    name, q, col = models[1]
    # a model whose predictions ARE the data-generating risk: calibration is
    # perfect by construction, so whatever ECE it scores is pure measurement error
    p_true, y_true = _risk_scores(60_000, 5)

    def mean_ece(p, y, n, m, draws=60, seed=0):
        rng = np.random.default_rng(seed)
        return float(np.mean([_ece(p[i], y[i], m) for i in
                              (rng.choice(len(y), n, replace=False)
                               for _ in range(draws))]))

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15],
                                          "wspace": 0.33})

    # ------------------------------- more bins, more noise mistaken for miscalibration
    ax = axes[0]
    ms = (5, 10, 15, 20, 30, 50, 80)
    perfect = [mean_ece(p_true, y_true, 500, m) for m in ms]
    nb = [mean_ece(q, yte, 500, m) for m in ms]
    ax.plot(ms, nb, "o-", color=col, lw=2.6, ms=6, label='uncalibrated model')
    ax.plot(ms, perfect, "o-", color=GREEN, lw=2.6, ms=6,  label='calibrated model')
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms], fontsize=9.5)
    ax.minorticks_off()
    ax.set_xlabel("number of bins  $M$")
    ax.set_ylabel("ECE  (500 rows)")
    ax.legend()
    ax.set_ylim(0, max(perfect + nb) * 1.25)
    ax.set_title("more bins => higher ECE",
                 fontsize=12, linespacing=1.5)

    # --------------------------------------------- the floor depends on the cohort
    ax = axes[1]
    ns = (250, 500, 1000, 2000, 5000, 20_000)
    floor = [mean_ece(p_true, y_true, n, 10, draws=40) for n in ns]
    nb_n = [mean_ece(q, yte, n, 10, draws=40) for n in ns]
    ax.plot(ns, nb_n, "o-", color=col, lw=2.6, ms=6)
    ax.plot(ns, floor, "o-", color=GREEN, lw=2.6, ms=6)
    ax.fill_between(ns, 0, floor, color=GREEN, alpha=0.12)
    ax.text(ns[0] * 1.15, floor[0] * 0.28, "noise floor", color=GREEN,
            fontsize=11, va="center")
    ax.text(ns[-1], nb_n[-1] + 0.01, 'uncalibrated model', color=col, fontsize=11,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=9, rotation=45,
                       ha="right")
    ax.minorticks_off()
    ax.set_xlabel("rows evaluated  ($M = 10$)")
    ax.set_ylabel("ECE")
    ax.set_ylim(0, max(nb_n) * 1.35)
    ax.set_title("ECE has intrinsic noise", fontsize=12, linespacing=1.5)

    # ------------------------------------- a useless model can score perfectly
    ax = axes[2]
    base = np.full_like(q, yte.mean())
    entries = (('uncalibrated model', q, col),
               ("predict base rate\nfor all rows", base, GREY))
    blank(ax)
    ax.set_title("model is worse than a constant", fontsize=12,
                 linespacing=1.5)
    for x, h in ((0.68, "ECE"), (0.92, "ROC-AUC")):
        ax.text(x, 0.82, h, fontsize=11.5, color=GREY, ha="center")
    ax.plot([0.0, 1.0], [0.75, 0.75], color=GREY, lw=1.1)
    for k, (lab, p, c) in enumerate(entries):
        yy = 0.62 - 0.20 * k
        ax.text(0.0, yy, lab, fontsize=11, color=c, va="center",
                linespacing=1.5)
        ax.text(0.68, yy, f"{_ece(p, yte):.3f}", fontsize=13, color=c,
                ha="center", va="center")
        ax.text(0.92, yy, f"{_auc(p, yte):.3f}", fontsize=13, color=c,
                ha="center", va="center")
    ax.plot([0.0, 1.0], [0.30, 0.30], color=FAINT, lw=1.4)
    ax.text(0.0, 0.15, '"a model saying nothing is never over-sure"',
                fontsize=11.5, color=AMBER, va="center", linespacing=1.8)
    ax.text(0.0, 0, "ECE is not a scoring rule",
                fontsize=11.5, color=RED, va="center", linespacing=1.8)
    save(fig, out, "ece_caveats")


def fig_cox_arithmetic(out: Path) -> None:
    """Where α and β come from: one logistic regression on the log-odds axis."""
    models, yte = _zoo_fit()
    name, q, col = models[1]                       # naive Bayes: β = 0.57
    a, b = _slope_intercept(q, yte)

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.1],
                                          "wspace": 0.30})

    # ------------------------------------------------------------- the recipe
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.80,
            r"$\mathrm{logit}\,\Pr(y=1) \;=\; \alpha \;+\; \beta\,"
            r"\mathrm{logit}(\hat{p})$",
            fontsize=21, color=INK, ha="center", va="center")
    ax.annotate("the intercept", xy=(0.555, 0.735), xytext=(0.4, 0.6),
                fontsize=11.5, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.1))
    ax.annotate("the slope", xy=(0.755, 0.735), xytext=(0.7, 0.6),
                fontsize=11.5, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.1))
    steps = (
        r"1.  turn each $\hat{p}$ into log-odds,  "
        r"$\log\,\hat{p}/(1-\hat{p})$",
        r"2.  logistic-regress $y$ on that ONE number",
        r"3.  the two fitted coefficients are $\alpha$ and $\beta$",
    )
    for k, t in enumerate(steps):
        ax.text(0.02, 0.44 - 0.115 * k, t, fontsize=12.5, color=INK,
                va="center")
    ax.text(0.5, 0.05,
            r"a calibrated model scores $\alpha = 0$ and $\beta = 1$",
            fontsize=12.5, color=GREY, ha="center", va="center",
            linespacing=1.8)

    # ------------------------------------------------- the fit, read off a plot
    ax = axes[1]
    xs_, ys_, ns = _bins(q, yte, 12)
    lx, ly = logit(xs_), logit(np.clip(ys_, 0.02, 0.98))
    lim = 4.6
    grid = np.linspace(-lim, lim, 200)
    ax.plot(grid, grid, ls="--", color=GREY, lw=1.5)
    ax.text(-lim + 0.25, lim - 0.7, r"honest:  $\alpha=0,\ \beta=1$",
            color=GREY, fontsize=11, ha="left")
    ax.plot(grid, a + b * grid, color=col, lw=3.0)
    ax.scatter(lx, ly, s=np.clip(ns, 10, 90), color=col, alpha=0.65,
               edgecolors="none", zorder=3)
    ax.axhline(0, color=FAINT, lw=1.4, zorder=0)
    ax.axvline(0, color=FAINT, lw=1.4, zorder=0)

    # α: where the fitted line crosses x = 0
    ax.plot([0], [a], "o", color=AMBER, ms=10, zorder=4)
    ax.annotate(rf"$\alpha = {a:+.2f}$",
                xy=(0, a), xytext=(-2.5, 1.9), fontsize=12, color=AMBER,
                ha="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.2))
    # β: a run-and-rise triangle on the fitted line
    x0, run = 1.2, 2.2
    y0, y1 = a + b * x0, a + b * (x0 + run)
    ax.plot([x0, x0 + run, x0 + run], [y0, y0, y1], color=TEAL, lw=2.0)
    ax.text(x0 + run / 2, y0 - 0.45, f"run {run:.1f}", fontsize=10.5,
            color=TEAL, ha="center", va="top")
    ax.text(x0 + run + 0.15, (y0 + y1) / 2, f"rise {y1 - y0:.2f}",
            fontsize=10.5, color=TEAL, va="center")
    ax.text(-0.3, -3.6, rf"$\beta = \dfrac{{{y1 - y0:.2f}}}{{{run:.1f}}}"
                        rf" = {b:.2f}$",
            fontsize=13.5, color=TEAL, ha="center", va="center")
    ax.set_xlabel(r"$\mathrm{logit}(\hat{p})$ (model prediction)")
    ax.set_ylabel("log-odds of reality")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"uncalibrated model", fontsize=12.5, color=col)
    save(fig, out, "cox_arithmetic")


def fig_cox_reading(out: Path) -> None:
    """What each of the two numbers does to a reliability curve."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.2],
                                          "wspace": 0.34})
    grid = np.linspace(0.004, 0.996, 400)
    lg = logit(grid)

    # --------------------------------------------------------------- the slope
    ax = axes[0]
    _diagonal(ax, label=False)
    for beta, c in ((0.5, RED), (1.0, GREY), (2.0, BLUE)):
        ax.plot(grid, sigmoid(beta * lg), color=c, lw=2.8,
                label=rf"$\beta = {beta}$")
    _calib_axes(ax)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.text(0.5, -0.20, "β < 1: too extreme —> shrink them\n"
                        "β > 1: too timid —> stretch them",
            transform=ax.transAxes, fontsize=11, color=TEAL, ha="center",
            va="top", linespacing=1.7)
    ax.set_title(r"the slope tilts the curve   ($\alpha = 0$)", fontsize=12.5,
                 color=TEAL)

    # ----------------------------------------------------------- the intercept
    ax = axes[1]
    _diagonal(ax, label=False)
    for alpha, c in ((-1.0, RED), (0.0, GREY), (1.0, BLUE)):
        ax.plot(grid, sigmoid(alpha + lg), color=c, lw=2.8,
                label=rf"$\alpha = {alpha:+.0f}$".replace("+0", "0"))
    _calib_axes(ax, ylabel="")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.text(0.5, -0.20, "α < 0: predictions too high\n"
                        "α > 0: predictions too low",
            transform=ax.transAxes, fontsize=11, color=AMBER, ha="center",
            va="top", linespacing=1.7)
    ax.set_title(r"the intercept shifts it   ($\beta = 1$)", fontsize=12.5,
                 color=AMBER)

    # ------------------------------------------------- the four models, placed
    ax = axes[2]
    models, yte = _zoo_fit()
    # hand-placed so four labels and a star share one small panel
    where = {"logistic regression": (0.16, 1.30, "left", "bottom"),
             "naive Bayes": (0.40, 0.52, "left", "center"),
             "random forest": (0.46, 2.02, "left", "center"),
             "linear SVC": (1.28, 3.56, "center", "top")}
    ax.axhline(1.0, color=GREY, ls="--", lw=1.4)
    ax.axvline(0.0, color=GREY, ls="--", lw=1.4)
    ax.plot([0], [1], "*", color=INK, ms=20, zorder=4)
    ax.text(-0.12, 0.80, "perfect\ncalibration", fontsize=11.5, color=INK, ha="center",
            va="top")
    for name, q, col in models:
        a, b = _slope_intercept(q, yte)
        ax.plot([a], [b], "o", color=col, ms=11, zorder=3)
        tx, ty, ha, va = where[name]
        ax.text(tx, ty, f"{name}\nα {a:+.2f},  β {b:.2f}", fontsize=10,
                color=col, ha=ha, va=va, linespacing=1.4)
    ax.set_xlabel(r"intercept  $\alpha$")
    ax.set_ylabel(r"slope  $\beta$")
    ax.set_xlim(-0.45, 2.15)
    ax.set_ylim(0.15, 4.5)
    ax.set_title("classic uncalibrated models", fontsize=12.5)
    save(fig, out, "cox_reading")


def _roc_curve(p, y):
    """FPR and TPR at every threshold, walking the scores from high to low."""
    o = np.argsort(-p, kind="mergesort")
    yy = y[o]
    tpr = np.r_[0, np.cumsum(yy) / max(yy.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - yy) / max((1 - yy).sum(), 1)]
    return fpr, tpr, np.r_[np.inf, p[o]]


def _rates_at(p, y, thr):
    pred = p >= thr
    return (pred & (y == 0)).sum() / (y == 0).sum(), \
           (pred & (y == 1)).sum() / (y == 1).sum()


def fig_auc_build(out: Path) -> None:
    """Sweep a threshold, trace a curve, measure the area under it."""
    models, yte = _zoo_fit()
    name, q, col = models[0]                       # logistic regression, AUC .896
    thr = 0.60
    fpr, tpr, _ = _roc_curve(q, yte)
    f0, t0 = _rates_at(q, yte, thr)

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0],
                                          "wspace": 0.30})

    # ------------------------------------------- 1. two populations, one cut
    ax = axes[0]
    bins = np.linspace(0, 1, 40)
    ax.hist(q[yte == 0], bins=bins, color=BLUE, alpha=0.55, label="labeled 0")
    ax.hist(q[yte == 1], bins=bins, color=RED, alpha=0.55, label="labeled 1")
    ax.axvline(thr, color=INK, lw=2.4, alpha=0.5)
    ax.set_xlabel("model score")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.text(0.95, 0.55, f"above the cut:\n{100 * t0:.0f}% TP\n{100 * f0:.0f}% FP",
            transform=ax.transAxes, fontsize=10.5, color=INK, ha="right",
            va="center", linespacing=1.6)

    # ---------------------------------------------- 2. sweep it, trace a curve
    ax = axes[1]
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.5)
    ax.plot(fpr, tpr, color=col, lw=3.0)
    ax.plot([f0], [t0], "o", color=INK, ms=11, zorder=4)
    ax.annotate(f"the cut at {thr:g}", xy=(f0, t0), xytext=(f0 + 0.25, t0 - 0.1),
                fontsize=11, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
    for t, lab in ((0.20, "0.2"), (0.85, "0.85")):
        fa, ta = _rates_at(q, yte, t)
        ax.plot([fa], [ta], "o", color=GREY, ms=7, zorder=3)
        ax.text(fa + 0.03, ta - 0.05, lab, fontsize=10, color=GREY)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("every threshold is a point on the curve", fontsize=12.5)

    # ------------------------------------------------- 3. the area under it
    ax = axes[2]
    ax.fill_between(fpr, 0, tpr, color=col, alpha=0.22)
    ax.plot(fpr, tpr, color=col, lw=3.0)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.5)
    ax.text(0.55, 0.30, f"AUC\n{_auc(q, yte):.3f}", fontsize=17, color=col,
            ha="center", va="center", linespacing=1.5)
    ax.text(0.97, 0.03, "coin flip = 0.50\nperfect = 1.00", fontsize=10.5,
            color=GREY, ha="right", va="bottom", linespacing=1.6)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("AUC: area under the curve", fontsize=12.5, color=col)
    save(fig, out, "auc_build")


def _recalibration_demo(n=6000, seed=2):
    """The raw scorer of the Platt slides, plus its two repaired versions."""
    s, praw, y = _uncalibrated(n=n, seed=seed)
    h = len(s) // 2
    platt, _ = _platt(s[:h], y[:h], s[h:])
    sx, sy = _pava(s[:h], y[:h])
    iso = _isotonic_predict(sx, sy, s[h:])
    return praw[h:], platt, iso, y[h:]


def fig_auc_monotone(out: Path) -> None:
    """A strictly increasing map cannot reorder anybody."""
    raw, platt, _, y = _recalibration_demo()
    # spread across the range rather than sampled at random: the raw model is
    # bimodal, so a random draw would leave the middle of the slopegraph empty
    o_all = np.argsort(raw)
    sub = o_all[(np.linspace(0.02, 0.98, 11) * (len(raw) - 1)).astype(int)]

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.05, 1.05],
                                          "wspace": 0.32})

    # ------------------------------------------- 1. nobody overtakes anybody
    ax = axes[0]
    blank(ax)
    ax.set_title("Calibration map moves all predictions", fontsize=12.5)
    for k, i in enumerate(sub):
        ax.plot([0.22, 0.78], [raw[i], platt[i]], color=BLUE, lw=1.6, alpha=0.8)
        ax.plot([0.22], [raw[i]], "o", color=BLUE, ms=6)
        ax.plot([0.78], [platt[i]], "o", color=BLUE, ms=6)
    ax.text(0.22, 1.06, "before", fontsize=12, color=INK, ha="center")
    ax.text(0.78, 1.06, "after Platt", fontsize=12, color=INK, ha="center")
    ax.set_ylim(-0.14, 1.16)
    ax.text(0.5, -0.09, "the lines never cross", fontsize=12.5, color=GREEN,
            ha="center", va="center")

    # -------------------------------------------------- 2. the map is increasing
    ax = axes[1]
    o = np.argsort(raw)
    ax.plot(raw[o], platt[o], color=PURPLE, lw=3.0)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.4)
    ax.plot(raw[sub], platt[sub], "o", color=PURPLE, ms=7, zorder=3)
    ax.set_xlabel("probability before")
    ax.set_ylabel("probability after")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration map is strictly increasing", fontsize=12.5)
    ax.text(0.97, 0.06, r"$s_i > s_j \Rightarrow \hat{p}_i > \hat{p}_j$",
            transform=ax.transAxes, fontsize=11, color=INK, ha="right",
            va="bottom", linespacing=1.7)

    # ------------------------------------------------ 3. so the ROC is the same
    ax = axes[2]
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.4)
    f1, t1, _ = _roc_curve(raw, y)
    f2, t2, _ = _roc_curve(platt, y)
    ax.plot(f1, t1, color=RED, lw=5.0, alpha=0.5)
    ax.plot(f2, t2, color=GREEN, lw=2.2, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ROC curve is unchanged", fontsize=12)
    ax.text(0.97, 0.22, f"before   AUC {_auc(raw, y):.4f}", transform=ax.transAxes,
            fontsize=11.5, color=RED, ha="right")
    ax.text(0.97, 0.12, f"after    AUC {_auc(platt, y):.4f}",
            transform=ax.transAxes, fontsize=11.5, color=GREEN, ha="right")
    save(fig, out, "auc_monotone")


def fig_auc_exception(out: Path) -> None:
    """Ties are the one way a calibrator can cost you discrimination."""
    raw, platt, iso, y = _recalibration_demo()

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.05],
                                          "wspace": 0.32})

    # --------------------------------------------------- 1. the staircase ties
    ax = axes[0]
    o = np.argsort(raw)
    ax.plot(raw[o], platt[o], color=RED, lw=2.4, label="Platt")
    ax.plot(raw[o], iso[o], color=AMBER, lw=2.4, label="isotonic")
    # shade the widest flat step, whichever it turns out to be
    lvl = np.unique(iso)
    spans = [(raw[iso == v].min(), raw[iso == v].max(), v) for v in lvl]
    lo, hi, flat = max(spans, key=lambda t: t[1] - t[0])
    ax.axvspan(lo, hi, color=FAINT, zorder=0)
    ax.set_xlabel("probability before")
    ax.set_ylabel("probability after")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.annotate("these rows all\nget the same value",
                xy=((lo + hi) / 2, flat), xytext=(0.70, 0.22), fontsize=10.5,
                color=AMBER, ha="center", va="top", linespacing=1.6,
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.1))

    # ------------------------------------------------ 2. ties cost a little AUC
    ax = axes[1]
    blank(ax)
    rows = (("raw", raw, GREY), ("Platt", platt, RED), ("isotonic", iso, AMBER))
    ax.text(0.58, 0.86, "distinct values", fontsize=11, color=GREY,
            ha="center")
    ax.text(0.92, 0.86, "AUC", fontsize=11, color=GREY, ha="center")
    ax.plot([0.0, 1.0], [0.79, 0.79], color=GREY, lw=1.1)
    for k, (lab, p, c) in enumerate(rows):
        yy = 0.67 - 0.15 * k
        ax.text(0.0, yy, lab, fontsize=12.5, color=c, va="center")
        ax.text(0.58, yy, f"{len(np.unique(p)):,}", fontsize=12.5, color=INK,
                ha="center", va="center")
        ax.text(0.92, yy, f"{_auc(p, y):.4f}", fontsize=12.5, color=c,
                ha="center", va="center")

    # ---------------------------------------------------- 3. the two axes
    ax = axes[2]
    blank(ax)
    box(ax, (0.24, 0.72), 0.45, 0.20,
        "is the rank\ncorrect?", ec=BLUE, fs=11, tc=BLUE)
    box(ax, (0.76, 0.72), 0.45, 0.20,
        "are numbers\nmeaningfull?", ec=PURPLE, fs=10.5, tc=PURPLE)
    ax.text(0.26, 0.50, "AUC", fontsize=15, color=BLUE, ha="center")
    ax.text(0.74, 0.50, "calibration", fontsize=15, color=PURPLE, ha="center")
    arrow(ax, (0.26, 0.60), (0.26, 0.55), color=BLUE, lw=1.4)
    arrow(ax, (0.74, 0.60), (0.74, 0.55), color=PURPLE, lw=1.4)
    save(fig, out, "auc_exception")


def _wilson(k, n, z=1.96):
    """Wilson interval for a bin's observed rate — honest when n is small or p→0."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return c - hw, c + hw


def _reliability_cohort(n=500, seed=3):
    """One clinic-sized slice of the random forest from the opening line-up."""
    models, yte = _zoo_fit()
    _, q, col = models[2]
    idx = np.random.default_rng(seed).choice(len(yte), n, replace=False)
    return q[idx], yte[idx], col


def fig_reliability_build(out: Path) -> None:
    """From a cloud of 0s and 1s to a curve, in two moves."""
    p, y, col = _reliability_cohort()
    rng = np.random.default_rng(0)
    jit = rng.normal(0, 0.028, len(y))
    m = 5
    edges = np.quantile(p, np.linspace(0, 1, m + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, m - 1)
    xs = np.array([p[idx == b].mean() for b in range(m)])
    ys = np.array([y[idx == b].mean() for b in range(m)])

    fig, axd = plt.subplot_mosaic(
        [["raw", "bin", "rel"], ["raw", "bin", "hist"]], figsize=WIDE,
        height_ratios=[3.0, 1.0],
        gridspec_kw={"wspace": 0.33, "hspace": 0.12})

    # ------------------------------------------- 1. what you actually have
    ax = axd["raw"]
    ax.scatter(p, y + jit, s=16, color=col, alpha=0.5, edgecolors="none")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["labeled 0", "labeled 1"], fontsize=10)
    ax.set_xlabel(r"predicted probability  $\hat{p}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.18, 1.18)
    ax.text(0.5, 0.5, "probability is never observed\nonly the result is observed",
            transform=ax.transAxes, fontsize=11.5, color=GREY, ha="center",
            va="center", linespacing=1.6)

    # --------------------------------------------- 2. group, then average
    ax = axd["bin"]
    for b in range(m):
        if b % 2 == 0:
            ax.axvspan(edges[b], edges[b + 1], color=FAINT, zorder=0)
    ax.scatter(p, y + jit, s=14, color=col, alpha=0.32, edgecolors="none")
    for b in range(m):
        ax.plot([edges[b], edges[b + 1]], [ys[b], ys[b]], color=RED, lw=2.4)
        ax.plot([xs[b]], [ys[b]], "o", color=RED, ms=10, zorder=4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=10)
    ax.set_xlabel(r"predicted probability  $\hat{p}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.18, 1.18)
    ax.set_title("bin and average each bin", fontsize=12.5)
    ax.text(0.97, 0.38, "x = mean prediction\ny = fraction of label 1",
            transform=ax.transAxes, fontsize=10.5, color=RED, va="center",
            ha="right", linespacing=1.6)

    # ------------------------------------------------ 3. the finished plot
    ax = axd["rel"]
    _diagonal(ax, label=False)
    ax.plot(xs, ys, "o-", color=col, lw=2.6, ms=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("observed frequency", fontsize=10.5)
    ax.set_xticklabels([])
    ax.set_title("reliability diagram", fontsize=12.5)

    ax = axd["hist"]
    ax.hist(p, bins=np.linspace(0, 1, 26), color=col, alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(r"$\hat{p}$", fontsize=10.5)
    ax.set_ylabel("how many", fontsize=9.5)
    for s_ in ("left",):
        ax.spines[s_].set_visible(False)
    save(fig, out, "reliability_build")


def fig_reliability_choices(out: Path) -> None:
    """The three decisions hiding inside a curve you were handed."""
    p, y, col = _reliability_cohort()

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"wspace": 0.33})

    def curve(p, y, m, strategy="quantile"):
        if strategy == "quantile":
            edges = np.quantile(p, np.linspace(0, 1, m + 1))
        else:
            edges = np.linspace(0, 1, m + 1)
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, m - 1)
        xs, ys, ns = [], [], []
        for b in range(m):
            sel = idx == b
            if sel.sum() == 0:
                continue
            xs.append(p[sel].mean()); ys.append(y[sel].mean())
            ns.append(int(sel.sum()))
        return np.array(xs), np.array(ys), np.array(ns)

    # ----------------------------------------------- how many bins
    ax = axes[0]
    _diagonal(ax, label=False)
    for m, c in ((5, BLUE), (10, GREEN), (25, AMBER)):
        xs, ys, _ = curve(p, y, m)
        ax.plot(xs, ys, "o-", color=c, lw=2.2, ms=5, label=f"{m} bins")
    _calib_axes(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_title("three bin counts", fontsize=12,
                 linespacing=1.5)

    # ----------------------------------------------- how the bins are drawn
    ax = axes[1]
    _diagonal(ax, label=False)
    smallest = {}
    for strat, c, lab in (("quantile", col, "equal mass"),
                          ("width", AMBER, "equal width")):
        xs, ys, ns = curve(p, y, 10, strat)
        ax.plot(xs, ys, "o-", color=c, lw=2.2, ms=5, label=lab)
        smallest[lab] = int(ns.min())
    _calib_axes(ax, ylabel="")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.text(0.97, 0.15, f"smallest bin: {smallest['equal mass']} rows",
            transform=ax.transAxes, fontsize=10.5, color=col, ha="right")
    ax.text(0.97, 0.05, f"smallest bin: {smallest['equal width']} rows",
            transform=ax.transAxes, fontsize=10.5, color=AMBER, ha="right")
    ax.set_title("bins types", fontsize=12)

    # ----------------------------------------------- and how sure each point is
    ax = axes[2]
    _diagonal(ax, label=False)
    xs, ys, ns = curve(p, y, 10)
    lo, hi = zip(*[_wilson(round(a * b), b) for a, b in zip(ys, ns)])
    ax.errorbar(xs, ys, yerr=[ys - np.array(lo), np.array(hi) - ys], fmt="o-",
                color=col, lw=2.2, ms=5, ecolor=col, elinewidth=1.6, capsize=4,
                alpha=0.95)
    _calib_axes(ax, ylabel="")
    ax.set_title("95% intervals", fontsize=12)
    ax.text(0.97, 0.05, "wiggle is mostly noise",
            transform=ax.transAxes, fontsize=10.5, color=INK, ha="right",
            va="bottom", linespacing=1.6)
    save(fig, out, "reliability_choices")


def fig_brier_arithmetic(out: Path) -> None:
    """The formula, and the same ten rows worked through it by hand."""
    ph, yv = BRIER_DEMO_P, BRIER_DEMO_Y
    diff = ph - yv
    pen = diff ** 2

    fig, axes = plt.subplots(1, 2, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.25],
                                          "wspace": 0.04})

    # ------------------------------------------------------------- the formula
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.74,
            r"$BS \;=\; \dfrac{1}{N}\sum_{i=1}^{N}"
            r"\left(\hat{p}_i - y_i\right)^2$",
            fontsize=27, color=INK, ha="center", va="center")
    ax.annotate("number of rows", xy=(0.325, 0.615),
                xytext=(0.13, 0.42), fontsize=11, color=GREY, ha="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.1))
    ax.annotate("the probability\nthe model reported", xy=(0.625, 0.655),
                xytext=(0.52, 0.42), fontsize=11, color=BLUE, ha="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1))
    ax.annotate("1 if it happened,\n0 if it did not", xy=(0.800, 0.655),
                xytext=(0.87, 0.42), fontsize=11, color=RED, ha="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.1))
    ax.text(0.5, 0.05, r"$0 \leq BS \leq 1$,  lower is better,  $0$ is perfect",
            fontsize=12.5, color=GREY, ha="center", va="center")

    # --------------------------------------------------------- worked by hand
    ax = axes[1]
    blank(ax)
    cols = (0.10, 0.34, 0.53, 0.75, 0.97)
    heads = (r"patient", r"$\hat{p}_i$", r"$y_i$", r"$\hat{p}_i - y_i$",
             r"$(\hat{p}_i - y_i)^2$")
    for x, h in zip(cols, heads):
        ax.text(x, 0.955, h, fontsize=12, color=GREY, ha="center", va="center")
    ax.plot([0.02, 1.0], [0.905, 0.905], color=GREY, lw=1.2)

    for k in range(len(ph)):
        yy = 0.860 - 0.070 * k
        col = RED if yv[k] else BLUE
        ax.text(cols[0], yy, f"{k + 1}", fontsize=11.5, color=GREY,
                ha="center", va="center")
        ax.text(cols[1], yy, f"{ph[k]:.2f}", fontsize=11.5, color=INK,
                ha="center", va="center")
        ax.text(cols[2], yy, f"{yv[k]}", fontsize=11.5, color=col,
                ha="center", va="center")
        ax.text(cols[3], yy, f"{diff[k]:+.2f}", fontsize=11.5, color=col,
                ha="center", va="center")
        ax.text(cols[4], yy, f"{pen[k]:.4f}", fontsize=11.5, color=INK,
                ha="center", va="center")

    ytot = 0.860 - 0.070 * len(ph) - 0.010
    ax.plot([0.60, 1.0], [ytot + 0.030, ytot + 0.030], color=GREY, lw=1.2)
    ax.text(cols[3], ytot - 0.012, "sum", fontsize=12, color=GREY,
            ha="center", va="center")
    ax.text(cols[4], ytot - 0.012, f"{pen.sum():.4f}", fontsize=12.5,
            color=INK, ha="center", va="center")
    ax.text(0.60, ytot - 0.095,
            rf"$BS = \dfrac{{{pen.sum():.4f}}}{{{len(ph)}}} = "
            rf"\mathbf{{{pen.mean():.3f}}}$",
            fontsize=15, color=PURPLE, ha="center", va="center")
    save(fig, out, "brier_arithmetic")


def fig_brier_score(out: Path) -> None:
    """One patient, then a cohort, then what the single number hides.

    The third panel runs on the four classifiers of the opening slide rather than
    a fresh example, so the decomposition lands on models the room has already
    met — and the naive-Bayes-beats-logistic paradox flagged there gets its answer.
    """
    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.1, 1.15],
                                          "wspace": 0.34})

    # ------------------------------------------------- one patient, one penalty
    ax = axes[0]
    grid = np.linspace(0, 1, 300)
    ax.plot(grid, (grid - 1) ** 2, color=RED, lw=2.8)
    ax.plot(grid, grid ** 2, color=BLUE, lw=2.8)
    ax.text(0.14, 0.45, "$y=1$", color=RED, fontsize=11.5,
            ha="center", va="center", linespacing=1.5)
    ax.text(0.86, 0.45, "$y=0$", color=BLUE, fontsize=11.5,
            ha="center", va="center", linespacing=1.5)
    ax.set_xlabel(r"predicted probability  $\hat{p}$")
    ax.set_ylabel("penalty")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_title("squared error, per row", fontsize=12.5)

    # ------------------------------------------ the cohort: average the penalties
    ax = axes[1]
    ph, yv = BRIER_DEMO_P, BRIER_DEMO_Y
    pen = (ph - yv) ** 2
    xs = np.arange(len(ph))
    ax.bar(xs, pen, color=[RED if v else BLUE for v in yv], width=0.62)
    bs = pen.mean()
    ax.axhline(bs, color=INK, lw=2.2, ls="--")
    ax.text(len(ph) - 0.45, bs + 0.02, f"Brier (mean) = {bs:.3f}",
            fontsize=12.5, color=INK, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:.2f}" for v in ph], fontsize=9, rotation=55,
                       ha="right")
    for t, v in zip(ax.get_xticklabels(), yv):
        t.set_color(RED if v else BLUE)
    ax.set_ylabel("squared error")
    ax.set_ylim(0, 0.75)
    ax.set_xlabel(r"$\hat{p}$")
    ax.set_title("average over the cohort", fontsize=12.5)

    # ------------------------------------------------- what the number hides
    models, yte = _zoo_fit()
    ybar = yte.mean()
    unc = ybar * (1 - ybar)

    def decompose(q):
        """Murphy: BS = miscalibration + (irreducible − resolution)."""
        xs_, ys_, ns = _bins(q, yte, 12)
        w = ns / ns.sum()
        return (float((w * (xs_ - ys_) ** 2).sum()),
                float((w * (ys_ - ybar) ** 2).sum()))

    ax = axes[2]
    names = [n for n, _, _ in models]
    parts = [decompose(q) for _, q, _ in models]
    rel = np.array([p[0] for p in parts])
    res = np.array([p[1] for p in parts])
    floor = unc - res
    x = np.arange(len(names))
    ax.bar(x, floor, color=FAINT, edgecolor=GREY, width=0.58,
           label="what its ranking cannot avoid")
    ax.bar(x, rel, bottom=floor, color=PURPLE, width=0.58,
           label="miscalibration")
    for k in range(len(names)):
        ax.text(k, floor[k] + rel[k] + 0.006, f"{floor[k] + rel[k]:.3f}",
                fontsize=11.5, color=INK, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=10,
                       linespacing=1.4)
    ax.set_ylabel("Brier score")
    ax.set_ylim(0, 0.25)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    save(fig, out, "brier_score")


def _overconfident_net(n=8000, seed=17, T_true=2.5):
    """A network-like scorer whose logits are inflated by a constant factor.

    Multiplying the honest log-odds by T_true is exactly the distortion a
    modern network acquires by training to near-zero training loss: the ranking
    is untouched, every logit is simply too large. It is also, conveniently, the
    one distortion a single temperature can undo exactly.
    """
    p, y = _risk_scores(n, seed)
    return T_true * logit(p), y


def _nll(z, y, T):
    q = np.clip(sigmoid(z / T), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _fit_temperature(z, y, grid=None):
    """One parameter, found by scanning — the search space is one-dimensional."""
    grid = np.linspace(0.30, 6.0, 800) if grid is None else grid
    losses = np.array([_nll(z, y, t) for t in grid])
    return grid[int(np.argmin(losses))], grid, losses


def fig_temperature_scaling(out: Path) -> None:
    """One knob, fitted on held-out logits by minimising the loss."""
    z, y = _overconfident_net()
    h = len(z) // 2
    T, grid, losses = _fit_temperature(z[:h], y[:h])

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.05],
                                          "wspace": 0.33})

    # --------------------------------------------------- what dividing by T does
    ax = axes[0]
    zz = np.linspace(-8, 8, 400)
    for t, c, lab in ((1.0, GREY, r"$T = 1$  (raw)"), (2.5, PURPLE, r"$T = 2.5$"),
                      (5.0, BLUE, r"$T = 5$")):
        ax.plot(zz, sigmoid(zz / t), color=c, lw=2.8, label=lab)
    ax.axhline(0.5, color=FAINT, lw=1.4, zorder=0)
    ax.axvline(0.0, color=FAINT, lw=1.4, zorder=0)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_xlabel("network logit  $z$")
    ax.set_ylabel(r"$\sigma(z/T)$")
    ax.set_ylim(-0.02, 1.02)

    # ------------------------------------------------------ how T is chosen
    ax = axes[1]
    ax.plot(grid, losses, color=TEAL, lw=2.8)
    ax.plot([T], [_nll(z[:h], y[:h], T)], "o", color=TEAL, ms=10, zorder=3)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.4)
    ax.annotate(rf"$\hat{{T}} = {T:.2f}$", xy=(T, _nll(z[:h], y[:h], T)),
                xytext=(T + 1.5, _nll(z[:h], y[:h], T) + 0.10), fontsize=13,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.2))
    ax.set_xlabel("temperature  $T$")
    ax.set_ylabel("log loss on the validation set")
    ax.set_xlim(0.3, 6.0)
    ax.set_ylim(losses.min() - 0.03, 1.25)

    # ------------------------------------------- what it does to the confidences
    ax = axes[2]
    raw, cal = sigmoid(z[h:]), sigmoid(z[h:] / T)
    conf_r, conf_c = np.maximum(raw, 1 - raw), np.maximum(cal, 1 - cal)
    bins = np.linspace(0.5, 1.0, 26)
    ax.hist(conf_r, bins=bins, color=RED, alpha=0.55, label="raw")
    ax.hist(conf_c, bins=bins, color=TEAL, alpha=0.65, label="after  $T$")
    ax.set_xlabel("confidence  $\\max(\\hat{p},\\,1-\\hat{p})$")
    ax.set_ylabel("rows")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    save(fig, out, "temperature_scaling")


def fig_temperature_effect(out: Path) -> None:
    """What it repairs, what it cannot, and how it relates to Platt."""
    z, y = _overconfident_net()
    h = len(z) // 2
    T, _, _ = _fit_temperature(z[:h], y[:h])
    raw, cal = sigmoid(z[h:]), sigmoid(z[h:] / T)
    yte = y[h:]
    # Platt, fitted on the same held-out half, for the side-by-side
    platt, (b0, b1) = _platt(z[:h], y[:h], z[h:])

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15],
                                          "wspace": 0.33})

    ax = axes[0]
    _diagonal(ax, label=False)
    for p, c, lab in ((raw, RED, "raw"), (cal, TEAL, f"T = {T:.2f}")):
        xs_, ys_, _ = _bins(p, yte, 12)
        ax.plot(xs_, ys_, "o-", color=c, lw=2.6, ms=6, label=lab)
    _calib_axes(ax)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title("on held-out rows", fontsize=12.5)

    # ------------------------------------------------------------- the scoreboard
    ax = axes[1]
    blank(ax)
    rows = (("ROC-AUC", _auc(raw, yte), _auc(cal, yte), "{:.3f}"),
            ("accuracy", np.mean((raw > 0.5) == yte),
             np.mean((cal > 0.5) == yte), "{:.3f}"),
            ("log loss", _nll(z[h:], yte, 1.0), _nll(z[h:], yte, T), "{:.3f}"),
            ("Brier", np.mean((raw - yte) ** 2), np.mean((cal - yte) ** 2),
             "{:.3f}"),
            ("slope β", _slope_intercept(raw, yte)[1],
             _slope_intercept(cal, yte)[1], "{:.2f}"))
    for x, head, c in ((0.58, "raw", RED), (0.88, f"T = {T:.2f}", TEAL)):
        ax.text(x, 0.88, head, fontsize=12.5, color=c, ha="center")
    ax.plot([0.0, 1.0], [0.81, 0.81], color=GREY, lw=1.1)
    for k, (lab, a, b, fmt) in enumerate(rows):
        yy = 0.70 - 0.125 * k
        ax.text(0.0, yy, lab, fontsize=12.5, color=INK, va="center")
        ax.text(0.58, yy, fmt.format(a), fontsize=12.5, color=RED, ha="center",
                va="center")
        ax.text(0.88, yy, fmt.format(b), fontsize=12.5, color=TEAL,
                ha="center", va="center")
    ax.text(0.0, 0.05,
            "AUC and accuracy untouched",
            fontsize=11, color=GREY, va="center", linespacing=1.7)

    # --------------------------------------------- the family it belongs to
    ax = axes[2]
    blank(ax)
    fam = ((r"$\sigma(a\cdot z + b)$", "Platt", "2", RED),
           (r"$\sigma(z / T)$", "temperature", "1", TEAL))
    ax.text(0.60, 0.90, "map", fontsize=11.5, color=GREY, ha="center")
    ax.text(0.93, 0.90, "params", fontsize=11.5, color=GREY, ha="center")
    ax.plot([0.0, 1.0], [0.83, 0.83], color=GREY, lw=1.1)
    for k, (expr, name, npar, c) in enumerate(fam):
        yy = 0.71 - 0.15 * k
        ax.text(0.0, yy, name, fontsize=12.5, color=c, va="center")
        ax.text(0.60, yy, expr, fontsize=14, color=INK, ha="center",
                va="center")
        ax.text(0.93, yy, npar, fontsize=12.5, color=c, ha="center",
                va="center")
    ax.plot([0.0, 1.0], [0.44, 0.44], color=FAINT, lw=1.4)
    a_raw, b_raw = _slope_intercept(raw, yte)
    ax.text(0.0, 0.06,
            rf"here  $\beta = {b_raw:.2f}$,  and"
            rf"$1/\beta = {1 / b_raw:.2f} \approx \hat{{T}} = {T:.2f}$",
            fontsize=12, color=TEAL, va="center", linespacing=1.8)
    save(fig, out, "temperature_effect")


def _cv_calibrate(forest, Xd, yd, Xt, K=5, seed=0):
    """CalibratedClassifierCV by hand: K models, each Platt-scaled out-of-fold.

    Returns the test-set predictions of the averaged ensemble, and the K fitted
    (b0, b1) pairs so the middle panel can draw the maps that were averaged.
    """
    folds = np.arange(len(yd)) % K
    np.random.default_rng(seed).shuffle(folds)
    coefs, preds = [], []
    for k in range(K):
        rf = forest(k).fit(Xd[folds != k], yd[folds != k])
        s_oof = rf.predict_proba(Xd[folds == k])[:, 1]
        _, b = _platt(s_oof, yd[folds == k], s_oof)
        coefs.append(b)
        preds.append(sigmoid(b[0] + b[1] * rf.predict_proba(Xt)[:, 1]))
    return np.mean(preds, axis=0), coefs


def _split_calibrate(forest, Xd, yd, Xt):
    """The honest single split: half the rows fit the model, half the calibrator."""
    m = len(yd) // 2
    rf = forest(0).fit(Xd[:m], yd[:m])
    p, _ = _platt(rf.predict_proba(Xd[m:])[:, 1], yd[m:],
                  rf.predict_proba(Xt)[:, 1])
    return p


def fig_calibrated_cv(out: Path) -> None:
    """Cross-validated calibration: every row calibrates a model that never saw it."""
    from sklearn.ensemble import RandomForestClassifier

    def forest(seed=0):
        # left deep on purpose: the averaging bias of section 1 is the thing
        # being repaired here, so the raw forest has to show it
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=1,
                                      random_state=seed)

    def cohort(n, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 6))
        p = sigmoid(1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * X[:, 2] - 0.6)
        return X, rng.binomial(1, p)

    Xt, yt = cohort(4000, 99)

    fig, axes = plt.subplots(1, 3, figsize=WIDE,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0],
                                          "wspace": 0.3})

    # ---------------------------------------------------------- the fold layout
    ax = axes[0]
    blank(ax)
    ax.set_title("each fold calibrates the model that\nnever saw it",
                 fontsize=12.5, linespacing=1.5)
    K = 5
    x0, cw, gap = 0.02, 0.122, 0.011
    for k in range(K):
        yy = 0.74 - 0.125 * k
        for j in range(K):
            held = j == k
            ax.add_patch(Rectangle(
                (x0 + j * (cw + gap), yy - 0.045), cw, 0.09,
                facecolor="#fdecec" if held else "#e9f0f8",
                edgecolor=RED if held else BLUE, lw=1.5))
        ax.text(0.70, yy, f"\u2192  calibrator {k + 1}", fontsize=11, color=RED,
                va="center")
    ax.text(x0, 0.86, "fit the model", fontsize=11, color=BLUE, va="center")
    ax.text(0.40, 0.86, "fit the calibrator", fontsize=11, color=RED,
            va="center")
    ax.text(0.5, 0.12, "average the 5 calibrated models", fontsize=12.5,
            color=PURPLE, ha="center")
    ax.text(0.5, 0.01,
            "every patient is used\nfor model and calibrator",
            fontsize=11, color=GREY, ha="center", va="center", linespacing=1.5)

    # ------------------------------------------------------ the K maps averaged
    Xd, yd = cohort(600, 11)
    _, coefs = _cv_calibrate(forest, Xd, yd, Xt[:1], K=K, seed=11)
    grid = np.linspace(0.02, 0.98, 200)
    curves = [sigmoid(b[0] + b[1] * grid) for b in coefs]

    ax = axes[1]
    for c in curves:
        ax.plot(grid, c, color=RED, lw=1.4, alpha=0.5)
    ax.plot(grid, np.mean(curves, axis=0), color=PURPLE, lw=3.0)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2)
    ax.text(0.03, 0.95, "5 fold calibrators", color=RED, fontsize=11.5,
            transform=ax.transAxes, va="top")
    ax.text(0.03, 0.87, "their average", color=PURPLE, fontsize=11.5,
            transform=ax.transAxes, va="top")
    ax.set_xlabel("score  s(x)")
    ax.set_ylabel("calibrated probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # -------------------------------------------- what the sacrificed half costs
    sizes = (200, 400, 800, 1600, 3200)
    sp, cv = [], []
    for n in sizes:
        a, b = [], []
        for r in range(4):
            Xd, yd = cohort(n, 200 + r)
            a.append(np.mean((_split_calibrate(forest, Xd, yd, Xt) - yt) ** 2))
            b.append(np.mean(
                (_cv_calibrate(forest, Xd, yd, Xt, K=K, seed=r)[0] - yt) ** 2))
        sp.append(np.mean(a))
        cv.append(np.mean(b))

    ax = axes[2]
    ax.plot(sizes, sp, "o-", color=RED, lw=2.6, ms=6)
    ax.plot(sizes, cv, "o-", color=PURPLE, lw=2.6, ms=6)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("rows in the development set")
    ax.set_ylabel("Brier score on new rows")
    ax.set_title("the smaller the study,\nthe more the split costs",
                 fontsize=12.5, linespacing=1.5)
    ax.text(sizes[-1] * 0.95, sp[-1] + 0.0006, "one 50/50 split", color=RED,
            fontsize=11, ha="right", va="bottom")
    ax.text(sizes[-1] * 0.95, cv[-1] - 0.0006, "CalibratedClassifierCV",
            color=PURPLE, fontsize=11, ha="right", va="top")
    ax.set_ylim(min(cv) - 0.0022, max(sp) + 0.0008)
    ax.annotate("", xy=(sizes[0], cv[0]), xytext=(sizes[0], sp[0]),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
    ax.text(sizes[0] * 1.12, (sp[0] + cv[0]) / 2,
            f"{100 * (sp[0] - cv[0]) / sp[0]:.0f}%", fontsize=11.5, color=GREY,
            va="center")
    save(fig, out, "calibrated_cv")


FIGURES = (
    fig_bernoulli_expfam, fig_expfam, fig_expfam_loss, fig_bernoulli_b,
    fig_log_loss_intro, fig_log_loss_likelihood, fig_forest_boundary_bias,
    fig_glm, fig_margin_bias, fig_model_zoo_calibration, fig_naive_bayes_bias,
    fig_platt_flow, fig_calibration_leak, fig_auc_build, fig_auc_monotone,
    fig_auc_exception, fig_reliability_build, fig_reliability_choices,
    fig_brier_arithmetic, fig_brier_score, fig_cox_arithmetic,
    fig_cox_reading, fig_ece_arithmetic, fig_ece_caveats, fig_platt_fit,
    fig_isotonic_fit, fig_platt_vs_isotonic, fig_temperature_scaling,
    fig_temperature_effect, fig_calibrated_cv,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course04/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
