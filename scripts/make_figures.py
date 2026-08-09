#!/usr/bin/env python3
"""Render the illustration figures used by the Course01 lecture slides.

Every figure is synthetic and seeded, so `make figures` reproduces byte-identical
output anywhere without touching `data/`. The one exception is the shortcut figure,
which composites a marker onto the real chest radiograph already in `Course01/img/`
to show how a spurious cue is learned — it is a constructed example, and the slide
says so.

Usage: make_figures.py [outdir]      (default: Course01/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

INK = "#1b1b1b"
GREY = "#9aa0a6"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#00ab0e"
AMBER = "#b7791f"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
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


def lda_boundary(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (w, b) of the LDA decision boundary w·x + b = 0."""
    mu0, mu1 = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
    cov = np.cov(x.T)
    w = np.linalg.solve(cov, mu1 - mu0)
    b = -w @ (mu0 + mu1) / 2
    return w, b


def draw_boundary(ax, w, b, color, label):
    xs = np.array(ax.get_xlim())
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=color, lw=2.5, label=label, zorder=3)


def ask(ax, question: str) -> None:
    """Replace a panel with the question it answers, for the predict-then-reveal stage.

    The panel keeps its footprint so the question and reveal images have the same
    aspect ratio and do not jump in size when you advance the slide.
    """
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0.04, 0.06),
            0.92,
            0.88,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=GREY,
            ls=(0, (6, 6)),
            lw=1.4,
        )
    )
    ax.text(
        0.5,
        0.60,
        "?",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=52,
        color=GREY,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.28,
        question,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        color=INK,
    )


def stage_name(base: str, reveal: bool) -> str:
    return f"{base}.png" if reveal else f"{base}_q.png"


# --------------------------------------------------------------------------- noise
def fig_noise(out: Path, reveal: bool = True) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = np.vstack(
        [rng.normal([-1.2, -0.4], 1.0, (n, 2)), rng.normal([1.2, 0.6], 1.0, (n, 2))]
    )
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)

    # the annotator is not sloppy, she is cautious: when unsure she calls it a finding,
    # so the errors all point one way. That is what moves the boundary.
    over_call = (y == 0) & (rng.random(len(y)) < 0.35)
    y_obs = np.where(over_call, 1, y)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), sharex=True, sharey=True)
    for ax, labels, title in (
        (axes[0], y, "If the label were the truth"),
        (axes[1], y_obs, "What the annotator wrote\n35% of negatives called positive"),
    ):
        for cls, color, name in ((0, BLUE, "no finding"), (1, RED, "finding")):
            m = labels == cls
            ax.scatter(
                x[m, 0],
                x[m, 1],
                s=15,
                c=color,
                alpha=0.6,
                edgecolors="none",
                label=name,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("feature 1")
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(-4.0, 4.0)
    axes[0].set_ylabel("feature 2")

    w_true, b_true = lda_boundary(x, y)
    w_obs, b_obs = lda_boundary(x, y_obs)
    draw_boundary(axes[0], w_true, b_true, INK, "boundary")
    draw_boundary(axes[1], w_true, b_true, GREY, "true boundary")
    if reveal:
        draw_boundary(axes[1], w_obs, b_obs, AMBER, "what you learn")
    else:
        axes[1].set_title(
            "What the annotator wrote\nwhere does the fitted line go?", fontsize=12
        )

    for ax in axes:
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.savefig(out / stage_name("noise", reveal))
    plt.close(fig)


# --------------------------------------------------------------------- missingness
def fig_missingness(out: Path, reveal: bool = True) -> None:
    rng = np.random.default_rng(1)
    n = 600
    severity = np.sort(rng.beta(1.6, 3.2, n))[::-1]

    variables = [
        ("age", 1.00),
        ("sex", 1.00),
        ("heart rate", 0.99),
        ("temperature", 0.97),
        ("creatinine", 0.90),
        ("white cell count", 0.88),
        ("bilirubin", 0.55),
        ("lactate", 0.00),
        ("troponin", 0.00),
        ("blood culture", 0.00),
        ("arterial pH", 0.00),
        ("cardiac output", 0.00),
    ]
    observed = np.zeros((n, len(variables)), dtype=float)
    for j, (_, base) in enumerate(variables):
        if base > 0:
            p = np.clip(base - 0.04 * (1 - severity), 0, 1)
        else:  # ordered only when the patient looks unwell
            p = np.clip(0.01 + 2.4 * severity, 0, 0.99)
        observed[:, j] = rng.random(n) < p

    fig, axes = plt.subplots(
        1, 2, figsize=(13.6, 4.0), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    ax = axes[0]
    ax.imshow(
        observed, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([v for v, _ in variables], rotation=42, ha="right", fontsize=9)
    ax.set_ylabel("patients, sorted by time spend in hospital")
    ax.set_yticks([])
    ax.set_title("black = recorded,  white = missing", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # mortality conditional on whether lactate was ever drawn
    lactate = observed[:, 7].astype(bool)
    died = rng.random(n) < np.clip(0.01 + 0.95 * severity**1.4, 0, 0.95)
    rates = [died[~lactate].mean(), died[lactate].mean()]

    ax = axes[1]
    if not reveal:
        ask(
            ax,
            "Who died more often:\nthe patients whose lactate\nwas drawn, or not drawn?",
        )
        fig.savefig(out / stage_name("missingness", reveal))
        plt.close(fig)
        return

    bars = ax.bar(
        ["lactate\nnever drawn", "lactate\ndrawn"],
        [r * 100 for r in rates],
        color=[GREY, RED],
        width=0.55,
    )
    for bar, r in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rates) * 100 * 0.03,
            f"{r * 100:.0f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax.set_ylabel("died in hospital (%)")
    ax.set_ylim(0, max(rates) * 100 * 1.28)
    fig.savefig(out / stage_name("missingness", reveal))
    plt.close(fig)


# ------------------------------------------------------------------------ imbalance
def fig_imbalance(out: Path, reveal: bool = True) -> None:
    n_neg, n_pos = 9945, 55

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 3.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    ax = axes[0]
    bars = ax.barh(["died", "survived"], [n_pos, n_neg], height=0.55)
    ax.bar_label(bars, labels=[f"{n_pos:,}", f"{n_neg:,}"], padding=8, fontsize=13)
    ax.set_xlim(0, n_neg * 1.24)
    ax.set_xlabel("admissions")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    if not reveal:
        ask(ax, "This model is 99.45% accurate.\nHands up if you would ship it.")
        fig.suptitle(
            "accuracy 99.45%", fontsize=13.5, fontweight="bold", color=RED, y=1.09
        )
        fig.savefig(out / stage_name("imbalance", reveal))
        plt.close(fig)
        return

    cm = np.array([[n_neg, 0], [n_pos, 0]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_neg)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if cm[i, j] > n_neg / 2 else INK,
            )
    ax.set_xticks([0, 1], ["predicted\nsurvived", "predicted\ndied"], fontsize=10)
    ax.set_yticks([0, 1], ["actually\nsurvived", "actually\ndied"], fontsize=10)
    ax.set_title('the model that always says "survived"', fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.suptitle(
        "accuracy 99.45%   |   recall 0%",
        fontsize=13.5,
        fontweight="bold",
        color=RED,
        y=1.09,
    )
    fig.savefig(out / stage_name("imbalance", reveal))
    plt.close(fig)


# ----------------------------------------------------------------------------- shift
def fig_shift(out: Path, reveal: bool = True) -> None:
    rng = np.random.default_rng(3)
    grid = np.linspace(20, 105, 400)

    def density(samples):
        h = 4.0
        return np.exp(-0.5 * ((grid[:, None] - samples[None, :]) / h) ** 2).sum(
            axis=1
        ) / (len(samples) * h * np.sqrt(2 * np.pi))

    train = rng.normal(58, 11, 4000)
    deploy = rng.normal(72, 13, 4000)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))

    ax = axes[0]
    ax.fill_between(
        grid, density(train), color=BLUE, alpha=0.45, label="training hospital"
    )
    ax.fill_between(
        grid, density(deploy), color=RED, alpha=0.45, label="deployment hospital"
    )
    ax.set_xlabel("patient age")
    ax.set_yticks([])
    ax.legend(fontsize=9.5, frameon=False)
    ax.spines["left"].set_visible(False)

    ax = axes[1]
    if not reveal:
        ask(ax, "It scored AUC 0.83 at validation.\nWhat does it score a year later?")
        fig.suptitle(
            "The model does not change; The world can.",
            fontsize=12.5,
            y=1.02,
            fontweight="bold",
        )
        fig.savefig(out / stage_name("shift", reveal))
        plt.close(fig)
        return

    months = np.arange(0, 13)
    auc = 0.83 - 0.013 * months - 0.055 * (months > 6) * (months - 6) / 6
    auc += rng.normal(0, 0.006, months.size)
    ax.plot(months, auc, "o-", color=RED, lw=2.2, ms=5)
    ax.axhline(0.83, color=GREY, ls="--", lw=1.4)
    ax.text(0.15, 0.836, "AUC at validation", color=GREY, fontsize=9.5)
    ax.set_xlabel("years since deployment")
    ax.set_ylabel("AUC in production")
    ax.set_ylim(0.6, 0.88)

    fig.suptitle(
        "The model does not change; The world can.",
        fontsize=12.5,
        y=1.02,
        fontweight="bold",
    )
    fig.savefig(out / stage_name("shift", reveal))
    plt.close(fig)


# -------------------------------------------------------------------------- shortcut
def fig_shortcut(out: Path, reveal: bool = True) -> None:
    from PIL import Image

    xray = (
        np.asarray(Image.open(out / "image2d_medical.jpg").convert("L"), dtype=float)
        / 255
    )
    h, w = xray.shape

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    mx, my = 0.30 * w, 0.115 * h  # marker centre, well inside the frame

    def show(ax, title, color):
        ax.imshow(xray, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12.5, color=color, fontweight="bold", pad=8)

    def marker(ax):
        # every "pneumonia" scan in this hospital was shot at the bedside, and the
        # portable unit stamps a marker in the corner
        ax.add_patch(
            Rectangle(
                (mx - 0.115 * w, my - 0.035 * h),
                0.23 * w,
                0.07 * h,
                facecolor="white",
                edgecolor="none",
            )
        )
        ax.text(mx, my, "PORTABLE", ha="center", va="center", fontsize=8, color="black")

    show(axes[0], 'labelled "pneumonia"', RED)
    marker(axes[0])

    show(axes[1], 'labelled "normal"', BLUE)

    if not reveal:
        ask(axes[2], "It gets every one right.\nWhat did it learn?")
        fig.suptitle(
            "100% accuracy on the test set", fontsize=13.5, y=1.04, fontweight="bold"
        )
        fig.savefig(out / stage_name("shortcut", reveal))
        plt.close(fig)
        return

    show(axes[2], "where the model looks", INK)
    marker(axes[2])
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - mx) ** 2 + (yy - my) ** 2) / (2 * (0.10 * w) ** 2)))
    axes[2].imshow(blob, cmap="inferno", alpha=0.6 * blob, vmin=0, vmax=1)

    fig.suptitle(
        "100% accuracy on the test set, 0% of it from the lungs",
        fontsize=13.5,
        y=1.04,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "effect shown by Zech et al. 2018",
        ha="center",
        fontsize=9,
        color=GREY,
    )
    fig.savefig(out / stage_name("shortcut", reveal))
    plt.close(fig)


# ===========================================================================
# Non-medical twins. Each one shows the same failure as the figure above it,
# in a domain nobody in the room can wave away as "a hospital problem". They
# are single-stage: the medical figure already ran the predict-then-reveal, so
# these land as the confirmation rather than a second quiz.
# ===========================================================================


def blur(a: np.ndarray, k: int = 9) -> np.ndarray:
    """Separable box blur, so the synthetic photos have texture instead of speckle."""
    ker = np.ones(k) / k
    for axis in (0, 1):
        a = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), axis, a)
    return a


# ----------------------------------------------------------------- noise (alt)
def fig_noise_alt(out: Path) -> None:
    """Two crowdworkers labelling the same reviews: the same ceiling, no radiologist."""
    rng = np.random.default_rng(10)
    classes = ["negative", "neutral", "positive"]
    n = 3000
    truth = rng.choice(3, n, p=[0.30, 0.35, 0.35])

    def label(slip: float) -> np.ndarray:
        seen = truth.copy()
        move = rng.random(n) < slip
        seen[move] = np.clip(truth[move] + rng.choice([-1, 1], move.sum()), 0, 2)
        return seen

    a, b = label(0.24), label(0.26)
    cm = np.array([[((a == i) & (b == j)).sum() for j in range(3)] for i in range(3)])

    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    ax = axes[0]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=11.5,
                color="white" if cm[i, j] > cm.max() / 2 else INK,
            )
    ax.set_xticks(range(3), classes, fontsize=10)
    ax.set_yticks(range(3), classes, fontsize=10)
    ax.set_xlabel("reviewer B")
    ax.set_ylabel("reviewer A")
    ax.set_title("same 3 000 reviews, two reviewers", fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    scores = [(a == b).mean(), (truth == b).mean()]
    ceiling = scores[-1] * 100
    bars = ax.bar(
        [
            "reviewer A/B\n(scored against B/A)",
            "best possible model\n(scored against A+B)",
        ],
        [s * 100 for s in scores],
        color=[GREY, AMBER],
        width=0.5,
    )
    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            f"{s * 100:.0f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.axhspan(ceiling, 100, color=RED, alpha=0.11)
    ax.axhline(ceiling, color=RED, lw=1.2, ls="--")
    ax.text(
        0.5,
        (ceiling + 100) / 2,
        "no model can reach here",
        ha="center",
        va="center",
        fontsize=10,
        color=RED,
    )
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    fig.savefig(out / "noise_alt.png")
    plt.close(fig)


# ---------------------------------------------------------------- errors (alt)
def fig_errors_alt(out: Path) -> None:
    """A delivery fleet's telemetry: sentinel values and mixed units, no clinician needed."""
    rng = np.random.default_rng(11)
    n = 900
    t = np.linspace(0, 1, n)
    lon = (
        2.350
        + 0.045 * np.sin(2 * np.pi * 1.3 * t)
        + 0.020 * np.cos(2 * np.pi * 2.1 * t)
    )
    lat = (
        48.856
        + 0.030 * np.cos(2 * np.pi * 1.1 * t)
        + 0.012 * np.sin(2 * np.pi * 3.0 * t)
    )
    lon += rng.normal(0, 0.0006, n)
    lat += rng.normal(0, 0.0006, n)
    # "no fix" is written to the database as a number, not as NULL
    lost = rng.random(n) < 0.06

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))

    ax = axes[0]
    ax.scatter(
        np.where(lost, 0.0, lon), np.where(lost, 0.0, lat), s=6, c=INK, alpha=0.7
    )
    # ax.scatter([0], [0], s=70, facecolors="none", edgecolors=RED, lw=1.8)
    ax.annotate(
        "6% of positions\nare (0, 0)",
        (0, 0),
        (0.28, 0.20),
        textcoords="axes fraction",
        fontsize=10,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    ax.annotate(
        "the whole route\nis in here",
        (lon.mean(), lat.mean()),
        (0.30, 0.66),
        textcoords="axes fraction",
        fontsize=10,
        color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN},
    )
    ax.set_title("raw GPS positions", fontsize=12)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    ax = axes[1]
    ax.plot(lon[~lost], lat[~lost], lw=1.2, color=BLUE)
    ax.set_title("the same data, zeros dropped", fontsize=12)
    ax.set_xlabel("longitude")

    fig.savefig(out / "errors_alt.png")
    plt.close(fig)


def fig_errors_alt_bis(out: Path) -> None:
    """Mixed units, seen one supplier at a time and then merged.

    The point of the three panels is that neither supplier looks wrong on its own.
    Every sanity check you would run per-source passes; the bug only exists in the
    concatenation, which is where nobody looks.
    """
    rng = np.random.default_rng(15)
    # both fleets drive the same roads at the same speeds, ~38 km/h
    speed_a = rng.normal(38, 6, 2600)
    speed_b = rng.normal(38, 6, 1400)
    recorded_b = speed_b / 1.609  # supplier B's firmware reports mph, nobody converted
    bins = np.linspace(5, 65, 61)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharex=True, sharey=True)

    panels = (
        (axes[0], recorded_b, AMBER, "supplier B alone", "1 400 vehicles, mean 24"),
        (axes[1], speed_a, BLUE, "supplier A alone", "2 600 vehicles, mean 38"),
    )
    for ax, values, color, title, caption in panels:
        ax.hist(values, bins=bins, color=color, alpha=0.85)
        ax.set_title(title, fontsize=12.5, color=color, fontweight="bold")
        ax.set_xlabel("speed as recorded")
        ax.text(
            0.5,
            0.94,
            caption,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=GREY,
        )
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    ax = axes[2]
    # grey, because a merged column arrives with no colour telling you which is which
    ax.hist(np.r_[recorded_b, speed_a], bins=bins, color=GREY)
    ax.set_title("one `speed` column", fontsize=12.5, color=RED, fontweight="bold")
    ax.set_xlabel("speed as recorded")
    ax.text(
        0.5,
        0.94,
        "two humps, one fleet",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color=RED,
    )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # one call: the y axis is shared, so scaling each in turn compounds
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.22)

    fig.suptitle(
        "Neither supplier looks wrong on its own",
        fontsize=13.5,
        y=1.03,
        fontweight="bold",
    )
    fig.savefig(out / "errors_alt_bis.png")
    plt.close(fig)


# ------------------------------------------------------------- imbalance (alt)
def fig_imbalance_alt(out: Path) -> None:
    """Transactions fraud: 492 in 284 807, and `return 0` still wins the accuracy contest."""
    n_ok, n_fraud = 284_315, 492

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 3.5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    ax = axes[0]
    bars = ax.barh(
        ["fraud", "legitimate"], [n_fraud, n_ok], height=0.55, color=[RED, BLUE]
    )
    ax.bar_label(bars, labels=[f"{n_fraud:,}", f"{n_ok:,}"], padding=8, fontsize=13)
    ax.set_xlim(0, n_ok * 1.26)
    ax.set_xticks([0, 100_000, 200_000, 300_000], ["0", "100k", "200k", "300k"])
    ax.set_xlabel("Transactions")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    bars = ax.bar(
        ["accuracy", "fraud caught"], [99.83, 0.0], color=[GREY, RED], width=0.55
    )
    for bar, v in zip(bars, (99.83, 0.0)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{v:.2f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    ax.set_ylim(0, 118)
    ax.set_ylabel("%")
    ax.set_title('model always says "legitimate"', fontsize=12)

    fig.suptitle(
        "accuracy 99.83%   |   every fraud, missed",
        fontsize=13.5,
        fontweight="bold",
        color=RED,
        y=1.09,
    )
    fig.savefig(out / "imbalance_alt.png")
    plt.close(fig)


# ----------------------------------------------------------------- shift (alt)
def fig_shift_alt(out: Path) -> None:
    """Demand forecasting across a regime break: P(Y|X) moved, the model did not."""
    rng = np.random.default_rng(13)
    weeks = np.arange(104)
    baseline = 1000 + 70 * np.sin(2 * np.pi * weeks / 52) + 2.2 * weeks
    break_at = 70
    shock = np.where(weeks >= break_at, -430 + 5.5 * (weeks - break_at), 0)
    actual = baseline + shock + rng.normal(0, 22, weeks.size)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.0), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    ax = axes[0]
    ax.plot(weeks, actual, color=INK, lw=1.6, label="units actually sold")
    ax.plot(
        weeks, baseline, color=BLUE, lw=2.0, ls="--", label="what the model forecasts"
    )
    ax.axvline(break_at, color=RED, lw=1.4)
    ax.set_ylim(top=actual.max() * 1.14)
    ax.text(
        break_at + 3,
        actual.max() * 1.02,
        "a competitor opens next door",
        color=RED,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    ax.axvspan(0, break_at * 0.75, color=GREY, alpha=0.13)
    ax.text(
        break_at * 0.75 / 2,
        actual.max() * 1,
        "training window",
        color=GREY,
        fontsize=10,
        ha="center",
    )
    ax.set_xlabel("week")
    ax.set_ylabel("weekly demand")
    ax.legend(fontsize=9.5, frameon=False, loc="lower left")

    ax = axes[1]
    err = 100 * np.abs(actual - baseline) / actual
    ax.plot(weeks, err, color=GREEN, lw=1.8)
    ax.axvline(break_at, color=RED, lw=1.0, ls=":")
    ax.set_xlabel("week")
    ax.set_ylabel("forecast error (%)")

    fig.savefig(out / "shift_alt.png")
    plt.close(fig)


# -------------------------------------------------------------- shortcut (alt)
def fig_shortcut_alt(out: Path) -> None:
    """Husky vs wolf: the classifier that learned snow (Ribeiro et al. 2016), schematically."""
    rng = np.random.default_rng(14)
    size = 220
    yy, xx = np.mgrid[0:size, 0:size]
    body = (((xx - 118) / 66) ** 2 + ((yy - 148) / 44) ** 2) < 1
    head = (((xx - 64) / 30) ** 2 + ((yy - 104) / 30) ** 2) < 1
    # ears, tail and legs, so it reads as an animal rather than a grey blob
    ears = ((np.abs(xx - 52) + 1.6 * np.abs(yy - 84) < 22) & (yy < 84)) | (
        (np.abs(xx - 80) + 1.6 * np.abs(yy - 86) < 20) & (yy < 86)
    )
    tail = (((xx - 186) / 26) ** 2 + ((yy - 128) / 9) ** 2) < 1
    legs = ((np.abs(xx - 96) < 9) | (np.abs(xx - 152) < 9)) & (yy > 140) & (yy < 196)
    animal = body | head | ears | tail | legs

    def scene(snowy: bool) -> np.ndarray:
        texture = blur(rng.normal(0, 1, (size, size)))
        texture /= np.abs(texture).max()
        if snowy:
            img = np.stack(
                [0.90 + 0.09 * texture, 0.92 + 0.09 * texture, 0.96 + 0.06 * texture],
                axis=-1,
            )
        else:
            img = np.stack(
                [0.34 + 0.16 * texture, 0.52 + 0.16 * texture, 0.22 + 0.13 * texture],
                axis=-1,
            )
        fur = 0.44 + 0.30 * texture
        img[animal] = np.stack([fur, fur, fur * 1.04], axis=-1)[animal]
        return np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    snow, grass = scene(True), scene(False)

    for ax, img, title, color in (
        (axes[0], snow, 'labelled "wolf"', RED),
        (axes[1], grass, 'labelled "husky"', BLUE),
        (axes[2], snow, "where the model looks", INK),
    ):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12.5, color=color, fontweight="bold", pad=8)

    # the model's attention is everywhere the animal is not: it is scoring the snow
    heat = blur(np.where(animal, 0.0, 1.0), 15)
    axes[2].imshow(heat, cmap="inferno", alpha=0.55 * heat, vmin=0, vmax=1.9)

    fig.suptitle(
        "Every wolf in the training set was photographed on snow;\nEvery husky in the training set was photographed on grass.",
        fontsize=13.5,
        y=1.1,
    )
    fig.text(
        0.5,
        0.02,
        "schematic of Ribeiro et al. 2016",
        ha="center",
        fontsize=9,
        color=GREY,
    )
    fig.savefig(out / "shortcut_alt.png")
    plt.close(fig)


STAGED = (fig_noise, fig_missingness, fig_imbalance, fig_shift, fig_shortcut)
SINGLE = (
    fig_noise_alt,
    fig_errors_alt,
    fig_errors_alt_bis,
    fig_imbalance_alt,
    fig_shift_alt,
    fig_shortcut_alt,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course01/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in STAGED:
        name = fn.__name__.removeprefix("fig_")
        # each figure ships in two stages: ask the room, then reveal the answer
        for reveal in (False, True):
            fn(out, reveal)
            print(f"{out / stage_name(name, reveal)}")
    for fn in SINGLE:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
