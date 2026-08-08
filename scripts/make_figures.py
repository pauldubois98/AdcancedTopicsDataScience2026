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
    ax.text(0.5, 0.60, "?", transform=ax.transAxes, ha="center", va="center",
            fontsize=52, color=GREY, fontweight="bold")
    ax.text(0.5, 0.28, question, transform=ax.transAxes, ha="center", va="center",
            fontsize=12.5, color=INK)


def stage_name(base: str, reveal: bool) -> str:
    return f"{base}.png" if reveal else f"{base}_q.png"


# --------------------------------------------------------------------------- noise
def fig_noise(out: Path, reveal: bool = True) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = np.vstack([rng.normal([-1.2, -0.4], 1.0, (n, 2)), rng.normal([1.2, 0.6], 1.0, (n, 2))])
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
            ax.scatter(x[m, 0], x[m, 1], s=15, c=color, alpha=0.6, edgecolors="none", label=name)
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
    ax.imshow(observed, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
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
        ask(ax, "Who died more often:\nthe patients whose lactate\nwas drawn, or not drawn?")
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

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})

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
        fig.suptitle("accuracy 99.45%", fontsize=13.5, fontweight="bold", color=RED, y=1.09)
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
        return np.exp(-0.5 * ((grid[:, None] - samples[None, :]) / h) ** 2).sum(axis=1) / (
            len(samples) * h * np.sqrt(2 * np.pi)
        )

    train = rng.normal(58, 11, 4000)
    deploy = rng.normal(72, 13, 4000)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))

    ax = axes[0]
    ax.fill_between(grid, density(train), color=BLUE, alpha=0.45, label="training hospital")
    ax.fill_between(grid, density(deploy), color=RED, alpha=0.45, label="deployment hospital")
    ax.set_xlabel("patient age")
    ax.set_yticks([])
    ax.legend(fontsize=9.5, frameon=False)
    ax.spines["left"].set_visible(False)

    ax = axes[1]
    if not reveal:
        ask(ax, "It scored AUC 0.83 at validation.\nWhat does it score a year later?")
        fig.suptitle(
            "The model does not change; The world can.",
            fontsize=12.5, y=1.02, fontweight="bold",
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
        "The model does not change; The world can.", fontsize=12.5, y=1.02, fontweight="bold"
    )
    fig.savefig(out / stage_name("shift", reveal))
    plt.close(fig)


# -------------------------------------------------------------------------- shortcut
def fig_shortcut(out: Path, reveal: bool = True) -> None:
    from PIL import Image

    xray = np.asarray(Image.open(out / "image2d_medical.jpg").convert("L"), dtype=float) / 255
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


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course01/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_noise, fig_missingness, fig_imbalance, fig_shift, fig_shortcut):
        name = fn.__name__.removeprefix("fig_")
        # each figure ships in two stages: ask the room, then reveal the answer
        for reveal in (False, True):
            fn(out, reveal)
            print(f"{out / stage_name(name, reveal)}")


if __name__ == "__main__":
    main()
