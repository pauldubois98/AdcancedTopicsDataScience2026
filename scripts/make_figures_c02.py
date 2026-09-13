#!/usr/bin/env python3
"""Render the illustration figures used by the Course02 main lecture slides.

Same contract as `make_figures_c02b.py` (the companion deck) and
`make_figures_c03.py`: every figure is synthetic and seeded, so `make figures`
reproduces the same output anywhere without touching `data/`. Figures land in
`Course02/img/` alongside the companion deck's, under names that do not collide.

Usage: make_figures_c02.py [outdir]      (default: Course02/img)
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

WIDE = (12.6, 5.2)
HALF = (10.8, 4.45)

# the .pptx content placeholder is 9.00in x 3.71in; pad every saved figure up to
# that aspect so pandoc does not shrink it and leave the slide two-thirds empty
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


def equation(out: Path, name: str, lines, fs=34, annots=()) -> None:
    """Render display math as an image, auto-sized to fill the slide width.

    Pandoc turns markdown math into OOXML that LibreOffice drops, taking the whole
    slide with it, so every formula in these decks ships as a picture instead.
    `annots` are (x, text, colour) callouts at explicit figure coordinates.
    """
    fig = plt.figure(figsize=(12.0, 12.0 / SLIDE_ASPECT))
    n = len(lines)
    top, bot = 0.86, (0.42 if annots else 0.10)
    handles = []
    for i, line in enumerate(lines):
        y = top - (i + 0.5) * (top - bot) / n
        handles.append(fig.text(0.5, y, line, ha="center", va="center",
                                fontsize=fs, color=INK))

    fig.canvas.draw()
    widest = max(h.get_window_extent(fig.canvas.get_renderer()).width
                 for h in handles) / fig.get_size_inches()[0] / fig.dpi
    for h in handles:
        h.set_fontsize(min(fs * 0.92 / widest, 44))

    for x, text, colour in annots:
        fig.text(x, 0.24, text, ha="center", va="center", fontsize=17, color=colour)
        fig.text(x, 0.34, "↑", ha="center", va="center", fontsize=17, color=colour)
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------- shared fake ward
def _hr_series(seed=0, hours=30):
    """One patient's heart rate, irregularly sampled and slowly deteriorating."""
    rng = np.random.default_rng(seed)
    t = [0.0]
    while t[-1] < hours:
        # the sicker the patient gets, the more often someone measures them
        gap = rng.uniform(0.4, 3.2) * (1 - 0.55 * t[-1] / hours)
        t.append(t[-1] + max(gap, 0.25))
    t = np.array(t[:-1])
    hr = 74 + 14 * (t / hours) ** 2 + 5 * np.sin(t / 2.2) + rng.normal(0, 3.0, len(t))
    return t, hr


# =========================================================== 1. time-aware features
def fig_cyclical(out: Path) -> None:
    """Hour 23 and hour 0 are one hour apart — unless you feed the raw number."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    ax = axes[0]
    ax.plot([0, 23], [0, 0], color=GREY, lw=2)
    for h in range(24):
        ax.plot([h], [0], "o", color=(RED if h in (0, 23) else FAINT), ms=9,
                zorder=3)
    for h in (0, 23):
        ax.text(h, 0.12, str(h), ha="center", fontsize=12, color=RED)
    ax.annotate("", (0, -0.14), (23, -0.14),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(11.5, -0.30, "the model sees 23 hours", ha="center", fontsize=12.5,
            color=RED)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("raw `hour`: a straight line", fontsize=12.5)
    blank(ax)
    ax.set_xlim(-2, 25)
    ax.set_ylim(-0.6, 0.6)

    ax2 = axes[1]
    th = 2 * np.pi * np.arange(24) / 24
    ax2.plot(np.sin(th), np.cos(th), color=FAINT, lw=2)
    ax2.scatter(np.sin(th), np.cos(th), s=60, color=FAINT, zorder=2)
    for h in (0, 23):
        a = 2 * np.pi * h / 24
        ax2.plot([np.sin(a)], [np.cos(a)], "o", color=RED, ms=11, zorder=3)
        ax2.text(1.22 * np.sin(a), 1.22 * np.cos(a), str(h), ha="center",
                 va="center", fontsize=12, color=RED)
    for h in (6, 12, 18):
        a = 2 * np.pi * h / 24
        ax2.text(1.22 * np.sin(a), 1.22 * np.cos(a), str(h), ha="center",
                 va="center", fontsize=11, color=GREY)
    ax2.set_title("(sin, cos): a circle", fontsize=12.5)
    ax2.text(0, -1.55, "23 and 0 are neighbours again", ha="center", fontsize=12.5,
             color=GREEN)
    ax2.set_xlim(-1.7, 1.7)
    ax2.set_ylim(-1.8, 1.6)
    ax2.set_aspect("equal")
    blank(ax2)
    ax2.set_xlim(-1.7, 1.7)
    ax2.set_ylim(-1.8, 1.6)

    ax3 = axes[2]
    g = np.linspace(0, 24, 400)
    ax3.plot(g, np.sin(2 * np.pi * g / 24), color=BLUE, lw=2.6, label="hour_sin")
    ax3.plot(g, np.cos(2 * np.pi * g / 24), color=AMBER, lw=2.6, label="hour_cos")
    ax3.axhline(0, color=FAINT, lw=1)
    ax3.set_xticks([0, 6, 12, 18, 24])
    ax3.set_xlabel("hour")
    ax3.legend(frameon=False, fontsize=11, loc="lower left")
    ax3.set_title("two columns, both needed", fontsize=12.5)
    ax3.text(12, 1.48, "sin alone: 6h and 18h look identical", ha="center",
             fontsize=11, color=GREY)
    ax3.set_ylim(-1.35, 1.75)
    save(fig, out, "cyclical")


def fig_elapsed(out: Path) -> None:
    """Time since, and time between: two columns hiding in the timestamps."""
    t, hr = _hr_series(1)
    now = t[-1]

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, hr, "-o", color=BLUE, lw=1.6, ms=5)
    ax.axvline(0, color=GREEN, lw=2)
    ax.text(0.3, hr.max() + 2, "admission", color=GREEN, fontsize=12)
    ax.axvline(now, color=RED, lw=2)
    ax.text(now - 0.3, hr.max() + 2, "now", color=RED, fontsize=12, ha="right")
    ax.annotate("", (0, 66), (now, 66),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2))
    ax.text(now / 2, 63.5, f"h_since_admit = {now:.1f}", ha="center", fontsize=12,
            color=AMBER)
    ax.annotate("", (t[-2], 70.5), (t[-1], 70.5),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=2))
    ax.text(t[-2] - 0.4, 70.5, f"h_since_obs = {t[-1] - t[-2]:.1f}", ha="right",
            va="center", fontsize=11.5, color=PURPLE)
    ax.set_ylabel("heart rate")
    ax.set_ylim(60, hr.max() + 8)
    ax.set_title("one patient, irregularly measured", fontsize=13)

    gaps = np.diff(t)
    axes[1].bar(t[1:], gaps, width=0.28, color=PURPLE)
    axes[1].set_ylabel("gap (h)")
    axes[1].set_xlabel("hours since admission")
    axes[1].set_ylim(0, gaps.max() * 1.6)
    axes[1].text(0.5, gaps.max() * 1.30, "gaps shrink: someone is worried — "
                 "the sampling rate is itself a feature",
                 fontsize=11.5, color=INK)
    save(fig, out, "elapsed")


def fig_windows(out: Path) -> None:
    """Collapse the recent past into fixed columns."""
    t, hr = _hr_series(1)
    now = t[-1]

    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    ax.axvspan(0, now, color=GREEN, alpha=0.08)
    ax.axvspan(now - 24, now, color=AMBER, alpha=0.13)
    ax.axvspan(now - 6, now, color=RED, alpha=0.16)
    ax.plot(t, hr, "-o", color=BLUE, lw=1.6, ms=5, zorder=3)

    top = hr.max() + 10
    for lo, colour, label in ((now - 6, RED, "6h"), (now - 24, AMBER, "24h"),
                              (0, GREEN, "expanding")):
        y = top - (0 if colour is RED else (2.6 if colour is AMBER else 5.2))
        ax.annotate("", (lo, y), (now, y),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=2))
        ax.text((lo + now) / 2, y + 0.9, label, color=colour, fontsize=11.5,
                ha="center", va="bottom")

    def win(lo):
        m = (t >= lo) & (t <= now)
        return hr[m]

    rows = (("hr_mean_6h", f"{win(now - 6).mean():.0f}", RED),
            ("hr_max_6h", f"{win(now - 6).max():.0f}", RED),
            ("hr_mean_24h", f"{win(now - 24).mean():.0f}", AMBER),
            ("n_obs_24h", f"{len(win(now - 24))}", AMBER),
            ("hr_mean_all", f"{hr.mean():.0f}", GREEN))
    for i, (k, v, colour) in enumerate(rows):
        ax.text(now + 1.2, hr.max() - i * 4.2, f"{k} = {v}", fontsize=12,
                color=colour, va="center")
    ax.text(now + 1.2, hr.max() - 5 * 4.2 - 1.5,
            "hr_mean_6h − hr_mean_24h\n= a trend, in one column", fontsize=11.5,
            color=INK, va="top")

    ax.set_xlim(-1, now + 14)
    ax.set_ylim(62, top + 3.5)
    ax.set_xlabel("hours since admission")
    ax.set_ylabel("heart rate")
    ax.set_title("three windows ending at the same moment", fontsize=13)
    save(fig, out, "windows")


def fig_window_types(out: Path) -> None:
    """Sliding, tumbling, expanding: same data, three different training tables."""
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    blank(ax)
    t0, t1 = 0.08, 0.78

    def timeline(y, label, sub):
        ax.plot([t0, t1], [y, y], color=GREY, lw=1.6)
        ax.text(0.005, y + 0.055, label, fontsize=13, color=INK)
        ax.text(t1 + 0.02, y, sub, fontsize=11.5, color=GREY, va="center")

    # sliding: overlapping windows, one row per observation
    timeline(0.80, "sliding (rolling)", "one row per moment,\nwindows overlap")
    for i in range(5):
        # staggered, so the overlap between consecutive windows is visible
        x = t0 + i * 0.085
        ax.add_patch(Rectangle((x, 0.835 + i * 0.030), 0.20, 0.026,
                               facecolor=BLUE, alpha=0.40, edgecolor=BLUE))
    # tumbling: adjacent, non-overlapping
    timeline(0.45, "tumbling", "one row per block,\nno shared information")
    for i in range(3):
        x = t0 + i * 0.235
        ax.add_patch(Rectangle((x, 0.48), 0.225, 0.06, facecolor=AMBER, alpha=0.35,
                               edgecolor=AMBER))
    # expanding: everything since admission
    timeline(0.10, "expanding", "everything since\nadmission")
    for i in range(4):
        ax.add_patch(Rectangle((t0, 0.13 + i * 0.052), 0.16 + i * 0.175, 0.045,
                               facecolor=GREEN, alpha=0.30, edgecolor=GREEN))
    ax.text(0.5, 0.005, "sliding for online prediction · tumbling when the training "
                        "rows must be near-independent",
            ha="center", fontsize=12, color=INK)
    ax.set_xlim(0, 1.0)
    save(fig, out, "window_types")


def fig_lags(out: Path) -> None:
    """Where it was, how much it moved, how fast."""
    t, hr = _hr_series(1)
    t, hr = t[5:13], hr[5:13]
    i = int(np.argmax(np.diff(hr))) + 1          # the clearest step to annotate

    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    ax.plot(t, hr, "-o", color=BLUE, lw=1.8, ms=8)
    ax.plot([t[i - 1], t[i]], [hr[i - 1], hr[i - 1]], color=GREY, lw=1.4, ls="--")
    ax.plot([t[i], t[i]], [hr[i - 1], hr[i]], color=RED, lw=3)

    lo = hr.min() - 4
    ax.annotate("", (t[i - 1], lo + 0.8), (t[i], lo + 0.8),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=2))
    ax.text((t[i - 1] + t[i]) / 2, lo + 1.4, f"h_since_obs = {t[i] - t[i - 1]:.1f}",
            ha="center", fontsize=12, color=PURPLE)
    ax.text(t[i - 1] - 0.12, hr[i - 1], f"hr_lag1 = {hr[i - 1]:.0f}", ha="right",
            va="center", fontsize=12.5, color=AMBER)
    ax.text(t[i] + 0.12, (hr[i] + hr[i - 1]) / 2,
            f"hr_delta = {hr[i] - hr[i - 1]:+.0f}", fontsize=12.5, color=RED,
            va="center")
    ax.text(t[i] + 0.12, hr[i] + 0.6,
            f"hr_rate = {(hr[i] - hr[i - 1]) / (t[i] - t[i - 1]):+.1f} bpm/h",
            fontsize=12.5, color=GREEN, va="bottom")
    ax.plot([t[0]], [hr[0]], "o", color=INK, ms=10, zorder=4)
    ax.text(t[0], hr[0] + 1.2, "first row: the lag is missing",
            fontsize=11.5, color=INK, ha="left", va="top")

    ax.set_xlabel("hours since admission")
    ax.set_ylabel("heart rate")
    ax.set_ylim(lo, hr.max() + 6)
    ax.set_title("the rate is usually the most useful of the three", fontsize=13)
    save(fig, out, "lags")


# ============================================================== 2. the catalogue
def fig_log_transform(out: Path) -> None:
    """A long right tail decides everything until you compress it."""
    rng = np.random.default_rng(0)
    los = np.clip(rng.lognormal(0.9, 0.85, 4000), 1, 90)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].hist(los, bins=60, color=BLUE)
    axes[0].set_xlabel("length of stay (days)")
    axes[0].set_ylabel("patients")
    axes[0].set_title("raw: half the axis is 30 patients", fontsize=13)
    axes[0].annotate("these few patients drive\nany squared error",
                     (22, 15), (17, 620), fontsize=11.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED))

    axes[1].hist(np.log1p(los), bins=60, color=GREEN)
    axes[1].set_xlabel("log1p(length of stay)")
    axes[1].set_title("after log1p: roughly symmetric", fontsize=13)
    axes[1].text(0.5, -0.22, "differences on this axis are RATIOS on the original one:"
                             "  +0.69 = twice as long",
                 transform=axes[1].transAxes, ha="center", fontsize=11.5, color=INK)
    fig.subplots_adjust(bottom=0.24)
    save(fig, out, "log_transform")


def fig_onehot(out: Path) -> None:
    """One binary column per level — and why the integer shortcut is wrong."""
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    blank(ax)
    levels = ("emergency", "elective", "transfer")
    rows = ("emergency", "transfer", "elective", "emergency")

    ax.text(0.10, 0.90, "admission_type", ha="center", fontsize=12.5, color=GREY)
    for i, r in enumerate(rows):
        box(ax, (0.10, 0.76 - i * 0.115), 0.17, 0.085, r, ec=FAINT, fs=11.5)
    arrow(ax, (0.21, 0.53), (0.29, 0.53), lw=2.2)
    ax.text(0.25, 0.58, "one-hot", ha="center", fontsize=11.5, color=GREY)

    for j, lv in enumerate(levels):
        ax.text(0.40 + j * 0.13, 0.90, f"is_{lv}", ha="center", fontsize=12,
                color=BLUE)
    for i, r in enumerate(rows):
        for j, lv in enumerate(levels):
            on = r == lv
            box(ax, (0.40 + j * 0.13, 0.76 - i * 0.115), 0.10, 0.085,
                "1" if on else "0", ec=(BLUE if on else FAINT),
                fc=("#dce7f3" if on else "white"), fs=12)

    ax.plot([0.79, 0.79], [0.08, 0.92], color=FAINT, lw=2)
    ax.text(0.90, 0.86, "the shortcut that is wrong", ha="center", fontsize=12.5,
            color=RED)
    ax.plot([0.83, 0.99], [0.62, 0.62], color=GREY, lw=1.8)
    for j, (lv, v) in enumerate(zip(levels, (1, 2, 3))):
        x = 0.84 + j * 0.07
        ax.plot([x], [0.62], "o", color=RED, ms=9)
        ax.text(x, 0.68, str(v), ha="center", fontsize=12, color=RED)
        ax.text(x, 0.53, lv[:4], ha="center", fontsize=10, color=GREY, rotation=30)
    ax.text(0.90, 0.36, "encoding 1 / 2 / 3 tells the model\n"
                        "transfer − elective = elective − emergency,\n"
                        "and that elective sits between them",
            ha="center", fontsize=11.5, color=INK)
    ax.text(0.90, 0.14, "none of that is true", ha="center", fontsize=12.5,
            color=RED)
    save(fig, out, "onehot")


def fig_standardize(out: Path) -> None:
    """Without scaling, the column with the biggest numbers wins every time."""
    rng = np.random.default_rng(2)
    hr = rng.normal(74, 12, 300)
    creat = rng.normal(1.1, 0.35, 300)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    axes[0].hist(hr, bins=40, color=BLUE, label="heart_rate")
    axes[0].hist(creat, bins=40, color=RED, label="creatinine")
    axes[0].set_xlim(-5, 120)
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title("raw units, same axis", fontsize=12.5)
    axes[0].annotate("creatinine is here", (1.1, 30), (22, 45), fontsize=11.5,
                     color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    axes[0].set_yticks([])

    z = lambda v: (v - v.mean()) / v.std()
    axes[1].hist(z(hr), bins=40, color=BLUE, alpha=0.75)
    axes[1].hist(z(creat), bins=40, color=RED, alpha=0.75)
    axes[1].set_title("after StandardScaler", fontsize=12.5)
    axes[1].set_xlabel("(x − mean) / std")
    axes[1].set_yticks([])

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.88, "a distance between two patients", ha="center", fontsize=12.5,
            color=GREY)
    ax.text(0.5, 0.68, "raw:   $\\sqrt{(\\Delta hr)^2 + (\\Delta creat)^2}$\n"
                       "= $\\sqrt{18^2 + 0.4^2}$ = 18.00",
            ha="center", fontsize=13, color=RED)
    ax.text(0.5, 0.42, "the creatinine difference changed\nthe distance by 0.004",
            ha="center", fontsize=12.5, color=RED)
    ax.text(0.5, 0.16, "scaled:  1.50 and 1.14 —\nboth columns now count",
            ha="center", fontsize=13, color=GREEN)
    save(fig, out, "standardize")


def fig_binning(out: Path) -> None:
    """A threshold effect a straight line cannot see."""
    rng = np.random.default_rng(3)
    age = rng.uniform(20, 95, 500)
    risk = 0.08 + 0.32 * (age > 65) + 0.22 * (age > 80) + rng.normal(0, 0.05, 500)
    edges = [20, 65, 80, 95]

    fig, ax = plt.subplots(figsize=HALF)
    ax.scatter(age, risk, s=14, color=FAINT, edgecolors="none")
    a, b = np.polyfit(age, risk, 1)
    g = np.linspace(20, 95, 2)
    ax.plot(g, a * g + b, color=RED, lw=2.8, label="linear in age")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (age >= lo) & (age < hi)
        ax.plot([lo, hi], [risk[m].mean()] * 2, color=GREEN, lw=3.4,
                label="binned" if lo == 20 else None)
        ax.axvline(hi, color=GREY, ls=":", lw=1.2)
    ax.set_xlabel("age")
    ax.set_ylabel("risk")
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax.set_title("the bands clinicians already use: <65, 65–79, 80+", fontsize=13)
    ax.text(0.5, -0.20, "the cut points are a modelling choice — justify them",
            transform=ax.transAxes, ha="center", fontsize=11.5, color=GREY)
    fig.subplots_adjust(bottom=0.22)
    save(fig, out, "binning")


def fig_polynomial(out: Path) -> None:
    """U-shaped in age: a straight line has to average it away."""
    rng = np.random.default_rng(4)
    age = rng.uniform(0, 95, 400)
    risk = 0.35 - 0.011 * age + 0.00013 * age ** 2 + rng.normal(0, 0.035, 400)
    g = np.linspace(0, 95, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, deg, title, colour in ((axes[0], 1, "age only", RED),
                                   (axes[1], 2, "age and age²", GREEN)):
        ax.scatter(age, risk, s=13, color=FAINT, edgecolors="none")
        p = np.polynomial.Polynomial.fit(age, risk, deg)
        ax.plot(g, p(g), color=colour, lw=3)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("age")
    axes[0].set_ylabel("in-hospital risk")
    axes[0].text(6, 0.34, "infants and the very old are\nboth high risk — one line\n"
                          "cannot say that", fontsize=11.5, color=RED)
    axes[1].text(30, 0.34, "standardise first, or age²\ndwarfs every other column",
                 fontsize=11.5, color=GREY)
    save(fig, out, "polynomial")


def fig_interaction(out: Path) -> None:
    """The effect of one column depends on another."""
    creat = np.linspace(0.6, 2.4, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, inter, title in ((axes[0], 0.0, "additive model: creatinine + age"),
                             (axes[1], 0.30, "with creatinine × age")):
        for age, colour, label in ((30, BLUE, "age 30"), (85, RED, "age 85")):
            base = 0.05 + 0.16 * (creat - 0.6) + 0.004 * (age - 30)
            ax.plot(creat, base + inter * (creat - 0.6) * (age - 30) / 55,
                    color=colour, lw=3, label=label)
        ax.axvline(1.4, color=GREY, ls=":", lw=1.4)
        ax.set_xlabel("creatinine (mg/dL)")
        ax.set_title(title, fontsize=13)
    axes[0].set_ylabel("risk")
    axes[0].legend(frameon=False, fontsize=11.5, loc="lower right")
    axes[0].text(0.03, 0.97, "the two lines stay parallel: the model is\n"
                             "forced to say creatinine means the same\n"
                             "thing at every age",
                 transform=axes[0].transAxes, va="top", fontsize=11, color=GREY)
    axes[1].text(0.65, 0.42, "1.4 is mild at 30 and alarming at 85 —\n"
                             "one extra column says so",
                 fontsize=11.5, color=GREEN)
    save(fig, out, "interaction")


def fig_group_relative(out: Path) -> None:
    """95 bpm means two different things in two different wards."""
    rng = np.random.default_rng(5)
    ward = rng.normal(72, 9, 700)
    icu = rng.normal(94, 11, 700)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].hist(ward, bins=45, color=BLUE, alpha=0.8, label="general ward")
    axes[0].hist(icu, bins=45, color=RED, alpha=0.8, label="ICU")
    axes[0].axvline(95, color=INK, lw=2.4)
    axes[0].text(96.5, axes[0].get_ylim()[1] * 0.55, "a patient at 95 bpm",
                 fontsize=11.5, color=INK, rotation=90, va="center",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axes[0].set_xlabel("heart rate")
    axes[0].set_yticks([])
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title("the same number, two populations", fontsize=13)

    zw = (95 - ward.mean()) / ward.std()
    zi = (95 - icu.mean()) / icu.std()
    ax = axes[1]
    ax.barh([1, 0], [zw, zi], color=[BLUE, RED], height=0.45)
    ax.axvline(0, color=GREY, lw=1.4)
    ax.set_yticks([1, 0], ["general ward", "ICU"], fontsize=12.5)
    ax.set_xlabel("hr_z_ward = (95 − ward mean) / ward sd")
    for yy, v in ((1, zw), (0, zi)):
        ax.text(v + 0.06, yy, f"{v:+.1f} σ", va="center", fontsize=13)
    ax.set_xlim(-0.6, 3.2)
    ax.set_title("group-relative: alarming on the ward, ordinary in the ICU",
                 fontsize=12.5)
    save(fig, out, "group_relative")


def fig_embedding(out: Path) -> None:
    """15 000 sparse columns, or 32 dense ones the model learns for itself."""
    rng = np.random.default_rng(6)
    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.16)

    ax = fig.add_subplot(gs[0])
    blank(ax)
    ax.text(0.5, 0.93, "one row, two encodings", ha="center", fontsize=13,
            color=GREY)
    for j in range(28):
        on = j == 11
        ax.add_patch(Rectangle((0.03 + j * 0.033, 0.62), 0.028, 0.11,
                               facecolor="#dce7f3" if on else "white",
                               edgecolor=BLUE if on else FAINT, lw=1.4))
    ax.text(0.03 + 11 * 0.033 + 0.014, 0.78, "1", ha="center", fontsize=12,
            color=BLUE)
    ax.text(0.5, 0.53, "one-hot: 15 000 columns, 14 999 of them zero",
            ha="center", fontsize=12.5, color=INK)
    ax.text(0.5, 0.44, "every code equally far from every other code",
            ha="center", fontsize=11.5, color=RED)
    vals = rng.normal(0, 1, 12)
    for j, v in enumerate(vals):
        ax.add_patch(Rectangle((0.19 + j * 0.055, 0.16), 0.05, 0.11,
                               facecolor=plt.cm.RdBu_r(0.5 + v / 6),
                               edgecolor=GREY, lw=0.8))
    ax.text(0.5, 0.07, "embedding: 32 numbers, learned with the classifier",
            ha="center", fontsize=12.5, color=GREEN)

    ax2 = fig.add_subplot(gs[1])
    groups = (("cardiac", (-1.1, 0.7), RED), ("renal", (1.0, 0.9), BLUE),
              ("respiratory", (0.2, -1.0), GREEN))
    for name, (cx, cy), colour in groups:
        pts = rng.normal((cx, cy), 0.30, (26, 2))
        ax2.scatter(pts[:, 0], pts[:, 1], s=26, color=colour, edgecolors="none",
                    alpha=0.85)
        ax2.text(cx, cy + 0.62, name, ha="center", fontsize=12.5, color=colour)
    ax2.set_title("the learned space: codes with similar outcomes end up close",
                  fontsize=12)
    ax2.set_xlabel("embedding dim 1")
    ax2.set_ylabel("embedding dim 2")
    ax2.set_xticks([])
    ax2.set_yticks([])
    save(fig, out, "embedding")


# ============================================================== 3. missing data
def _ward_table(seed=0, n=60, p=9):
    """A patient x feature table with realistic, structured holes."""
    rng = np.random.default_rng(seed)
    names = ["age", "sex", "heart_rate", "bp", "creatinine", "troponin",
             "lactate", "walk_test", "hr_lag1"]
    M = np.zeros((n, p), bool)
    M[:, 5] = rng.random(n) < 0.62          # troponin: only ordered sometimes
    M[:, 6] = rng.random(n) < 0.45          # lactate: ICU only
    M[:, 7] = rng.random(n) < 0.78          # walk test: rarely done
    M[:, 4] = rng.random(n) < 0.12
    M[:, 3] = rng.random(n) < 0.08
    M[:2, 8] = True                          # lag features: missing at the start
    M[rng.choice(n, 6, replace=False), 8] = True
    return names, M


def fig_missing_map(out: Path) -> None:
    """Look at the holes before you fill them: they have structure."""
    from matplotlib.colors import ListedColormap

    names, M = _ward_table()
    fig = plt.figure(figsize=(12.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1], wspace=0.04)

    ax = fig.add_subplot(gs[0])
    ax.imshow(M.T, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#f4f6f8", INK]), vmin=0, vmax=1)
    ax.set_yticks(range(len(names)), names, fontsize=11.5)
    ax.set_xticks([])
    ax.set_xlabel("one column per patient")
    ax.set_title("the missingness map: black = missing", fontsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)

    frac = M.mean(0) * 100
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    ax2.barh(np.arange(len(names)), frac,
             color=[RED if f > 50 else (AMBER if f > 20 else BLUE) for f in frac],
             height=0.6)
    for i, f in enumerate(frac):
        ax2.text(f + 3, i, f"{f:.0f}%", va="center", fontsize=11)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 50, 100], ["0", "50", "100%"], fontsize=10.5)
    ax2.set_title("% missing", fontsize=12)
    ax2.tick_params(labelleft=False)

    fig.text(0.5, 0.045, "walk_test and troponin go missing in blocks, not at random\n"
                         "— that pattern is itself a feature "
                         "(`add_indicator=True`)",
             ha="center", fontsize=12.5, color=INK)
    fig.subplots_adjust(bottom=0.20)
    save(fig, out, "missing_map")


def fig_mechanisms(out: Path) -> None:
    """MCAR, MAR, MNAR — and the bias each one leaves behind."""
    rng = np.random.default_rng(1)
    n = 320
    age = rng.uniform(25, 90, n)
    trop = 0.02 * age + rng.normal(0, 0.45, n) + 0.6

    drop_mcar = rng.random(n) < 0.45
    drop_mar = rng.random(n) < np.clip(1.25 - 0.018 * age, 0.03, 0.95)
    drop_mnar = rng.random(n) < np.clip((trop - 0.9) * 1.4, 0.02, 0.95)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True)
    cases = (("MCAR — a broken analyser", drop_mcar, "nothing"),
             ("MAR — ordered for older patients", drop_mar, "age (observed)"),
             ("MNAR — too sick to be tested", drop_mnar, "the value itself"))
    for ax, (title, drop, dep) in zip(axes, cases):
        ax.scatter(age[~drop], trop[~drop], s=18, color=BLUE, edgecolors="none",
                   label="observed")
        ax.scatter(age[drop], trop[drop], s=26, color=RED, marker="x", lw=1.2,
                   label="missing")
        ax.axhline(trop.mean(), color=INK, lw=2)
        ax.axhline(trop[~drop].mean(), color=RED, lw=2, ls="--")
        bias = trop[~drop].mean() - trop.mean()
        ax.set_title(f"{title}\nP(missing) depends on {dep}", fontsize=12)
        ax.set_xlabel("age")
        ax.text(0.03, 0.03, f"mean of what you kept: {bias:+.2f}",
                transform=ax.transAxes, fontsize=11.5,
                color=(GREEN if abs(bias) < 0.06 else RED))
    axes[0].set_ylabel("troponin")
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.text(0.5, 0.005, "solid line = the true mean · dashed = the mean of the rows "
                         "you still have · only the third one is beyond repair",
             ha="center", fontsize=11.5, color=INK)
    fig.subplots_adjust(bottom=0.20)
    save(fig, out, "mechanisms")


def fig_mean_impute(out: Path) -> None:
    """Filling with the mean invents a fake, over-confident cohort."""
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(0, 1, n)
    y = 1.1 * x + rng.normal(0, 0.6, n)
    drop = rng.random(n) < 0.35
    y_imp = y.copy()
    y_imp[drop] = y[~drop].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    axes[0].hist(y, bins=45, color=BLUE, alpha=0.55, label="complete data")
    axes[0].hist(y_imp, bins=45, color=RED, alpha=0.75, label="after mean fill")
    axes[0].set_xlabel("creatinine (standardised)")
    axes[0].set_yticks([])
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title(f"variance {y.std():.2f} → {y_imp.std():.2f}", fontsize=13)
    axes[0].annotate("a spike that exists\nin no patient", (0, 120), (1.1, 150),
                     fontsize=11.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED))

    axes[1].scatter(x[~drop], y[~drop], s=16, color=BLUE, edgecolors="none")
    axes[1].scatter(x[drop], y_imp[drop], s=22, color=RED, edgecolors="none")
    axes[1].axhline(y[~drop].mean(), color=RED, lw=1.4, ls="--")
    r0 = np.corrcoef(x, y)[0, 1]
    r1 = np.corrcoef(x, y_imp)[0, 1]
    axes[1].set_xlabel("age (standardised)")
    axes[1].set_ylabel("creatinine")
    axes[1].set_title(f"correlation with age: {r0:.2f} → {r1:.2f}", fontsize=13)
    axes[1].text(0.03, 0.05, "every imputed patient is given\nthe same value, "
                             "whatever their age",
                 transform=axes[1].transAxes, fontsize=11.5, color=RED)
    save(fig, out, "mean_impute")


def fig_mice(out: Path) -> None:
    """Impute m times, analyse m times, pool — and keep the extra uncertainty."""
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(13.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.34)

    ax = fig.add_subplot(gs[0])
    blank(ax)
    ax.text(0.5, 0.95, "one table with holes  →  m completed tables", ha="center",
            fontsize=13, color=INK)
    base = rng.random((6, 5))
    holes = [(1, 2), (3, 0), (4, 3), (2, 4), (5, 1)]
    for d in range(3):
        x0 = 0.06 + d * 0.33
        for i in range(6):
            for j in range(5):
                miss = (i, j) in holes
                ax.add_patch(Rectangle((x0 + j * 0.045, 0.66 - i * 0.075), 0.042,
                                       0.068,
                                       facecolor=(plt.cm.Purples(0.25 + 0.5 * rng.random())
                                                  if miss else "#eef1f4"),
                                       edgecolor=(PURPLE if miss else FAINT),
                                       lw=1.2))
        ax.text(x0 + 0.11, 0.72, f"dataset {d + 1}", ha="center", fontsize=11.5,
                color=PURPLE)
    ax.text(0.5, 0.13, "purple cells = drawn from a model of\nthat column given all "
                       "the others;\neverything else is identical",
            ha="center", fontsize=12, color=INK)

    ax2 = fig.add_subplot(gs[1])
    est = np.array([0.55, 0.78, 0.57, 0.74, 0.66])
    se = np.array([0.09, 0.10, 0.09, 0.09, 0.10])
    ax2.errorbar(est, np.arange(5) + 1, xerr=1.96 * se, fmt="o", color=PURPLE,
                 capsize=4, lw=1.8, ms=7)
    pooled = est.mean()
    T = (se ** 2).mean() + (1 + 1 / 5) * est.var(ddof=1)
    ax2.errorbar([pooled], [0], xerr=1.96 * np.sqrt(T), fmt="o", color=INK,
                 capsize=5, lw=2.6, ms=10)
    ax2.errorbar([pooled], [-0.9], xerr=1.96 * np.sqrt((se ** 2).mean()), fmt="o",
                 color=RED, capsize=5, lw=2.2, ms=9)
    ax2.set_yticks([-0.9, 0] + list(range(1, 6)),
                   ["single imputation", "pooled (Rubin)"] + [f"m = {i}" for i in
                                                              range(1, 6)],
                   fontsize=11.5)
    ax2.set_xlabel("estimated log-odds of readmission")
    ax2.set_title("the m answers disagree — that spread is real information",
                  fontsize=12.5)
    ax2.text(pooled + 0.02, -1.7, "too narrow: it never knew\nthe values were invented",
             fontsize=11, color=RED, ha="center")
    ax2.set_ylim(-2.4, 5.8)
    save(fig, out, "mice")


def fig_eq_rubin(out: Path) -> None:
    equation(out, "eq_rubin", [
        r"$\bar{Q}=\frac{1}{m}\sum_{i=1}^{m} Q_i$"
        r"$\qquad\qquad T=\bar{U}+\left(1+\frac{1}{m}\right)B$",
    ], fs=30,
       annots=((0.30, "the pooled\nestimate", BLUE),
               (0.63, "within-imputation\nvariance", GREEN),
               (0.88, "between-imputation\nvariance", RED)))


FIGURES = (
    fig_cyclical, fig_elapsed, fig_windows, fig_window_types, fig_lags,
    fig_log_transform, fig_onehot, fig_standardize, fig_binning, fig_polynomial,
    fig_interaction, fig_group_relative, fig_embedding,
    fig_missing_map, fig_mechanisms, fig_mean_impute, fig_mice,
    fig_eq_rubin,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course02/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
