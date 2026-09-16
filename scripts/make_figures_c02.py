#!/usr/bin/env python3
"""Render the illustration figures used by the Course02 slides.

The deck covers feature engineering and missing data first, then "how models
work"; the two halves were once separate decks, and the figures are still
grouped that way below. Everything lands in `Course02/img/`.

Same contract as `make_figures_c03.py`: every figure is synthetic and seeded, so
`make figures` reproduces the same output anywhere without touching `data/`.
Only numpy / matplotlib / scikit-learn are needed.

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


def bare(ax):
    """Keep the axes box but drop ticks: used by every decision-boundary panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def equation(out: Path, name: str, lines, fs=34, annots=()) -> None:
    """Render display math as an image, auto-sized to fill the slide width.

    Built at the slide aspect rather than padded up to it, and the font size is
    fitted by measuring the drawn text: a hand-picked size either overflows or
    leaves the equation tiny on the slide. `annots` are (x, text, colour) callouts
    at explicit figure coordinates.
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

# =========================== part 1: feature engineering and missing data
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

# ============================================ part 2: how the models work
# --------------------------------------------------------------- shared datasets
def _moons(n=260, noise=0.22, seed=0):
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n, noise=noise, random_state=seed)


def _blobs2(n=200, seed=1):
    """Two roughly linearly separable classes: the logistic-regression example."""
    rng = np.random.default_rng(seed)
    a = rng.normal((-1.0, -0.6), (0.85, 0.75), (n // 2, 2))
    b = rng.normal((1.2, 0.9), (0.85, 0.75), (n // 2, 2))
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    return X, y


def _scatter_classes(ax, X, y, s=22, alpha=1.0):
    for cls, colour, marker in ((0, BLUE, "o"), (1, RED, "^")):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=s, color=colour, marker=marker,
                   edgecolors="none", alpha=alpha, zorder=3)


def _field(ax, model, X, cmap="RdBu_r", res=300):
    """Draw the predicted-probability field of a fitted classifier over the data."""
    pad = 0.4
    gx, gy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, res),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, res),
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    z = model.predict_proba(grid)[:, 1].reshape(gx.shape)
    ax.imshow(z, extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin="lower",
              cmap=cmap, vmin=0, vmax=1, alpha=0.72, aspect="auto",
              interpolation="bilinear")
    ax.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.4)
    ax.set_aspect("equal")
    return z


def _house_data(n=60, seed=4):
    """The running example: surface (m²) → price (k€), with honest noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(25, 145, n)
    y = 1.9 * x + 60 + rng.normal(0, 26, n)
    return x, y


# ================================================================ 0. the setup
def fig_map(out: Path) -> None:
    """The whole deck on one slide: every model, and what its prediction is."""
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    blank(ax)
    rows = (
        ("Linear regression", "a weighted sum of the columns", BLUE),
        ("Logistic regression", "the same sum, squashed into a probability", BLUE),
        ("Decision tree", "a sequence of yes/no questions", AMBER),
        ("Random forest", "hundreds of disagreeing trees, averaged", GREEN),
        ("Boosting", "many small trees, each fixing the last one's mistakes", RED),
        ("Support vector machine",
         "the widest street between the classes — bent by a kernel", PURPLE),
        ("k-nearest neighbours", "whatever the closest rows in the table say", INK),
    )
    ax.text(0.24, 0.94, "the model", ha="center", fontsize=12.5, color=GREY)
    ax.text(0.70, 0.94, "what its prediction actually is", ha="center",
            fontsize=12.5, color=GREY)
    for i, (name, what, colour) in enumerate(rows):
        y = 0.845 - i * 0.128
        box(ax, (0.24, y), 0.36, 0.080, name, ec=colour, fs=12.5)
        ax.text(0.455, y, "→", ha="center", va="center", fontsize=15, color=GREY)
        ax.text(0.495, y, what, ha="left", va="center", fontsize=12.5, color=INK)
    save(fig, out, "map")


def fig_supervised(out: Path) -> None:
    """Every model in this deck is the same box with different insides."""
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    blank(ax)

    ax.text(0.13, 0.90, "features  X", ha="center", fontsize=13, color=BLUE)
    cells = (("surface", "78 m²"), ("rooms", "3"), ("floor", "4"), ("year", "1974"))
    for i, (k, v) in enumerate(cells):
        y = 0.74 - i * 0.155
        box(ax, (0.13, y), 0.20, 0.105, f"{k}   {v}", ec=FAINT, fs=11.5)

    box(ax, (0.47, 0.50), 0.22, 0.26, "model\n$f_\\theta$", ec=INK, fs=17)
    arrow(ax, (0.25, 0.50), (0.35, 0.50), lw=2.2)
    arrow(ax, (0.59, 0.50), (0.70, 0.50), lw=2.2)
    box(ax, (0.82, 0.50), 0.20, 0.14, "310 k€", ec=RED, fs=15)
    ax.text(0.82, 0.66, "prediction  $\\hat y$", ha="center", fontsize=13, color=RED)

    ax.text(0.47, 0.20, "$\\theta$ is chosen to make $\\hat y$ close to the\n"
                        "true $y$ on the training rows",
            ha="center", fontsize=12.5, color=GREY)
    ax.annotate("", (0.47, 0.30), (0.47, 0.36),
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.text(0.13, 0.06, "what changes between models:\nthe SHAPE of $f$ and the LOSS we minimise",
            ha="left", fontsize=12, color=INK)
    save(fig, out, "supervised")


def fig_tasks(out: Path) -> None:
    """Regression predicts a number, classification predicts a label."""
    x, y = _house_data()
    Xc, yc = _blobs2()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].scatter(x, y, s=26, color=BLUE, edgecolors="none")
    axes[0].set_title("regression — the target is a NUMBER", fontsize=13)
    axes[0].set_xlabel("surface (m²)")
    axes[0].set_ylabel("price (k€)")

    _scatter_classes(axes[1], Xc, yc, s=26)
    axes[1].set_title("classification — the target is a LABEL", fontsize=13)
    axes[1].set_xlabel("feature 1")
    axes[1].set_ylabel("feature 2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    save(fig, out, "tasks")


# ======================================================== 1. linear regression
def fig_linreg_fit(out: Path) -> None:
    """A line, and the residuals it is judged on."""
    x, y = _house_data()
    a, b = np.polyfit(x, y, 1)
    grid = np.linspace(20, 150, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, show in zip(axes, (False, True)):
        ax.scatter(x, y, s=26, color=BLUE, edgecolors="none", zorder=3)
        ax.plot(grid, a * grid + b, color=RED, lw=2.6, zorder=2)
        if show:
            for xi, yi in zip(x, y):
                ax.plot([xi, xi], [yi, a * xi + b], color=AMBER, lw=1.2, zorder=1)
        ax.set_xlabel("surface (m²)")
    axes[0].set_ylabel("price (k€)")
    axes[0].set_title("a line: $\\hat y = w\\,x + b$", fontsize=13)
    axes[1].set_title("the residuals $y_i - \\hat y_i$ it is scored on",
                      fontsize=13, color=AMBER)
    save(fig, out, "linreg_fit")


def fig_linreg_which(out: Path) -> None:
    """Three candidate lines and the one number that separates them."""
    x, y = _house_data()
    grid = np.linspace(20, 150, 2)
    a, b = np.polyfit(x, y, 1)
    cands = ((1.2, 130, GREY), (a, b, RED), (2.7, -20, PURPLE))

    fig, ax = plt.subplots(figsize=HALF)
    ax.scatter(x, y, s=26, color=BLUE, edgecolors="none", zorder=3)
    for w, c, colour in cands:
        mse = np.mean((y - (w * x + c)) ** 2)
        ax.plot(grid, w * grid + c, color=colour, lw=2.6)
        ax.text(151, w * 150 + c, f"  MSE {mse:,.0f}", color=colour,
                fontsize=12, va="center")
    ax.set_xlim(20, 185)
    ax.set_xlabel("surface (m²)")
    ax.set_ylabel("price (k€)")
    ax.set_title("“best” means: smallest mean squared error", fontsize=13)
    save(fig, out, "linreg_which")


def fig_eq_mse(out: Path) -> None:
    equation(out, "eq_mse", [
        r"$\mathcal{L}(w,b)\;=\;\frac{1}{n}\sum_{i=1}^{n}"
        r"\left(y_i-(w\,x_i+b)\right)^2$",
    ])


def fig_linreg_loss(out: Path) -> None:
    """The loss is a bowl, and the bowl has exactly one bottom."""
    x, y = _house_data()
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    ws = np.linspace(-1.5, 2.5, 200)
    bs = np.linspace(-1.6, 1.6, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    loss_w = [np.mean((y - (w * x)) ** 2) for w in ws]
    axes[0].plot(ws, loss_w, color=BLUE, lw=2.8)
    k = int(np.argmin(loss_w))
    axes[0].plot([ws[k]], [loss_w[k]], "o", color=RED, ms=9)
    axes[0].axvline(ws[k], color=GREY, ls="--", lw=1.2)
    axes[0].set_xlabel("slope $w$   (b fixed at 0)")
    axes[0].set_ylabel("mean squared error")
    axes[0].set_title("one parameter: a parabola", fontsize=13)

    W, B = np.meshgrid(ws, bs)
    L = np.mean((y[:, None, None] - (W[None] * x[:, None, None] + B[None])) ** 2, 0)
    cs = axes[1].contourf(W, B, L, levels=22, cmap="Blues_r")
    axes[1].contour(W, B, L, levels=22, colors="white", linewidths=0.5)
    j, i = np.unravel_index(np.argmin(L), L.shape)
    axes[1].plot([W[j, i]], [B[j, i]], "o", color=RED, ms=10)
    axes[1].set_xlabel("slope $w$")
    axes[1].set_ylabel("intercept $b$")
    axes[1].set_title("two parameters: a bowl, seen from above", fontsize=13)
    fig.colorbar(cs, ax=axes[1], shrink=0.85, label="MSE")
    save(fig, out, "linreg_loss")


def fig_linreg_gd(out: Path) -> None:
    """Two ways down the bowl: solve it, or walk it."""
    x, y = _house_data()
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    ws = np.linspace(-1.0, 2.2, 200)
    bs = np.linspace(-1.4, 1.4, 200)
    W, B = np.meshgrid(ws, bs)
    L = np.mean((y[:, None, None] - (W[None] * x[:, None, None] + B[None])) ** 2, 0)

    fig, ax = plt.subplots(figsize=HALF)
    ax.contour(W, B, L, levels=18, colors=[GREY], linewidths=0.8)
    w, b, path = -0.8, 1.2, []
    for _ in range(28):
        path.append((w, b))
        r = y - (w * x + b)
        w += 0.25 * 2 * np.mean(r * x)
        b += 0.25 * 2 * np.mean(r)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=RED, lw=2.0, ms=4.5)
    ax.plot([path[0, 0]], [path[0, 1]], "o", color=RED, ms=10)
    ax.text(path[0, 0], path[0, 1] + 0.12, "random start", color=RED, fontsize=11.5,
            ha="center")
    j, i = np.unravel_index(np.argmin(L), L.shape)
    ax.plot([W[j, i]], [B[j, i]], "*", color=GREEN, ms=20, zorder=4)
    ax.text(W[j, i] + 0.06, B[j, i] - 0.15,
            "the minimum\n(closed form: $\\hat\\theta=(X^\\top X)^{-1}X^\\top y$)",
            color=GREEN, fontsize=11.5)
    ax.set_xlabel("slope $w$")
    ax.set_ylabel("intercept $b$")
    ax.set_title("gradient descent: repeat  $\\theta \\leftarrow \\theta - "
                 "\\eta\\,\\nabla\\mathcal{L}$", fontsize=13)
    save(fig, out, "linreg_gd")


def fig_linreg_multi(out: Path) -> None:
    """Two features: the line becomes a plane; p features, a hyperplane."""
    rng = np.random.default_rng(2)
    n = 90
    x1 = rng.uniform(25, 145, n)
    x2 = rng.uniform(0, 8, n)
    y = 1.7 * x1 + 14 * x2 + 40 + rng.normal(0, 22, n)

    fig = plt.figure(figsize=(12.4, 5.0))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(x1, x2, y, s=18, color=BLUE, depthshade=False)
    g1, g2 = np.meshgrid(np.linspace(25, 145, 12), np.linspace(0, 8, 12))
    ax.plot_surface(g1, g2, 1.7 * g1 + 14 * g2 + 40, color=RED, alpha=0.35,
                    edgecolor=RED, linewidth=0.4)
    ax.set_xlabel("surface")
    ax.set_ylabel("floor")
    ax.set_zlabel("price")
    ax.set_title("2 features → a plane", fontsize=13)
    ax.view_init(18, -60)

    ax2 = fig.add_subplot(122)
    blank(ax2)
    ax2.text(0.5, 0.72, r"$\hat y \;=\; w_1x_1+w_2x_2+\cdots+w_px_p+b$",
             ha="center", fontsize=19, color=INK)
    ax2.text(0.5, 0.45, "one weight per column\n"
                        "each weight = “how much $\\hat y$ moves\n"
                        "when this column moves by 1, all else fixed”",
             ha="center", fontsize=13, color=GREY)
    ax2.text(0.5, 0.15, "this is why linear models are readable —\n"
                        "and why the units of your columns matter",
             ha="center", fontsize=12.5, color=AMBER)
    save(fig, out, "linreg_multi")


def fig_linreg_basis(out: Path) -> None:
    """“Linear” means linear in the WEIGHTS, not in x."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 45))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 45)
    grid = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, d, title in zip(axes, (1, 3, 12),
                            ("$x$ only", "$x, x^2, x^3$", "up to $x^{12}$")):
        p = np.polynomial.Polynomial.fit(x, y, d)
        ax.scatter(x, y, s=24, color=BLUE, edgecolors="none", zorder=3)
        ax.plot(grid, p(grid), color=RED, lw=2.6)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylim(-2.0, 2.0)
    axes[0].set_ylabel("y")
    save(fig, out, "linreg_basis")


def fig_logistic(out: Path) -> None:
    """From an unbounded score to a probability."""
    from sklearn.linear_model import LogisticRegression

    X, y = _blobs2()
    m = LogisticRegression().fit(X, y)
    z = np.linspace(-6, 6, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].plot(z, 1 / (1 + np.exp(-z)), color=PURPLE, lw=3)
    axes[0].axhline(0.5, color=GREY, ls="--", lw=1.2)
    axes[0].axvline(0, color=GREY, ls="--", lw=1.2)
    axes[0].set_xlabel("score  $z = w^\\top x + b$")
    axes[0].set_ylabel("$P(y=1)$")
    axes[0].set_title("the sigmoid  $\\sigma(z)=1/(1+e^{-z})$", fontsize=13)
    axes[0].text(-5.6, 0.86, "any real number in…", fontsize=11.5, color=GREY)
    axes[0].text(-5.6, 0.76, "…a probability out", fontsize=11.5, color=GREY)

    _field(axes[1], m, X)
    _scatter_classes(axes[1], X, y)
    axes[1].set_title("$z=0$ is a straight boundary; the colour is $P(y=1)$",
                      fontsize=13)
    bare(axes[1])
    save(fig, out, "logistic")


def fig_eq_logistic(out: Path) -> None:
    equation(out, "eq_logistic", [
        r"$P(y=1\mid x)\;=\;\sigma(w^\top x+b)$",
        r"$\mathcal{L}\;=\;-\frac{1}{n}\sum_i\left[y_i\log \hat p_i"
        r"+(1-y_i)\log(1-\hat p_i)\right]$",
    ], fs=28)


def fig_linreg_limits(out: Path) -> None:
    """Where a straight boundary simply cannot go."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    Xm, ym = _moons(240, 0.20, 0)
    t = rng.uniform(0, 2 * np.pi, 240)
    r = np.r_[rng.normal(0.7, 0.14, 120), rng.normal(1.9, 0.16, 120)]
    Xc = np.c_[r * np.cos(t), r * np.sin(t)]
    yc = np.r_[np.zeros(120, int), np.ones(120, int)]
    Xx = rng.normal(0, 0.8, (240, 2))
    yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    for ax, (X, y, title) in zip(axes, ((Xm, ym, "two moons"),
                                        (Xc, yc, "one class inside the other"),
                                        (Xx, yx, "XOR"))):
        m = LogisticRegression().fit(X, y)
        _field(ax, m, X)
        _scatter_classes(ax, X, y, s=18)
        acc = m.score(X, y)
        ax.set_title(f"{title} — accuracy {acc:.0%}", fontsize=12.5)
        bare(ax)
    save(fig, out, "linreg_limits")


# ============================================================ 2. decision trees
def _tree_toy(seed=0, n=240):
    """A dataset whose true boundary really is axis-aligned: trees shine here."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    y = ((X[:, 0] > 0.55) | ((X[:, 1] > 0.62) & (X[:, 0] > 0.22))).astype(int)
    flip = rng.random(n) < 0.06
    y[flip] = 1 - y[flip]
    return X, y


def fig_tree_questions(out: Path) -> None:
    """The tree and the partition it draws are the same object."""
    X, y = _tree_toy()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax = axes[0]
    blank(ax)
    box(ax, (0.5, 0.92), 0.36, 0.12, "$x_1 > 0.55$ ?", ec=INK, fs=13)
    ax.text(0.32, 0.78, "no", fontsize=11.5, color=GREY, ha="right")
    ax.text(0.715, 0.77, "yes", fontsize=11.5, color=GREY)
    arrow(ax, (0.42, 0.86), (0.26, 0.70))
    arrow(ax, (0.58, 0.86), (0.76, 0.70))
    box(ax, (0.24, 0.62), 0.34, 0.12, "$x_2 > 0.62$ ?", ec=INK, fs=13)
    box(ax, (0.78, 0.62), 0.26, 0.12, "class 1", ec=RED, fs=13, fc="#fdeaea")
    ax.text(0.10, 0.48, "no", fontsize=11.5, color=GREY, ha="right")
    ax.text(0.375, 0.47, "yes", fontsize=11.5, color=GREY)
    arrow(ax, (0.17, 0.56), (0.11, 0.40))
    arrow(ax, (0.31, 0.56), (0.42, 0.40))
    box(ax, (0.10, 0.32), 0.26, 0.12, "class 0", ec=BLUE, fs=13, fc="#e9f0f8")
    box(ax, (0.46, 0.32), 0.30, 0.12, "$x_1 > 0.22$ ?", ec=INK, fs=12.5)
    ax.text(0.31, 0.18, "no", fontsize=11.5, color=GREY, ha="right")
    ax.text(0.615, 0.17, "yes", fontsize=11.5, color=GREY)
    arrow(ax, (0.40, 0.26), (0.32, 0.14))
    arrow(ax, (0.52, 0.26), (0.62, 0.14))
    box(ax, (0.30, 0.07), 0.20, 0.11, "class 0", ec=BLUE, fs=12, fc="#e9f0f8")
    box(ax, (0.66, 0.07), 0.20, 0.11, "class 1", ec=RED, fs=12, fc="#fdeaea")

    ax2 = axes[1]
    ax2.axvspan(0.55, 1.0, color="#fdeaea")
    ax2.add_patch(plt.Rectangle((0, 0), 0.55, 0.62, color="#e9f0f8"))
    ax2.add_patch(plt.Rectangle((0, 0.62), 0.22, 0.38, color="#e9f0f8"))
    ax2.add_patch(plt.Rectangle((0.22, 0.62), 0.33, 0.38, color="#fdeaea"))
    ax2.axvline(0.55, color=INK, lw=2)
    ax2.plot([0, 0.55], [0.62, 0.62], color=INK, lw=2)
    ax2.plot([0.22, 0.22], [0.62, 1.0], color=INK, lw=2)
    _scatter_classes(ax2, X, y, s=20)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_title("every leaf is a rectangle of feature space", fontsize=13)
    save(fig, out, "tree_questions")


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


def fig_tree_split(out: Path) -> None:
    """How the first question is chosen: try every threshold, score each one."""
    X, y = _tree_toy()
    ths = np.linspace(0.03, 0.97, 200)

    def score(col):
        out_ = []
        for t in ths:
            m = X[:, col] <= t
            n = len(y)
            out_.append(m.sum() / n * _gini(y[m]) + (~m).sum() / n * _gini(y[~m]))
        return np.array(out_)

    s1, s2 = score(0), score(1)
    k = int(np.argmin(s1))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    _scatter_classes(axes[0], X, y, s=20)
    axes[0].axvline(ths[k], color=GREEN, lw=2.6)
    for t in (0.2, 0.35, 0.8):
        axes[0].axvline(t, color=GREY, lw=1.2, ls="--")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("candidate thresholds (dashed) — one wins", fontsize=13)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].plot(ths, s1, color=BLUE, lw=2.6, label="split on $x_1$")
    axes[1].plot(ths, s2, color=AMBER, lw=2.6, label="split on $x_2$")
    axes[1].plot([ths[k]], [s1[k]], "o", color=GREEN, ms=10, zorder=4)
    axes[1].annotate("best split of all\n(feature AND threshold)",
                     (ths[k], s1[k]), (ths[k] - 0.06, s1[k] + 0.10),
                     fontsize=11.5, color=GREEN, ha="center",
                     arrowprops=dict(arrowstyle="->", color=GREEN))
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("weighted Gini impurity after the split")
    axes[1].legend(frameon=False)
    axes[1].set_title("greedy: pick the split that drops impurity most", fontsize=13)
    save(fig, out, "tree_split")


def fig_impurity(out: Path) -> None:
    """What “pure” means, for the three usual measures."""
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].plot(p, 2 * p * (1 - p), color=BLUE, lw=2.8, label="Gini  $2p(1-p)$")
    axes[0].plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2, color=AMBER,
                 lw=2.8, label="entropy (÷2)")
    axes[0].plot(p, np.minimum(p, 1 - p), color=GREY, lw=2.4, ls="--",
                 label="misclassification")
    axes[0].set_xlabel("proportion of class 1 in the node  $p$")
    axes[0].set_ylabel("impurity")
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title("maximal at 50/50, zero when the node is pure", fontsize=13)

    ax = axes[1]
    blank(ax)
    for i, (frac, label) in enumerate(((0.5, "impure: 50/50"),
                                       (0.8, "better: 80/20"),
                                       (1.0, "pure: 100/0"))):
        y0 = 0.78 - i * 0.31
        rng = np.random.default_rng(i)
        xs = rng.uniform(0.34, 0.94, 40)
        ys = y0 + rng.uniform(-0.07, 0.07, 40)
        lab = (rng.random(40) < frac).astype(int)
        for cls, colour, marker in ((0, BLUE, "o"), (1, RED, "^")):
            m = lab == cls
            ax.scatter(xs[m], ys[m], s=34, color=colour, marker=marker)
        ax.text(0.30, y0, label, ha="right", va="center", fontsize=12.5)
        ax.text(0.97, y0, f"Gini {2 * lab.mean() * (1 - lab.mean()):.2f}",
                ha="left", va="center", fontsize=12, color=GREY)
    ax.set_xlim(0, 1.25)
    ax.set_title("a split is good when both children are purer", fontsize=13)
    save(fig, out, "impurity")


def fig_tree_grow(out: Path) -> None:
    """The same tree, stopped at four depths."""
    from sklearn.tree import DecisionTreeClassifier

    X, y = _tree_toy()
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.9), sharey=True)
    for ax, d in zip(axes, (1, 2, 4, None)):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        _field(ax, m, X)
        _scatter_classes(ax, X, y, s=14)
        leaves = m.get_n_leaves()
        title = f"depth {d}" if d else "unlimited"
        ax.set_title(f"{title} — {leaves} leaves\ntrain acc {m.score(X, y):.0%}",
                     fontsize=11.5)
        bare(ax)
    save(fig, out, "tree_grow")


def fig_tree_regression(out: Path) -> None:
    """A regression tree predicts a constant per leaf: staircases, not lines."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 80))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.18, 80)
    grid = np.linspace(0, 1, 600)[:, None]

    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, d in zip(axes, (1, 3, 12)):
        m = DecisionTreeRegressor(max_depth=d).fit(x[:, None], y)
        ax.scatter(x, y, s=22, color=BLUE, edgecolors="none", zorder=3)
        ax.plot(grid.ravel(), m.predict(grid), color=RED, lw=2.6)
        ax.plot(grid.ravel(), np.sin(2 * np.pi * grid.ravel()), color=GREY,
                lw=1.6, ls="--")
        ax.set_title(f"max_depth = {d}", fontsize=13)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, out, "tree_regression")


def fig_tree_overfit(out: Path) -> None:
    """Depth is the capacity knob, and it overfits exactly as advertised."""
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = _tree_toy(0, 700)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.6, random_state=0)
    depths = np.arange(1, 21)
    tr = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
          .score(Xtr, ytr) for d in depths]
    te = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
          .score(Xte, yte) for d in depths]

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(depths, 1 - np.array(tr), "o-", color=BLUE, lw=2.6, ms=5,
            label="training error")
    ax.plot(depths, 1 - np.array(te), "o-", color=RED, lw=2.6, ms=5,
            label="test error")
    k = int(np.argmax(te))
    ax.axvline(depths[k], color=GREY, ls="--", lw=1.2)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("error rate")
    ax.set_xticks(depths[::2])
    ax.legend(frameon=False)
    ax.set_title("an unpruned tree can always reach 0 training error", fontsize=13)
    save(fig, out, "tree_overfit")


def fig_tree_instability(out: Path) -> None:
    """Change 10% of the rows, get a different tree: this is the variance problem."""
    from sklearn.tree import DecisionTreeClassifier

    X, y = _moons(300, 0.22, 0)
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for i, ax in enumerate(axes):
        idx = rng.choice(len(X), len(X), replace=True)
        m = DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])
        _field(ax, m, X)
        _scatter_classes(ax, X[idx], y[idx], s=14)
        ax.set_title(f"same distribution, resample #{i + 1}", fontsize=12.5)
        bare(ax)
    save(fig, out, "tree_instability")


def fig_forest_idea(out: Path) -> None:
    """Averaging many noisy-but-unbiased guesses."""
    rng = np.random.default_rng(0)
    truth = 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    single = rng.normal(truth, 1.0, 4000)
    avg = rng.normal(truth, 1.0, (4000, 50)).mean(1)
    bins = np.linspace(-3.5, 3.5, 60)
    axes[0].hist(single, bins=bins, color=BLUE, alpha=0.8)
    axes[0].axvline(truth, color=INK, lw=2)
    axes[0].set_title(f"one tree: spread {single.std():.2f}", fontsize=13)
    axes[1].hist(avg, bins=bins, color=GREEN, alpha=0.85)
    axes[1].axvline(truth, color=INK, lw=2)
    axes[1].set_title(f"average of 50 independent trees: spread {avg.std():.2f}",
                      fontsize=13)
    for ax in axes:
        ax.set_xlabel("prediction error at one test point")
        ax.set_yticks([])
    save(fig, out, "forest_idea")


def _row(fig, frags, y, fs, colours, gaps=None, frac=0.86, max_fs=44):
    """Lay mathtext fragments out left to right, centred; return their centres.

    Same measure-then-place trick as `equation_parts`: the space *between*
    fragments is given here in em, never as `\\;` inside a fragment, because
    mathtext drops trailing space from a text's bounding box.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    figw = fig.get_size_inches()[0] * fig.dpi
    texts = [fig.text(0.0, y, f, fontsize=fs, color=c, ha="left", va="baseline")
             for f, c in zip(frags, colours)]
    gaps = [0.4] * (len(frags) - 1) if gaps is None else list(gaps)

    def widths():
        fig.canvas.draw()
        return [t.get_window_extent(r).width for t in texts]

    em = lambda size: size * fig.dpi / 72.0
    w = widths()
    fs_final = min(fs * frac * figw / (sum(w) + em(fs) * sum(gaps)), max_fs)
    for t in texts:
        t.set_fontsize(fs_final)
    w = widths()
    gap_px = [g * em(fs_final) for g in gaps] + [0.0]

    x = (figw - sum(w) - sum(gap_px)) / 2
    centres = []
    for t, wi, gi in zip(texts, w, gap_px):
        t.set_position((x / figw, y))
        centres.append((x + wi / 2) / figw)
        x += wi + gi
    return centres


def fig_eq_forest_variance(out: Path) -> None:
    """The formula, with both of its symbols defined in words."""
    fig = plt.figure(figsize=(12.6, 12.6 / SLIDE_ASPECT))
    centres = _row(fig, [
        r"$\mathrm{Var}\left(\bar f\right)=$",
        r"$\frac{\sigma^2}{M}$",
        r"$+$",
        r"$\frac{M-1}{M}\,\rho\,\sigma^2$",
    ], y=0.80, fs=34, colours=(INK, GREEN, INK, RED), gaps=(0.45, 0.45, 0.45))

    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    blank(ax)

    cards = (
        (0.26, GREEN, r"$\sigma^2$  —  how much ONE tree moves",
         "refit a single tree on a fresh sample:\nhow far does its prediction jump?",
         "divided by M  →  averaged away", 1),
        (0.74, RED, r"$\rho$  —  how much TWO trees agree",
         "take two trees of the forest:\ndo they get the same rows wrong?",
         "not divided by anything  →  the floor", 3),
    )
    for cx, colour, title, body, foot, term in cards:
        box(ax, (cx, 0.36), 0.44, 0.30, "", ec=colour, lw=2.0)
        ax.text(cx, 0.46, title, ha="center", va="center", fontsize=17,
                color=colour)
        ax.text(cx, 0.345, body, ha="center", va="center", fontsize=13.5,
                color=INK)
        ax.text(cx, 0.25, foot, ha="center", va="center", fontsize=13,
                color=colour)
        arrow(ax, (cx, 0.515), (centres[term], 0.70), color=colour, lw=1.6)

    ax.text(0.5, 0.06, "deep, unstable trees make $\\sigma^2$ big   ·   "
                       "trees grown on the same rows and columns make $\\rho$ big",
            ha="center", va="center", fontsize=13, color=GREY)
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / "eq_forest_variance.png")
    plt.close(fig)


def fig_forest_variance(out: Path) -> None:
    """Averaging buys you a floor, not zero — and the floor is set by rho."""
    M = np.arange(1, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

    ax = axes[0]
    for rho, colour in ((0.0, GREEN), (0.2, BLUE), (0.5, AMBER), (0.9, RED)):
        ax.plot(M, 1 / M + (M - 1) / M * rho, color=colour, lw=2.6,
                label=rf"$\rho$ = {rho}")
        ax.plot([M.min(), M.max()], [rho, rho], color=colour, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("number of trees M  (log scale)")
    ax.set_ylabel(r"Var of the average  (in units of $\sigma^2$)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 8000)          # empty gutter on the right for the legend
    ax.set_title("the variance stops falling at the floor $\\rho\\sigma^2$",
                 fontsize=13)
    ax.text(1200, 0.05, "dotted line =\nthe floor $\\rho\\sigma^2$", fontsize=10.5,
            color=GREY)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))

    ax2 = axes[1]
    for rho, colour in ((0.0, GREEN), (0.2, BLUE), (0.5, AMBER), (0.9, RED)):
        eff = M / (1 + (M - 1) * rho)
        ax2.plot(M, eff, color=colour, lw=2.6)
        if rho:
            ax2.plot([M.min(), M.max()], [1 / rho, 1 / rho], color=colour, ls=":",
                     lw=1.2)
            ax2.text(1300, 1 / rho, rf"  $1/\rho$ = {1 / rho:.1f}", color=colour,
                     fontsize=11, va="center")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 6000)
    ax2.set_xlabel("number of trees M  (log scale)")
    ax2.set_ylabel("trees that actually count  $M/(1+(M-1)\\rho)$")
    ax2.set_title("1000 correlated trees are worth a handful", fontsize=13)
    ax2.annotate(r"$\rho$ = 0.5, M = 1000" "\n" r"$\to$ worth 2 independent trees",
                 (1000, 1000 / (1 + 999 * 0.5)), (28, 30), fontsize=11.5,
                 color=AMBER, arrowprops=dict(arrowstyle="->", color=AMBER))

    # both axes are stated in the two symbols, so restate what they are
    fig.subplots_adjust(bottom=0.26, wspace=0.3)
    fig.text(0.28, 0.06, "$\\sigma^2$ = variance of ONE tree\n"
                         "(how far its prediction moves when it is refitted)",
             ha="center", va="center", fontsize=12.5, color=INK)
    fig.text(0.76, 0.06, "$\\rho$ = correlation between two trees of the forest\n"
                         "(how often they are wrong on the same rows)",
             ha="center", va="center", fontsize=12.5, color=INK)
    save(fig, out, "forest_variance")


def fig_forest_diagram(out: Path) -> None:
    """Two sources of randomness, one vote."""
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    blank(ax)
    box(ax, (0.5, 0.90), 0.24, 0.10, "training data", ec=INK, fs=12.5)
    for i in range(5):
        x = 0.13 + i * 0.185
        arrow(ax, (0.5, 0.845), (x, 0.70), color=GREY, lw=1.2)
        box(ax, (x, 0.615), 0.155, 0.145,
            f"bootstrap\nsample {i + 1}", ec=BLUE, fs=10.5)
        arrow(ax, (x, 0.54), (x, 0.44), color=GREY, lw=1.2)
        box(ax, (x, 0.355), 0.155, 0.145, f"deep tree {i + 1}", ec=GREEN, fs=10.5)
        arrow(ax, (x, 0.28), (0.5, 0.17), color=GREY, lw=1.2)
    box(ax, (0.5, 0.10), 0.30, 0.10, "average  /  majority vote", ec=GREEN, fs=12.5)
    ax.text(0.965, 0.615, "\u2460 rows drawn\nwith replacement", fontsize=11.5,
            color=BLUE, va="center", ha="left")
    ax.text(0.965, 0.355, "\u2461 at every split,\nonly \u221ap random\ncolumns are tried",
            fontsize=11.5, color=GREEN, va="center", ha="left")
    ax.set_xlim(0, 1.30)
    save(fig, out, "forest_diagram")


def fig_forest_boundary(out: Path) -> None:
    """One tree, ten trees, five hundred: the boundary stops twitching."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = _moons(260, 0.22, 0)
    models = (
        ("1 tree", DecisionTreeClassifier(random_state=0)),
        ("10 trees", RandomForestClassifier(10, random_state=0)),
        ("500 trees", RandomForestClassifier(500, random_state=0)),
    )
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, (title, m) in zip(axes, models):
        m.fit(X, y)
        _field(ax, m, X)
        _scatter_classes(ax, X, y, s=16)
        ax.set_title(title, fontsize=13)
        bare(ax)
    axes[1].text(0.5, -0.06, "same bias, less and less variance — "
                             "and the probabilities become usable",
                 transform=axes[1].transAxes, ha="center", fontsize=11.5,
                 color=GREY)
    save(fig, out, "forest_boundary")


def fig_max_features(out: Path) -> None:
    """What the knob literally does: a fresh draw of columns at every node."""
    rng = np.random.default_rng(7)
    p_cols, k = 12, 4
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.95, f"p = {p_cols} columns,  max_features = {k}", ha="center",
            fontsize=13, color=INK)

    nodes = ("root", "left child", "right child")
    for r, name in enumerate(nodes):
        y = 0.74 - r * 0.215
        drawn = np.sort(rng.choice(p_cols, k, replace=False))
        winner = drawn[rng.integers(k)]
        ax.text(0.0, y, name, ha="left", va="center", fontsize=12, color=GREY)
        for c in range(p_cols):
            x = 0.200 + c * 0.046
            on = c in drawn
            ax.add_patch(FancyBboxPatch(
                (x - 0.020, y - 0.050), 0.040, 0.100,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor="#dce7f3" if on else "white",
                edgecolor=GREEN if c == winner else (BLUE if on else FAINT),
                linewidth=2.6 if c == winner else 1.4))
            ax.text(x, y, f"$x_{{{c + 1}}}$", ha="center", va="center",
                    fontsize=10.5, color=INK if on else GREY)
        ax.text(1.0, y, f"best split: $x_{{{winner + 1}}}$", ha="right",
                va="center", fontsize=12, color=GREEN)
    ax.text(0.5, 0.14, "the winner is the best of the 4 drawn, never of all 12",
            ha="center", fontsize=12, color=GREY)
    ax.text(0.5, 0.04, "a NEW draw at every node — not once per tree",
            ha="center", fontsize=13, color=RED)

    ax2 = axes[1]
    blank(ax2)
    ax2.text(0.5, 0.94, "at every split, the tree:", ha="center", fontsize=13,
             color=INK)
    steps = (("1.", "draw max_features columns at random"),
             ("2.", "try every threshold, on those only"),
             ("3.", "keep the best, then recurse"))
    for i, (n, t) in enumerate(steps):
        y = 0.80 - i * 0.10
        ax2.text(0.04, y, n, fontsize=12.5, color=BLUE, va="center")
        ax2.text(0.12, y, t, fontsize=12.5, color=INK, va="center")

    ax2.text(0.5, 0.42, "what to set it to", ha="center", fontsize=13, color=GREY)
    rows = ((r"$\sqrt{p}$", "classification default", GREEN),
            (r"$p/3$  or  all", "regression defaults", GREEN),
            ("1", "maximum diversity, weakest trees", AMBER),
            ("None = all p", "no column randomness → plain bagging", RED))
    for i, (val, what, colour) in enumerate(rows):
        y = 0.31 - i * 0.088
        ax2.text(0.30, y, val, ha="right", fontsize=12.5, color=colour, va="center")
        ax2.text(0.35, y, what, fontsize=12, color=INK, va="center")
    save(fig, out, "max_features")


def fig_forest_features(out: Path) -> None:
    """Why max_features is the knob that makes a forest a forest."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # 25 columns, many of them informative but redundant: exactly the situation
    # where every unrestricted tree would keep choosing the same strong split
    X, y = make_classification(n_samples=1200, n_features=25, n_informative=12,
                               n_redundant=8, class_sep=0.7, flip_y=0.05,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    ks = [1, 2, 3, 5, 8, 12, 18, 25]
    err = np.array([[1 - RandomForestClassifier(300, max_features=k, random_state=s)
                     .fit(Xtr, ytr).score(Xte, yte) for k in ks] for s in range(5)])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].plot(ks, err.mean(0), "o-", color=GREEN, lw=2.8, ms=7)
    axes[0].fill_between(ks, err.mean(0) - err.std(0), err.mean(0) + err.std(0),
                         color=GREEN, alpha=0.15)
    axes[0].axvline(np.sqrt(25), color=GREY, ls="--", lw=1.4)
    axes[0].text(np.sqrt(25) + 0.4, err.max() * 0.99,
                 "the $\\sqrt{p}$ default", fontsize=11.5, color=GREY)
    axes[0].text(1, err.mean(0)[0], "  trees too weak", fontsize=11.5, color=AMBER,
                 va="center")
    axes[0].text(24.5, err.mean(0)[-1] - 0.0012, "all columns\n= plain bagging",
                 fontsize=11.5, color=AMBER, ha="right", va="top")
    axes[0].set_xlabel("max_features  (columns tried per split, out of 25)")
    axes[0].set_ylabel("test error")
    axes[0].set_title("25 columns, 12 informative and 8 redundant", fontsize=13)

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.86, "both ends are worse than the middle", ha="center",
            fontsize=14, color=GREEN)
    ax.text(0.5, 0.62, "all columns allowed \u2192 every tree picks\n"
                       "the same strong split first \u2192 $\\rho$ large\n"
                       "\u2192 averaging buys almost nothing",
            ha="center", fontsize=13, color=RED)
    ax.text(0.5, 0.30, "one column allowed \u2192 the trees disagree,\n"
                       "but each one is barely better than chance\n"
                       "\u2192 low $\\rho$, high individual error",
            ha="center", fontsize=13, color=AMBER)
    ax.text(0.5, 0.06, "$\\sqrt{p}$ is a default, not a law \u2014 tune it",
            ha="center", fontsize=12.5, color=GREY)
    save(fig, out, "forest_features")


def fig_forest_correlation(out: Path) -> None:
    """rho is not a metaphor: measure it, and watch it fight tree strength."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1200, n_features=25, n_informative=12,
                               n_redundant=8, class_sep=0.7, flip_y=0.05,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)

    ks = [1, 2, 3, 5, 8, 12, 18, 25]
    rho, solo, forest = [], [], []
    for k in ks:
        f = RandomForestClassifier(60, max_features=k, random_state=0).fit(Xtr, ytr)
        # every tree's own votes on the test rows: correlated trees make
        # correlated mistakes, which is exactly the rho of the formula
        preds = np.array([t.predict(Xte) for t in f.estimators_])
        c = np.corrcoef(preds)
        rho.append(c[np.triu_indices_from(c, k=1)].mean())
        solo.append(np.mean([1 - np.mean(p == yte) for p in preds]))
        forest.append(1 - f.score(Xte, yte))
    rho, solo, forest = np.array(rho), np.array(solo), np.array(forest)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7))
    # the twin y-axis needs room, or its label lands on the right panel
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    ax.plot(ks, rho, "o-", color=RED, lw=2.8, ms=6)
    ax.set_ylabel(r"mean pairwise correlation $\rho$", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.set_xlabel("max_features  (columns tried per split, out of 25)")
    ax.set_ylim(0, 0.42)
    ax2 = ax.twinx()
    ax2.plot(ks, solo, "s--", color=BLUE, lw=2.4, ms=6)
    ax2.set_ylabel("error of ONE tree in the forest", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(0.30, 0.42)
    ax2.spines["right"].set_visible(True)
    ax.set_title("the two effects pull against each other", fontsize=13)
    ax.text(12.5, 0.10, "more columns →\nstronger trees (blue ↓)\n"
                        "but more alike (red ↑)", fontsize=11.5, color=INK)

    ax3 = axes[1]
    ax3.plot(rho, forest, "-", color=GREY, lw=1.6, zorder=1)
    ax3.scatter(rho, forest, s=90, color=GREEN, zorder=3)
    for k, r, e in zip(ks, rho, forest):
        ax3.annotate(f"{k}", (r, e), (0, 11), textcoords="offset points",
                     ha="center", fontsize=11, color=INK)
    j = int(np.argmin(forest))
    ax3.scatter([rho[j]], [forest[j]], s=200, facecolor="none", edgecolor=RED,
                lw=2.2, zorder=4)
    ax3.set_xlabel(r"mean pairwise correlation $\rho$  (labels = max_features)")
    ax3.set_ylabel("test error of the 60-tree forest")
    ax3.set_title("the best forest is not the most diverse one", fontsize=13,
                  pad=14)
    ax3.margins(y=0.16)
    ax3.text(0.02, 0.52, "too diverse:\nevery tree is weak", transform=ax3.transAxes,
             ha="left", va="top", fontsize=11.5, color=AMBER)
    ax3.text(0.99, 0.04, "too alike:\nnothing to average",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=11.5,
             color=AMBER)
    save(fig, out, "forest_correlation")


def fig_forest_ntrees(out: Path) -> None:
    """More trees never overfits — it just stops helping."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = _moons(900, 0.32, 7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.55, random_state=0)
    ms = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400]
    curves = []
    for seed in range(6):
        curves.append([1 - RandomForestClassifier(m, random_state=seed)
                       .fit(Xtr, ytr).score(Xte, yte) for m in ms])
    curves = np.array(curves)

    fig, ax = plt.subplots(figsize=HALF)
    for c in curves:
        ax.plot(ms, c, color=GREY, lw=1.0, alpha=0.6)
    ax.plot(ms, curves.mean(0), "o-", color=GREEN, lw=3, ms=6,
            label="mean over 6 seeds")
    ax.set_xscale("log")
    ax.set_xlabel("number of trees (log scale)")
    ax.set_ylabel("test error")
    ax.legend(frameon=False)
    ax.set_title("n_estimators is a budget knob, not a capacity knob:\n"
                 "flat is fine, you just pay for it", fontsize=13)
    save(fig, out, "forest_ntrees")


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
            axes[1, m].set_title("residuals + the tree fitted to them", fontsize=11)
            F = F + eta * h.predict(x[:, None])
            Fg = Fg + eta * h.predict(grid[:, None])
        else:
            axes[1, m].set_title("residuals: nearly flat", fontsize=11)
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("residual")
    save(fig, out, "boosting_stages")


def fig_bagging_vs_boosting(out: Path) -> None:
    """Two ways to spend many trees, and the two errors they attack."""
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    blank(ax)
    ax.text(0.25, 0.95, "FOREST — parallel", ha="center", fontsize=13, color=GREEN)
    ax.text(0.75, 0.95, "BOOSTING — sequential", ha="center", fontsize=13,
            color=AMBER)
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
    box(ax, (0.75, 0.27), 0.20, 0.10, r"weighted sum  $\sum \eta\, h_m$",
        ec=AMBER, fs=11.5)
    ax.text(0.75, 0.06, "each tree corrects the running model\n→ attacks BIAS",
            ha="center", fontsize=11.5, color=INK)
    save(fig, out, "bagging_vs_boosting")


def fig_boosting_overfit(out: Path) -> None:
    """Unlike a forest, adding boosting rounds eventually hurts."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = _moons(700, 0.38, 11)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.55, random_state=0)

    gb = GradientBoostingClassifier(n_estimators=600, learning_rate=0.25,
                                    max_depth=4, random_state=0).fit(Xtr, ytr)
    stages = np.arange(1, 601)
    gb_te = np.array([1 - np.mean((p[:, 1] > 0.5).astype(int) == yte)
                      for p in gb.staged_predict_proba(Xte)])
    gb_tr = np.array([1 - np.mean((p[:, 1] > 0.5).astype(int) == ytr)
                      for p in gb.staged_predict_proba(Xtr)])
    ms = [1, 2, 5, 10, 25, 50, 100, 200, 400, 600]
    rf_te = [1 - RandomForestClassifier(m, random_state=0).fit(Xtr, ytr)
             .score(Xte, yte) for m in ms]

    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(stages, gb_tr, color=AMBER, lw=1.6, ls="--", label="boosting — train")
    ax.plot(stages, gb_te, color=AMBER, lw=2.6, label="boosting — test")
    ax.plot(ms, rf_te, "o-", color=GREEN, lw=2.6, ms=5, label="forest — test")
    k = int(np.argmin(gb_te))
    ax.plot([stages[k]], [gb_te[k]], "o", color=RED, ms=9, zorder=4)
    ax.annotate("early stopping goes here", (stages[k], gb_te[k]),
                (stages[k] + 90, gb_te[k] + 0.05), fontsize=11.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xscale("log")
    ax.set_xlabel("number of trees (log scale)")
    ax.set_ylabel("error rate")
    ax.legend(frameon=False)
    ax.set_title("more trees: harmless in a forest, a real risk in boosting",
                 fontsize=13)
    save(fig, out, "boosting_overfit")


# ================================================== 5. putting them side by side
# ================================================== 5. support vector machines
def _sep_blobs(n=90, seed=3, gap=2.6):
    """Two clearly separable clouds: the slide only works if a gap exists."""
    rng = np.random.default_rng(seed)
    a = rng.normal((-gap / 2, -gap / 2 + 0.3), 0.62, (n // 2, 2))
    b = rng.normal((gap / 2, gap / 2 - 0.3), 0.62, (n // 2, 2))
    X = np.vstack([a, b])
    return X, np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]


def _svm_field(ax, m, X, res=340, pad=0.7, margins=True):
    """Shade by the decision function, and draw the street: f = -1, 0, +1."""
    gx, gy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, res),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, res),
    )
    z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.imshow(1 / (1 + np.exp(-z)), extent=(gx.min(), gx.max(), gy.min(), gy.max()),
              origin="lower", cmap="RdBu_r", vmin=0, vmax=1, alpha=0.7,
              aspect="auto", interpolation="bilinear")
    ax.contour(gx, gy, z, levels=[0], colors=[INK], linewidths=1.8)
    if margins:
        ax.contour(gx, gy, z, levels=[-1, 1], colors=[INK], linewidths=1.1,
                   linestyles="dashed")
    ax.set_aspect("equal")


def _mark_sv(ax, m, s=180, lw=2.2):
    ax.scatter(m.support_vectors_[:, 0], m.support_vectors_[:, 1], s=s,
               facecolors="none", edgecolors=GREEN, linewidths=lw, zorder=4)


def fig_svm_which(out: Path) -> None:
    """Every one of these lines has zero training error. Which one?"""
    X, y = _sep_blobs()
    lo, hi = X.min(0) - 0.7, X.max(0) + 0.7
    grid = np.linspace(lo[0], hi[0], 2)

    fig, ax = plt.subplots(figsize=HALF)
    # all four are genuine separators of this data; only their margin differs
    for a, b in ((-1.506, -0.727), (-1.49, -0.11), (-0.95, 0.10), (-2.6, -0.55)):
        ax.plot(grid, a * grid + b, color=GREY, lw=2.0, ls="--")
    _scatter_classes(ax, X, y, s=34)
    ax.set_title("all four separate the training data perfectly", fontsize=13)
    ax.text(0.5, -0.09, "so the training error cannot choose between them — "
                        "something else has to",
            transform=ax.transAxes, ha="center", fontsize=12, color=RED)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    bare(ax)
    save(fig, out, "svm_which")


def fig_svm_margin(out: Path) -> None:
    """The answer: the line with the widest empty street around it."""
    from sklearn.svm import SVC

    X, y = _sep_blobs()
    m = SVC(kernel="linear", C=1000).fit(X, y)
    best = 2 / np.linalg.norm(m.coef_[0])

    lo = (X.min(0) - 0.7)
    hi = (X.max(0) + 0.7)
    grid = np.linspace(lo[0], hi[0], 2)

    # a valid but badly placed separator; its street is the distance to the
    # closest point on either side, measured rather than drawn by eye
    a, c = -1.506, -0.727
    d = np.abs(a * X[:, 0] - X[:, 1] + c) / np.hypot(a, 1)
    half = d.min()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    ax = axes[0]
    ax.plot(grid, a * grid + c, color=INK, lw=1.8)
    band = half * np.hypot(a, 1)
    ax.fill_between(grid, a * grid + c - band, a * grid + c + band,
                    color=AMBER, alpha=0.30)
    _scatter_classes(ax, X, y, s=34)
    ax.set_title(f"a valid line — street {2 * half:.2f} wide", fontsize=13)
    ax.text(0.5, -0.09, "it squeezes past the blue cloud — one new point flips it",
            transform=ax.transAxes, ha="center", fontsize=12, color=AMBER)

    _svm_field(axes[1], m, X)
    _scatter_classes(axes[1], X, y, s=34)
    _mark_sv(axes[1], m)
    axes[1].set_title(f"the SVM line — street {best:.2f} wide", fontsize=13)
    axes[1].text(0.5, -0.09,
                 "circled = the support vectors: the only points that matter",
                 transform=axes[1].transAxes, ha="center", fontsize=12, color=GREEN)
    for a_ in axes:
        a_.set_xlim(lo[0], hi[0])
        a_.set_ylim(lo[1], hi[1])
        a_.set_aspect("equal")
        bare(a_)
    save(fig, out, "svm_margin")


def fig_eq_svm(out: Path) -> None:
    equation_parts(out, "eq_svm", [
        r"$\min_{w,b}$",
        r"$\frac{1}{2}\|w\|^2$",
        r"$+$",
        r"$C\,\sum_i \max\left(0,\,1-y_i(w^\top x_i+b)\right)$",
    ], gaps=(0.45, 0.45, 0.45), fs=30,
       annots=((1, "make the street\nas WIDE as possible", GREEN),
               (3, "pay C for every point inside it\nor on the wrong side", RED)))


def fig_svm_c(out: Path) -> None:
    """C is the regularization knob: how much are violations allowed to cost?"""
    from sklearn.svm import SVC

    X, y = _sep_blobs(160, seed=5, gap=1.5)
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, C in zip(axes, (0.02, 1.0, 100.0)):
        m = SVC(kernel="linear", C=C).fit(X, y)
        _svm_field(ax, m, X)
        _scatter_classes(ax, X, y, s=16)
        _mark_sv(ax, m, s=52, lw=1.2)
        width = 2 / np.linalg.norm(m.coef_[0])
        ax.set_title(f"C = {C:g}\nstreet {width:.2f} wide, "
                     f"{len(m.support_vectors_)} support vectors", fontsize=12)
        bare(ax)
    axes[0].text(0.5, -0.10, "soft: wide street, many violations tolerated",
                 transform=axes[0].transAxes, ha="center", fontsize=11.5,
                 color=GREEN)
    axes[2].text(0.5, -0.10, "hard: narrow street, fits the training points",
                 transform=axes[2].transAxes, ha="center", fontsize=11.5, color=RED)
    save(fig, out, "svm_c")


def fig_svm_kernel(out: Path) -> None:
    """The trick: a circle in 2D is a flat plane one dimension up."""
    rng = np.random.default_rng(1)
    n = 150
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.r_[rng.normal(0.85, 0.16, n // 2), rng.normal(2.1, 0.20, n // 2)]
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    z = (X ** 2).sum(1)

    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    _scatter_classes(ax, X, y, s=22)
    ax.set_title("no straight line works", fontsize=12.5)
    ax.set_aspect("equal")
    bare(ax)

    ax2 = fig.add_subplot(gs[1], projection="3d")
    for cls, colour, marker in ((0, BLUE, "o"), (1, RED, "^")):
        m_ = y == cls
        ax2.scatter(X[m_, 0], X[m_, 1], z[m_], s=18, color=colour, marker=marker,
                    depthshade=False)
    g1, g2 = np.meshgrid(np.linspace(-2.6, 2.6, 8), np.linspace(-2.6, 2.6, 8))
    ax2.plot_surface(g1, g2, np.full_like(g1, 2.2), color=GREY, alpha=0.35,
                     edgecolor=GREY, linewidth=0.3)
    ax2.set_title("lift: add a third column $x_1^2+x_2^2$", fontsize=12.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.view_init(16, -62)

    ax3 = fig.add_subplot(gs[2])
    from sklearn.svm import SVC
    m = SVC(kernel="rbf", C=10, gamma=0.5).fit(X, y)
    _svm_field(ax3, m, X, margins=False)
    _scatter_classes(ax3, X, y, s=22)
    ax3.set_title("back in 2D: the plane is a circle", fontsize=12.5)
    bare(ax3)
    fig.text(0.5, 0.15, "the kernel computes the lifted dot products without ever "
                        "building the extra columns",
             ha="center", fontsize=12.5, color=INK)
    fig.subplots_adjust(bottom=0.30, top=0.96)
    save(fig, out, "svm_kernel")


def fig_svm_similarity(out: Path) -> None:
    """A kernel is a similarity, and the prediction is a vote of similarities."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

    d = np.linspace(0, 3, 300)
    for g, colour in ((0.2, BLUE), (1.0, GREEN), (5.0, RED)):
        axes[0].plot(d, np.exp(-g * d ** 2), color=colour, lw=2.8,
                     label=rf"$\gamma$ = {g}")
    axes[0].set_xlabel("distance between two points  $\\|x-x'\\|$")
    axes[0].set_ylabel("$K(x,x')$  —  how similar they count as")
    axes[0].set_title(r"RBF:  $K(x,x')=\exp(-\gamma\|x-x'\|^2)$", fontsize=13)
    axes[0].legend(frameon=False)
    axes[0].text(2.05, 0.58, "small $\\gamma$: far points\nstill count as similar",
                 fontsize=11, color=BLUE)
    axes[0].text(0.85, 0.10, "large $\\gamma$: only\nnear-neighbours count",
                 fontsize=11, color=RED)

    ax = axes[1]
    sv = ((-2.0, 1), (-0.8, 1), (0.5, -1), (1.4, -1), (2.4, 1))
    grid = np.linspace(-3.4, 3.6, 500)
    total = np.zeros_like(grid)
    for x0, sign in sv:
        bump = sign * np.exp(-1.6 * (grid - x0) ** 2)
        total += bump
        ax.plot(grid, bump, color=(RED if sign > 0 else BLUE), lw=1.3, alpha=0.55)
        ax.plot([x0], [0], "^" if sign > 0 else "o",
                color=(RED if sign > 0 else BLUE), ms=9, clip_on=False, zorder=4)
    ax.plot(grid, total, color=INK, lw=3)
    ax.axhline(0, color=GREY, lw=1.2, ls="--")
    ax.fill_between(grid, 0, total, where=total > 0, color=RED, alpha=0.10)
    ax.fill_between(grid, 0, total, where=total < 0, color=BLUE, alpha=0.10)
    ax.set_title(r"$f(x)=\sum_i \alpha_i y_i K(x,x_i)+b$   —   "
                 r"one bump per support vector", fontsize=12.5)
    ax.set_xlabel("x")
    ax.set_yticks([])
    ax.text(0.02, 0.04, "predict the sign of the sum", transform=ax.transAxes,
            fontsize=11.5, color=INK)
    save(fig, out, "svm_similarity")


def fig_svm_kernel_zoo(out: Path) -> None:
    """The four kernels a student will actually meet, and when to use them."""
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    blank(ax)
    ax.text(0.13, 0.93, "kernel", ha="center", fontsize=12.5, color=GREY)
    ax.text(0.40, 0.93, r"$K(x,x')$", ha="center", fontsize=12.5, color=GREY)
    ax.text(0.63, 0.93, "knobs", ha="center", fontsize=12.5, color=GREY)
    ax.text(0.85, 0.93, "reach for it when", ha="center", fontsize=12.5, color=GREY)

    rows = (
        ("linear", r"$x^\top x'$", "C only",
         "p is large, n is huge,\ntext / sparse features", BLUE),
        ("polynomial", r"$(\gamma\,x^\top x' + r)^d$", r"C, $\gamma$, r, d",
         "you want explicit\nfeature interactions", PURPLE),
        ("RBF / Gaussian", r"$\exp(-\gamma\|x-x'\|^2)$", r"C, $\gamma$",
         "the default —\nstart here", GREEN),
        ("sigmoid", r"$\tanh(\gamma\,x^\top x' + r)$", r"C, $\gamma$, r",
         "almost never:\nnot always a valid kernel", AMBER),
    )
    for i, (name, formula, knobs, when, colour) in enumerate(rows):
        y = 0.76 - i * 0.185
        ax.plot([0.02, 0.98], [y + 0.10, y + 0.10], color=FAINT, lw=1.2)
        ax.text(0.13, y, name, ha="center", va="center", fontsize=13.5,
                color=colour)
        ax.text(0.40, y, formula, ha="center", va="center", fontsize=16, color=INK)
        ax.text(0.63, y, knobs, ha="center", va="center", fontsize=12.5, color=INK)
        ax.text(0.85, y, when, ha="center", va="center", fontsize=12, color=GREY)
    ax.text(0.5, 0.04, "any symmetric, positive-semidefinite similarity is a legal "
                       "kernel — including ones written for strings, graphs or "
                       "sequences",
            ha="center", fontsize=12.5, color=INK)
    save(fig, out, "svm_kernel_zoo")


def fig_svm_kernels(out: Path) -> None:
    """Four kernels, four shapes of boundary — and what gamma does."""
    from sklearn.svm import SVC

    X, y = _moons(260, 0.22, 0)
    models = (
        ("linear", SVC(kernel="linear", C=1)),
        ("polynomial, degree 3", SVC(kernel="poly", degree=3, C=1, coef0=1)),
        (r"RBF, $\gamma$ = 0.5  (good)", SVC(kernel="rbf", C=1, gamma=0.5)),
        (r"RBF, $\gamma$ = 50  (memorised)", SVC(kernel="rbf", C=1, gamma=50)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.0), sharey=True)
    for ax, (title, m) in zip(axes, models):
        m.fit(X, y)
        _svm_field(ax, m, X, margins=False)
        _scatter_classes(ax, X, y, s=14)
        ax.set_title(title, fontsize=12)
        bare(ax)
    axes[3].text(0.5, -0.08, "big $\\gamma$ = each point defends its own island",
                 transform=axes[3].transAxes, ha="center", fontsize=11, color=RED)
    save(fig, out, "svm_kernels")


def fig_zoo(out: Path) -> None:
    """One dataset, every model of the deck, in the order we met them."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    X, y = _moons(300, 0.22, 0)
    models = (
        ("logistic regression", LogisticRegression()),
        ("logistic + degree-3 terms",
         make_pipeline(PolynomialFeatures(3), StandardScaler(),
                       LogisticRegression(max_iter=2000))),
        ("decision tree", DecisionTreeClassifier(random_state=0)),
        ("random forest, 300 trees", RandomForestClassifier(300, random_state=0)),
        ("gradient boosting", GradientBoostingClassifier(random_state=0)),
        ("SVM, linear kernel", SVC(kernel="linear", C=1)),
        (r"SVM, RBF kernel", SVC(kernel="rbf", C=1, gamma=0.5)),
        ("k-NN, k = 15", KNeighborsClassifier(15)),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6))
    fig.subplots_adjust(hspace=0.22, wspace=0.10)
    for ax, (title, m) in zip(axes.ravel(), models):
        m.fit(X, y)
        if hasattr(m, "decision_function") and isinstance(m, SVC):
            _svm_field(ax, m, X, margins=False)
        else:
            _field(ax, m, X)
        _scatter_classes(ax, X, y, s=11)
        ax.set_title(title, fontsize=11.5)
        bare(ax)
    save(fig, out, "zoo")


def fig_knn(out: Path) -> None:
    """The model with no training at all, for contrast."""
    from sklearn.neighbors import KNeighborsClassifier

    X, y = _moons(260, 0.22, 0)
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, k in zip(axes, (1, 15, 90)):
        m = KNeighborsClassifier(k).fit(X, y)
        _field(ax, m, X)
        _scatter_classes(ax, X, y, s=16)
        ax.set_title(f"k = {k}", fontsize=13)
        bare(ax)
    axes[0].text(0.5, -0.05, "memorises", transform=axes[0].transAxes,
                 ha="center", fontsize=11.5, color=RED)
    axes[2].text(0.5, -0.05, "smooths everything away",
                 transform=axes[2].transAxes, ha="center", fontsize=11.5,
                 color=RED)
    save(fig, out, "knn")


def fig_knn_predict(out: Path) -> None:
    """The whole algorithm: look up the k closest rows, then vote or average."""
    from sklearn.neighbors import KNeighborsRegressor

    X, y = _moons(160, 0.30, 4)
    q = np.array([0.35, 0.28])
    k = 5
    d = np.linalg.norm(X - q, axis=1)
    near = np.argsort(d)[:k]
    votes = np.bincount(y[near], minlength=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax = axes[0]
    for i in near:
        ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color=GREY, lw=1.2, zorder=1)
    ax.add_patch(plt.Circle(q, d[near[-1]], facecolor=GREEN, alpha=0.10,
                            edgecolor=GREEN, lw=2, zorder=1))
    _scatter_classes(ax, X, y, s=70)
    ax.plot([q[0]], [q[1]], "*", color=INK, ms=20, zorder=5)
    ax.text(q[0], q[1] + 0.09, "new patient", fontsize=12, color=INK, ha="center",
            va="bottom")
    # zoom on the neighbourhood: at full extent the circle is a dot on the slide
    r = d[near[-1]] * 2.6
    ax.set_xlim(q[0] - r, q[0] + r)
    ax.set_ylim(q[1] - r, q[1] + r)
    winner = "red" if votes[1] > votes[0] else "blue"
    ax.set_title(f"classification: k = {k} nearest, "
                 f"{votes[1]} red vs {votes[0]} blue → {winner}", fontsize=12.5)
    ax.set_aspect("equal")
    bare(ax)

    rng = np.random.default_rng(0)
    xr = np.sort(rng.uniform(0, 1, 60))
    yr = np.sin(2 * np.pi * xr) + rng.normal(0, 0.22, 60)
    ax2 = axes[1]
    grid = np.linspace(0, 1, 400)[:, None]
    ax2.plot(grid.ravel(), KNeighborsRegressor(5).fit(xr[:, None], yr)
             .predict(grid), color=GREY, lw=2.0, ls="--", label="k-NN, k = 5")
    x0 = 0.42
    near_r = np.argsort(np.abs(xr - x0))[:5]
    ax2.scatter(xr, yr, s=26, color=BLUE, edgecolors="none", zorder=2)
    ax2.scatter(xr[near_r], yr[near_r], s=90, facecolors="none", edgecolors=GREEN,
                linewidths=2.0, zorder=3)
    pred = yr[near_r].mean()
    ax2.plot([x0], [pred], "*", color=RED, ms=22, zorder=5)
    ax2.plot([xr[near_r].min(), xr[near_r].max()], [pred, pred], color=RED, lw=2.4)
    ax2.axvline(x0, color=GREY, lw=1.0, ls=":")
    ax2.set_title("regression: predict the MEAN of the k neighbours", fontsize=12.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(frameon=False, fontsize=11, loc="upper right")

    fig.text(0.5, 0.015, "no training step at all — the training set IS the model",
             ha="center", fontsize=12.5, color=RED)
    fig.subplots_adjust(bottom=0.16)
    save(fig, out, "knn_predict")


def fig_knn_pitfalls(out: Path) -> None:
    """Distances are only as meaningful as the units and the dimension."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 220
    age = rng.normal(58, 9, n)                       # years
    income = rng.normal(38000, 15000, n)             # euros: 1600x the spread
    sick = ((age - 58) / 9 + (income - 38000) / 15000 + rng.normal(0, 0.7, n)) > 0
    X = np.c_[age, income]
    y = sick.astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    panels = (
        ("raw columns: years vs euros", KNeighborsClassifier(15)),
        ("after standardising", make_pipeline(StandardScaler(),
                                              KNeighborsClassifier(15))),
    )
    gx, gy = np.meshgrid(np.linspace(age.min() - 2, age.max() + 2, 280),
                         np.linspace(income.min() - 3000, income.max() + 3000, 280))
    for ax, (title, model) in zip(axes[:2], panels):
        model.fit(X, y)
        z = model.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
        ax.imshow(z, extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin="lower",
                  cmap="RdBu_r", vmin=0, vmax=1, alpha=0.7, aspect="auto",
                  interpolation="bilinear")
        ax.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.4)
        _scatter_classes(ax, X, y, s=16)
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("income (€)  →")
    axes[0].set_xlabel("age (years)  →")
    axes[1].set_xlabel("age (years)  →")
    axes[0].text(0.5, -0.19, "income swamps every distance:\nthe boundary ignores age",
                 transform=axes[0].transAxes, ha="center", fontsize=11.5, color=RED)
    axes[1].text(0.5, -0.19, "both columns count the same",
                 transform=axes[1].transAxes, ha="center", fontsize=11.5,
                 color=GREEN)

    # distance concentration: the nearest neighbour creeps toward the farthest
    ps = np.array([1, 2, 5, 10, 20, 50, 100, 300, 1000])
    ratio = []
    for p_ in ps:
        r = np.random.default_rng(int(p_))
        vals = []
        for _ in range(30):
            A = r.random((500, int(p_)))
            d = np.linalg.norm(A - r.random(int(p_)), axis=1)
            vals.append(d.min() / d.max())
        ratio.append(np.mean(vals))
    axes[2].plot(ps, ratio, "o-", color=PURPLE, lw=2.8, ms=7)
    axes[2].axhline(1.0, color=GREY, ls="--", lw=1.2)
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_xlabel("number of columns p  (log scale)")
    axes[2].set_ylabel("nearest distance / farthest distance")
    axes[2].set_title("and distance itself stops meaning much", fontsize=12.5)
    axes[2].text(0.03, 0.99, "at 1.0 the closest point is\nas far as the farthest —\n"
                             "every neighbour is a stranger",
                 transform=axes[2].transAxes, ha="left", va="top", fontsize=11,
                 color=PURPLE)
    fig.subplots_adjust(bottom=0.28, wspace=0.30)
    save(fig, out, "knn_pitfalls")

FIGURES = (
    # part 1: feature engineering and missing data
    fig_cyclical, fig_elapsed, fig_windows, fig_window_types, fig_lags,
    fig_log_transform, fig_onehot, fig_standardize, fig_binning, fig_polynomial,
    fig_interaction, fig_group_relative, fig_embedding,
    fig_missing_map, fig_mechanisms, fig_mean_impute, fig_mice,
    fig_eq_rubin,
    # part 2: how the models work
    fig_map, fig_supervised, fig_tasks, fig_linreg_fit, fig_linreg_which,
    fig_eq_mse, fig_linreg_loss, fig_linreg_gd, fig_linreg_multi,
    fig_linreg_basis, fig_logistic, fig_eq_logistic, fig_linreg_limits,
    fig_tree_questions, fig_tree_split, fig_impurity, fig_tree_grow,
    fig_tree_regression, fig_tree_overfit, fig_tree_instability,
    fig_forest_idea, fig_eq_forest_variance, fig_forest_variance,
    fig_forest_diagram, fig_forest_boundary, fig_max_features,
    fig_forest_features, fig_forest_correlation, fig_forest_ntrees,
    fig_boosting_stages, fig_bagging_vs_boosting, fig_boosting_overfit,
    fig_svm_which, fig_svm_margin, fig_eq_svm, fig_svm_c, fig_svm_kernel,
    fig_svm_similarity, fig_svm_kernel_zoo, fig_svm_kernels, fig_zoo,
    fig_knn, fig_knn_predict, fig_knn_pitfalls,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course02/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
