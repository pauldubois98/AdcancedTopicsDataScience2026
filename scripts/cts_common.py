"""Shared plumbing for the CourseTS figure scripts.

Palette, slide geometry, the save/annotate helpers and the time-series estimators
used by both halves of the deck. Imported by make_figures_cTS1.py (structure) and
make_figures_cTS2.py (forecasting and learning); it renders nothing itself.
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

INK = "#1b1b1b"
GREY = "#9aa0a6"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#00ab0e"
AMBER = "#b7791f"
PURPLE = "#6b46c1"
FAINT = "#e8eaed"

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


def eq(ax, y, text, fs=17, x=0.5, color=INK, ha="center"):
    """One display equation on a blank axis."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)


def items(ax, pairs, top=0.70, step=0.15, x=0.06, fs=11.5, color=INK):
    """A stacked list of (label, gloss): label on one line, gloss indented under it."""
    for i, (a, b) in enumerate(pairs):
        yy = top - i * step
        ax.text(x, yy, a, fontsize=fs, color=color, va="center")
        if b:
            ax.text(x + 0.02, yy - step * 0.38, b, fontsize=fs - 1.0, color=GREY,
                    va="center")


def tidy(ax, xlabel="time", ylabel="value"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ------------------------------------------------------------------ time series
M = 12          # the seasonal period used everywhere: 12 "months" per cycle
N = 144         # 12 cycles of it


def _trend(t, slope=0.35, curve=0.0):
    return 20.0 + slope * t + curve * (t / 100.0) ** 2


def _season(t, period=M, amp=6.0, shape=(1.0, 0.45, 0.2)):
    """A seasonal shape built from a few harmonics, so it is not a pure sine."""
    s = np.zeros_like(t, dtype=float)
    for k, a in enumerate(shape, start=1):
        s += a * np.sin(2 * np.pi * k * t / period + 0.7 * k)
    return amp * s / np.abs(shape).sum()


def ar1(n, phi, sigma=1.0, seed=0, x0=0.0):
    """One realisation of x_t = phi x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = x0 + e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def arp(n, phis, sigma=1.0, seed=0, burn=200):
    """AR(p) with coefficients `phis` = (phi_1, ..., phi_p)."""
    rng = np.random.default_rng(seed)
    p = len(phis)
    e = rng.normal(0, sigma, n + burn)
    x = np.zeros(n + burn)
    for i in range(p, n + burn):
        x[i] = np.dot(phis, x[i - p:i][::-1]) + e[i]
    return x[burn:]


def maq(n, thetas, sigma=1.0, seed=0):
    """MA(q): x_t = eps_t + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}."""
    rng = np.random.default_rng(seed)
    q = len(thetas)
    e = rng.normal(0, sigma, n + q)
    w = np.concatenate(([1.0], np.asarray(thetas, float)))
    return np.array([np.dot(w, e[i + q::-1][:q + 1]) for i in range(n)])


def series(n=N, slope=0.35, amp=6.0, sigma=1.6, seed=5, curve=0.0, phi=0.0):
    """The reference series: trend + seasonality + (optionally correlated) noise."""
    t = np.arange(n, dtype=float)
    noise = ar1(n, phi, sigma, seed) if phi else np.random.default_rng(seed).normal(0, sigma, n)
    return t, _trend(t, slope, curve) + _season(t, amp=amp) + noise


def random_walk(n, sigma=1.0, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(drift, sigma, n))


# ------------------------------------------------------------------ estimators
def acf(x, nlags=36):
    """Sample autocorrelation, the standard biased (1/n) estimator."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    return np.array([1.0] + [np.dot(x[k:], x[:-k]) / n / c0 for k in range(1, nlags + 1)])


def corr_stem(ax, vals, n, title="", color=BLUE, band=True, start=1):
    """A correlogram: stems, zero line, and the +-2/sqrt(n) white-noise band."""
    lags = np.arange(len(vals))
    ax.axhline(0, color=GREY, lw=1.0)
    if band:
        b = 1.96 / np.sqrt(n)
        ax.fill_between([start - 0.5, len(vals) - 0.5], -b, b, color=BLUE, alpha=0.12, lw=0)
    ax.vlines(lags[start:], 0, vals[start:], color=color, lw=2.4)
    ax.plot(lags[start:], vals[start:], "o", color=color, ms=3.5)
    ax.set_xlim(start - 0.5, len(vals) - 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("lag")
    if title:
        ax.set_title(title, fontsize=12)


def fit_ar(x, p):
    """Least-squares AR(p) fit; returns (intercept, phis, residuals)."""
    x = np.asarray(x, float)
    X = np.column_stack([x[p - j - 1:len(x) - j - 1] for j in range(p)])
    X = np.column_stack([np.ones(len(X)), X])
    y = x[p:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0], beta[1:], y - X @ beta

def ljung_box(x, mlags=10, k=0):
    """Ljung-Box Q, its degrees of freedom and p-value."""
    from scipy.stats import chi2
    x = np.asarray(x, float)
    n = len(x)
    r = acf(x, mlags)
    h = np.arange(1, mlags + 1)
    q = n * (n + 2) * np.sum(r[1:] ** 2 / (n - h))
    df = max(mlags - k, 1)
    return q, df, float(chi2.sf(q, df))
