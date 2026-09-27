"""Render the illustration figures for CourseTS/slides1.md (structure).

Every figure is synthetic and seeded, so `make figures` reproduces the same output
anywhere without touching `data/`. Shared palette, helpers and estimators live in
cts_common.py.

Usage: make_figures_cTS1.py [outdir]      (default: CourseTS/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cts_common import (
    AMBER, BLUE, FAINT, GREEN, GREY, INK, M, N, PURPLE,
    RED, WIDE, _season, _trend, acf, ar1, arp, arrow, blank, box, corr_stem,
    eq, fit_ar, items, maq, random_walk, save, series, tidy,
)


def pacf(x, nlags=24):
    """Partial autocorrelation via the Durbin-Levinson recursion."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = [1.0]
    phi[1, 1] = r[1]
    out.append(r[1])
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out.append(phi[k, k])
    return np.array(out)


def moving_average(x, w):
    """Centred moving average; even windows get the 2xw form, as in classical decomposition."""
    x = np.asarray(x, float)
    if w % 2 == 0:
        k = np.ones(w + 1) / w
        k[0] = k[-1] = 1 / (2 * w)
    else:
        k = np.ones(w) / w
    out = np.full(len(x), np.nan)
    h = len(k) // 2
    for i in range(h, len(x) - h):
        out[i] = np.dot(k, x[i - h:i + h + 1])
    return out


def classical_decompose(y, period=M):
    """Trend by centred MA, seasonal by averaging detrended values per phase."""
    trend = moving_average(y, period)
    detr = y - trend
    seas = np.zeros(period)
    for ph in range(period):
        vals = detr[ph::period]
        seas[ph] = np.nanmean(vals)
    seas -= seas.mean()
    season = seas[np.arange(len(y)) % period]
    return trend, season, y - trend - season


def _df_null(n=200, reps=4000, seed=0):
    """Simulate the Dickey-Fuller null: t-ratios from random walks, constant included."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        y = np.cumsum(rng.normal(0, 1, n + 1))
        dy = np.diff(y)
        X = np.column_stack([np.ones(n), y[:-1]])
        beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
        resid = dy - X @ beta
        s2 = resid @ resid / (n - 2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        out[r] = beta[1] / se
    return out


def adf_stat(y, lags=None, regression="c"):
    """Augmented Dickey-Fuller t-ratio, computed by hand.

    Fits  dy_t = a + b t + g y_{t-1} + sum_j d_j dy_{t-j} + e  and returns
    (t-ratio on g, lags used). `regression` is "n", "c" or "ct".
    """
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy)
    if lags is None:                       # Schwert's rule, as used by the packages
        lags = int(np.ceil(12 * (len(y) / 100) ** 0.25))
        lags = max(0, min(lags, n // 3))
    Y = dy[lags:]
    cols = [y[lags:n]]                                     # the lagged level
    for j in range(1, lags + 1):
        cols.append(dy[lags - j:n - j])
    if regression in ("c", "ct"):
        cols.append(np.ones(len(Y)))
    if regression == "ct":
        cols.append(np.arange(len(Y), dtype=float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(Y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(s2 * np.linalg.pinv(X.T @ X)[0, 0])
    return beta[0] / se, lags


def kpss_stat(y, regression="c", lags=None):
    """KPSS LM statistic: partial sums of the residuals over a long-run variance."""
    y = np.asarray(y, float)
    n = len(y)
    tt = np.arange(n, dtype=float)
    X = np.ones((n, 1)) if regression == "c" else np.column_stack([np.ones(n), tt])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    S = np.cumsum(e)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** 0.25))
    s2 = e @ e / n                                  # Bartlett-kernel long-run variance
    for j in range(1, lags + 1):
        s2 += 2 * (1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    return (S ** 2).sum() / (n ** 2 * s2), lags


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


def _tricube(u):
    return (1 - np.abs(np.clip(u, -1, 1)) ** 3) ** 3


def _loess_at(t, y, t0, span, rw=None):
    """One local weighted linear fit, evaluated at t0. Returns (value, weights, a, b).

    `rw` are optional external weights (STL's robustness weights) that multiply the
    tricube neighbourhood weights.
    """
    d = np.abs(t - t0)
    k = max(2, int(np.ceil(span * len(t))))
    bw = np.sort(d)[min(k - 1, len(d) - 1)]
    bw = max(bw, 1e-9)
    w = _tricube(d / bw)
    if rw is not None:
        w = w * rw
    if w.sum() <= 0:
        return float(np.mean(y)), w, float(np.mean(y)), 0.0
    X = np.column_stack([np.ones_like(t), t - t0])
    W = np.diag(w)
    try:
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float(np.average(y, weights=w)), w, 0.0, 0.0
    return beta[0], w, beta[0], beta[1]


def _loess(t, y, span=0.3, rw=None, at=None):
    """Loess evaluated at `at` (default: at the data points)."""
    pts = t if at is None else np.asarray(at, float)
    return np.array([_loess_at(t, y, t0, span, rw)[0] for t0 in pts])


def _ma(x, w):
    """Simple trailing-free moving average of order w, shortening the array."""
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def stl_decompose(y, period, ns=0.75, nt=0.45, nl=0.45, ni=2, no=0, want=None):
    """A faithful-enough STL. Returns (T, S, R) and, if `want` is a dict, fills it
    with the intermediates of the FIRST inner pass so the slides can show them."""
    n = len(y)
    t = np.arange(n, dtype=float)
    T = np.zeros(n)                      # <- the initialisation the slide asks about
    S = np.zeros(n)
    rw = np.ones(n)
    first = True
    for _ in range(no + 1):
        for _ in range(ni):
            d = y - T                                            # 1. detrend
            # 2. smooth each cycle-subseries, extended one cycle each side
            C = np.empty(n + 2 * period)
            for ph in range(period):
                idx = np.arange(ph, n, period)
                ts = idx.astype(float)
                at = np.concatenate(([ph - period], ts, [ts[-1] + period]))
                sm = _loess(ts, d[idx], ns, rw[idx], at=at)
                pos = np.arange(len(at)) * period + (ph - period) + period
                C[pos] = sm
            # 3. low-pass the extended cycle-subseries and remove it
            lp = _ma(_ma(_ma(C, period), period), 3)
            L = _loess(t, lp, nl)
            S_new = C[period:period + n] - L                      # 4. the season
            ds = y - S_new                                        # 5. deseasonalise
            T_new = _loess(t, ds, nt, rw)                         # 6. re-smooth trend
            if first and want is not None:
                want.update(d=d.copy(), C=C.copy(), L=L.copy(), S=S_new.copy(),
                            ds=ds.copy(), T=T_new.copy())
                first = False
            S, T = S_new, T_new
        R = y - T - S
        h = 6 * np.median(np.abs(R - np.median(R)))
        rw = _tricube(np.abs(R) / max(h, 1e-9)) if h > 0 else np.ones(n)
        rw = np.clip((1 - np.clip(np.abs(R) / max(h, 1e-9), 0, 1) ** 2) ** 2, 0, 1)
    return T, S, y - T - S


def _df_regress(y, const=True):
    """Plain Dickey-Fuller regression of the change on the level."""
    dy = np.diff(y)
    lev = y[:-1]
    X = np.column_stack([np.ones(len(lev)), lev]) if const else lev[:, None]
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    s2 = resid @ resid / (len(dy) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.pinv(X.T @ X)[-1, -1])
    return beta[-1], beta[-1] / se, resid, lev, dy


def _stl_demo_series():
    """Growing seasonal amplitude, a clear trend, and one planted outlier."""
    t = np.arange(N, dtype=float)
    rng = np.random.default_rng(31)
    grow = np.linspace(0.45, 1.6, N)
    y = _trend(t, 0.3) + grow * _season(t, amp=6) + rng.normal(0, 1.0, N)
    y[95] += 16.0
    return t, y


def _test_cases():
    """Five series whose true status we know, for the two recipe slides."""
    rng = np.random.default_rng(0)
    n = 300
    return [
        ("white noise", rng.normal(0, 1, n), "stationary", GREEN),
        (r"AR(1), $\phi = 0.6$", ar1(n, 0.6, seed=1), "stationary", GREEN),
        (r"AR(1), $\phi = 0.95$", ar1(n, 0.95, seed=2), "stationary", GREEN),
        ("random walk", random_walk(n, seed=3), "NOT stationary", RED),
        ("trend + noise", 0.05 * np.arange(n) + rng.normal(0, 1, n),
         "NOT stationary", RED),
    ]


# ================================================================== 0. opening
def fig_ts_zoo(out: Path) -> None:
    """Four series that all look like "data over time" and need different tools."""
    fig, axes = plt.subplots(1, 4, figsize=WIDE)
    t = np.arange(180, dtype=float)
    rng = np.random.default_rng(3)

    panels = [
        ("trend", _trend(t, 0.22) + rng.normal(0, 1.6, len(t)), BLUE,
         "the level moves"),
        ("seasonality", 25 + _season(t, amp=7) + rng.normal(0, 1.0, len(t)), GREEN,
         "fixed, known period"),
        ("cycle", 25 + 7 * np.sin(2 * np.pi * t / 63 + 0.4)
         + 3 * np.sin(2 * np.pi * t / 41) + rng.normal(0, 0.9, len(t)), AMBER,
         "wanders, no fixed period"),
        ("shock + noise", 25 + np.where(t > 110, -8.0, 0.0)
         + rng.normal(0, 1.6, len(t)), RED, "one-off level change"),
    ]
    for ax, (name, y, col, sub) in zip(axes, panels):
        ax.plot(t, y, color=col, lw=1.6)
        ax.set_title(f"{name}\n{sub}", fontsize=12)
        ax.set_xlabel("t")
        ax.set_yticks([])
    axes[0].set_ylabel("value")
    fig.tight_layout()
    save(fig, out, "ts_zoo")


def fig_iid_broken(out: Path) -> None:
    """Shuffling rows destroys everything, which is the whole point."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    t, y = series(120, seed=11)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))

    ax = axes[0]
    ax.plot(t, y, color=BLUE, lw=1.7)
    ax.set_title("the series", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    ax.plot(t, y[perm], color=RED, lw=1.0, alpha=0.9)
    ax.set_title("the same rows, shuffled", fontsize=12.5)
    tidy(ax, "row order")

    ax = axes[2]
    blank(ax)
    rows = [
        ("mean", GREEN, "unchanged"),
        ("variance", GREEN, "unchanged"),
        ("histogram", GREEN, "unchanged"),
        ("trend", RED, "gone"),
        ("seasonality", RED, "gone"),
        ("autocorrelation", RED, "gone"),
    ]
    for i, (name, col, verdict) in enumerate(rows):
        yy = 0.76 - i * 0.125
        ax.text(0.06, yy, name, fontsize=11.5, color=INK, va="center")
        ax.text(0.94, yy, verdict, fontsize=11.5, color=col, va="center",
                ha="right", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "iid_broken")


# ============================================================ 1. decomposition
def fig_components(out: Path) -> None:
    """y = T + S + R, stacked, on the reference series."""
    t, y = series()
    trend = _trend(t)
    seas = _season(t)
    resid = y - trend - seas

    fig, axes = plt.subplots(4, 1, figsize=(12.6, 6.4), sharex=True)
    for ax, (v, col, name) in zip(axes, [
        (y, INK, "observed  $y_t$"),
        (trend, BLUE, "trend  $T_t$"),
        (seas, GREEN, "seasonality  $S_t$"),
        (resid, RED, "remainder  $R_t$"),
    ]):
        ax.plot(t, v, color=col, lw=1.6)
        ax.set_ylabel(name, fontsize=11)
        ax.set_yticks([])
    axes[-1].axhline(0, color=GREY, lw=0.9)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    save(fig, out, "components")


def fig_eq_decomposition(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    eq(ax, 0.86, r"$y_t \;=\; T_t \;+\; S_t \;+\; R_t$", fs=26, color=BLUE)
    ax.text(0.5, 0.73, "additive:  the seasonal swing is the same size every year",
            ha="center", fontsize=12.5, color=GREY)
    eq(ax, 0.52, r"$y_t \;=\; T_t \times S_t \times R_t$", fs=26, color=PURPLE)
    ax.text(0.5, 0.39, "multiplicative:  the swing is a fixed percentage of the level",
            ha="center", fontsize=12.5, color=GREY)
    eq(ax, 0.20, r"$\log y_t \;=\; \log T_t + \log S_t + \log R_t$", fs=21, color=GREEN)
    ax.text(0.5, 0.06,
            "take logs and multiplicative becomes additive",
            ha="center", fontsize=12, color=INK)
    save(fig, out, "eq_decomposition")


def fig_add_vs_mult(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    t = np.arange(N, dtype=float)
    rng = np.random.default_rng(7)
    add = _trend(t, 0.45) + _season(t, amp=6) + rng.normal(0, 1.2, N)
    # the multiplicative twin needs an EXPONENTIAL trend: log of a linear trend is
    # concave, so a linear one would leave the right panel visibly bent
    growth = 20.0 * np.exp(0.0105 * t)
    mult = growth * (1 + _season(t, amp=6) / 22) * np.exp(rng.normal(0, 0.035, N))

    axes[0].plot(t, add, color=BLUE, lw=1.6)
    axes[0].set_title("additive\nconstant swing", fontsize=12.5)
    axes[1].plot(t, mult, color=PURPLE, lw=1.6)
    axes[1].set_title("multiplicative\nswing grows with the level", fontsize=12.5)
    axes[2].plot(t, np.log(mult), color=GREEN, lw=1.6)
    axes[2].plot(t, np.log(growth), color=GREY, lw=1.4, ls="--")
    axes[2].set_title("log of the middle panel", fontsize=12.5)
    for ax in axes:
        tidy(ax, "t")
    axes[2].set_ylabel("log value")
    fig.tight_layout()
    save(fig, out, "add_vs_mult")


def fig_moving_average(out: Path) -> None:
    """The window length is the whole decision."""
    t, y = series(seed=21)
    fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True)
    for ax, w, col, note in [
        (axes[0], 3, AMBER, "too short\nseasonality survives"),
        (axes[1], M, BLUE, "w = period = 12\nseasonality averages out"),
        (axes[2], 37, RED, "too long\nthe trend is flattened\nand cropped too much"),
    ]:
        ax.plot(t, y, color=FAINT, lw=1.4, zorder=1)
        ax.plot(t, moving_average(y, w), color=col, lw=2.6, zorder=3)
        ax.set_title(f"{note}", fontsize=12)
        ax.set_xlabel("t")
    axes[0].set_ylabel("value")
    fig.suptitle("centred moving average", fontsize=13)
    fig.tight_layout()
    save(fig, out, "moving_average")


def fig_classical_decomp(out: Path) -> None:
    """The four steps with their formulas written out, and step 3 worked through."""
    t, y = series(seed=21)
    trend = moving_average(y, M)
    detr = y - trend

    phase = 3
    idx = [i for i in range(phase, len(y), M) if not np.isnan(detr[i])]
    vals = detr[idx]
    raw = vals.mean()
    allmeans = np.array([np.nanmean(detr[j::M]) for j in range(M)])
    centre = allmeans.mean()

    fig = plt.figure(figsize=WIDE)
    ax = fig.add_subplot(1, 1, 1)
    blank(ax)
    steps = [
        ("1.", "trend: centred moving average", BLUE,
         r"$\hat T_t = \frac{1}{m}\left(\frac{1}{2}y_{t-m/2} + y_{t-m/2+1}"
         r" + \cdots + \frac{1}{2}y_{t+m/2}\right)$"),
        ("2.", "detrend", BLUE, r"$d_t = y_t - \hat T_t$"),
        ("3.", "average the detrended values of each phase", GREEN,
         r"$\bar s_j = \frac{1}{n_j}\!\!\sum_{t \,\equiv\, j \;(\mathrm{mod}\; m)}"
         r"\!\! d_t$" + "        "
         r"$\hat S_t = \bar s_{(t\,\mathrm{mod}\,m)}"
         r" - \frac{1}{m}\sum_{k=0}^{m-1}\bar s_k$"),
        ("4.", "remainder", RED, r"$\hat R_t = y_t - \hat T_t - \hat S_t$"),
    ]
    # step 3 carries a two-level subscript, so the rows are not evenly spaced
    for (num, title, col, formula), yy in zip(steps, (0.84, 0.60, 0.36, 0.10)):
        ax.text(0.09, yy + 0.060, num, fontsize=21, color=col, fontweight="bold",
                va="center")
        ax.text(0.15, yy + 0.063, title, fontsize=17, color=INK, va="center")
        ax.text(0.15, yy - 0.072, formula, fontsize=18, color=col, va="center")
    fig.tight_layout()
    save(fig, out, "classical_decomp")


def fig_stl(out: Path) -> None:
    """STL as a recipe, in the same format as the classical decomposition."""
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1])

    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(0.5, 0.965, "STL inner loop — repeat $n_i$ times", ha="center",
            fontsize=13, color=INK)
    steps = [
        ("1.", "detrend with the current trend", BLUE,
         r"$d_t = y_t - \hat T_t$" + "        "
         + r"(the loop starts from $\hat T_t = 0$)"),
        ("2.", "smooth each cycle-subseries", GREEN,
         r"loess of span $n_s$ on all the $d_t$ sharing a phase $\;\rightarrow\; C_t$"),
        ("3.", "low-pass filter that", AMBER,
         r"MA$_m \rightarrow$ MA$_m \rightarrow$ MA$_3 \rightarrow$ loess$(n_l)$"
         r"$\;\rightarrow\; L_t$"),
        ("4.", "the season is what survives it", GREEN,
         r"$\hat S_t = C_t - L_t$"),
        ("5.", "deseasonalise", BLUE,
         r"$y_t - \hat S_t$"),
        ("6.", "re-smooth the trend", BLUE,
         r"$\hat T_t = $ loess of span $n_t$ on $\;y_t - \hat S_t$"),
    ]
    for i, (num, title, col, formula) in enumerate(steps):
        yy = 0.87 - i * 0.143
        ax.text(0.03, yy, num, fontsize=13, color=col, fontweight="bold", va="center")
        ax.text(0.09, yy + 0.028, title, fontsize=11.5, color=INK, va="center")
        ax.text(0.09, yy - 0.040, formula, fontsize=11.5, color=col, va="center")
    ax.text(0.5, 0.015, r"step 3 removes any trend that leaked into $C_t$, "
                        r"so that $\hat S_t$ averages to zero",
            ha="center", fontsize=10.5, color=GREY)

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(0.5, 0.965, "outer loop — repeat $n_o$ times", ha="center", fontsize=13,
            color=RED)
    ax.text(0.06, 0.865, "residual:", fontsize=11.5, color=GREY, va="center")
    ax.text(0.40, 0.865, r"$\hat R_t = y_t - \hat T_t - \hat S_t$", fontsize=12.5,
            color=INK, va="center")
    ax.text(0.06, 0.755, "weight:", fontsize=11.5, color=GREY, va="center")
    ax.text(0.40, 0.755, r"$\rho_t = B\!\left(\frac{|\hat R_t|}"
                         r"{6\,\mathrm{median}|\hat R|}\right)$",
            fontsize=13, color=RED, va="center")
    ax.text(0.06, 0.635, "bisquare:", fontsize=11.5, color=GREY, va="center")
    ax.text(0.40, 0.635, r"$B(u) = (1 - u^2)^2$ for $u < 1$, else $0$",
            fontsize=11.5, color=RED, va="center")
    ax.text(0.5, 0.545, r"then run the inner loop again with every loess weighted by $\rho_t$",
            ha="center", fontsize=11, color=GREY)

    ax.plot([0.04, 0.96], [0.475, 0.475], color=FAINT, lw=1.5)
    ax.text(0.5, 0.415, "the knobs", ha="center", fontsize=12.5, color=INK)
    knobs = [
        (r"$m$", "the period"),
        (r"$n_s$", "seasonal window: how fast the shape may change"),
        (r"$n_t$", "trend window: how wiggly the trend may be"),
        (r"$n_l$", "low-pass window, usually left at $m$"),
        (r"$n_i, n_o$", "inner / outer iterations, 2 and 0 if no outliers"),
    ]
    for i, (a, b) in enumerate(knobs):
        yy = 0.335 - i * 0.068
        ax.text(0.10, yy, a, fontsize=12, color=BLUE, va="center", ha="right")
        ax.text(0.15, yy, b, fontsize=10.5, color=INK, va="center")
    fig.tight_layout()
    save(fig, out, "stl")


def fig_eq_stationarity(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.92, "strict stationarity", ha="center", fontsize=14, color=PURPLE)
    eq(ax, 0.72,
       r"$(y_{t_1},\dots,y_{t_k}) \;\overset{d}{=}\; (y_{t_1+h},\dots,y_{t_k+h})$",
       fs=17)
    ax.text(0.5, 0.56, "the whole joint distribution is unchanged by a time shift",
            ha="center", fontsize=11.5, color=GREY)
    
    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.92, "weak (second-order) stationarity", ha="center", fontsize=14,
            color=GREEN)
    eq(ax, 0.74, r"$\mathbb{E}[y_t] = \mu$" + "      (constant)", fs=16)
    eq(ax, 0.58, r"$\mathrm{Var}[y_t] = \sigma^2 < \infty$" + "   (constant)", fs=16)
    eq(ax, 0.42, r"$\mathrm{Cov}[y_t, y_{t+h}] = \gamma(h)$", fs=16)
    ax.text(0.5, 0.30, "depends on the gap $h$", ha="center", fontsize=11.5, color=GREY)
    fig.tight_layout()
    save(fig, out, "eq_stationarity")


def fig_stationary_zoo(out: Path) -> None:
    """Ask the room before revealing: which of these six are stationary?"""
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.6))
    n = 260
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(4)
    panels = [
        ("white noise", rng.normal(0, 1, n), True, "yes"),
        # an AR(2) with complex roots: genuinely stationary, genuinely no fixed
        # period. Labelled "cycle" because AR is only introduced later in the deck.
        ("cycle", arp(n, (1.8, -0.88), seed=2), True, "yes"),
        ("random walk", random_walk(n, seed=3), False, "no — variance grows"),
        ("linear trend", 0.03 * t + rng.normal(0, 1, n), False, "no — mean moves"),
        ("seasonal", 2.4 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.6, n),
         False, "no — mean moves"),
        ("variance shift", rng.normal(0, 1, n) * np.where(t < 130, 0.5, 2.2),
         False, "no — variance moves"),
    ]
    for ax, (name, y, ok, verdict) in zip(axes.ravel(), panels):
        ax.plot(t, y, color=GREEN if ok else RED, lw=1.2)
        ax.set_title(name, fontsize=11.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.98, 0.04, verdict, transform=ax.transAxes, ha="right",
                fontsize=10.5, color=GREEN if ok else RED)
    fig.tight_layout()
    save(fig, out, "stationary_zoo")


def fig_why_stationarity(out: Path) -> None:
    """The estimation argument: one realisation, and time has to stand in for repeats."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    n = 200
    t = np.arange(n, dtype=float)

    ax = axes[0]
    for s in range(6):
        ax.plot(t, ar1(n, 0.7, seed=40 + s), color=BLUE, lw=0.9, alpha=0.45)
    ax.set_title("many realisations at each $t$", fontsize=12)
    tidy(ax, "t")

    ax = axes[1]
    ax.plot(t, ar1(n, 0.7, seed=40), color=BLUE, lw=1.7)
    ax.set_title("only one available", fontsize=12)
    tidy(ax, "t")

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.88, "Averaging over time", ha="center",
            fontsize=13, color=INK)
    ax.text(0.5, 0.70, r"$\hat\mu = \frac{1}{n}\sum_t y_t$", ha="center", fontsize=19,
            color=BLUE)
    ax.text(0.5, 0.53,
            "is legitimate only if every $y_t$ has the same mean",
            ha="center", fontsize=12, color=INK)
    ax.text(0.5, 0.33,
            "i.e., is stationary",
            ha="center", fontsize=13.5, color=GREEN)
    fig.tight_layout()
    save(fig, out, "why_stationarity")


def fig_random_walk(out: Path) -> None:
    """The variance fan: the one picture that makes a unit root visible."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    n = 250
    t = np.arange(n, dtype=float)

    ax = axes[0]
    for s in range(40):
        ax.plot(t, ar1(n, 0.7, seed=100 + s), color=BLUE, lw=0.6, alpha=0.30)
    ax.plot(t, 2 * np.sqrt(1 / (1 - 0.7 ** 2)) * np.ones(n), color=INK, lw=1.6, ls="--")
    ax.plot(t, -2 * np.sqrt(1 / (1 - 0.7 ** 2)) * np.ones(n), color=INK, lw=1.6, ls="--")
    ax.set_ylim(-16, 16)
    ax.set_title(r"AR(1), $\phi = 0.7$" + "\nspread settles down", fontsize=12)
    tidy(ax, "t")

    ax = axes[1]
    for s in range(40):
        ax.plot(t, random_walk(n, seed=100 + s), color=RED, lw=0.6, alpha=0.30)
    ax.plot(t, 2 * np.sqrt(t), color=INK, lw=1.6, ls="--")
    ax.plot(t, -2 * np.sqrt(t), color=INK, lw=1.6, ls="--", label=r"$\pm 2\sqrt{t}$")
    ax.set_ylim(-16, 16)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.set_title(r"random walk, $\phi = 1$" + "\nspread never stops growing", fontsize=12)
    tidy(ax, "t")

    ax = axes[2]
    blank(ax)
    eq(ax, 0.88, r"$y_t = \phi\, y_{t-1} + \varepsilon_t$", fs=19, color=INK)
    eq(ax, 0.66, r"$\mathrm{Var}[y_t] = \dfrac{\sigma^2}{1-\phi^2}$", fs=18, color=BLUE)
    ax.text(0.5, 0.52, r"finite only while $|\phi| < 1$", ha="center", fontsize=12,
            color=GREY)
    eq(ax, 0.30, r"$\phi = 1 \;\Rightarrow\; y_t = \sum_{s \leq t}\varepsilon_s$", fs=18,
           color=RED)
    eq(ax, 0.15, r"$\mathrm{Var}[y_t] = t\,\sigma^2$", fs=18, color=RED)
    fig.tight_layout()
    save(fig, out, "random_walk")


def fig_unit_root(out: Path) -> None:
    """phi crossing 1 is a cliff, not a slope."""
    fig, axes = plt.subplots(1, 4, figsize=WIDE, sharex=True)
    n = 200
    t = np.arange(n, dtype=float)
    for ax, (phi, col, note) in zip(axes, [
        (0.25, GREEN, "forgets fast"),
        (0.75, AMBER, "forgets slowly"),
        (1.0, RED, "random walk"),
        (1.25, PURPLE, "explosive"),
    ]):
        ax.plot(t, ar1(n, phi, seed=9), color=col, lw=1.5)
        ax.set_title(rf"$\phi = {phi}$" + f"\n{note}", fontsize=12)
        ax.set_xlabel("t")
        # the y scale IS the message: the panels differ by three orders of magnitude
        ax.tick_params(axis="y", labelsize=9)
    axes[0].set_ylabel("value")
    fig.tight_layout()
    save(fig, out, "unit_root")


def fig_spurious_regression(out: Path) -> None:
    """Two independent random walks, and a beautiful meaningless regression."""
    n = 220
    a = random_walk(n, seed=17, drift=0.05)
    b = random_walk(n, seed=93, drift=-0.04)
    r = np.corrcoef(a, b)[0, 1]
    da, db = np.diff(a), np.diff(b)
    rd = np.corrcoef(da, db)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    ax = axes[0]
    ax.plot(a, color=BLUE, lw=1.7, label="series A")
    ax.plot(b, color=RED, lw=1.7, label="series B")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("two random walks,\ngenerated independently", fontsize=12)
    tidy(ax, "t")

    ax = axes[1]
    ax.scatter(a, b, s=12, color=PURPLE, alpha=0.7)
    ax.set_title(f"levels:  r = {r:.2f}", fontsize=12.5)
    ax.set_xlabel("A")
    ax.set_ylabel("B")

    ax = axes[2]
    ax.scatter(da, db, s=12, color=GREEN, alpha=0.7)
    ax.set_title(f"differences:  r = {rd:.2f}", fontsize=12.5)
    ax.set_xlabel(r"$\Delta$A")
    ax.set_ylabel(r"$\Delta$B")
    fig.tight_layout()
    save(fig, out, "spurious_regression")


def fig_differencing(out: Path) -> None:
    """The two differences, and what each one removes."""
    t, y = series(seed=23)
    d1 = np.diff(y)
    ds = y[M:] - y[:-M]
    both = np.diff(ds)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 5.6))
    ax = axes[0, 0]
    ax.plot(t, y, color=INK, lw=1.4)
    ax.set_title(r"$y_t$" + " — trend and seasonality", fontsize=12)

    ax = axes[0, 1]
    ax.plot(t[1:], d1, color=BLUE, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.set_title(r"$\Delta y_t = y_t - y_{t-1}$" + " — trend gone, season stays",
                 fontsize=12)

    ax = axes[1, 0]
    ax.plot(t[M:], ds, color=GREEN, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.set_title(r"$\Delta_{12} y_t = y_t - y_{t-12}$" + " — season gone", fontsize=12)

    ax = axes[1, 1]
    ax.plot(t[M + 1:], both, color=PURPLE, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.set_title(r"$\Delta \Delta_{12} y_t$" + " — both gone", fontsize=12)

    for ax in axes.ravel():
        ax.set_yticks([])
        ax.set_xlabel("t")
    fig.tight_layout()
    save(fig, out, "differencing")


def fig_over_differencing(out: Path) -> None:
    """Differencing is not free: each pass inflates variance and plants an MA root."""
    n = 400
    # phi must be below 0.5 for the variance to actually grow: Var(dx) = 2(1-phi)Var(x),
    # so a strongly autocorrelated series is SMOOTHED by differencing, not roughened
    phi = 0.2
    x = ar1(n, phi, seed=8)
    d1 = np.diff(x)
    d2 = np.diff(d1)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.6))
    for j, (v, name, col) in enumerate([
        (x, rf"$x_t$: AR(1), $\phi = {phi}$, already stationary", GREEN),
        (d1, r"$\Delta x_t$", AMBER),
        (d2, r"$\Delta^2 x_t$", RED),
    ]):
        ax = axes[0, j]
        ax.plot(v, color=col, lw=1.0)
        ax.set_title(f"{name}\nvariance = {v.var():.2f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        corr_stem(axes[1, j], acf(v, 16), len(v), color=col)
        axes[1, j].set_ylabel("ACF" if j == 0 else "")
    axes[0, 1].text(0.5, -0.20, r"$\mathrm{Var}(\Delta x) = 2(1-\phi)\,\mathrm{Var}(x)$"
                    + f"   so here {2 * (1 - phi):.1f}" + r"$\times$",
                    transform=axes[0, 1].transAxes, ha="center", fontsize=12, color=AMBER)
    fig.suptitle("variance grows & large negative lag-1 autocorrelation", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "over_differencing")


def fig_adf_kpss(out: Path) -> None:
    """Two tests with opposite nulls, and the 2x2 that actually gets used."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.93, "opposite nulls", ha="center", fontsize=13.5)
    box(ax, (0.5, 0.72), 0.86, 0.20,
        "Augmented Dickey-Fuller\n"
        r"$H_0$: there is a unit root  (non-stationary)",
        ec=BLUE, fs=12)
    box(ax, (0.5, 0.44), 0.86, 0.20,
        "Kwiatkowski Phillips Schmidt Shin\n" r"$H_0$: the series is stationary",
        ec=GREEN, fs=12)
    ax.text(0.5, 0.20,
            "a small p-value rejects the null.\n"
            "ADF: small p is good news.  KPSS: small p is bad news.",
            ha="center", fontsize=12, color=INK, linespacing=1.6)
    # ax.text(0.5, 0.03, "failing to reject is not evidence of the null",
    #         ha="center", fontsize=11.5, color=RED)

    ax = axes[1]
    blank(ax)
    cells = [
        ((0.30, 0.63), "ADF rejects\nKPSS does not", "stationary", GREEN),
        ((0.72, 0.63), "ADF does not\nKPSS rejects", "difference it", BLUE),
        ((0.30, 0.30), "both reject", "look at the plots", AMBER),
        ((0.72, 0.30), "neither rejects", "not enough data", RED),
    ]
    for (x, y), cond, verdict, col in cells:
        box(ax, (x, y), 0.36, 0.24, "", ec=col, fs=11)
        ax.text(x, y + 0.055, cond, ha="center", fontsize=11, color=GREY,
                linespacing=1.3)
        ax.text(x, y - 0.062, verdict, ha="center", fontsize=11.5, color=col,
                linespacing=1.3)
    fig.tight_layout()
    save(fig, out, "adf_kpss")


# ========================================================== 3. autocorrelation
def fig_eq_acf(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    eq(ax, 0.76, r"$\gamma(h) \;=\; \mathrm{Cov}(y_t,\, y_{t+h})$", fs=22, color=BLUE)
    ax.text(0.5, 0.64, "autocovariance at lag $h$; stationarity gives $\\gamma(h)$ and not $\\gamma(t, h)$",
            ha="center", fontsize=11.5, color=GREY)
    eq(ax, 0.46, r"$\rho(h) \;=\; \dfrac{\gamma(h)}{\gamma(0)} \;\in\; [-1, 1]$",
       fs=22, color=GREEN)
    ax.text(0.5, 0.31, "autocorrelation is unit-free", ha="center", fontsize=11.5, color=GREY)
    eq(ax, 0.05, r"$\hat\rho(h) = \dfrac{\sum_{t=1}^{n-h}(y_t-\bar y)(y_{t+h}-\bar y)}"
                 r"{\sum_{t=1}^{n}(y_t-\bar y)^2}$", fs=19, color=INK)
    # ax.text(0.94, 0.1, "note the denominator:\nn terms, not n−h",
    #         ha="center", fontsize=11, color=RED, linespacing=1.4)
    save(fig, out, "eq_acf")


def fig_acf_build(out: Path) -> None:
    """From a lag scatterplot to one point of the correlogram."""
    x = ar1(300, 0.75, seed=14)
    fig, axes = plt.subplots(1, 4, figsize=WIDE)
    for ax, h in zip(axes[:3], (1, 2, 8)):
        ax.scatter(x[:-h], x[h:], s=9, color=BLUE, alpha=0.6)
        r = acf(x, 10)[h]
        ax.set_title(f"lag {h}:  r = {r:.2f}", fontsize=12.5)
        ax.set_xlabel(r"$y_t$")
        ax.set_ylabel(rf"$y_{{t+{h}}}$")
        ax.set_aspect("equal")
    corr_stem(axes[3], acf(x, 24), len(x), title="all the lags at once")
    axes[3].set_ylabel(r"$\hat\rho(h)$")
    fig.suptitle("the correlogram is one scatterplot per lag, squashed to a single number each",
                 fontsize=13)
    fig.tight_layout()
    save(fig, out, "acf_build")


def fig_acf_zoo(out: Path) -> None:
    """The four signatures worth recognising on sight."""
    n = 300
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(6)
    cases = [
        ("white noise", rng.normal(0, 1, n), "nothing outside the band"),
        ("trend", 0.05 * t + rng.normal(0, 1, n), "slow decay, stays positive"),
        ("seasonal, period 12", _season(t, amp=3) + rng.normal(0, 0.8, n),
         "spikes at 12, 24, 36"),
        ("AR(1), $\\phi = 0.8$", ar1(n, 0.8, seed=5), "geometric decay"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.6))
    for j, (name, y, note) in enumerate(cases):
        axes[0, j].plot(t, y, color=INK, lw=1.0)
        axes[0, j].set_title(name, fontsize=12)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        corr_stem(axes[1, j], acf(y, 40), n)
        axes[1, j].set_title(note, fontsize=11, color=BLUE)
    axes[1, 0].set_ylabel("ACF")
    fig.tight_layout()
    save(fig, out, "acf_zoo")


def fig_acf_bands(out: Path) -> None:
    """The band is a white-noise band, and it is a per-lag band."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    corr_stem(axes[0], acf(x, 40), 200, title="pure white noise, n = 200")
    axes[0].set_ylabel("ACF")
    hits = np.abs(acf(x, 40)[1:]) > 1.96 / np.sqrt(200)
    axes[0].text(2.5, -0.9, f"{hits.sum()} of 40 lags outside the band",
                 fontsize=11, color=RED)

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.88, "Suppose", ha="center", fontsize=13)
    eq(ax, 0.68, r"$\hat\rho(h) \;\approx\; \mathcal{N}(0,; 1/n)$",
       fs=19, color=BLUE)
    ax.text(0.5, 0.52, "i.e. the series is white noise",
            ha="center", fontsize=11.5, color=GREY)
    eq(ax, 0.34, r"band $= \pm\, 1.96/\sqrt{n}$", fs=18, color=INK)
    ax.text(0.5, 0.16, "the band narrows when collecting more data",
            ha="center", fontsize=11.5, color=AMBER, linespacing=1.5)

    fig.tight_layout()
    save(fig, out, "acf_bands")


def fig_pacf_idea(out: Path) -> None:
    """What "partial" removes, on an AR(1) where lag 2 is entirely second-hand."""
    x = ar1(600, 0.8, seed=15)
    a = acf(x, 20)
    p = pacf(x, 20)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.90, r"AR(1):  $y_t = 0.8\, y_{t-1} + \varepsilon_t$",
            ha="center", fontsize=13)
    for i, (x0, lab) in enumerate([(0.18, r"$y_{t-2}$"), (0.5, r"$y_{t-1}$"),
                                   (0.82, r"$y_t$")]):
        box(ax, (x0, 0.58), 0.18, 0.14, lab, ec=BLUE, fs=14)
    arrow(ax, (0.28, 0.58), (0.40, 0.58), color=BLUE, lw=2.4)
    arrow(ax, (0.60, 0.58), (0.72, 0.58), color=BLUE, lw=2.4)
    ax.annotate("", xy=(0.76, 0.45), xytext=(0.24, 0.44),
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 2.0,
                                "connectionstyle": "arc3,rad=0.35", "ls": "--"})
    ax.text(0.5, 0.30, "no direct arrow from $y_{t-2}$ to $y_t$", ha="center",
            fontsize=12, color=RED)
    ax.text(0.5, 0.10, "yet they are correlated, because both pass through $y_{t-1}$",
            ha="center", fontsize=11.5, color=GREY)

    corr_stem(axes[1], a, len(x), title=r"ACF: $\rho(2) \approx 0.8^2 = 0.64$")
    axes[1].set_ylabel("ACF")
    corr_stem(axes[2], p, len(x), title="PACF: lag 2 is zero", color=PURPLE)
    axes[2].set_ylabel("PACF")
    fig.suptitle("the PACF at lag h is the correlation left after regressing out lags 1..h-1",
                 fontsize=13)
    fig.tight_layout()
    save(fig, out, "pacf_idea")


def fig_acf_pacf_signature(out: Path) -> None:
    """The identification table, with the two processes that prove it."""
    ar = arp(600, (0.9, -0.25), seed=19)
    ma = maq(600, (0.8, 0.5), seed=21)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.8))

    corr_stem(axes[0, 0], acf(ar, 20), len(ar), title="AR(2) — ACF decays")
    corr_stem(axes[0, 1], pacf(ar, 20), len(ar), title="AR(2) — PACF cuts off at 2",
              color=PURPLE)
    corr_stem(axes[1, 0], acf(ma, 20), len(ma), title="MA(2) — ACF cuts off at 2")
    corr_stem(axes[1, 1], pacf(ma, 20), len(ma), title="MA(2) — PACF decays",
              color=PURPLE)
    axes[0, 0].set_ylabel("ACF")
    axes[1, 0].set_ylabel("ACF")

    for ax in (axes[0, 2], axes[1, 2]):
        ax.remove()
    ax = fig.add_subplot(1, 3, 3)
    blank(ax)
    ax.text(0.5, 0.92, "the identification table", ha="center", fontsize=13.5)
    hdr = [("", 0.16), ("ACF", 0.50), ("PACF", 0.82)]
    for txt, x in hdr:
        ax.text(x, 0.76, txt, ha="center", fontsize=12, color=GREY)
    rows = [
        ("AR(p)", "AutoRegressive", "tails off", "cuts off\nafter p", BLUE),
        ("MA(q)", "Moving Average", "cuts off\nafter q", "tails off", GREEN),
        ("ARMA", "AutoRegressive Moving Average", "tails off", "tails off",
         AMBER),
    ]
    for i, (name, full, a, b, col) in enumerate(rows):
        y = 0.63 - i * 0.215
        ax.text(0.16, y + 0.035, name, ha="center", fontsize=13, color=col)
        ax.text(0.16, y - 0.045, full, ha="center", fontsize=9, color=GREY)
        ax.text(0.50, y, a, ha="center", fontsize=11.5, linespacing=1.3)
        ax.text(0.82, y, b, ha="center", fontsize=11.5, linespacing=1.3)
    ax.text(0.5, 0.03, "clean on paper, murky on real data",
            ha="center", fontsize=11.5, color=RED, linespacing=1.5)
    fig.tight_layout()
    save(fig, out, "acf_pacf_signature")


def fig_ma_edges(out: Path) -> None:
    """A centred window has no data at the ends: NaN, or you invent something."""
    t, y = series(72, seed=21)
    w = M
    h = w // 2
    ma = moving_average(y, w)

    # the three standard ways of not having a NaN
    pad_edge = np.concatenate([np.full(h, y[0]), y, np.full(h, y[-1])])
    pad_refl = np.concatenate([y[h:0:-1], y, y[-2:-h - 2:-1]])
    filled = {}
    for name, padded in (("repeat the edge", pad_edge), ("reflect", pad_refl)):
        k = np.ones(w + 1) / w
        k[0] = k[-1] = 1 / (2 * w)
        filled[name] = np.array([np.dot(k, padded[i:i + w + 1])
                                 for i in range(len(y))])
    trailing = np.array([np.nan if i < w else y[i - w:i].mean()
                         for i in range(len(y))])

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    ax.plot(t, y, color=FAINT, lw=1.6)
    ax.plot(t, ma, color=BLUE, lw=2.6)
    ax.axvspan(t[0], t[h], color=RED, alpha=0.12, lw=0)
    ax.axvspan(t[-h - 1], t[-1], color=RED, alpha=0.12, lw=0)
    ax.text(t[h] + 1, y.max(), f"{h} points\nwith no value", fontsize=11, color=RED,
            va="top", linespacing=1.4)
    ax.set_title(r"the centred window needs $y_{t\pm 6}$", fontsize=12)
    tidy(ax, "t")

    ax = axes[1]
    z = 22                                   # zoom on the tail, where they differ
    ax.plot(t[-z:], y[-z:], color=FAINT, lw=1.8)
    ax.plot(t[-z:], ma[-z:], color=BLUE, lw=3.4, label="NaN at the ends")
    for name, col in (("repeat the edge", AMBER), ("reflect", GREEN)):
        ax.plot(t[-z:], filled[name][-z:], color=col, lw=2.0, ls="--", label=name)
    ax.plot(t[-z:], trailing[-z:], color=PURPLE, lw=2.0, label="trailing window")
    ax.axvspan(t[-h - 1], t[-1], color=RED, alpha=0.10, lw=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title(f"zoom on the last {z} points", fontsize=12)
    tidy(ax, "t")

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.92, "which one to use", ha="center", fontsize=13.5)
    items(ax, [
        ("NaN", "honest, and what pandas does by default"),
        ("repeat or reflect", "invents data, and biases the last trend point"),
        ("trailing window", "no future needed, but it lags by $w/2$"),
        # ("loess", "refits a local line, so the ends are estimated"),
    ], top=0.78, step=0.165)
    fig.tight_layout()
    save(fig, out, "ma_edges")


def fig_loess_build(out: Path) -> None:
    """Loess is one weighted straight line per point, and then you move the point."""
    rng = np.random.default_rng(4)
    t = np.linspace(0, 10, 90)
    y = np.sin(t * 0.8) * 3 + 0.35 * t + rng.normal(0, 0.55, len(t))
    span = 0.3

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    for ax, t0 in zip(axes[:2], (3.0, 8.4)):
        val, w, a, b = _loess_at(t, y, t0, span)
        inw = w > 0
        ax.scatter(t[~inw], y[~inw], s=14, color=GREY, alpha=0.30, zorder=1)
        ax.scatter(t[inw], y[inw], s=16 + 90 * w[inw], color=BLUE, alpha=0.75,
                   zorder=2)
        tt = np.linspace(t[inw].min(), t[inw].max(), 20)
        ax.plot(tt, a + b * (tt - t0), color=RED, lw=2.4, zorder=3)
        ax.plot([t0], [val], "o", color=RED, ms=11, zorder=4)
        ax.axvline(t0, color=RED, lw=1.0, ls=":", zorder=0)
        ax.set_title(rf"$t_0 = {t0}$", fontsize=12)
        tidy(ax, "t")

    ax = axes[2]
    ax.scatter(t, y, s=16, color=GREY, alpha=0.45)
    ax.plot(t, _loess(t, y, span), color=RED, lw=2.8)
    ax.set_title("do that at every $t_0$",
                 fontsize=12)
    tidy(ax, "t")
    fig.suptitle("Fit one local line per point:\n" + \
                 "weight the neighbours, fit a line, keep one point", fontsize=13)
    fig.tight_layout()
    save(fig, out, "loess_build")


def fig_loess_span(out: Path) -> None:
    """The weight function, the span dial, and why the ends survive."""
    rng = np.random.default_rng(4)
    t = np.linspace(0, 10, 90)
    y = np.sin(t * 0.8) * 3 + 0.35 * t + rng.normal(0, 0.55, len(t))

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    u = np.linspace(-1.3, 1.3, 400)
    ax.plot(u, _tricube(u), color=PURPLE, lw=2.8)
    ax.axvline(-1, color=GREY, lw=1.0, ls=":")
    ax.axvline(1, color=GREY, lw=1.0, ls=":")
    ax.set_xlabel(r"$u_i = (t_i - t_0)\,/\,$bandwidth")
    ax.set_ylabel("weight")
    ax.set_title("tricube weight\n" + r"$w_i = (1 - |u_i|^3)^3$", fontsize=12)
    ax.text(0.5, 0.25, "zero outside\nthe window", transform=ax.transAxes,
            fontsize=10.5, color=GREY, ha="center", linespacing=1.4)

    ax = axes[1]
    ax.scatter(t, y, s=14, color=GREY, alpha=0.4)
    for span, col in ((0.12, AMBER), (0.35, GREEN), (0.9, RED)):
        ax.plot(t, _loess(t, y, span), color=col, lw=2.2, label=f"span = {span}")
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("span is a smoothness dial", fontsize=12)
    tidy(ax, "t")

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.92, "Loess benefits compared\n" + \
            "to classical decomposition", ha="center", fontsize=13.5, color=GREEN)
    items(ax, [
        ("it estimates the ends", "the window just becomes one-sided, no NaN"),
        ("the span is continuous", "unlike an integer $w$"),
        ("robust", "to large residuals and resampling"),
    ], top=0.72, step=0.175)
    fig.tight_layout()
    save(fig, out, "loess_span")


def fig_white_noise(out: Path) -> None:
    """The reference process: no memory at all."""
    rng = np.random.default_rng(12)
    n = 300
    x = rng.normal(0, 1, n)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.90, "white noise", ha="center", fontsize=15, color=GREEN)
    eq(ax, 0.72, r"$\varepsilon_t \;\sim\; \mathcal{N}(0,\, \sigma^2)$", fs=20)
    ax.text(0.5, 0.60, "same independent distribution for all $t$",
            ha="center", fontsize=12, color=GREY)
    items(ax, [
        (r"$\mathbb{E}[\varepsilon_t] = 0$", "constant mean"),
        (r"$\mathrm{Var}[\varepsilon_t] = \sigma^2$", "constant variance"),
        (r"$\mathrm{Cov}(\varepsilon_t, \varepsilon_{t+h}) = 0,\; h \neq 0$",
         "no memory"),
    ], top=0.44, step=0.145, fs=12.5)

    ax = axes[1]
    ax.plot(x, color=GREEN, lw=0.9)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_title("one realisation", fontsize=12.5)
    tidy(ax, "t", r"$\varepsilon_t$")

    corr_stem(axes[2], acf(x, 30), n, title="ACF: nothing to predict",
              color=GREEN)
    axes[2].set_ylabel("AutoCorrelation Function")
    fig.tight_layout()
    save(fig, out, "white_noise")


def fig_ar_intro(out: Path) -> None:
    """Feed white noise through a one-line recursion and it acquires memory."""
    n = 200
    phi = 0.8
    rng = np.random.default_rng(7)
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    ax.plot(e, color=GREEN, lw=0.9)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_title("white noise", fontsize=12.5)
    tidy(ax, "t", r"$\varepsilon_t$")

    ax = axes[1]
    blank(ax)
    eq(ax, 0.88, r"$y_t \;=\; \phi\, y_{t-1} \;+\; \varepsilon_t$", fs=22, color=BLUE)
    ax.text(0.5, 0.53, "fraction $\\phi$ of previous value plus noise",
            ha="center", fontsize=12, color=INK, linespacing=1.5)
    ax.text(0.5, 0.14, r"stationary as long as $|\phi| < 1$", ha="center",
            fontsize=13.5, color=RED)

    ax = axes[2]
    ax.plot(x, color=BLUE, lw=1.3)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_title(rf"AR(1) with $\phi = {phi}$", fontsize=12.5)
    tidy(ax, "t", r"$y_t$")
    fig.tight_layout()
    save(fig, out, "ar_intro")


def fig_ar_memory(out: Path) -> None:
    """phi is a memory dial: how long one shock survives."""
    n = 240
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    h = np.arange(0, 21)
    for phi, col in ((0.3, GREEN), (0.6, AMBER), (0.9, RED)):
        ax.plot(h, phi ** h, "o-", color=col, lw=2.2, ms=4,
                label=rf"$\phi = {phi}$")
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel("steps after the shock")
    ax.set_ylabel("what is left of the shock")
    ax.set_title(r"one shock decays as $\phi^h$", fontsize=12.5)

    ax = axes[1]
    for k, (phi, col) in enumerate(((0.3, GREEN), (0.6, AMBER), (0.9, RED))):
        ax.plot(ar1(n, phi, seed=3) + k * 9, color=col, lw=1.1)
        ax.text(2, k * 9 + 4.6, rf"$\phi = {phi}$", fontsize=11, color=col)
    ax.set_yticks([])
    ax.set_xlabel("t")
    ax.set_title("the same shocks, three values of $\\phi$", fontsize=12.5)

    ax = axes[2]
    hh = np.arange(0, 25)
    for phi, col in ((0.3, GREEN), (0.6, AMBER), (0.9, RED)):
        x = ar1(3000, phi, seed=11)
        ax.plot(hh, acf(x, 24), "o", color=col, ms=4.5,
                label=rf"$\phi = {phi}$")
        ax.plot(hh, phi ** hh, color=col, lw=1.4, ls="--", alpha=0.8)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_xlabel("lag $h$")
    ax.set_ylabel(r"$\rho(h)$")
    ax.set_title("ACF estimated from data\n"
                 r"dashed: the theory $\phi^{|h|}$", fontsize=12)
    fig.tight_layout()
    save(fig, out, "ar_memory")


def fig_random_walk_intro(out: Path) -> None:
    """phi = 1: the shocks stop decaying and simply pile up."""
    n = 200
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, n)
    x = np.cumsum(e)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.90, r"set $\phi = 1$", ha="center", fontsize=15, color=RED)
    eq(ax, 0.72, r"$y_t \;=\; y_{t-1} \;+\; \varepsilon_t$", fs=21, color=RED)
    ax.text(0.5, 0.58, "unroll it:", ha="center", fontsize=12, color=GREY)
    eq(ax, 0.45, r"$y_t \;=\; \sum_{s \leq t} \varepsilon_s$", fs=21, color=INK)
    ax.text(0.5, 0.18, "shocks never decay", ha="center", fontsize=12, color=INK, linespacing=1.5)

    ax = axes[1]
    ax.vlines(np.arange(n), 0, e, color=GREEN, lw=0.8)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_title("white noise", fontsize=12.5)
    tidy(ax, "t", r"$\varepsilon_t$")

    ax = axes[2]
    ax.plot(x, color=RED, lw=1.6)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_title("running total", fontsize=12.5)
    tidy(ax, "t", r"$y_t$")
    fig.tight_layout()
    save(fig, out, "random_walk_intro")


def fig_lag_def(out: Path) -> None:
    """What "lag h" means: the same series, slid along."""
    vals = np.array([12, 15, 11, 18, 21, 17, 24, 27], dtype=float)
    n = len(vals)

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    ax.plot(np.arange(n), vals, "o-", color=INK, lw=2.0, ms=7, label=r"$y_t$")
    ax.plot(np.arange(1, n + 1), vals, "o--", color=BLUE, lw=1.8, ms=6,
            alpha=0.85, label=r"$y_{t-1}$  (lag 1)")
    ax.plot(np.arange(3, n + 3), vals, "o--", color=AMBER, lw=1.8, ms=6,
            alpha=0.85, label=r"$y_{t-3}$  (lag 3)")
    ax.legend(frameon=False, fontsize=11.5)
    ax.set_xlim(-0.6, n + 2.6)
    ax.set_title("a lag is the same series, shifted right", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.95, r"$y_{t-h}$  is the value $h$ steps earlier",
            ha="center", fontsize=14, color=INK)
    cols = [(r"$t$", 0.13, GREY), (r"$y_t$", 0.33, INK),
            (r"$y_{t-1}$", 0.55, BLUE), (r"$y_{t-3}$", 0.79, AMBER)]
    for name, x, col in cols:
        ax.text(x, 0.80, name, ha="center", fontsize=13, color=col)
    for r in range(6):
        yy = 0.68 - r * 0.098
        ax.text(0.13, yy, f"{r}", ha="center", fontsize=11.5, color=GREY)
        ax.text(0.33, yy, f"{vals[r]:.0f}", ha="center", fontsize=11.5)
        ax.text(0.55, yy, f"{vals[r-1]:.0f}" if r >= 1 else "—",
                ha="center", fontsize=11.5, color=BLUE if r >= 1 else RED)
        ax.text(0.79, yy, f"{vals[r-3]:.0f}" if r >= 3 else "—",
                ha="center", fontsize=11.5, color=AMBER if r >= 3 else RED)
    ax.text(0.5, 0.045, "a lag-$h$ column costs you the first $h$ rows",
            ha="center", fontsize=12, color=RED)
    fig.tight_layout()
    save(fig, out, "lag_def")


def fig_acf_def(out: Path) -> None:
    """The autocorrelation function, and what one value of it means."""
    n = 260
    rng = np.random.default_rng(19)
    cases = [
        ("smooth", ar1(n, 0.85, seed=31), "$\\rho(1)$ near $+1$:\nneighbours are alike"),
        ("no memory", rng.normal(0, 1, n), "$\\rho(1)$ near $0$:\nthe past says nothing"),
        ("alternating", ar1(n, -0.75, seed=33),
         "$\\rho(1)$ near $-1$:\nup is followed by down"),
    ]

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.75)

    ax = fig.add_subplot(gs[:, 0])
    blank(ax)
    ax.text(0.5, 0.93, "autocorrelation function", ha="center", fontsize=14,
            color=INK)
    ax.text(0.5, 0.78, r"$\rho(h) \;=\; \mathrm{Corr}\left(y_t,\; y_{t-h}\right)$",
            ha="center", fontsize=23, color=BLUE)
    ax.text(0.5, 0.63, "the correlation between the series\nand its own lag-$h$ copy",
            ha="center", fontsize=12, color=GREY, linespacing=1.5)
    ax.text(0.5, 0.45, r"$\rho(0) = 1$" + "        " + r"$-1 \leq \rho(h) \leq 1$",
            ha="center", fontsize=14, color=INK)
    ax.text(0.5, 0.25, r"ACF: $h \mapsto \rho(h)$",
            ha="center", fontsize=12.5, color=INK, linespacing=1.5)
    ax.text(0.5, 0.10, "it only means something if the series\n"
                       "is stationary: otherwise $\\rho$ depends on $t$ too",
            ha="center", fontsize=11.5, color=RED, linespacing=1.5)

    for i, (name, y, note) in enumerate(cases):
        ax = fig.add_subplot(gs[i, 1])
        r1 = acf(y, 1)[1]
        ax.plot(y[:120], color=[GREEN, GREY, PURPLE][i], lw=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}:   " + rf"$\rho(1) = {r1:+.2f}$", fontsize=12)
        ax.text(1.02, 0.5, note, transform=ax.transAxes, fontsize=10,
                color=GREY, va="center", linespacing=1.4)
    save(fig, out, "acf_def")


def fig_df_intuition(out: Path) -> None:
    """Does the level predict the next change? That single question is the test."""
    n = 400
    stat = ar1(n, 0.6, seed=12)
    walk = random_walk(n, seed=16)

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1], hspace=0.55, wspace=0.34)

    ax = fig.add_subplot(gs[:, 0])
    blank(ax)
    ax.text(0.5, 0.9, "a stationary series is pulled\nback to its mean",
            ha="center", fontsize=12, color=GREEN, linespacing=1.5)
    ax.text(0.5, 0.715, "high values tend to be followed by a fall\n"
                        "low values by a rise",
            ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.555), 0.88, 0.135,
        r"$y_t$ predicts $\delta y_t$" +"\ncoefficient is negative",
        ec=GREEN, fs=11)
    ax.text(0.5, 0.4, "a random walk has no mean",
            ha="center", fontsize=12, color=RED)
    ax.text(0.5, 0.28, "high/low values are not impacting next step",
            ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.125), 0.88, 0.115,
        "the coefficient is ZERO", ec=RED, fs=11.5)

    for col, (y, name, c) in enumerate(((stat, r"stationary: AR(1), $\phi = 0.6$", GREEN),
                                        (walk, "random walk", RED))):
        ax = fig.add_subplot(gs[0, col + 1])
        ax.plot(y, color=c, lw=1.0)
        ax.axhline(y.mean(), color=GREY, lw=1.0, ls=":")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=11.5, color=c)

        g, t, _, lev, dy = _df_regress(y)
        ax = fig.add_subplot(gs[1, col + 1])
        ax.scatter(lev, dy, s=7, color=c, alpha=0.45)
        xs = np.linspace(lev.min(), lev.max(), 20)
        ax.plot(xs, np.polyval(np.polyfit(lev, dy, 1), xs), color=INK, lw=2.2)
        ax.axhline(0, color=GREY, lw=0.9)
        ax.set_xlabel(r"level  $y_{t-1}$")
        if col == 0:
            ax.set_ylabel(r"change  $\Delta y_t$")
        ax.set_title(rf"slope $= {g:+.3f}$", fontsize=11.5, color=c)
    fig.suptitle("regress the change on the level & inspect the sign", fontsize=13)
    save(fig, out, "df_intuition")


def fig_df_test(out: Path) -> None:
    """The plain Dickey-Fuller test: the algebra, the variants, the null."""
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.955, "turn it into a coefficient test", ha="center", fontsize=13)
    ax.text(0.5, 0.83, r"$y_t = \phi\, y_{t-1} + \varepsilon_t$", ha="center",
            fontsize=16, color=INK)
    ax.text(0.5, 0.725, r"subtract $y_{t-1}$ from both sides", ha="center",
            fontsize=10.5, color=GREY)
    ax.text(0.5, 0.615, r"$\Delta y_t = (\phi - 1)\, y_{t-1} + \varepsilon_t$",
            ha="center", fontsize=16, color=BLUE)
    ax.text(0.5, 0.515, r"and write $\gamma = \phi - 1$", ha="center", fontsize=10.5,
            color=GREY)
    box(ax, (0.5, 0.355), 0.94, 0.20,
        r"$H_0:\; \gamma = 0$   unit root, $\phi = 1$" + "\n"
        r"$H_1:\; \gamma < 0$   stationary, $|\phi| < 1$", ec=RED, fs=12.5)
    ax.text(0.5, 0.17, r"$\mathrm{DF} = \hat\gamma \,/\, \mathrm{se}(\hat\gamma)$",
            ha="center", fontsize=15, color=GREEN)

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.955, "three versions", ha="center", fontsize=13)
    variants = [
        ('"n"', r"$\Delta y_t = \gamma y_{t-1} + \varepsilon_t$",
         "no constant: the series is\ncentred on zero", GREY),
        ('"c"', r"$\Delta y_t = \alpha + \gamma y_{t-1} + \varepsilon_t$",
         "a constant: the series has\na non-zero mean", BLUE),
        ('"ct"', r"$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \varepsilon_t$",
         "and a trend: testing against\nTREND-stationarity", PURPLE),
    ]
    for i, (tag, eqn, note, col) in enumerate(variants):
        yy = 0.81 - i * 0.255
        ax.text(0.05, yy, tag, fontsize=14, color=col, va="center")
        ax.text(0.17, yy + 0.045, eqn, fontsize=12.5, color=col, va="center")
        ax.text(0.17, yy - 0.065, note, fontsize=10, color=GREY, va="center",
                linespacing=1.4)
    ax.text(0.5, 0.025, "critical values differ\nbetween versions",
            ha="center", fontsize=10.5, color=RED, linespacing=1.5)

    ax = axes[2]
    stats = _df_null(n=200, reps=4000, seed=1)
    crit = np.quantile(stats, 0.05)
    ax.hist(stats, bins=70, density=True, color=BLUE, alpha=0.75,
            label="DF null\n(4000 simulated\nrandom walks)")
    z = np.linspace(-5, 4, 400)
    ax.axvline(crit, color=RED, lw=2.0)
    ax.axvline(-1.96, color=GREY, lw=1.4, ls=":")
    ax.set_xlim(-5, 4)
    ax.annotate(f"5% critical\nvalue {crit:.2f}", xy=(crit, 0.06),
                xycoords=("data", "axes fraction"), xytext=(0.03, 0.30),
                textcoords="axes fraction", color=RED, fontsize=10,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(-1.90, 0.015, r"$-1.96$", transform=ax.get_xaxis_transform(),
            color=GREY, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlabel(r"$t$-ratio on $\hat\gamma$")
    ax.set_yticks([])
    ax.set_title("$H_0$: 4000 simulated random walks", fontsize=11)
    # fig.suptitle('the Dickey-Fuller test:  $H_0$ is "there is a unit root"',
    #              fontsize=13)
    fig.tight_layout()
    save(fig, out, "df_test")


def fig_adf_test(out: Path) -> None:
    """Why the plain test is not enough: it rejects a TRUE null far too often."""
    crit = float(np.quantile(_df_null(n=200, reps=4000, seed=1), 0.05))

    cases = [("iid", 0.0, GREEN), (r"$\psi = -0.5$", -0.5, AMBER),
             (r"$\psi = -0.8$", -0.8, RED)]
    plain, aug = [], []
    for _, psi, _c in cases:
        tp, ta = [], []
        for k in range(600):
            # H0 is TRUE here: a unit root whose INCREMENTS are autocorrelated
            y = np.cumsum(ar1(201, psi, seed=k) if psi else
                          np.random.default_rng(k).normal(0, 1, 201))
            tp.append(_df_regress(y)[1])
            ta.append(adf_stat(y, lags=8)[0])
        plain.append(np.mean(np.array(tp) < crit))
        aug.append(np.mean(np.array(ta) < crit))

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.955, "DF assumes", ha="center", fontsize=13)
    ax.text(0.5, 0.845, r"$\Delta y_t = \alpha + \gamma y_{t-1} + \varepsilon_t$",
            ha="center", fontsize=14.5, color=BLUE)
    box(ax, (0.5, 0.735), 0.90, 0.105,
        r"$\varepsilon_t$ must be white noise", ec=RED, fs=12.5)
    ax.text(0.5, 0.535, "a real series is rarely an AR(1), so its\n"
                        "increments are usually correlated",
            ha="center", fontsize=11, color=INK, linespacing=1.6)
    ax.text(0.5, 0.395, "and se$(\\hat\\gamma)$ is computed\nfrom a wrong assumption",
            ha="center", fontsize=11, color=GREY, linespacing=1.6)
    ax.text(0.5, 0.225, "wrong se  $\\Rightarrow$  wrong $t$\n"
                        r"$\Rightarrow$  wrong decision",
            ha="center", fontsize=12.5, color=RED, linespacing=1.6)

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.955, "add lags of $\\Delta y$", ha="center", fontsize=13,
            color=GREEN)
    ax.text(0.5, 0.835, r"$\Delta y_t = \alpha + \beta t + \gamma\, y_{t-1}"
                        r" + \sum_{j=1}^{p} \delta_j \Delta y_{t-j}"
                        r" + \varepsilon_t$",
            ha="center", fontsize=12.5, color=BLUE)
    ax.text(0.5, 0.655, "the extra terms absorb the correlation,\n"
                        "so what is left really is white",
            ha="center", fontsize=11, color=INK, linespacing=1.5)
    items(ax, [
        (r"only the $\delta_j$ are new", "everything else is the plain DF"),
        (r"$\gamma$ still answers the question", "same $H_0$, same one-sided test"),
        ("the null distribution is unchanged", "the same Dickey-Fuller table"),
        (r"choose $p$ by Akaike Information Criterion,\nor Schwert's rule", ""),
        #  r"$p = \lceil 12\,(n/100)^{1/4} \rceil$"),
    ], top=0.545, step=0.125)
    fig.tight_layout()
    save(fig, out, "adf_test")


def fig_kpss_intuition(out: Path) -> None:
    """Do the deviations cancel, or pile up? That is the whole test."""
    n = 400
    stat = ar1(n, 0.6, seed=12)
    walk = random_walk(n, seed=16)

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1], hspace=0.55, wspace=0.34,
                          top=0.87, bottom=0.07, left=0.03, right=0.98)

    ax = fig.add_subplot(gs[:, 0])
    blank(ax)
    ax.text(0.5, 0.95, "a stationary series keeps\ncoming back to its mean",
            ha="center", fontsize=12, color=GREEN, linespacing=1.5)
    ax.text(0.5, 0.7, "so its deviations keep flipping sign,\n"
                        "and a running total of them keeps\nbeing cancelled out",
            ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.55), 0.88, 0.115,
        "the partial sums stay small", ec=GREEN, fs=11.5)
    ax.text(0.5, 0.415, "a wandering series does not",
            ha="center", fontsize=12, color=RED)
    ax.text(0.5, 0.305, "its deviations keep the same sign for\n"
                        "long stretches, so the total piles up",
            ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.135), 0.88, 0.115,
        "the partial sums run away", ec=RED, fs=11.5)

    for col, (y, name, c) in enumerate(((stat, r"stationary: AR(1), $\phi = 0.6$", GREEN),
                                        (walk, "random walk", RED))):
        e = y - y.mean()
        flips = int((np.diff(np.sign(e)) != 0).sum())

        ax = fig.add_subplot(gs[0, col + 1])
        ax.fill_between(np.arange(n), 0, e, where=e >= 0, color=c, alpha=0.55, lw=0)
        ax.fill_between(np.arange(n), 0, e, where=e < 0, color=GREY, alpha=0.55, lw=0)
        ax.axhline(0, color=INK, lw=1.0)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel('t')
        if col == 0:
            ax.set_ylabel('deviations from the mean')
        ax.set_title(name, fontsize=11, color=c)
        ax.text(0.98, 0.05, f"{flips} sign changes", transform=ax.transAxes,
                ha="right", fontsize=10, color=INK)

        ax = fig.add_subplot(gs[1, col + 1])
        S = np.cumsum(e)
        ax.plot(S, color=c, lw=2.0)
        ax.axhline(0, color=GREY, lw=1.0)
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel(r"running total  $S_t$")
        lm, _ = kpss_stat(y)
        ax.set_title(rf"$|S_t|$ reaches {np.abs(S).max():.0f}" + "\n"
                     rf"LM $= {lm:.2f}$", fontsize=11, color=c)
    # fig.suptitle("the KPSS intuition: add up the deviations and see whether they "
    #              "cancel", fontsize=13, y=0.975)
    save(fig, out, "kpss_intuition")


def fig_stl_inner(out: Path) -> None:
    """One pass of the inner loop, with the actual series at every stage."""
    t, y = _stl_demo_series()
    w = {}
    stl_decompose(y, M, want=w)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.8))
    tc = np.arange(-M, N + M, dtype=float)

    ax = axes[0, 0]
    ax.plot(t, y, color=GREY, lw=3.2, alpha=0.55, label=r"$y_t$")
    ax.axhline(0, color=RED, lw=2.0, label=r"$\hat T_t = 0$  (initialisation)")
    ax.plot(t, w["d"], color=BLUE, lw=1.2, label=r"$d_t = y_t - \hat T_t$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.97, 0.12, "first pass, so $d_t = y_t$ exactly",
            transform=ax.transAxes, ha="right", fontsize=9, color=RED)
    ax.set_title(r"1. detrend", fontsize=11.5, color=BLUE)

    ax = axes[0, 1]
    ax.plot(t, w["d"], color=FAINT, lw=1.0)
    for ph in (0, 5, 9):
        idx = np.arange(ph, N, M)
        ax.plot(idx, w["d"][idx], "o", ms=3, alpha=0.75,
                color=[GREEN, AMBER, PURPLE][(ph // 4) % 3])
    ax.plot(tc, w["C"], color=GREEN, lw=1.8)
    ax.set_title("2. smooth each phase separately\n"
                 r"loess per cycle-subseries $\rightarrow C_t$", fontsize=10.5,
                 color=GREEN)

    ax = axes[0, 2]
    ax.plot(tc, w["C"], color=FAINT, lw=1.4)
    ax.plot(t, w["L"], color=AMBER, lw=2.4)
    ax.set_title("3. the trend that leaked into $C_t$\n"
                 r"low-pass filter $\rightarrow L_t$", fontsize=10.5, color=AMBER)

    ax = axes[1, 0]
    ax.plot(t, w["S"], color=GREEN, lw=1.6)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.set_title(r"4. $\hat S_t = C_t - L_t$" + "\nnow it averages to zero",
                 fontsize=10.5, color=GREEN)

    ax = axes[1, 1]
    ax.plot(t, y, color=FAINT, lw=1.0)
    ax.plot(t, w["ds"], color=BLUE, lw=1.2)
    ax.set_title(r"5. deseasonalise:  $y_t - \hat S_t$", fontsize=10.5, color=BLUE)

    ax = axes[1, 2]
    ax.plot(t, w["ds"], color=FAINT, lw=1.0)
    ax.plot(t, w["T"], color=RED, lw=2.6)
    ax.set_title("6. re-smooth the trend\n"
                 r"loess $\rightarrow \hat T_t$, then repeat", fontsize=10.5,
                 color=RED)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("one pass of the STL inner loop — the trend starts at zero and "
                 "each pass sharpens both parts", fontsize=13)
    fig.tight_layout()
    save(fig, out, "stl_inner")


def fig_stl_outer(out: Path) -> None:
    """The robustness loop: one bad point, and what the weights do about it."""
    t, y = _stl_demo_series()
    # a narrow trend window, so that one bad point can actually bend the trend;
    # with the default wide window the robustness pass is nearly invisible
    nt = 0.10
    T0, S0, R0 = stl_decompose(y, M, nt=nt, no=0)
    T1, S1, R1 = stl_decompose(y, M, nt=nt, no=2)

    h = 6 * np.median(np.abs(R0 - np.median(R0)))
    rw = np.clip((1 - np.clip(np.abs(R0) / h, 0, 1) ** 2) ** 2, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    ax.plot(t, R0, color=INK, lw=1.1)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.axhline(h, color=RED, lw=1.4, ls="--")
    ax.axhline(-h, color=RED, lw=1.4, ls="--")
    ax.text(2, h + 0.6, r"$\pm\, 6 \times$ median $|\hat R|$", color=RED, fontsize=10.5)
    ax.annotate("outlier", xy=(95, R0[95]), xytext=(30, R0[95] - 1.5),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_title(r"residuals $\hat R_t$", fontsize=12)
    tidy(ax, "t", "residual")

    ax = axes[1]
    ax.plot(t, rw, color=PURPLE, lw=1.6)
    ax.plot([95], [rw[95]], "o", color=RED, ms=9)
    ax.set_ylim(-0.05, 1.12)
    ax.text(0.03, 0.12, "outlier weight\nclose to 0 instead of 1",
            transform=ax.transAxes, fontsize=10.5, color=RED, linespacing=1.4)
    ax.set_title(r"bisquare weight  $\rho_t = B(|\hat R_t| / 6\,$median$)$",
                 fontsize=11)
    tidy(ax, "t", r"$\rho_t$")

    ax = axes[2]
    lo, hi = 62, 128
    m_ = (t >= lo) & (t <= hi)
    ax.plot(t[m_], y[m_], color=FAINT, lw=1.2)
    ax.plot(t[m_], T0[m_], color=RED, lw=2.6,
            label=r"$n_o = 0$: trend bends to outliers")
    ax.plot(t[m_], T1[m_], color=GREEN, lw=2.6, label=r"$n_o = 2$: robust")
    ax.axvline(95, color=GREY, lw=1.0, ls=":")
    k = int(np.argmax(np.abs(T0 - T1)))
    ax.annotate("", xy=(t[k], T0[k]), xytext=(t[k], T1[k]),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.4})
    ax.text(t[k] + 2, (T0[k] + T1[k]) / 2, f"{abs(T0[k] - T1[k]):.1f}", fontsize=11,
            color=INK, va="center")
    ax.set_ylim(T1[m_].min() - 1.5, T0[m_].max() + 3.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("inner loop with new weights", fontsize=11.5)
    tidy(ax, "t")
    fig.tight_layout()
    save(fig, out, "stl_outer")


def fig_ma_intro(out: Path) -> None:
    """MA(q): the other way to build memory out of white noise."""
    th = (0.8, -0.4)
    n = 120
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, n + len(th))
    w = np.concatenate(([1.0], np.asarray(th, float)))
    x = np.array([np.dot(w, e[i + len(th)::-1][:len(th) + 1]) for i in range(n)])
    long = maq(4000, th, seed=1)
    r = acf(long, 8)
    t1, t2 = th
    den = 1 + t1 ** 2 + t2 ** 2
    theory = np.array([1.0, (t1 + t1 * t2) / den, t2 / den, 0, 0, 0, 0, 0, 0])

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.845, r"$y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}"
                        r" + \dots + \theta_q \varepsilon_{t-q}$",
            ha="center", fontsize=14, color=GREEN)
    ax.text(0.5, 0.735, "a weighted sum of the last $q$ SHOCKS", ha="center",
            fontsize=11, color=GREY)
    items(ax, [
        (r"$\varepsilon_t$ are never observed", r"only $y_t$ is observed"),
        ("always stationary", "as finite sum of finite-variance terms"),
    ], top=0.68, step=0.125)

    ax = axes[1]
    z = 60                                    # fewer points: the shocks must be visible
    ax.vlines(np.arange(z), 0, e[len(th):len(th) + z], color=GREY, lw=1.6,
              alpha=0.9)
    ax.plot(np.arange(z), x[:z], color=GREEN, lw=2.2)
    ax.axhline(0, color=GREY, lw=0.8)
    lo = 34
    ax.axvspan(lo - len(th) - 0.4, lo + 0.4, color=AMBER, alpha=0.30, lw=0)
    ax.set_ylim(x[:z].min() * 1.15, x[:z].max() * 1.45)
    # ax.annotate(r"$y_{34}$ is built from" + "\nthese 3 shocks",
    #             xy=(lo - 1, x[:z].max() * 0.6),
    #             xytext=(lo + 5, x[:z].max() * 1.20),
    #             fontsize=10, color=AMBER, linespacing=1.4,
    #             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6))
    ax.set_xlim(-1, z)
    ax.set_xlabel("t")
    ax.set_title(r"grey: the shocks $\varepsilon_t$ (not observed)" + "\n"
                 rf"green: MA(2), $\theta = ({t1},\, {t2})$", fontsize=11.5)
    ax.set_yticks([])

    fig.suptitle("white noise with a short memory", fontsize=13)
    fig.tight_layout()
    save(fig, out, "ma_intro")


def fig_ar_p(out: Path) -> None:
    """AR(1) generalises: p lags instead of one."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.8, r"$Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots"
                       r" + \phi_p Y_{t-p} + \varepsilon_t$",
            ha="center", fontsize=14.5, color=BLUE)
    ax.text(0.5, 0.60, "linear regression;\nwith $p$ features",
            ha="center", fontsize=12, color=GREY, linespacing=1.5)
    items(ax, [
        (r"$p = 1$", "markov"),
        (r"$p \geq 2$", "can produce a cycle"),
        (r"$p = m$", "can reach back a season"),
    ], top=0.44, step=0.13, fs=12)
    # ax.text(0.5, 0.02, r"stationarity is no longer just $|\phi| < 1$",
    #         ha="center", fontsize=12, color=RED)

    ax = axes[1]
    x = arp(240, (0.75,), seed=61)
    ax.plot(x, color=BLUE, lw=1.2)
    ax.set_title(r"AR(1), $\phi = 0.75$", fontsize=11.5)
    tidy(ax, "t")

    ax = axes[2]
    x = arp(240, (1.5, -0.8), seed=61)
    ax.plot(x, color=PURPLE, lw=1.2)
    ax.set_title(r"AR(2), $\phi = (1.5,\, -0.8)$", fontsize=11.5)
    tidy(ax, "t")
    fig.tight_layout()
    save(fig, out, "ar_p")


def fig_backshift(out: Path) -> None:
    """B is bookkeeping: it turns a recursion into a polynomial."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.955, "the backshift operator", ha="center", fontsize=14,
            color=INK)
    ax.text(0.5, 0.845, r"$B \;:\; \mathbb{R}^{\mathbb{Z}} \;\longrightarrow\;"
                        r" \mathbb{R}^{\mathbb{Z}}$", ha="center", fontsize=17,
            color=PURPLE)
    ax.text(0.5, 0.745, "it takes a SERIES and returns a SERIES",
            ha="center", fontsize=11, color=GREY, linespacing=1.4)
    ax.text(0.5, 0.630, r"$(B\,Y)_t = Y_{t-1}$" + "          "
                        + r"$(B^k Y)_t = Y_{t-k}$", ha="center", fontsize=17,
            color=BLUE)
    ax.text(0.5, 0.545, "linear and invertible on the infinite space",
            ha="center", fontsize=10.5, color=GREY)
    items(ax, [
        (r"$(1 - B)\,Y_t = Y_t - Y_{t-1}$", "the first difference"),
        (r"$(1 - B^m)\,Y_t$", "the seasonal difference"),
        (r"$(1 - B)^d$", "differencing $d$ times"),
    ], top=0.46, step=0.125, fs=12.5)
    ax.text(0.5, 0.075, "on a finite sample of length $n$ the image is shorter:",
            ha="center", fontsize=10.5, color=RED)
    ax.text(0.5, 0.005, r"$B^k$ costs you the first $k$ observations",
            ha="center", fontsize=11.5, color=RED)

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.92, "AR(p) can be rewritten", ha="center", fontsize=15, color=INK)
    ax.text(0.5, 0.78, r"$Y_t = \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p}"
                       r" + \varepsilon_t$", ha="center", fontsize=15)
    ax.text(0.5, 0.53, r"$\left(1 - \phi_1 B - \dots - \phi_p B^p\right) Y_t"
                       r" = \varepsilon_t$", ha="center", fontsize=15, color=BLUE)
    ax.text(0.5, 0.39, "name that bracket", ha="center", fontsize=10.5,
            color=GREY)
    ax.text(0.5, 0.26, r"$\Phi(B)\, Y_t = \varepsilon_t$", ha="center", fontsize=20,
            color=PURPLE)
    fig.tight_layout()
    save(fig, out, "backshift")


def fig_char_eq(out: Path) -> None:
    """Treat the polynomial as a polynomial in z and look at its roots."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    # ax.text(0.5, 0.93, "the characteristic equation", ha="center", fontsize=14)
    ax.text(0.5, 0.85, r"$\Phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \dots"
                       r" - \phi_p z^p = 0$", ha="center", fontsize=17.5, color=PURPLE)
    ax.text(0.5, 0.65, "replace the operator $B$ by a complex number $z$and solve",
            ha="center", fontsize=15, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.42), 0.90, 0.20,
        "the process is stationary\n"
        r"$\Longleftrightarrow$" + "  every root has $|z| > 1$", ec=GREEN, fs=13)
    ax.text(0.5, 0.20, "for AR(1) this condition simplifies to:",
            ha="center", fontsize=15, color=INK)
    ax.text(0.5, 0.08, r"$1 - \phi z = 0 \;\Rightarrow\; z = 1/\phi$,"
                       r"  so $|z| > 1 \Leftrightarrow |\phi| < 1$",
            ha="center", fontsize=15, color=BLUE)

    ax = axes[1]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=2.0)
    ax.fill(np.cos(th), np.sin(th), color=RED, alpha=0.07)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    for z, col, lab in ((2.0, GREEN, r"$\phi = 0.5 \Rightarrow z = 2$"),
                        (1.0, AMBER, r"$\phi = 1 \Rightarrow z = 1$"),
                        (0.5, RED, r"$\phi = 2 \Rightarrow z = 0.5$")):
        ax.plot([z], [0], "o", color=col, ms=8)
        ax.annotate(lab, xy=(z, 0), xytext=(z, 0.3 if z != 1.0 else 0.95),
                    ha="center", fontsize=10.5, color=col,
                    arrowprops={"arrowstyle": "->", "color": col, "lw": 1.2})
    ax.text(0, -0.55, "inside:explosive", ha="center", fontsize=10.5, color=RED,
            linespacing=1.3)
    ax.text(2.0, -0.55, "outside: stationary", ha="center", fontsize=10.5,
            color=GREEN)
    ax.set_xlim(-1.6, 2.9)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("Re $z$")
    ax.set_ylabel("Im $z$")
    ax.set_title("the unit circle", fontsize=12.5)
    fig.tight_layout()
    save(fig, out, "char_eq")


def fig_root_cases(out: Path) -> None:
    """The three cases, each with the arithmetic and the picture."""
    n = 150
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.55)

    rows = [
        (r"$|z| > 1$", "stationary and stable", GREEN,
         r"$Y_t = 0.5\, Y_{t-1} + \varepsilon_t$",
         r"$1 - 0.5z = 0 \;\Rightarrow\; z = 2 > 1$",
         "shocks decay away", 0.5),
        (r"$|z| = 1$", "unit root", AMBER,
         r"$Y_t = 1.0\, Y_{t-1} + \varepsilon_t$",
         r"$1 - z = 0 \;\Rightarrow\; z = 1$",
         "shocks never decay", 1.0),
        (r"$|z| < 1$", "explosive", RED,
         r"$Y_t = 2.0\, Y_{t-1} + \varepsilon_t$",
         r"$1 - 2z = 0 \;\Rightarrow\; z = 0.5 < 1$",
         "shocks compound", 2.0),
    ]

    for i, (cond, verdict, col, model, solve, meaning, phi) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        blank(ax)
        ax.text(0.015, 0.76, cond, fontsize=17, color=col, va="center")
        ax.text(0.17, 0.76, verdict, fontsize=12.5, color=col, va="center")
        ax.text(0.17, 0.33, meaning, fontsize=10.5, color=GREY, va="center")
        ax.text(0.645, 0.76, model, fontsize=11.5, color=INK, va="center")
        ax.text(0.645, 0.33, solve, fontsize=11.5, color=col, va="center")
        if i < 2:
            ax.plot([0.01, 0.99], [0.02, 0.02], color=FAINT, lw=1.4)

        ax = fig.add_subplot(gs[i, 1])
        x = ar1(n, phi, seed=9)
        if phi > 1:
            # 2^150 is 1e45: on a linear axis this is a flat line and a cliff
            ax.semilogy(np.abs(x) + 1e-3, color=col, lw=1.3)
            ax.text(0.97, 0.12, r"$|Y_t|$, log scale", transform=ax.transAxes,
                    ha="right", fontsize=9, color=col)
        else:
            ax.plot(x, color=col, lw=1.3)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)
        if i == 0:
            ax.set_title("one realisation", fontsize=11, color=INK)
    # fig.suptitle(r"the roots of $\Phi(z)$ decide everything", fontsize=13.5)
    save(fig, out, "root_cases")


def fig_box_cox_lambda(out: Path) -> None:
    """What the parameter actually does to the scale."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    yy = np.linspace(0.05, 6, 400)
    for lam, col in ((1.0, RED), (0.5, AMBER), (0.0, GREEN), (-1.0, PURPLE)):
        w = np.log(yy) if lam == 0 else (yy ** lam - 1) / lam
        ax.plot(yy, w, color=col, lw=2.4, label=rf"$\lambda = {lam}$")
    ax.axhline(0, color=GREY, lw=0.9)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    ax.set_xlabel(r"original value $y$")
    ax.set_ylabel(r"transformed value $w$")
    ax.set_ylim(-4, 6)
    ax.set_title(r"$\lambda$ chooses how hard to bend the scale", fontsize=12)

    ax = axes[1]
    pairs = np.array([1.0, 2.0, 4.0, 8.0])
    for k, (lam, col) in enumerate(((1.0, RED), (0.5, AMBER), (0.0, GREEN))):
        w = np.log(pairs) if lam == 0 else (pairs ** lam - 1) / lam
        ax.plot(w, np.full_like(w, -k), "o-", color=col, lw=1.6, ms=9)
        for v, orig in zip(w, pairs):
            ax.text(v, -k + 0.22, f"{orig:.0f}", ha="center", fontsize=9.5,
                    color=col)
        ax.text(-1.2, -k, rf"$\lambda = {lam}$", ha="right", fontsize=11, color=col,
                va="center")
    ax.set_xlim(-2.6, 8.5)
    ax.set_ylim(-2.7, 0.7)
    ax.set_yticks([])
    # ax.set_xlabel("where 1, 2, 4 and 8 land after the transform")
    ax.set_title("the smaller $\\lambda$,\nthe more large values are pulled",
                 fontsize=11.5)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.92, r"$\lambda$ is a dial", ha="center", fontsize=14, color=PURPLE)
    ax.text(0.5, 0.8, "Compression strength", ha="center", fontsize=14, color=INK)
    items(ax, [
        (r"$\lambda = 1$: nothing happens", "identity shifted"),
        (r"$\lambda < 1$: large values shrink", "fix a fanning spread"),
        (r"$\lambda = 0$: the log", "the limit (often used)"),
    ], top=0.6, step=0.15)
    fig.tight_layout()
    save(fig, out, "box_cox_lambda")


def fig_adf_recipe(out: Path) -> None:
    """The recipe, and step 1 shown to matter on one concrete series."""
    n = 300
    t = np.arange(n, dtype=float)
    trend = 0.05 * t + np.random.default_rng(0).normal(0, 1, n)
    t_c, _ = adf_stat(trend, regression="c")
    t_ct, _ = adf_stat(trend, regression="ct")

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.18, top=0.86, bottom=0.05,
                          left=0.03, right=0.98)

    ax = fig.add_subplot(gs[:, 0])
    blank(ax)
    ax.text(0.5, 0.965, "running an ADF test", ha="center", fontsize=13.5,
            color=BLUE)
    steps = [
        ("1.", "choose the deterministic part", INK,
         r'"n" none    "c" constant    "ct" constant + trend'),
        ("2.", r"choose the augmentation order $p$", INK,
         r"Schwert: $p = \lceil 12\,(n/100)^{1/4}\rceil$, or by AIC"),
        ("3.", "fit it by ordinary least squares", BLUE,
         r"$\Delta y_t = \alpha + \beta t + \gamma y_{t-1}"
         r" + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t$"),
        ("4.", r"take the $t$-ratio of $\hat\gamma$", BLUE,
         r"$\mathrm{ADF} = \hat\gamma \,/\, \mathrm{se}(\hat\gamma)$"),
        ("5.", "compare with the Dickey-Fuller table", RED,
         r'at $n = 300$:  $-2.87$ for "c",  $-3.42$ for "ct"'),
        ("6.", "read it", GREEN,
         r"more negative $\Rightarrow$ reject the unit root"),
    ]
    for i, (num, title, col, detail) in enumerate(steps):
        yy = 0.845 - i * 0.145
        ax.text(0.03, yy, num, fontsize=14, color=col, fontweight="bold",
                va="center")
        ax.text(0.09, yy + 0.028, title, fontsize=11.5, color=INK, va="center")
        ax.text(0.09, yy - 0.042, detail, fontsize=10, color=col, va="center")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, trend, color=AMBER, lw=1.0)
    ax.plot(t, 0.05 * t, color=INK, lw=2.0, ls="--")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("a straight trend plus noise\n"
                 "(not stationary, but trend-stationary)", fontsize=11.5)

    ax = fig.add_subplot(gs[1, 1])
    blank(ax)
    for i, (tag, val, cv, verdict, col) in enumerate((
            ('"c"', t_c, -2.87, 'no reject: "it has a unit root"', RED),
            ('"ct"', t_ct, -3.42, 'reject: "trend-stationary"', GREEN))):
        x = 0.27 + i * 0.46
        box(ax, (x, 0.42), 0.43, 0.62, "", ec=col, lw=2.0)
        ax.text(x, 0.605, tag, ha="center", fontsize=13.5, color=col)
        ax.text(x, 0.435, rf"ADF $= {val:.2f}$   vs   ${cv}$", ha="center",
                fontsize=11.5)
        ax.text(x, 0.205, verdict, ha="center", fontsize=10, color=col)
    save(fig, out, "adf_recipe")


def fig_kpss_recipe(out: Path) -> None:
    """How to actually run a KPSS test, on the same five series."""
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(0.5, 0.965, "KPSS test", ha="center", fontsize=13.5,
            color=GREEN)
    steps = [
        ("1.", "choose what \"stationary\" means here", INK,
         r'"c" around a level   "ct" around a trend'),
        ("2.", "regress $y_t$ on that, keep the residuals", INK,
         r"$\hat e_t = y_t - \hat\mu$   (or $-\hat\mu - \hat\xi t$)"),
        ("3.", "form the running total", BLUE,
         r"$S_t = \sum_{i \leq t} \hat e_i$"),
        ("4.", "estimate the long-run variance", BLUE,
         r"Bartlett kernel, bandwidth $\ell = \lfloor 4(n/100)^{1/4}\rfloor$"),
        ("5.", "the statistic", GREEN,
         r"$\mathrm{LM} = \sum_t S_t^2 \,/\, (n^2 \hat\sigma^2)$"),
        ("6.", "compare with the UPPER tail", RED,
         r'5%:  $0.463$ for "c",  $0.146$ for "ct"'),
    ]
    for i, (num, title, col, detail) in enumerate(steps):
        yy = 0.845 - i * 0.145
        ax.text(0.03, yy, num, fontsize=14, color=col, fontweight="bold",
                va="center")
        ax.text(0.09, yy + 0.028, title, fontsize=11.5, color=INK, va="center")
        ax.text(0.09, yy - 0.040, detail, fontsize=10.5, color=col, va="center")
    ax.text(0.5, 0.01, "large LM means the partial sums wander, so the level moves",
            ha="center", fontsize=10.5, color=INK)

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    crit = 0.463
    heads = [("series", 0.02, "left"), ("LM", 0.55, "center"),
             ("reject?", 0.72, "center"), ("truth", 0.92, "center")]
    for txt, x, al in heads:
        ax.text(x, 0.845, txt, fontsize=11, color=GREY, ha=al)
    for i, (name, y, truth, tcol) in enumerate(_test_cases()):
        k, _ = kpss_stat(y)
        rej = k > crit
        yy = 0.73 - i * 0.135
        ax.text(0.02, yy, name, fontsize=11.5, va="center")
        ax.text(0.55, yy, f"{k:.2f}", fontsize=11.5, ha="center", va="center",
                color=RED if rej else GREEN)
        ax.text(0.72, yy, "yes" if rej else "no", fontsize=11.5, ha="center",
                va="center", color=RED if rej else GREEN)
        ax.text(0.92, yy, truth, fontsize=10.5, ha="center", va="center", color=tcol)
    ax.text(0.5, 0.045, "rejecting KPSS is the \"bad\" outcome (the null is stationarity)",
            ha="center", fontsize=11, color=RED)
    ax.text(0.5, -0.02, r"$\phi = 0.95$ wrongly flagged", ha="center", fontsize=11, color=AMBER)
    fig.tight_layout()
    save(fig, out, "kpss_recipe")


def fig_box_cox_why(out: Path) -> None:
    """What the transform is FOR: a series whose spread no longer tracks its level."""
    n = 200
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(12)
    y = np.exp(0.013 * t + 1.6) * np.exp(rng.normal(0, 0.16, n))
    w = np.log(y)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    def spread_marks(ax, v, fmt):
        out = []
        for a, b, dx, al in ((0, 60, 6, "left"), (140, 200, -6, "right")):
            seg = v[a:b]
            x0 = a + 30
            ax.annotate("", xy=(x0, seg.max()), xytext=(x0, seg.min()),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
            ax.text(x0 + dx, seg.mean(), f"spread\n{np.ptp(seg):{fmt}}", fontsize=10.5,
                    color=INK, va="center", ha=al, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            out.append(np.ptp(seg))
        return out

    ax = axes[0]
    ax.plot(t, y, color=RED, lw=1.3)
    before = spread_marks(ax, y, ".0f")
    ax.set_title("before: the spread grows with the level", fontsize=12, color=RED)
    tidy(ax, "t")

    ax = axes[1]
    ax.plot(t, w, color=GREEN, lw=1.3)
    after = spread_marks(ax, w, ".2f")
    ax.set_title(r"after ($\lambda = 0$): the spread is the same", fontsize=12,
                 color=GREEN)
    tidy(ax, "t", "transformed value")
    print(f"  box_cox_why spreads: before {before[0]:.0f} -> {before[1]:.0f}, "
          f"after {after[0]:.2f} -> {after[1]:.2f}")

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.94, "Goal", ha="center", fontsize=13, color=PURPLE)
    box(ax, (0.5, 0.76), 0.94, 0.16, "Variance does not depend on value", ec=PURPLE, fs=12.5)
    # ax.text(0.5, 0.55, "Usually assumed:",
    #         ha="center", fontsize=11, color=GREY)
    # items(ax, [
    #     ("stationarity", "constant variance is one of its three conditions"),
    #     ("additive decomposition", "a constant seasonal swing, not a growing one"),
    #     ("prediction intervals", "one error size, reused at every horizon"),
    # ], top=0.46, step=0.145)
    ax.text(0.5, 0.50, "Box-Cox", ha="center", fontsize=14, color=PURPLE)
    eq(ax, 0.32, r"$w_t = (y_t^{\lambda} - 1)\,/\,\lambda$", fs=19)
    eq(ax, 0.18, r"$w_t = \log y_t$" + "     when " + r"$\lambda = 0$", fs=15, color=GREY)
    fig.tight_layout()
    save(fig, out, "box_cox_why")


def fig_box_cox_choice(out: Path) -> None:
    """Choosing lambda by profile likelihood, step by step."""
    n = 200
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(12)
    y = np.exp(0.013 * t + 1.6) * np.exp(rng.normal(0, 0.16, n))

    def bc(v, lam):
        return np.log(v) if abs(lam) < 1e-12 else (v ** lam - 1) / lam

    def profile_ll(lam):
        w = bc(y, lam)
        X = np.column_stack([np.ones(n), t])          # the model we intend to fit
        beta, *_ = np.linalg.lstsq(X, w, rcond=None)
        r = w - X @ beta
        s2 = r @ r / n
        return -0.5 * n * np.log(s2) + (lam - 1) * np.log(y).sum()

    lams = np.linspace(-0.6, 1.2, 241)
    ll = np.array([profile_ll(l) for l in lams])
    best = lams[int(np.argmax(ll))]
    drop = ll.max() - 1.92                            # 95% interval, chi2(1)/2
    inside = lams[ll >= drop]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.95, "how $\\lambda$ is chosen", ha="center", fontsize=13.5,
            color=PURPLE)
    steps = [
        ("1.", "pick a candidate $\\lambda$", r"anywhere on the ladder"),
        ("2.", "transform the series", r"$w_t(\lambda) = (y_t^\lambda - 1)/\lambda$"),
        ("3.", "fit the model you intend to use",
         r"here: a straight line in $t$"),
        ("4.", "record how well it fits",
         r"$\hat\sigma^2_\lambda$ = residual variance"),
        ("5.", "correct for the change of scale",
         r"$+\,(\lambda - 1)\sum_t \log y_t$"),
        ("6.", "repeat, and keep the best $\\lambda$", r"the maximum of the curve"),
    ]
    for i, (num, title, detail) in enumerate(steps):
        yy = 0.83 - i * 0.145
        ax.text(0.03, yy, num, fontsize=13.5, color=PURPLE, fontweight="bold",
                va="center")
        ax.text(0.09, yy + 0.028, title, fontsize=11.5, color=INK, va="center")
        ax.text(0.09, yy - 0.040, detail, fontsize=10.5, color=GREY, va="center")

    ax = axes[1]
    ax.plot(lams, ll, color=PURPLE, lw=2.6)
    ax.axvline(best, color=RED, lw=1.8)
    ax.axhline(drop, color=GREY, lw=1.2, ls=":")
    ax.axvspan(inside.min(), inside.max(), color=RED, alpha=0.10, lw=0)
    ax.plot([best], [ll.max()], "o", color=RED, ms=9)
    ax.text(best + 0.05, ll.max(), f"  $\\hat\\lambda$ = {best:.2f}", fontsize=12,
            color=RED, va="center")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("profile log-likelihood")
    ax.set_title("step 6: the curve, and its maximum", fontsize=12)
    ax.text(0.03, 0.06, f"95% interval  [{inside.min():.2f}, {inside.max():.2f}]",
            transform=ax.transAxes, fontsize=10.5, color=RED)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.95, "step 5", ha="center", fontsize=13,
            color=INK)
    ax.text(0.5, 0.80, r"$\ell(\lambda) = -\frac{n}{2}\log\hat\sigma^2_\lambda"
                       r" + (\lambda - 1)\sum_t \log y_t$",
            ha="center", fontsize=14, color=PURPLE)
    ax.text(0.5, 0.55, "The second term prevents picking\n"
                       "$\\lambda$ that squashes the series flattest",
            ha="center", fontsize=11, color=RED, linespacing=1.5)
    fig.tight_layout()
    save(fig, out, "box_cox_choice")


def fig_pacf_def(out: Path) -> None:
    """The formal definition, checked against an actual sequence of AR fits."""
    x = arp(4000, (0.9, -0.25), seed=19)
    pac = pacf(x, 8)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.96, "the definition", ha="center", fontsize=14, color=PURPLE)
    ax.text(0.5, 0.855, "regress $y_t$ on its first $h$ lags:", ha="center",
            fontsize=11.5, color=GREY)
    ax.text(0.5, 0.735, r"$y_t = \phi_{h1} y_{t-1} + \phi_{h2} y_{t-2}"
                        r" + \dots + \phi_{hh}\, y_{t-h} + e_t$",
            ha="center", fontsize=13.5, color=BLUE)
    box(ax, (0.5, 0.575), 0.92, 0.135,
        r"$\mathrm{PACF}(h) \;=\; \phi_{hh}$", ec=PURPLE, fs=17)
    ax.text(0.5, 0.4, "keep only the last coefficient",
            ha="center", fontsize=11.5, color=GREY)
    # ax.text(0.5, 0.345, "equivalently, strip out the lags in between first:",
    #         ha="center", fontsize=10.5, color=GREY)
    # ax.text(0.5, 0.235, r"$\phi_{hh} = \mathrm{Corr}\left(y_t - \hat y_t,\;"
    #                     r" y_{t-h} - \hat y_{t-h}\right)$",
    #         ha="center", fontsize=12.5, color=INK)
    # ax.text(0.5, 0.125, r"where $\hat{\cdot}$ is the best linear predictor from"
    #                     r" $y_{t-1},\dots,y_{t-h+1}$",
            # ha="center", fontsize=10, color=GREY)
    ax.text(0.5, 0.01, r"at $h = 1$, $\phi_{11} = \rho(1)$",
            ha="center", fontsize=11.5, color=GREEN)

    ax = axes[1]
    blank(ax)
    # ax.text(0.5, 0.96, "so fit them, one order at a time", ha="center", fontsize=13)
    ax.text(0.5, 0.875, r"series: AR(2), $\phi = (0.9,\, -0.25)$", ha="center",
            fontsize=11, color=GREY)
    ax.text(0.10, 0.775, "fit", fontsize=11, color=GREY)
    # ax.text(0.52, 0.775, "last coefficient", fontsize=11, color=GREY, ha="center")
    ax.text(0.90, 0.775, "PACF($h$)", fontsize=11, color=GREY, ha="center")
    for h in range(1, 6):
        _, phis, _ = fit_ar(x, h)
        yy = 0.645 - (h - 1) * 0.113
        col = PURPLE if h == 2 else GREY
        ax.text(0.10, yy, f"AR({h})", fontsize=12, color=col, va="center")
        # ax.text(0.52, yy, f"{phis[-1]:+.3f}", fontsize=12, color=col, va="center", ha="center")
        ax.text(0.90, yy, f"{pac[h]:+.3f}", fontsize=12, color=col, va="center", ha="center")
    # ax.text(0.5, 0.015, "the two columns are the same number,\n"
    #                     "computed two different ways",
    #         ha="center", fontsize=11.5, color=INK, linespacing=1.5)

    ax = axes[2]
    corr_stem(ax, pac, len(x), color=PURPLE)
    ax.axvline(2.5, color=RED, lw=1.4, ls="--")
    ax.set_ylabel("PACF")
    # ax.set_title("and this is why it cuts off", fontsize=12.5)
    ax.text(0.97, 0.90, "AR(2) needs no third lag,\nso $\\phi_{hh} = 0$ for $h > 2$",
            transform=ax.transAxes, ha="right", fontsize=10.5, color=RED,
            linespacing=1.4)
    fig.suptitle(r"PACF($h$): coefficient on $y_{t-h}$ once the lags $<h$ are killed", fontsize=13)
    fig.tight_layout()
    save(fig, out, "pacf_def")


def fig_ljung_box_intuition(out: Path) -> None:
    """Counting bars does not work; adding them up does."""
    from scipy.stats import chi2
    n, M = 240, 24
    band = 1.96 / np.sqrt(n)
    thr = chi2.ppf(0.95, M)
    wn = np.random.default_rng(0).normal(0, 1, n)
    ar = ar1(n, 0.3, seed=5)
    h = np.arange(1, M + 1)

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1], hspace=0.62, wspace=0.3,
                          top=0.835, bottom=0.09, left=0.03, right=0.98)

    ax = fig.add_subplot(gs[:, 0])
    blank(ax)
    ax.text(0.5, 0.75, "a few bars always poke out of the band\n"
                        "by chance, so counting them tells you\nalmost nothing",
            ha="center", fontsize=10.5, color=RED, linespacing=1.5)
    box(ax, (0.5, 0.545), 0.90, 0.115,
        "individual bars are not reliable", ec=RED, fs=11.5)
    ax.text(0.5, 0.35, "square every bar and add them up",
            ha="center", fontsize=12, color=GREEN)
    # ax.text(0.5, 0.295, "one big spike or many small consistent\n"
    #                     "ones both make the total large",
    #         ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.135), 0.90, 0.115,
        "that total is $Q$", ec=GREEN, fs=12)

    for col, (x, name, c) in enumerate(((wn, "white noise", GREEN),
                                        (ar, r"mild AR(1), $\phi = 0.3$", RED))):
        r = acf(x, M)
        outside = int((np.abs(r[1:]) > band).sum())

        ax = fig.add_subplot(gs[0, col + 1])
        corr_stem(ax, r, n, color=c)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f"{name}\n{outside} of {M} bars outside the band",
                     fontsize=11, color=c)
        if col == 0:
            ax.set_ylabel("ACF")

        ax = fig.add_subplot(gs[1, col + 1])
        Q = np.cumsum(n * (n + 2) * r[1:] ** 2 / (n - h))
        ax.plot(h, Q, color=c, lw=2.4)
        ax.axhline(thr, color=INK, lw=1.6, ls="--")
        ax.set_ylim(0, 62)
        ax.set_xlabel("lag")
        if col == 0:
            ax.set_ylabel("running total of $Q$")
            ax.text(M, thr + 2, f"5% threshold: {thr:.1f}", ha="right", fontsize=9.5,
                    color=INK)
        ax.set_title(rf"$Q = {Q[-1]:.1f}$", fontsize=11, color=c)
    # fig.suptitle("the Ljung-Box intuition: stop judging bars, measure the total",
    #              fontsize=13, y=0.975)
    save(fig, out, "ljung_box_intuition")


def fig_ljung_box_null(out: Path) -> None:
    """What the test actually asks, and why the answer is a chi-squared."""
    from scipy.stats import chi2
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    # ax.text(0.5, 0.96, "tests", ha="center", fontsize=14, color=INK)
    box(ax, (0.5, 0.79), 0.94, 0.17,
        r"$H_0:\;\; \rho(1) = \rho(2) = \dots = \rho(m) = 0$" + "\n"
        "all at once", ec=BLUE, fs=13)
    ax.text(0.5, 0.6, "not \"is lag X significant\"", ha="center", fontsize=11.5,
            color=GREY)
    ax.text(0.5, 0.5, "\"is there any autocorrelation in the first $m$ lags\"",
            ha="center", fontsize=12, color=INK)
    ax.text(0.5, 0.15, "when it rejects, read the correlogram\nto find out where\n(it does not say which lag is guilty)",
            ha="center", fontsize=10.5, color=RED, linespacing=1.4)

    ax = axes[1]
    blank(ax)
    # ax.text(0.5, 0.96, "where the statistic comes from", ha="center", fontsize=13)
    ax.text(0.5, 0.845, r"under $H_0$: $\;\mathrm{Var}[\hat\rho(h)]"
                        r" \approx \dfrac{n-h}{n(n+2)}$", ha="center", fontsize=13.5,
            color=BLUE)
    ax.text(0.5, 0.705, "so standardise and square:", ha="center",
            fontsize=11, color=GREY)
    ax.text(0.5, 0.585, r"$\left(\dfrac{\hat\rho(h)}{\mathrm{sd}}\right)^2"
                        r" = \dfrac{n(n+2)}{n-h}\,\hat\rho(h)^2$",
            ha="center", fontsize=14, color=INK)
    ax.text(0.5, 0.445, "sum up:", ha="center", fontsize=11, color=GREY)
    ax.text(0.5, 0.325, r"$Q = n(n+2)\sum_{h=1}^{m}\dfrac{\hat\rho(h)^2}{n-h}$",
            ha="center", fontsize=16, color=GREEN)
    ax.text(0.5, 0.15, "sum of $m$ squared standard normals,\n"
                        "i.e. a $\\chi^2_m$",
            ha="center", fontsize=12, color=INK, linespacing=1.5)

    ax = axes[2]
    rng = np.random.default_rng(0)
    n, mm, reps = 200, 10, 4000
    hh = np.arange(1, mm + 1)
    Q = np.array([n * (n + 2) * np.sum(acf(rng.normal(0, 1, n), mm)[1:] ** 2
                                       / (n - hh)) for _ in range(reps)])
    ax.hist(Q, bins=60, density=True, color=BLUE, alpha=0.75,
            label=f"simulated\n({reps} white-noise\nseries, $n$ = {n})")
    zz = np.linspace(0, 40, 400)
    ax.plot(zz, chi2.pdf(zz, mm), color=RED, lw=2.6, label=rf"$\chi^2_{{{mm}}}$")
    ax.axvline(chi2.ppf(0.95, mm), color=INK, lw=1.6, ls="--")
    ax.text(chi2.ppf(0.95, mm) + 0.8, ax.get_ylim()[1] * 0.38,
            f"5%: {chi2.ppf(0.95, mm):.1f}", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_xlabel("$Q$")
    ax.set_yticks([])
    # ax.set_title("and it really is one", fontsize=12.5)
    fig.tight_layout()
    save(fig, out, "ljung_box_null")


def fig_ljung_box_recipe(out: Path) -> None:
    """Running it on residuals, with the numbers."""
    n = 220
    t = np.arange(n, dtype=float)
    y = _trend(t, 0.2) + _season(t, amp=5) + ar1(n, 0.6, 1.0, seed=3)
    bad = y - np.polyval(np.polyfit(t, y, 1), t)          # trend only
    good = classical_decompose(y)[2]
    good = good[~np.isnan(good)]

    fig = plt.figure(figsize=WIDE)
    # gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05])
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(0.5, 0.965, "Ljung-Box test", ha="center", fontsize=13.5,
            color=GREEN)
    steps = [
        ("1.", "fit the model, keep the residuals", INK,
         "never run it on the raw series"),
        ("2.", "choose how many lags $m$", INK,
         r"$\approx 10$ non-seasonal, or $\approx 2$ seasons"),
        ("3.", "compute the first $m$ autocorrelations", BLUE,
         r"$\hat\rho(1), \dots, \hat\rho(m)$   of the residuals"),
        ("4.", "form $Q$", GREEN,
         r"$Q = n(n+2)\sum_h \hat\rho(h)^2/(n-h)$"),
        ("5.", "p-value from the upper tail", RED,
         r"$P(\chi^2_{\,m-k} > Q)$"),
    ]
    for i, (num, title, col, detail) in enumerate(steps):
        yy = 0.845 - i * 0.145
        ax.text(0.03, yy, num, fontsize=14, color=col, fontweight="bold",
                va="center")
        ax.text(0.09, yy + 0.028, title, fontsize=11.5, color=INK, va="center")
        ax.text(0.09, yy - 0.040, detail, fontsize=10.5, color=col, va="center")

    # ax = fig.add_subplot(gs[0, 1])
    # blank(ax)
    # ax.text(0.5, 0.965, "one series, two models", ha="center", fontsize=13.5)
    # bad, good = _ljung_demo()
    # rows = [("trend fitted,\nseasonality forgotten", bad, RED),
    #         ("trend + seasonal dummies", good, GREEN)]
    # for i, (name, r, col) in enumerate(rows):
    #     q, df, pv = ljung_box(r, 24, k=0)
    #     y0 = 0.70 - i * 0.40
    #     box(ax, (0.5, y0), 0.95, 0.34, "", ec=col, lw=2.0)
    #     ax.text(0.06, y0 + 0.105, name, fontsize=11.5, color=col, va="center",
    #             linespacing=1.3)
    #     cells = [(r"$m$", "24"), (r"$k$", "0"), ("df", str(df)),
    #              (r"$Q$", f"{q:.1f}"),
    #              (r"$p$", "< 0.001" if pv < 1e-3 else f"{pv:.2f}")]
    #     for j, (lab, val) in enumerate(cells):
    #         x = 0.12 + j * 0.195
    #         ax.text(x, y0 - 0.020, lab, fontsize=10.5, color=GREY, ha="center")
    #         ax.text(x, y0 - 0.095, val, fontsize=13, color=INK, ha="center")
    #     ax.text(0.94, y0 + 0.105,
    #             "reject: not white" if pv < 0.05 else "no evidence against white",
    #             fontsize=11, color=col, ha="right", va="center")
    # ax.text(0.5, 0.015, "a large $p$ is not a pass mark — on a short series the test\n"
    #                     "has little power, so read the correlogram too",
    #         ha="center", fontsize=11, color=RED, linespacing=1.5)
    fig.tight_layout()
    save(fig, out, "ljung_box_recipe")

FIGURES = (
    fig_ts_zoo,
    fig_iid_broken,
    fig_components,
    fig_eq_decomposition,
    fig_add_vs_mult,
    fig_moving_average,
    fig_classical_decomp,
    fig_stl,
    fig_eq_stationarity,
    fig_stationary_zoo,
    fig_why_stationarity,
    fig_random_walk,
    fig_unit_root,
    fig_spurious_regression,
    fig_differencing,
    fig_over_differencing,
    fig_adf_kpss,
    fig_eq_acf,
    fig_acf_build,
    fig_acf_zoo,
    fig_acf_bands,
    fig_pacf_idea,
    fig_acf_pacf_signature,
    fig_ma_edges,
    fig_loess_build,
    fig_loess_span,
    fig_white_noise,
    fig_ar_intro,
    fig_ar_memory,
    fig_random_walk_intro,
    fig_lag_def,
    fig_acf_def,
    fig_df_intuition,
    fig_df_test,
    fig_adf_test,
    fig_kpss_intuition,
    fig_stl_inner,
    fig_stl_outer,
    fig_ma_intro,
    fig_ar_p,
    fig_backshift,
    fig_char_eq,
    fig_root_cases,
    fig_box_cox_lambda,
    fig_adf_recipe,
    fig_kpss_recipe,
    fig_box_cox_why,
    fig_box_cox_choice,
    fig_pacf_def,
    fig_ljung_box_intuition,
    fig_ljung_box_null,
    fig_ljung_box_recipe,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "CourseTS/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
