"""Render the illustration figures for CourseTS/slides2.md (forecasting and learning).

Every figure is synthetic and seeded, so `make figures` reproduces the same output
anywhere without touching `data/`. Shared palette, helpers and estimators live in
cts_common.py.

Usage: make_figures_cTS2.py [outdir]      (default: CourseTS/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cts_common import ( AMBER, BLUE, FAINT, GREEN, GREY, INK, M, PURPLE, RED,
    WIDE, acf, arp, arrow, blank, box, corr_stem, eq, fit_ar, items,
    ljung_box, maq, random_walk, save, series, tidy,
)
from matplotlib.patches import Rectangle

TEAL = "#0f766e"


# ======================================================== 4. forecasting setup
def fig_forecast_task(out: Path) -> None:
    """Vocabulary: origin, horizon, information set."""
    t, y = series(120, seed=33)
    T = 96
    lo, hi = y.min(), y.max()
    span = hi - lo

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(t[:T], y[:T], color=BLUE, lw=1.8)
    ax.plot(t[T - 1:], y[T - 1:], color=GREY, lw=1.6, alpha=0.55)
    # the marker stops below the text band, so the caption never crosses it
    ax.axvline(T - 1, ymin=0, ymax=0.76, color=INK, lw=1.6, ls="--")
    ax.axvspan(T - 1, t[-1], ymax=0.76, color=FAINT, alpha=0.7, zorder=0)

    # all the text lives in a clear band above the data, never on top of it
    ax.set_ylim(lo - 0.30 * span, hi + 0.62 * span)
    ax.set_xlim(t[0] - 2, t[-1] + 12)

    ax.text(0.5, 0.97, "everything to the right of the line must be unknown when fitting",
            transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
            color=RED, linespacing=1.5)
    ax.text(T - 3, hi + 0.16 * span, "forecast origin $T$", ha="right",
            fontsize=12.5)
    ax.text(T + 3, hi + 0.16 * span, r"horizon $h = 1, 2, \dots, H$", ha="left",
            fontsize=12.5, color=GREY)
    ax.annotate("", xy=(t[-1], hi + 0.05 * span), xytext=(T - 1, hi + 0.05 * span),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))

    ax.text(0.015, 0.06,
            r"$\hat y_{T+h|T} = \mathbb{E}[\,y_{T+h} \mid y_1,\dots,y_T\,]$",
            transform=ax.transAxes, fontsize=14, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=FAINT))
    tidy(ax, "t")
    fig.tight_layout()
    save(fig, out, "forecast_task")


def fig_baselines(out: Path) -> None:
    """The four forecasts nobody publishes and everybody should beat."""
    t, y = series(132, seed=41)
    T = 108
    h = np.arange(T, len(t))
    tr = y[:T]

    mean_f = np.full(len(h), tr.mean())
    naive_f = np.full(len(h), tr[-1])
    snaive_f = tr[-M:][(h - T) % M]
    drift = (tr[-1] - tr[0]) / (T - 1)
    drift_f = tr[-1] + drift * (h - T + 1)

    fig, axes = plt.subplots(1, 4, figsize=WIDE, sharey=True)
    for ax, (f, name, col, formula) in zip(axes, [
        (mean_f, "mean", AMBER, r"$\hat y = \bar y$"),
        (naive_f, "naive", BLUE, r"$\hat y = y_T$"),
        (snaive_f, "seasonal naive", GREEN, r"$\hat y = y_{T+h-m}$"),
        (drift_f, "drift", PURPLE, r"$\hat y = y_T + h\,\frac{y_T - y_1}{T-1}$"),
    ]):
        ax.plot(t[:T], tr, color=GREY, lw=1.2)
        ax.plot(t[T:], y[T:], color=INK, lw=1.4, alpha=0.5)
        ax.plot(t[T:], f, color=col, lw=2.6)
        mae = np.abs(f - y[T:]).mean()
        ax.set_title(f"{name}\n{formula}\nMAE {mae:.2f}", fontsize=11.5)
        ax.set_xlabel("t")
    axes[0].set_ylabel("value")
    # fig.suptitle("four one-line forecasts", fontsize=13)
    fig.tight_layout()
    save(fig, out, "baselines")


def fig_naive_def(out: Path) -> None:
    """What the naive forecast actually is, and what its errors turn out to be."""
    t, y = series(132, seed=41)
    lo = 72                                   # zoom: the one-step offset must be visible
    d1 = np.diff(y)
    dm = y[M:] - y[:-M]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.985, "the naive forecast", ha="center", fontsize=14, color=BLUE)
    ax.text(0.5, 0.835, "whatever happened last,\nassume it keeps happening",
            ha="center", fontsize=12, color=INK, linespacing=1.5)
    ax.text(0.5, 0.715, r"$\hat y_{T+h|T} = y_T$   for every $h$", ha="center",
            fontsize=15, color=BLUE)
    ax.text(0.5, 0.615, "one flat line, however far ahead you look",
            ha="center", fontsize=10.5, color=GREY)
    box(ax, (0.5, 0.475), 0.86, 0.115,
        "no parameters, nothing fitted", ec=BLUE, fs=11.5)
    ax.text(0.5, 0.345, "seasonal naive", ha="center", fontsize=13, color=GREEN)
    ax.text(0.5, 0.235, r"$\hat y_{T+h|T} = y_{T+h-m}$", ha="center", fontsize=15,
            color=GREEN)
    ax.text(0.5, 0.135, "the same point in the previous cycle",
            ha="center", fontsize=10.5, color=GREY)

    for ax, (fc, err, name, col, lab) in zip(axes[1:], (
            (y[lo - 1:-1], d1[lo - 1:], "naive", BLUE, r"$\Delta y_t$"),
            (y[lo - M:len(y) - M], dm[lo - M:], "seasonal naive", GREEN,
             r"$\Delta_{12} y_t$"))):
        xs = t[lo:]
        ax.plot(xs, y[lo:], color=INK, lw=1.8, label="actual")
        ax.plot(xs, fc, color=col, lw=1.8, ls="--", label=f"{name} forecast")
        ax.vlines(xs, fc, y[lo:], color=RED, lw=1.0, alpha=0.65)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.set_xlabel("t")
        ax.set_title(f"{name}: shift the series along\n"
                     f"MAE {np.abs(err).mean():.2f}", fontsize=11.5, color=col)
        # clear band under the data so the caption is never drawn over the series
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo - 0.20 * (yhi - ylo), yhi)
        ax.text(0.5, 0.035, "red = the errors, and they are exactly  " + lab,
                transform=ax.transAxes, ha="center", fontsize=10.5, color=RED)
    axes[1].set_ylabel("value")
    fig.tight_layout()
    save(fig, out, "naive_def")


def fig_naive_strong(out: Path) -> None:
    """Why naive wins so often, and why "it tracks the data" proves nothing."""
    n = 160
    y = 50 + random_walk(n, sigma=1.0, seed=77)
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    ax.plot(y, color=INK, lw=1.6, label="series")
    ax.plot(np.arange(1, n), y[:-1], color=RED, lw=1.6, ls="--",
            label="naive forecast")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("looks like an excellent model", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    ax.scatter(y[:-1], y[1:], s=10, color=BLUE, alpha=0.6)
    ax.set_title(r"$R^2 = $" + f"{np.corrcoef(y[:-1], y[1:])[0,1]**2:.3f}"
                 + "\non the levels", fontsize=12.5)
    ax.set_xlabel(r"$y_{t-1}$")
    ax.set_ylabel(r"$y_t$")

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.88, "In case of a random walk,\nnaive forecast is optimal", ha="center", fontsize=14, color=RED)
    ax.text(0.5, 0.60, "naive forecast has no skill", ha="center", fontsize=12.5, color=INK, linespacing=1.5)
    fig.tight_layout()
    save(fig, out, "naive_strong")


def fig_metrics(out: Path) -> None:
    """The metric zoo, and the one trap in each."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    e = np.linspace(-4, 4, 400)
    ax = axes[0]
    ax.plot(e, np.abs(e), color=BLUE, lw=2.6, label="MAE: $|e|$")
    ax.plot(e, e ** 2 / 2, color=RED, lw=2.6, label="MSE: $e^2$")
    ax.legend(frameon=False, fontsize=11.5)
    ax.set_xlabel("error $e$")
    ax.set_ylabel("penalty")
    ax.set_title("MAE targets the median,\nRMSE targets the mean", fontsize=12)

    ax = axes[1]
    yy = np.linspace(0.2, 10, 300)
    ax.plot(yy, 100 * 1.0 / yy, color=AMBER, lw=2.6)
    ax.set_ylim(0, 260)
    ax.set_xlabel("actual value $y$")
    ax.set_ylabel("MAPE (%) for a fixed error of 1")
    ax.set_title("MAPE explodes near zero,\nand punishes over-forecasts harder",
                 fontsize=12)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.92, "so which one", ha="center", fontsize=13.5)
    items(ax, [
        ("RMSE", "same units, dominated by the worst points"),
        ("MAE", "same units, robust, not comparable across series"),
        ("MAPE", "scale-free, but needs $y \\gg 0$ and is asymmetric"),
        # ("MASE", "scale-free, safe at zero, baseline built in"),
    ], top=0.72, step=0.18)
    # ax.text(0.5, 0.02, "never compare a metric across different test periods",
    #         ha="center", fontsize=11.5, color=RED)
    fig.tight_layout()
    save(fig, out, "metrics")


def fig_eq_mase(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.5, 0.93, "scale the error by the baseline's error", ha="center",
            fontsize=13.5)
    eq(ax, 0.72, r"$\mathrm{MASE} \;=\; \dfrac{\frac{1}{H}\sum_h |y_{T+h} - \hat y_{T+h}|}"
                 r"{\frac{1}{n-m}\sum_{t>m} |y_t - y_{t-m}|}$", fs=22, color=BLUE)
    ax.text(0.5, 0.48, "numerator: your error on the test set", ha="center",
            fontsize=12, color=GREY)
    ax.text(0.5, 0.40, "denominator: the seasonal naive error on the TRAINING set",
            ha="center", fontsize=12, color=GREY)
    for x, txt, col in [
        (0.22, "MASE < 1\nbetter than naive", GREEN),
        (0.50, "MASE = 1\nno skill", AMBER),
        (0.78, "MASE > 1\nworse than naive", RED),
    ]:
        box(ax, (x, 0.18), 0.24, 0.17, txt, ec=col, fs=12, tc=col)
    save(fig, out, "eq_mase")


def fig_backtest(out: Path) -> None:
    """Rolling origin: the only honest way to score a forecaster."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    n = 30
    for ax, (title, expanding) in zip(axes, [("expanding window", True),
                                             ("sliding window", False)]):
        blank(ax)
        ax.set_title(title, fontsize=13)
        for k in range(5):
            y0 = 0.82 - k * 0.165
            end = 14 + k * 3
            start = 0 if expanding else end - 14
            ax.add_patch(Rectangle((0.06 + start * 0.028, y0 - 0.045),
                                   (end - start) * 0.028, 0.09,
                                   facecolor=BLUE, alpha=0.75, lw=0))
            ax.add_patch(Rectangle((0.06 + end * 0.028, y0 - 0.045), 3 * 0.028, 0.09,
                                   facecolor=RED, alpha=0.85, lw=0))
            ax.text(0.03, y0, f"fold {k+1}", fontsize=10.5, ha="right", va="center",
                    color=GREY)
        ax.text(0.30, 0.95, "train", color=BLUE, fontsize=12, ha="center")
        ax.text(0.62, 0.95, "test", color=RED, fontsize=12, ha="center")
        ax.text(0.5, 0.02,
                "every test block is strictly after its training block"
                if expanding else "old data is dropped: use it when the process drifts",
                ha="center", fontsize=11.5, color=INK)
    fig.suptitle("Refit at each origin",
                 fontsize=13)
    fig.tight_layout()
    save(fig, out, "backtest")


def fig_cv_wrong(out: Path) -> None:
    """Random k-fold on a time series, and the size of the lie."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    rng = np.random.default_rng(1)
    n = 60
    ax = axes[0]
    blank(ax)
    ax.set_title("random 5-fold", fontsize=12.5, color=RED)
    fold = rng.integers(0, 5, n)
    for k in range(5):
        for i in range(n):
            ax.add_patch(Rectangle((0.05 + i * 0.0148, 0.76 - k * 0.15), 0.0135, 0.10,
                                   facecolor=RED if fold[i] == k else BLUE,
                                   alpha=0.85, lw=0))
    ax.text(0.5, 0.06, "test points sit BETWEEN training points", ha="center",
            fontsize=11.5, color=RED)

    ax = axes[1]
    blank(ax)
    ax.set_title("leaks", fontsize=12.5)
    items(ax, [
        ("neighbouring points are nearly identical",
         "predicting $y_t$ from $y_{t-1}$ and $y_{t+1}$ is interpolation"),
        ("the model sees the future",
         "of every test point, in every fold"),
        ("rolling statistics cross the boundary",
         "a 7-step mean contains test rows"),
        # ("scaling on the full series",
        #  "the mean and std already leaked"),
    ], top=0.76, step=0.19)

    ax = axes[2]
    xs = ["random\n5-fold", "rolling\norigin"]
    vals = [0.42, 1.31]
    ax.bar(xs, vals, color=[RED, GREEN], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=13)
    ax.set_ylabel("reported test MAE")
    ax.set_title("same model, same data", fontsize=12.5)
    ax.set_ylim(0, 1.6)
    fig.tight_layout()
    save(fig, out, "cv_wrong")


def fig_intervals(out: Path) -> None:
    """Uncertainty compounds with the horizon, and the interval is the deliverable."""
    n, T, H = 120, 96, 24
    t, y = series(n, seed=52)
    tr = y[:T]
    phi = 0.7
    sig = 2.2
    h = np.arange(1, H + 1)
    point = tr[-1] + 0.35 * h
    sd = sig * np.sqrt(np.cumsum(phi ** (2 * (h - 1))))

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    ax.plot(t[:T], tr, color=BLUE, lw=1.6)
    ax.plot(t[T:], point, color=RED, lw=2.4, label="point forecast")
    for z, a in [(1.28, 0.30), (1.96, 0.18)]:
        ax.fill_between(t[T:], point - z * sd, point + z * sd, color=RED, alpha=a, lw=0)
    ax.plot(t[T:], y[T:], color=INK, lw=1.2, alpha=0.6, label="what happened")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title("80% and 95% intervals", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    ax.plot(h, sd, color=PURPLE, lw=2.8)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("forecast standard deviation")
    ax.set_title("uncertainty accumulates with $h$", fontsize=12.5)
    fig.tight_layout()
    save(fig, out, "intervals")


# =================================================================== 5. ARIMA
def arma11(n, phi, th, seed=0, burn=500):
    """One realisation of y_t = phi y_{t-1} + eps_t + theta eps_{t-1}."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n + burn)
    y = np.zeros(n + burn)
    for i in range(1, n + burn):
        y[i] = phi * y[i - 1] + e[i] + th * e[i - 1]
    return y[burn:]


def fig_arma_recap(out: Path) -> None:
    """The two building blocks from part 1, side by side, before combining them."""
    n = 220
    ar = arp(n, (0.75,), seed=61)
    ma = maq(n, (0.9, 0.6), seed=3)

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.28,
                          top=0.84, bottom=0.10, left=0.05, right=0.97)

    for col, (x, name, eqn, memory, col_) in enumerate((
            (ar, "AR(p)", r"$y_t = \phi_1 y_{t-1} + \dots + \phi_p y_{t-p}"
                          r" + \varepsilon_t$",
             "regress on your own past", BLUE),
            (ma, "MA(q)", r"$y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}"
                          r" + \dots + \theta_q \varepsilon_{t-q}$",
             "add up the last few shocks", GREEN))):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(x, color=col_, lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{memory}", fontsize=12, color=col_)
        ax.text(0.5, -0.17, eqn, transform=ax.transAxes, ha="center", fontsize=10.5,
                color=col_)

        ax = fig.add_subplot(gs[1, col])
        corr_stem(ax, acf(x, 18), n, color=col_)
        ax.set_ylim(-0.5, 1.0)
        ax.set_title("ACF tails off" if col == 0 else "ACF cuts off at $q$",
                     fontsize=11, color=col_)
        if col == 0:
            ax.set_ylabel("ACF")

    ax = fig.add_subplot(gs[:, 2])
    blank(ax)
    items(ax, [
        ("AR: long, fading memory", r"a shock decays as $\phi^h$, forever"),
        ("MA: short, exact memory", r"a shock is gone after $q$ steps"),
        ("AR needs a stationarity check", "the roots of $\\Phi(z)$"),
        ("MA is always stationary", "no conditions on $\\theta$"),
    ], top=0.80, step=0.145)
    box(ax, (0.5, 0.15), 0.94, 0.165,
        "real series have both:\na fading level AND short shocks", ec=AMBER, fs=12)
    save(fig, out, "arma_recap")


def fig_arima_intuition(out: Path) -> None:
    """Why bother with MA terms? Because AR alone needs far more of them."""
    y = arma11(600, 0.6, 0.7, seed=3)
    ps = [1, 2, 3, 5, 8, 12]
    rows = []
    for pp in ps:
        _, _, r = fit_ar(y, pp)
        q, df, pv = ljung_box(r, 20, k=pp)
        rows.append((pp, q, pv))
    first_white = next(pp for pp, _, pv in rows if pv > 0.05)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.985, "AR can imitate MA", ha="center", fontsize=14, color=INK)
    ax.text(0.5, 0.835, "but needs many terms", ha="center", fontsize=12, color=INK, linespacing=1.5)
    ax.text(0.5, 0.695, r"$y_t = 0.6\, y_{t-1} + \varepsilon_t + 0.7\, \varepsilon_{t-1}$",
            ha="center", fontsize=13, color=AMBER)
    ax.text(0.5, 0.60, "ARMA(1,1):  two parameters", ha="center", fontsize=11,
            color=GREY)

    ax = axes[1]
    xs = np.arange(len(ps))
    cols = [GREEN if pv > 0.05 else RED for _, _, pv in rows]
    ax.bar(xs, [q for _, q, _ in rows], color=cols, width=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f"AR({p})" for p in ps])
    ax.set_ylabel("residual Ljung-Box $Q(20)$")
    for x, (_, q, _) in zip(xs, rows):
        ax.text(x, q + 4, f"{q:.0f}", ha="center", fontsize=10.5)
    ax.set_title("red: residuals autocorrelated\ngreen: white noise", fontsize=11.5)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.965, "MA brings parsimony", ha="center",
            fontsize=12.5, color=GREEN)
    for x, lab, val, col in ((0.28, "ARMA(1,1)", "2", GREEN),
                             (0.72, f"AR({first_white})", str(first_white), RED)):
        box(ax, (x, 0.66), 0.38, 0.30, "", ec=col, lw=2.0)
        ax.text(x, 0.75, lab, ha="center", fontsize=13, color=col)
        ax.text(x, 0.605, f"{val} parameters", ha="center", fontsize=12)
    items(ax, [
        ("fewer parameters, hence less variance", "each one is estimated from the same $n$"),
        ("easier to interpret", "two numbers instead of eight"),
    ], top=0.30, step=0.155)
    fig.tight_layout()
    save(fig, out, "arima_intuition")


def fig_eq_arima(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    # ax.text(0.5, 0.95, "ARIMA(p, d, q)", ha="center", fontsize=14)
    eq(ax, 0.76, r"$\phi_p(B)\,(1-B)^d\, y_t \;=\; \theta_q(B)\,\varepsilon_t$",
       fs=26, color=INK)
    labels = [
        (0.265, "AR part\n$p$ lags of $y$", BLUE),
        (0.435, "differencing\n$d$ times", AMBER),
        (0.665, "MA part\n$q$ lags of $\\varepsilon$", GREEN),
    ]
    for x, txt, col in labels:
        ax.annotate("", xy=(x, 0.70), xytext=(x, 0.57),
                    arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.8})
        ax.text(x, 0.475, txt, ha="center", fontsize=11.5, color=col, linespacing=1.5)
    ax.text(0.5, 0.31, r"$B$ is the backshift operator:   $B y_t = y_{t-1}$,"
                       r"   $(1-B) y_t = y_t - y_{t-1}$",
            ha="center", fontsize=13, color=GREY)
    ax.text(0.5, 0.0,
            "'I' is a preprocessing step:\n"
            "difference until stationary\n"
            "(and undo the differencing on the forecast)",
            ha="center", fontsize=12, color=INK, linespacing=1.6)
    save(fig, out, "eq_arima")


def fig_arima_pipeline(out: Path) -> None:
    """d first, then p and q — the order matters and explains the letters."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    t, y = series(200, seed=71)
    d = np.diff(y[M:] - y[:-M])

    axes[0].plot(t, y, color=RED, lw=1.3)
    axes[0].set_title(r"$y_t$: not stationary" + "\nchoose $d$", fontsize=12)
    tidy(axes[0], "t")

    axes[1].plot(d, color=GREEN, lw=1.1)
    axes[1].axhline(0, color=GREY, lw=0.9)
    axes[1].set_title(r"$\Delta^d y_t$: stationary" + "\nfit ARMA",
                      fontsize=12)
    tidy(axes[1], "t")

    corr_stem(axes[2], acf(d, 24), len(d), title="ACF/PACF tells you $q$/$p$")
    axes[2].set_ylabel("ACF")
    # fig.suptitle("ARIMA fits a stationary model to a differenced series, "
    #              "then integrates the forecast back up", fontsize=13)
    fig.tight_layout()
    save(fig, out, "arima_pipeline")


def fig_sarima_problem(out: Path) -> None:
    """Plain ARIMA cannot reach back a whole season without paying for every lag."""
    t, y = series(240, seed=71)
    d1 = np.diff(y)
    ps = [1, 3, 6, 9, 12]
    rho12 = []
    for pp in ps:
        _, _, r = fit_ar(d1, pp)
        rho12.append(acf(r, 13)[12])
    _, _, r3 = fit_ar(d1, 3)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    ax.plot(t, y, color=INK, lw=1.0)
    ax.set_title("a series with a season of 12\nfit ARIMA(3,1,0)", fontsize=11.5)
    tidy(ax, "t")

    ax = axes[1]
    a = acf(r3, 26)
    corr_stem(ax, a, len(r3), color=GREY)
    for k in (12, 24):
        ax.vlines(k, 0, a[k], color=RED, lw=3.0)
        ax.plot([k], [a[k]], "o", color=RED, ms=6)
        ax.text(k, a[k] + 0.07, f"{a[k]:+.2f}", ha="center", fontsize=10.5,
                color=RED)
    ax.set_ylim(-0.5, 0.75)
    ax.set_ylabel("ACF of the residuals")
    ax.set_title("seasonality remains\nspikes at 12 and 24", fontsize=11.5,
                 color=RED)

    ax = axes[2]
    xs = np.arange(len(ps))
    cols = [RED if abs(v) > 0.25 else GREEN for v in rho12]
    ax.bar(xs, rho12, color=cols, width=0.6)
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p={p_}" for p_ in ps])
    ax.set_ylabel(r"residual $\rho(12)$")
    ax.set_ylim(-0.3, 0.72)
    for x, v in zip(xs, rho12):
        ax.text(x, v + (0.04 if v > 0 else -0.08), f"{v:+.2f}", ha="center",
                fontsize=10.5)
    ax.set_title("raising $p$ does not help\nuntil $p$ reaches the season itself",
                 fontsize=11.5)
    ax.text(0.34, 0.09, "12 parameters to\ncapture one yearly effect",
            transform=ax.transAxes, ha="center", fontsize=11, color=RED,
            linespacing=1.4)
    # fig.suptitle("the problem ARIMA has with a seasonal component", fontsize=13)
    fig.tight_layout()
    save(fig, out, "sarima_problem")


def fig_sarima(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.92, "SARIMA$(p,d,q)(P,D,Q)_m$", ha="center", fontsize=16, color=INK)
    eq(ax, 0.72, r"$\Phi_P(B^m)\,\phi_p(B)\,(1-B^m)^D(1-B)^d\, y_t"
                 r" \;=\; \Theta_Q(B^m)\,\theta_q(B)\,\varepsilon_t$", fs=15)
    ax.text(0.5, 0.56, "same three parameter, but in steps of $m$",
            ha="center", fontsize=12, color=GREY)
    items(ax, [
        (r"$m$", "the period: 12 monthly, 7 daily, 24 hourly"),
        (r"$D$", "seasonal differencing, almost always 0 or 1"),
        (r"$P, Q$", "seasonal AR and MA, read off the ACF at lags $m, 2m, \\dots$"),
    ], top=0.38, step=0.155, fs=12.5)

    ax = axes[1]
    d = np.diff(series(300, seed=81)[1])
    corr_stem(ax, acf(d, 40), 300, title="ACF before modeling seasonality")
    for k in (12, 24, 36):
        ax.axvline(k, color=RED, lw=1.0, ls=":")
    ax.set_ylabel("ACF")
    ax.text(0.5, 0.06, "the spikes at $m$, $2m$, $3m$ yield seasonal component",
            transform=ax.transAxes, ha="center", fontsize=11.5, color=RED)
    fig.tight_layout()
    save(fig, out, "sarima")


def fig_box_jenkins(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    # evenly spaced with real gaps; the last box used to run off the right edge
    xs = (0.135, 0.375, 0.615, 0.855)
    w, top, bot = 0.180, 0.62, 0.40
    stages = [
        ("1. identify", "plot, transform,\ndifference, read\nACF and PACF", BLUE),
        ("2. estimate", "maximum likelihood\nfor the chosen\n(p, d, q)", GREEN),
        ("3. diagnose", "residuals: white?\nLjung-Box,\nnormal QQ", AMBER),
        ("4. forecast", "point + interval,\nback-transformed", PURPLE),
    ]
    for x, (title, body, col) in zip(xs, stages):
        box(ax, (x, top), w, 0.40, "", ec=col, lw=2.0)
        ax.text(x, top + 0.135, title, ha="center", fontsize=13, color=col)
        ax.text(x, top - 0.045, body, ha="center", fontsize=10.5, color=INK,
                linespacing=1.5)
    for a_, b_ in zip(xs[:-1], xs[1:]):
        arrow(ax, (a_ + w / 2 + 0.005, top), (b_ - w / 2 - 0.005, top))

    # the feedback arrow runs BELOW the boxes, bowing downwards
    ax.annotate("", xy=(xs[0], bot - 0.03), xytext=(xs[2], bot - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text((xs[0] + xs[2]) / 2, 0.055,
            "if the residuals are not white noise, update the orders",
            ha="center", fontsize=12.5, color=RED)
    # ax.text(0.5, 0.955, "the Box-Jenkins loop", ha="center", fontsize=14)
    save(fig, out, "box_jenkins")


def fig_ic_motivation(out: Path) -> None:
    """Why a penalty is needed at all: fit alone always picks the biggest model."""
    x = arp(400, (0.6, 0.25), seed=91)
    tr, te = x[:300], x[300:]
    ps = np.arange(0, 11)
    rss, oos = [], []
    for pp in ps:
        if pp == 0:
            r = tr - tr.mean()
            pred = np.full(len(te), tr.mean())
        else:
            c, phis, r = fit_ar(tr, pp)
            X = np.column_stack([x[300 - j - 1:400 - j - 1] for j in range(pp)])
            pred = c + X @ phis
        rss.append(float(r @ r))
        oos.append(float(np.mean((te - pred) ** 2)))
    rss, oos = np.array(rss), np.array(oos)
    best = int(oos.argmin())

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.9, "Adding a lag never makes fit worse", ha="center", fontsize=12, color=INK, linespacing=1.5)
    # ax.text(0.5, 0.685, "the old model is still available:\n"
    #                     "set the new coefficient to zero",
    #         ha="center", fontsize=10.5, color=GREY, linespacing=1.5)
    box(ax, (0.5, 0.65), 0.94, 0.135, "\"most parameters\" implies \"best fit\"", ec=RED, fs=11.5)
    # ax.text(0.5, 0.375, "but the fit is measured on data\nyou have already seen",
    #         ha="center", fontsize=11.5, color=INK, linespacing=1.5)
    ax.text(0.5, 0.375, "For best generalization,\nfewer parameters is better",
            ha="center", fontsize=11.5, color=INK, linespacing=1.5)
    box(ax, (0.5, 0.195), 0.94, 0.16,
        "Charge for each parameter:\nUsing AIC or BIC", ec=GREEN, fs=11.5)

    ax = axes[1]
    ax.plot(ps, rss, "o-", color=RED, lw=2.4, ms=5)
    # p = 0 sits at ~600 and would flatten the monotone decline that is the point
    ax.set_ylim(232, 272)
    ax.set_xlabel("AR order p")
    ax.set_ylabel("in-sample error")
    # ax.set_title("fit on the training data\nnever stops improving", fontsize=11.5, color=RED)
    ax.annotate("", xy=(9.7, rss[-1] + 1), xytext=(9.7, rss[3] - 1),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(9.4, (rss[3] + rss[-1]) / 2, "always\ndown", ha="right", fontsize=10,
            color=RED, va="center", linespacing=1.3)

    ax = axes[2]
    ax.plot(ps, oos, "o-", color=GREEN, lw=2.4, ms=5)
    ax.axvline(best, color=INK, lw=1.4, ls="--")
    ax.plot([best], [oos[best]], "o", color=INK, ms=11, mfc="none", mew=2.0)
    ax.set_xlabel("AR order p")
    ax.set_ylabel("error on held-out data")
    # ax.set_title(f"but out of sample it turns up\nbest at p = {best}", fontsize=11.5, color=GREEN)
    ax.text(0.35, 0.93, "best", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=INK, linespacing=1.4)
    fig.tight_layout()
    save(fig, out, "ic_motivation")


def fig_aic_bic(out: Path) -> None:
    """Model selection by information criterion, and what auto_arima automates."""
    # fig, axes = plt.subplots(1, 3, figsize=WIDE)
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    rng = np.random.default_rng(5)
    x = arp(400, (0.6, 0.25), seed=91)
    orders = np.arange(0, 9)
    aic, bic = [], []
    for p in orders:
        if p == 0:
            r = x - x.mean()
            k = 1
        else:
            _, _, r = fit_ar(x, p)
            k = p + 1
        n = len(r)
        ll = -0.5 * n * (np.log(2 * np.pi * r.var()) + 1)
        aic.append(-2 * ll + 2 * k)
        bic.append(-2 * ll + np.log(n) * k)
    aic, bic = np.array(aic), np.array(bic)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.965, "Information criteria", ha="center", fontsize=13.5)
    eq(ax, 0.865, r"$\mathrm{AIC} = -2\log L + 2k$", fs=16, color=BLUE)
    ax.text(0.5, 0.795, "Akaike Information Criterion", ha="center", fontsize=10,
            color=BLUE)
    eq(ax, 0.685, r"$\mathrm{BIC} = -2\log L + k\log n$", fs=16, color=PURPLE)
    ax.text(0.5, 0.615, "Bayesian Information Criterion", ha="center", fontsize=10,
            color=PURPLE)
    ax.text(0.5, 0.495, r"$L$: likelihood", ha="center", fontsize=12.5)
    ax.text(0.5, 0.425, '"how probable the data is under the model"',
            ha="center", fontsize=10, color=GREY)
    ax.text(0.5, 0.335, r"$k$ parameters fitted        $n$ observations",
            ha="center", fontsize=11.5)
    ax.text(0.5, 0.215, "BIC picks smaller models\nAIC tends to over-select",
            ha="center", fontsize=11, color=PURPLE)
    ax.text(0.5, 0.05, "BIC penalises harder than AIC",
            ha="center", fontsize=10.5, color=INK, linespacing=1.5)

    ax = axes[1]
    ax.plot(orders, aic - aic.min(), "o-", color=BLUE, lw=2.2, label="AIC")
    ax.plot(orders, bic - bic.min(), "s-", color=PURPLE, lw=2.2, label="BIC")
    ax.axvline(2, color=GREEN, lw=1.4, ls="--")
    # p = 0 is ~330 above the minimum and would flatten everything else
    ax.set_ylim(-9, 38)
    ax.text(2.2, 33, "true\np = 2", color=GREEN, fontsize=11, linespacing=1.3)
    # ax.text(0.1, 6.5, f"p = 0 is off the scale\n(AIC {aic[0]-aic.min():.0f} above)",
    #         fontsize=9.5, color=GREY, linespacing=1.3)
    for crit, col, nm, dy in ((aic, BLUE, "AIC", -7.0), (bic, PURPLE, "BIC", -7.0)):
        k = int(crit.argmin())
        ax.plot([k], [0], "o", color=col, ms=13, mfc="none", mew=2.2)
        ax.annotate(f"{nm} picks {k}", xy=(k, -0.8), xytext=(k, dy),
                    ha="center", fontsize=10.5, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel("AR order p")
    ax.set_ylabel("criterion − its minimum")
    # ax.set_title("BIC finds the true order; AIC overshoots", fontsize=12.5)

    # ax = axes[2]
    # blank(ax)
    # ax.text(0.5, 0.88, "AutoARIMA", ha="center", fontsize=13.5,
    #         color=GREEN)
    # items(ax, [
    #     ("tests for $d$ and $D$", "KPSS, and a seasonal strength test"),
    #     ("steps through (p, q)", "a greedy walk, not a full grid"),
    #     ("keeps the best AICc", "corrected for small samples"),
    #     ("you still read the residuals", "automation picks orders, not sense"),
    # ], top=0.70, step=0.175)
    fig.tight_layout()
    save(fig, out, "aic_bic")


def fig_residual_diag(out: Path) -> None:
    """The four-panel residual check you run after every fit."""
    n = 300
    x = arp(n, (0.7, 0.2), seed=101)
    _, _, r = fit_ar(x, 2)
    fig, axes = plt.subplots(1, 4, figsize=WIDE)
    axes[0].plot(r, color=INK, lw=0.9)
    axes[0].axhline(0, color=GREY, lw=0.9)
    axes[0].set_title("residuals over time\nno pattern, no drift", fontsize=11.5)
    axes[0].set_xlabel("t")

    corr_stem(axes[1], acf(r, 24), len(r), title="ACF\ninside the band")

    axes[2].hist(r, bins=24, color=BLUE, alpha=0.75)
    axes[2].set_title("histogram\nroughly symmetric", fontsize=11.5)
    axes[2].set_yticks([])

    q = np.sort(r)
    from scipy.stats import norm
    theo = norm.ppf((np.arange(len(q)) + 0.5) / len(q)) * r.std()
    axes[3].scatter(theo, q, s=8, color=PURPLE, alpha=0.7)
    lim = [theo.min(), theo.max()]
    axes[3].plot(lim, lim, color=GREY, lw=1.2, ls="--")
    axes[3].set_title("QQ plot\ntails are where it fails", fontsize=11.5)
    axes[3].set_xlabel("theoretical")
    axes[3].set_ylabel("observed")
    fig.suptitle("if the residuals still have structure, the model has left signal on the table",
                 fontsize=13)
    fig.tight_layout()
    save(fig, out, "residual_diag")


def fig_arimax(out: Path) -> None:
    """What ARIMAX adds: a regressor you already know the future of."""
    rng = np.random.default_rng(11)
    n = 240
    events = [(c, c + 8) for c in (40, 95, 150, 205)]
    x = np.zeros(n)
    for a_, b_ in events:
        x[a_:b_] = 1.0
    y = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + 4.0 * x[i] + e[i]

    lag, tgt, xx = y[:-1], y[1:], x[1:]
    t = np.arange(1, n, dtype=float)

    def ols(X):
        b, *_ = np.linalg.lstsq(X, tgt, rcond=None)
        return b, tgt - X @ b

    _, r_plain = ols(np.column_stack([np.ones(n - 1), lag]))
    beta, r_x = ols(np.column_stack([np.ones(n - 1), lag, xx]))

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.975, "ARIMAX adds one ordinary regressor",
            ha="center", fontsize=13, color=GREY)
    ax.text(0.5, 0.75, r"$\phi_p(B)\,(1-B)^d\, y_t = \theta_q(B)\,\varepsilon_t$",
            ha="center", fontsize=13, color=GREY)
    ax.text(0.5, 0.675, "becomes", ha="center", fontsize=10.5, color=GREY)
    ax.text(0.5, 0.55, r"$\phi_p(B)\,(1-B)^d\,\left(y_t - \beta^{\top} x_t\right)"
                        r" = \theta_q(B)\,\varepsilon_t$",
            ha="center", fontsize=13, color=BLUE)
    box(ax, (0.5, 0.35), 0.96, 0.145,
        "ARIMA now models what is\nleft after the regression",
        ec=BLUE, fs=11)
    ax.text(0.5, 0.02, r"$x_t$ can be anything known:"
                       "\nevents, prices, weather, policy changes",
            ha="center", fontsize=10.5, color=GREY, linespacing=1.5)

    for ax, r, name, col in ((axes[1], r_plain, "AR only", RED),
                             (axes[2], r_x, r"AR $+\; x_t$", GREEN)):
        for a_, b_ in events:
            ax.axvspan(a_, b_, color=AMBER, alpha=0.30, lw=0)
        ax.plot(t, r, color=col, lw=1.1)
        ax.axhline(0, color=GREY, lw=0.9)
        ax.set_ylim(-4.5, 6.5)
        ax.set_xlabel("t")
        c = np.corrcoef(r, xx)[0, 1]
        ax.set_title(f"{name}: residual sd {r.std():.2f}\n"
                     rf"corr(residual, $x$) = {c:+.2f}", fontsize=11.5, color=col)
    axes[1].set_ylabel("residual")
    axes[1].text(0.5, 0.94, "shaded: the events.",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=10,
                 color=RED, linespacing=1.4)
    axes[2].text(0.5, 0.94, rf"$\hat\beta$ = {beta[2]:.2f} (true 4.00)" + "\n",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=10,
                 color=GREEN, linespacing=1.4)
    axes[2].text(0.5, 0.06, "$x$ be known for predictions",
                 transform=axes[2].transAxes, ha="center", fontsize=10.5, color=RED)
    fig.suptitle("same model, with something already known", fontsize=13)
    fig.tight_layout()
    save(fig, out, "arimax")


def fig_arima_limits(out: Path) -> None:
    """Where the classical machinery runs out."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    n, T = 160, 120
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(13)
    y = 20 + 0.25 * t + np.where(t > 70, 9.0, 0.0) + rng.normal(0, 1.2, n)

    ax = axes[0]
    ax.plot(t[:T], y[:T], color=BLUE, lw=1.6)
    ax.plot(t[T:], y[T:], color=GREY, lw=1.4)
    d = np.diff(y[:T])
    f = y[T - 1] + np.cumsum(np.full(n - T, d.mean()))
    ax.plot(t[T:], f, color=RED, lw=2.4, label="ARIMA(0,1,0) with drift")
    ax.axvline(70, color=AMBER, lw=1.4, ls=":")
    ax.text(71, y.min() + 1, "level shift", color=AMBER, fontsize=11)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title("one shift contaminates the drift estimate", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.9, "ARIMA is not good at:", ha="center", fontsize=13.5,
            color=RED)
    items(ax, [
        ("multiple seasonalities", "daily AND weekly AND yearly at once"),
        ("many series at once", "one fit per series, no shared component"),
        ("non-linearity", "ARIMA is linear in the regressors"),
        ("missing timestamps and irregular gaps", "the lag structure assumes a fixed grid"),
        ("level shifts", "in training: contaminates the estimates | in prediction: drift from reality"),
    ], top=0.74, step=0.16)
    fig.tight_layout()
    save(fig, out, "arima_limits")


# ================================================================= 6. Prophet
def fig_prophet_model(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    # ax.text(0.5, 0.92, "Prophet is a curve fitted to time", ha="center", fontsize=14)
    eq(ax, 0.72, r"$y(t) \;=\; g(t) \;+\; s(t) \;+\; h(t) \;+\; \varepsilon_t$",
       fs=22, color=INK)
    for x, txt, col in [(0.3, "trend", BLUE), (0.55, "seasonality", GREEN),
                        (0.75, "events", AMBER)]:
        ax.annotate("", xy=(x, 0.66), xytext=(x, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.text(x, 0.49, txt, ha="center", fontsize=12, color=col)
    ax.text(0.5, 0.32,
            "ARIMA: $y_t$ depends on $y_{t-1}$:\n"
            "PROPHET: $t$ is a regressor (nothing depends on past values)",
            ha="center", fontsize=12, color=RED, linespacing=1.6)

    ax = axes[1]
    t = np.arange(400, dtype=float)
    rng = np.random.default_rng(15)
    g = 10 + np.piecewise(t, [t < 130, (t >= 130) & (t < 280), t >= 280],
                          [lambda u: 0.05 * u,
                           lambda u: 0.05 * 130 + 0.16 * (u - 130),
                           lambda u: 0.05 * 130 + 0.16 * 150 - 0.02 * (u - 280)])
    s = 3.0 * np.sin(2 * np.pi * t / 30) + 1.2 * np.cos(2 * np.pi * t / 7)
    ax.plot(t, g, color=BLUE, lw=2.4, label="$g(t)$ trend")
    ax.plot(t, g + s, color=GREEN, lw=1.0, alpha=0.8, label="$+\\, s(t)$")
    ax.plot(t, g + s + rng.normal(0, 0.8, len(t)), color=GREY, lw=0.7, alpha=0.6,
            label="$+\\, \\varepsilon_t$")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title("Each parts is explainable", fontsize=12.5)
    tidy(ax, "t")
    fig.tight_layout()
    save(fig, out, "prophet_model")


def fig_prophet_trend(out: Path) -> None:
    """Piecewise linear with changepoints, plus the saturating alternative."""
    # fig, axes = plt.subplots(1, 3, figsize=WIDE)
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    t = np.arange(300, dtype=float)
    rng = np.random.default_rng(21)
    cps = [80, 170, 240]
    slopes = [0.06, 0.20, 0.02, -0.08]
    g = np.zeros_like(t)
    cur, last = 10.0, 0
    for i, cp in enumerate(cps + [len(t)]):
        seg = np.arange(last, cp)
        g[last:cp] = cur + slopes[i] * (seg - last)
        cur = g[cp - 1]
        last = cp
    y = g + rng.normal(0, 0.9, len(t))

    ax = axes[0]
    ax.plot(t, y, color=GREY, lw=0.8)
    ax.plot(t, g, color=BLUE, lw=2.6)
    for cp in cps:
        ax.axvline(cp, color=RED, lw=1.2, ls="--")
    ax.set_title("piecewise linear:\nthe slope changes at changepoints", fontsize=12)
    tidy(ax, "t")

    ax = axes[1]
    blank(ax)
    ax.text(0.5, 0.90, "how the changepoints are chosen", ha="center", fontsize=12.5)
    items(ax, [
        ("put many candidates down", "25 by default, over the first 80% of history"),
        ("give each a slope change $\\delta_j$", "one extra parameter per candidate"),
        ("shrink them", r"Laplace prior: $\delta_j \sim \mathrm{Laplace}(0, \tau)$"),
        ("$\\tau$ is the flexibility dial", "large $\\tau$ overfits, small $\\tau$ underfits"),
    ], top=0.70, step=0.18)

    # ax = axes[2]
    # tt = np.linspace(0, 300, 300)
    # C = 40.0
    # k = 0.035
    # sat = C / (1 + np.exp(-k * (tt - 130)))
    # ax.plot(tt, sat, color=PURPLE, lw=2.6)
    # ax.axhline(C, color=RED, lw=1.4, ls="--")
    # ax.text(5, C + 1, "capacity $C$ — you must supply it", color=RED, fontsize=11)
    # ax.set_ylim(0, C * 1.2)
    # ax.set_title("logistic growth:\nfor a series with a known ceiling", fontsize=12)
    # tidy(ax, "t")
    fig.tight_layout()
    save(fig, out, "prophet_trend")


def fig_prophet_trend_eq(out: Path) -> None:
    """The piecewise-linear trend written out, including the join-up term."""
    k, m, sj, dj = 0.05, 10.0, 120.0, 0.18
    t = np.arange(0, 300, dtype=float)
    a = (t >= sj).astype(float)
    g_no = (k + a * dj) * t + m
    g_yes = (k + a * dj) * t + (m + a * (-sj * dj))

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.975,
            r"$g(t) = \left(k + \mathbf{a}(t)^{\top}\boldsymbol{\delta}\right)\, t"
            r" + \left(m + \mathbf{a}(t)^{\top}\boldsymbol{\gamma}\right)$",
            ha="center", fontsize=14, color=BLUE)
    ax.text(0.5, 0.835, r"$a_j(t) = \mathbb{1}\left[\, t \geq s_j \,\right]$",
            ha="center", fontsize=14, color=PURPLE)
    ax.text(0.5, 0.715, "\"has changepoint $j$ happened yet?\"", ha="center",
            fontsize=10, color=GREY)
    items(ax, [
        (r"$k$", "base growth rate"),
        (r"$\delta_j$", "growth rate change at $s_j$"),
        (r"$m$", "base offset"),
        (r"$\gamma_j$", "offset change at $s_j$"),
    ], top=0.545, step=0.12, fs=12)
    ax.text(0.5, 0.02, "First bracket correspond to growth rate ; second bracket to offset.",
            ha="center", fontsize=10.5, color=INK)

    ax = axes[1]
    ax.step(t, a, color=PURPLE, lw=2.4, where="post")
    ax.axvline(sj, color=GREY, lw=1.2, ls=":")
    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 1])
    ax.set_xlabel("t")
    ax.set_ylabel(r"$a_j(t)$")
    ax.set_title(r"$\mathbf{a}(t)$ is just a switch per changepoint"
                 "\nthe rate becomes $k + \\delta_j$ after $s_j$", fontsize=11.5,
                 color=PURPLE)
    ax.text(sj + 6, 0.5, r"$s_j$", fontsize=12, color=GREY)

    ax = axes[2]
    ax.plot(t, g_no, color=RED, lw=2.2, label=r"without $\gamma$")
    ax.plot(t, g_yes, color=GREEN, lw=2.2, label=r"with $\gamma_j = -s_j \delta_j$")
    ax.axvline(sj, color=GREY, lw=1.2, ls=":")
    ax.annotate("", xy=(sj, g_no[int(sj)]), xytext=(sj, g_yes[int(sj)]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(sj - 6, (g_no[int(sj)] + g_yes[int(sj)]) / 2,
            f"jump of\n$s_j\\,\\delta_j$ = {sj * dj:.1f}", ha="right", fontsize=10.5,
            va="center", linespacing=1.4)
    ax.set_ylim(None, g_no.max() * 1.12)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_xlabel("t")
    ax.set_title(r"$\gamma$ allows jumps", fontsize=11.5)
    # fig.suptitle(r"changing the slope also moves the line, so each $\delta_j$ needs "
    #              r"a matching $\gamma_j$", fontsize=13)
    fig.tight_layout()
    save(fig, out, "prophet_trend_eq")


def _changepoint_demo():
    """A piecewise-linear trend with three real slope changes, plus the basis."""
    rng = np.random.default_rng(21)
    n = 300
    t = np.arange(n, dtype=float)
    cps_true = [80, 170, 240]
    slopes = [0.06, 0.20, 0.02, -0.08]
    g = np.zeros(n)
    cur, last = 10.0, 0
    for i, cp in enumerate(cps_true + [n]):
        seg = np.arange(last, cp)
        g[last:cp] = cur + slopes[i] * (seg - last)
        cur = g[cp - 1]
        last = cp
    y = g + rng.normal(0, 0.9, n)
    cand = np.linspace(0, 0.8 * n, 25).astype(int)
    H = np.column_stack([np.clip(t - c, 0, None) for c in cand]) / n
    X = np.column_stack([np.ones(n), t / n, H])
    return t, y, g, cps_true, cand, X


def _cp_fit(X, y, alpha):
    """MAP under a Laplace prior on the slope changes == lasso on the deltas."""
    import warnings
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Lasso(alpha=alpha, fit_intercept=False, max_iter=200000,
                  tol=1e-6).fit(X, y)
    return m.coef_[2:], X @ m.coef_


def fig_laplace_prior(out: Path) -> None:
    """What Laplace(0, tau) is, and why its log has a corner at zero."""
    from scipy.stats import norm, laplace
    d = np.linspace(-3, 3, 1001)

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    for tau, col in ((0.25, RED), (0.6, PURPLE), (1.5, BLUE)):
        ax.plot(d, laplace.pdf(d, 0, tau), color=col, lw=2.4,
                label=rf"$\tau$ = {tau}")
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel(r"$\delta$")
    ax.set_yticks([])
    ax.set_title(r"Laplace$(0, \tau)$:  $p(\delta) \propto e^{-|\delta| / \tau}$"
                 "\n\"a sharp peak at zero\"", fontsize=12, color=PURPLE)
    ax.text(0.03, 0.55, "small $\\tau$ $\\equiv$ \"expect zero\"",
            transform=ax.transAxes, fontsize=10.5, color=INK, linespacing=1.4)

    ax = axes[1]
    ax.plot(d, laplace.pdf(d, 0, 0.7), color=PURPLE, lw=2.6, label="Laplace")
    ax.plot(d, norm.pdf(d, 0, 0.7), color=GREY, lw=2.6, ls="--", label="Gaussian")
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlabel(r"$\delta$")
    ax.set_yticks([])
    ax.set_title("against a Gaussian of the same width", fontsize=12)
    ax.text(0.03, 0.62, "more mass at exactly\nzero, and fatter tails",
            transform=ax.transAxes, fontsize=10.5, color=INK, linespacing=1.4)

    fig.tight_layout()
    save(fig, out, "laplace_prior")


def fig_map_example(out: Path) -> None:
    """MAP with a Laplace prior, worked through: it is soft-thresholding."""
    from scipy.stats import norm, laplace
    sigma, tau = 0.5, 0.6
    lam = sigma ** 2 / tau
    grid = np.linspace(-3, 4, 40001)

    def numeric_map(mle):
        post = norm.pdf(grid, mle, sigma) * laplace.pdf(grid, 0, tau)
        return grid[int(post.argmax())]

    cases = [2.00, 1.00, 0.35, -0.90]

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.975, "For each changepoint:", ha="center",
            fontsize=13, color=INK)
    steps = [
        ("1.", "data:", r"$\hat\delta \pm \sigma$,  here $\sigma = 0.5$"),
        ("2.", "prior:", r"$\delta \sim$ Laplace$(0, \tau)$,  $\tau = 0.6$"),
        ("3.", "multiply & take minus the log",
         r"$(\delta - \hat\delta)^2 / 2\sigma^2 \;+\; |\delta| / \tau$"),
        ("4.", "minimise", "(it has a closed form)"),
    ]
    for i, (num, title, detail) in enumerate(steps):
        yy = 0.845 - i * 0.145
        ax.text(0.04, yy, num, fontsize=13.5, color=BLUE, fontweight="bold",
                va="center")
        ax.text(0.11, yy + 0.025, title, fontsize=11.5, color=INK, va="center")
        if detail:
            ax.text(0.11, yy - 0.035, detail, fontsize=11.5, color=BLUE,
                    va="center")
    ax.text(0.5, 0.235, r"$\hat\delta_{\mathrm{MAP}} = \mathrm{sign}(\hat\delta)\,"
                        r"\max\left(|\hat\delta| - \lambda,\; 0\right)$",
            ha="center", fontsize=14, color=GREEN)
    
    ax = axes[1]
    xs = np.linspace(-3, 3, 601)
    ax.plot(xs, xs, color=GREY, lw=1.6, ls=":", label=r"no prior ($\hat\delta$)")
    ax.plot(xs, np.sign(xs) * np.maximum(np.abs(xs) - lam, 0), color=GREEN, lw=2.8,
            label="MAP")
    ax.axvspan(-lam, lam, color=RED, alpha=0.18, lw=0)
    ax.axhline(0, color=GREY, lw=0.9)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_xlabel(r"what the data says,  $\hat\delta$")
    ax.set_ylabel(r"what MAP keeps,  $\hat\delta_{\mathrm{MAP}}$")
    ax.set_title("soft-thresholding", fontsize=12, color=GREEN)
    ax.text(0.5, 0.08, rf"dead zone: $|\hat\delta| < {lam:.2f}$" "\nbecomes zero",
            transform=ax.transAxes, ha="center", fontsize=10, color=RED,
            linespacing=1.4)

    ax = axes[2]
    blank(ax)
    ax.text(0.5, 0.975, "Changepoint candidates", ha="center", fontsize=13)
    heads = [(r"$\hat\delta$", 0.13), ("MAP", 0.38)]
    for txt, x in heads:
        ax.text(x, 0.845, txt, ha="center", fontsize=11, color=GREY)
    for i, mle in enumerate(cases):
        num = numeric_map(mle)
        soft = np.sign(mle) * max(abs(mle) - lam, 0.0)
        col = RED if abs(soft) < 1e-9 else GREEN
        yy = 0.715 - i * 0.135
        ax.text(0.13, yy, f"{mle:+.2f}", ha="center", fontsize=12.5, va="center")
        ax.text(0.38, yy, f"{num:+.3f}", ha="center", fontsize=12.5, va="center",
                color=col)
        if abs(soft) < 1e-9:
            ax.text(0.64, yy, "killed", ha="right", fontsize=10, color=RED, va="center")
    fig.suptitle(r"MAP with a Laplace prior has a closed form: it is a threshold", fontsize=13)
    fig.tight_layout()
    save(fig, out, "map_example")


def fig_prophet_shrinkage(out: Path) -> None:
    """How the changepoints are shrunk, and what the flexibility dial does."""
    t, y, g, cps_true, cand, X = _changepoint_demo()
    mid = 0.02
    d_mid, _ = _cp_fit(X, y, mid)

    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.665, "Prophet lays down 25 candidates changepoints\n"
                        "over the first 80% of the history, and gives\n"
                        r"each one a slope change $\delta_j$",
            ha="center", fontsize=13, color=INK, linespacing=1.6)
    ax.text(0.5, 0.45, r"$\delta_j \;\sim\; \mathrm{Laplace}(0,\, \tau)$",
            ha="center", fontsize=13, color=PURPLE)
    ax.text(0.5, 0.25, "Penalty on the $\\delta$s sets\nmost of them to EXACTLY zero", ha="center", fontsize=13, color=PURPLE)
    ax.text(0.5, 0.025, r"$\tau$ is the flexibility dial", ha="center", fontsize=13, color=GREEN)

    ax = axes[1]
    ax.axhline(0, color=GREY, lw=1.0)
    ax.vlines(cand, 0, d_mid, color=PURPLE, lw=2.6)
    ax.plot(cand, d_mid, "o", color=PURPLE, ms=4)
    for c in cps_true:
        ax.axvline(c, color=GREEN, lw=1.3, ls="--")
    kept = int((np.abs(d_mid) > 1e-4).sum())
    ax.set_xlabel("candidate changepoint")
    ax.set_ylabel(r"fitted $\delta_j$")
    ax.set_title(f"at $\\tau$ = {mid}: {kept} of 25 survive\n"
                 "dashed green are real breaks", fontsize=11.5)
    ax.text(0.97, 0.05, "the rest are exactly zero", transform=ax.transAxes,
            ha="right", fontsize=10, color=PURPLE)

    ax = axes[2]
    ax.plot(t, y, color=FAINT, lw=1.0)
    for alpha, col, lab in ((0.5, RED, "stiff"), (0.02, GREEN, "about right"),
                            (0.001, AMBER, "floppy")):
        d, fit = _cp_fit(X, y, alpha)
        k = int((np.abs(d) > 1e-4).sum())
        ax.plot(t, fit, color=col, lw=2.2,
                label=rf"$\tau$ = {alpha}: {k} kept  ({lab})")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlabel("t")
    ax.set_title("small $\\tau$ forces a straight trend,\n"
                 "large $\\tau$ lets it chase the noise", fontsize=11.5)
    # fig.suptitle(r"the changepoint prior scale $\tau$ is one number that decides "
    #              "how bendy the trend may be", fontsize=13)
    fig.tight_layout()
    save(fig, out, "prophet_shrinkage")


def fig_fourier_seasonality(out: Path) -> None:
    """Seasonality as a truncated Fourier series, and the order as a smoothness dial."""
    t = np.linspace(0, 2, 600)
    P = 1.0
    target = (np.sin(2 * np.pi * t) + 0.6 * np.sin(4 * np.pi * t + 0.8)
              + 0.35 * np.sin(6 * np.pi * t + 1.9) + 0.2 * np.sin(10 * np.pi * t))

    def fourier_fit(K):
        cols = [np.ones_like(t)]
        for k in range(1, K + 1):
            cols += [np.sin(2 * np.pi * k * t / P), np.cos(2 * np.pi * k * t / P)]
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        return X @ beta

    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.text(0.5, 0.88, "seasonality as a sum of waves", ha="center", fontsize=13)
    eq(ax, 0.64,
       r"$s(t) = \sum_{k=1}^{K}\left[a_k \cos\frac{2\pi k t}{P}"
       r" + b_k \sin\frac{2\pi k t}{P}\right]$", fs=16, color=GREEN)
    ax.text(0.5, 0.44, "$P$ is the period\n"
                       "$K$ is \"how wiggly\" the phenomenon may be", ha="center",
            fontsize=12, color=GREY, linespacing=1.5)
    ax.text(0.5, 0.20, "$2K$ parameters\nfit with ordinary regression",
            ha="center", fontsize=12, color=INK, linespacing=1.5)

    ax = axes[1]
    ax.plot(t, target, '--', color=GREY, lw=5, label="true shape")
    for K, col in [(1, RED), (3, AMBER), (10, GREEN)]:
        ax.plot(t, fourier_fit(K), color=col, lw=1, label=f"K = {K}")
    ax.legend(frameon=False, fontsize=10.5, ncol=2)
    ax.set_title("K$ is a smoothness dial", fontsize=12.5)
    ax.set_xlabel("$t$ (in periods)")

    ax = axes[2]
    for K, col in [(1, RED), (3, AMBER), (10, GREEN)]:
        ax.plot(t, target - fourier_fit(K), color=col, lw=1.6, label=f"K = {K}")
    ax.axhline(0, color=GREY, lw=1.0)
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("residuals", fontsize=12.5)
    ax.set_xlabel("$t$ (in periods)")
    fig.tight_layout()
    save(fig, out, "fourier_seasonality")


def fig_prophet_events(out: Path) -> None:
    """Known irregular events as indicator regressors, with a window."""
    n = 300
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(9)
    events = [60, 150, 245]
    bump = np.zeros(n)
    for e in events:
        for d, w in [(-1, 0.4), (0, 1.0), (1, 0.7), (2, 0.3)]:
            if 0 <= e + d < n:
                bump[e + d] += 7.0 * w
    base = 20 + 0.03 * t + 2.0 * np.sin(2 * np.pi * t / 30)
    y = base + bump + rng.normal(0, 0.8, n)

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    ax.plot(t, y, color=INK, lw=1.1)
    for e in events:
        ax.axvline(e, color=AMBER, lw=1.4, ls="--")
    ax.set_title("three known, irregular, repeated events", fontsize=12.5)
    tidy(ax, "t")

    ax = axes[1]
    blank(ax)
    # ax.text(0.5, 0.90, "$h(t)$: a column per event", ha="center", fontsize=13, color=AMBER)
    eq(ax, 0.70, r"$h(t) = \sum_j \kappa_j \, \mathbf{1}[\,t \in D_j\,]$", fs=19)
    items(ax, [
        ("$D_j$ is the window of the event", ""),
        ("Past & future dates must be supplied", "the model cannot guess event dates"),
        ("One coefficient per event", "regularised like the changepoints"),
    ], top=0.52, step=0.155)
    fig.tight_layout()
    save(fig, out, "prophet_events")


def fig_prophet_vs_arima(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.28, 0.94, "ARIMA", ha="center", fontsize=15, color=BLUE)
    ax.text(0.72, 0.94, "Prophet", ha="center", fontsize=15, color=GREEN)
    rows = [
        ("models", "dependence on recent values",
         "shape of the curve against $t$"),
        ("strength", "short horizons, clean seasonality",
         "long horizons, several seasonalities, events"),
        ("stationarity", "needed (tune $d$)", "not needed"),
        ("missing timestamps", "breaks", "no problem"),
        # ("outliers", "propagate through the lags", "to be dropped"),
        ("interpretable", "coefficients, not components",
         "trend, seasonality and events"),
        # ("tuning", "via AIC or a search", "priors: flexibility dials"),
    ]
    for i, (label, a, b) in enumerate(rows):
        y = 0.82 - i * 0.115
        ax.text(0.015, y, label, fontsize=11.5, color=GREY, va="center")
        ax.text(0.28, y, a, fontsize=11.5, color=INK, va="center", ha="center")
        ax.text(0.72, y, b, fontsize=11.5, color=INK, va="center", ha="center")
        if i:
            ax.plot([0.01, 0.99], [y + 0.058, y + 0.058], color=FAINT, lw=1.0)
    ax.text(0.5, 0.02, "in published benchmarks neither dominates", ha="center", fontsize=12, color=RED)
    save(fig, out, "prophet_vs_arima")

FIGURES = (
    fig_forecast_task,
    fig_baselines,
    fig_naive_def,
    fig_naive_strong,
    fig_metrics,
    fig_eq_mase,
    fig_backtest,
    fig_cv_wrong,
    fig_intervals,
    fig_arma_recap,
    fig_arima_intuition,
    fig_eq_arima,
    fig_arima_pipeline,
    fig_sarima_problem,
    fig_sarima,
    fig_box_jenkins,
    fig_ic_motivation,
    fig_aic_bic,
    fig_residual_diag,
    fig_arimax,
    fig_arima_limits,
    fig_prophet_model,
    fig_prophet_trend,
    fig_prophet_trend_eq,
    fig_laplace_prior,
    fig_map_example,
    fig_prophet_shrinkage,
    fig_fourier_seasonality,
    fig_prophet_events,
    fig_prophet_vs_arima,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "CourseTS/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
