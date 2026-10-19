#!/usr/bin/env python3
"""Render the illustration figures used by the Course10 lecture slides.

Session 9 introduces recurrent networks, then uses one to predict the next word:
one-hot inputs and outputs are impractical, so each word is embedded, either with a
frozen pre-trained embedding (our CBOW of Session 8, the official word2vec) or with
one learned end to end. The concept figures are synthetic and seeded; the result
figures plot the numbers measured in Course10/rnn_next_word-solutions.ipynb.

Usage: make_figures_c10.py [outdir]      (default: Course10/img)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def box(ax, xy, w, h, text, fc="white", ec=BLUE, fs=11, lw=1.8, tc=INK, ls="-"):
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
            linestyle=ls,
        )
    )
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.4)


def arrow(ax, a, b, color=GREY, lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
            shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}",
        )
    )


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ================================================================ I. sequences


def fig_sequences(out: Path) -> None:
    """Text, signals and DNA: the input is a sequence of variable length."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=WIDE)

    ax = axes[0]
    blank(ax)
    ax.set_title("text")
    for sent, y in (("the patient was\nadmitted on monday", 0.74),
                    ("no fever", 0.50),
                    ("blood pressure was\nstable during the\nnight", 0.20)):
        ax.text(0.02, y, sent, fontsize=11.5, va="center", family="monospace")

    ax = axes[1]
    t = np.linspace(0, 4, 800)
    ecg = np.zeros_like(t)
    for beat in np.arange(0.3, 4, 0.8):
        ecg += 1.2 * np.exp(-((t - beat) / 0.018) ** 2)
        ecg -= 0.25 * np.exp(-((t - beat - 0.04) / 0.02) ** 2)
        ecg += 0.25 * np.exp(-((t - beat - 0.3) / 0.06) ** 2)
    ax.plot(t, ecg + 0.03 * rng.standard_normal(len(t)), color=RED, lw=1.3)
    ax.set_title("signal (ECG)")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])

    ax = axes[2]
    blank(ax)
    ax.set_title("DNA")
    seq = "".join(rng.choice(list("ACGT"), 60))
    for row in range(3):
        ax.text(0.02, 0.72 - 0.24 * row, " ".join(seq[20 * row:20 * row + 20][i:i + 5]
                                                  for i in range(0, 20, 5)),
                fontsize=13, family="monospace", va="center")
    fig.suptitle("ordered, and of any length", color=GREY, y=0.02, fontsize=12)
    save(fig, out, "sequences")


def fig_fixed_window(out: Path) -> None:
    """A dense network needs a fixed-size input, and learns each position separately."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    words = ["the", "patient", "was", "admitted", "on", "?"]
    xs = np.linspace(0.06, 0.56, len(words))
    for x, w in zip(xs, words):
        box(ax, (x, 0.2), 0.085, 0.12, w, ec=RED if w == "?" else GREY, fs=12)
    box(ax, (0.30, 0.68), 0.46, 0.18, "dense layer: a different weight\nfor every position",
        ec=BLUE, fs=12.5)
    for x in xs[:-1]:
        arrow(ax, (x, 0.27), (0.30 + (x - 0.30) * 0.6, 0.58), color=BLUE)
    arrow(ax, (0.53, 0.68), (xs[-1], 0.27), color=RED, rad=-0.4)
    ax.text(0.68, 0.78, "Input size is fixed",
            fontsize=13, va="center")
    ax.text(0.68, 0.50, "Words positions are learned separately", fontsize=13, va="center")
    save(fig, out, "fixed_window")


# ================================================================ II. RNN


def fig_rnn_cell(out: Path) -> None:
    """The recurrent cell: the state is fed back to the cell at the next step."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    box(ax, (0.22, 0.5), 0.14, 0.2, "RNN\ncell", ec=BLUE, fs=14)
    box(ax, (0.22, 0.1), 0.09, 0.1, r"$x_t$", ec=GREY, fs=15)
    box(ax, (0.22, 0.9), 0.09, 0.1, r"$h_t$", ec=GREEN, fs=15)
    arrow(ax, (0.22, 0.15), (0.22, 0.4), color=GREY)
    arrow(ax, (0.22, 0.6), (0.22, 0.85), color=GREEN)
    ax.add_patch(FancyArrowPatch((0.29, 0.55), (0.29, 0.45), arrowstyle="-|>",
                                 mutation_scale=14, color=RED, lw=2.0,
                                 connectionstyle="arc3,rad=-2.2"))
    ax.text(0.40, 0.5, r"$h_{t-1}$", fontsize=15, color=RED, va="center")
    ax.text(0.52, 0.66, r"$h_t = \tanh\left(x_t\, W_{xh} + h_{t-1}\, W_{hh} + b\right)$",
            fontsize=20)
    ax.text(0.52, 0.44, r"$h_0 = 0$", fontsize=16)
    ax.text(0.52, 0.24, "the hidden state h summarises\neverything read so far",
            fontsize=13, color=GREY)
    save(fig, out, "rnn_cell")


def _unrolled(ax, n=4, back=False, right=0.88):
    xs = np.linspace(0.14, right, n)
    labels = [str(t) for t in range(1, n)] + ["T"]
    for x, lab in zip(xs, labels):
        box(ax, (x, 0.5), 0.1, 0.2, "cell", ec=BLUE, fs=13)
        box(ax, (x, 0.1), 0.07, 0.1, rf"$x_{lab}$", ec=GREY, fs=14)
        box(ax, (x, 0.9), 0.07, 0.1, rf"$h_{lab}$", ec=GREEN, fs=14)
        arrow(ax, (x, 0.15), (x, 0.4), color=GREY)
        arrow(ax, (x, 0.6), (x, 0.85), color=GREEN)
    box(ax, (0.04, 0.5), 0.05, 0.1, r"$h_0$", ec=GREY, fs=13)
    arrow(ax, (0.065, 0.5), (xs[0] - 0.05, 0.5), color=INK)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.05, 0.5), (b - 0.05, 0.5), color=INK)
        ax.text((a + b) / 2, 0.55, r"$W_{hh}$", ha="center", fontsize=12)
        if back:
            arrow(ax, (b - 0.05, 0.40), (a + 0.05, 0.40), color=RED, lw=2.2)
    return xs


def fig_rnn_unrolled(out: Path) -> None:
    """Unrolled in time, an RNN is a deep network that shares its weights."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    _unrolled(ax)
    ax.text(0.5, -0.05, "the same cell (same weights) at every step  ·  as many steps "
            "as the sequence is long", ha="center", fontsize=12.5, color=GREY)
    save(fig, out, "rnn_unrolled")


def fig_rnn_tasks(out: Path) -> None:
    """Many-to-one, many-to-many, one-to-many."""
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    configs = (("many to one\n(classify a sequence)", [1, 1, 1, 1], [0, 0, 0, 1]),
               ("many to many\n(tag each step)", [1, 1, 1, 1], [1, 1, 1, 1]),
               ("one to many\n(generate new steps)", [1, 0, 0, 0], [1, 1, 1, 1]))
    for ax, (title, ins, outs) in zip(axes, configs):
        blank(ax)
        ax.set_title(title, fontsize=12.5)
        xs = np.linspace(0.14, 0.86, 4)
        for x, i, o in zip(xs, ins, outs):
            box(ax, (x, 0.5), 0.14, 0.16, "", ec=BLUE)
            if i:
                box(ax, (x, 0.12), 0.1, 0.1, "", ec=GREY, fc=FAINT)
                arrow(ax, (x, 0.17), (x, 0.42), color=GREY)
            if o:
                box(ax, (x, 0.88), 0.1, 0.1, "", ec=GREEN, fc="#e6f6e8")
                arrow(ax, (x, 0.58), (x, 0.83), color=GREEN)
        for a, b in zip(xs, xs[1:]):
            arrow(ax, (a + 0.07, 0.5), (b - 0.07, 0.5), color=INK)
    save(fig, out, "rnn_tasks")


def fig_bptt(out: Path) -> None:
    """Backpropagation through time: the gradient flows back through every step."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    xs = _unrolled(ax, back=True, right=0.80)
    box(ax, (0.94, 0.9), 0.08, 0.1, "loss", ec=RED, fs=12)
    arrow(ax, (xs[-1] + 0.035, 0.9), (0.90, 0.9), color=RED)
    ax.text(0.5, -0.05, "to reach the first input, the gradient is multiplied by "
            r"$W_{hh}$ (and $\tanh'$) once per step", ha="center", fontsize=12.5,
            color=RED)
    save(fig, out, "bptt")


def fig_vanishing(out: Path) -> None:
    """A product of T factors shrinks or explodes exponentially."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    steps = np.arange(0, 60)
    ax = axes[0]
    for lam, col in ((0.8, BLUE), (1.0, GREY), (1.2, RED)):
        ax.semilogy(steps, lam ** steps, color=col, lw=2.4, label=rf"factor {lam} per step")
    ax.set_xlabel("number of steps back in time")
    ax.set_ylabel("gradient size (log scale)")
    ax.set_ylim(1e-6, 1e5)
    ax.legend(frameon=False)
    ax.set_title("vanishing  ·  stable  ·  exploding")

    ax = axes[1]
    blank(ax)
    ax.text(0.02, 0.84, "vanishing (the common case)", fontsize=14, color=BLUE)
    ax.text(0.04, 0.70, "inputs more than ~20 steps back get\nalmost no gradient: "
            "no long-term memory", fontsize=12.5, va="top")
    ax.text(0.02, 0.40, "exploding", fontsize=14, color=RED)
    ax.text(0.04, 0.26, "one huge step ruins the training\n"
            "fix: gradient clipping (Session 6)", fontsize=12.5, va="top")
    save(fig, out, "vanishing")


def fig_vanishing_measured(out: Path) -> None:
    """Gradient of the last state w.r.t. each input, for MyRNN: the code of the slides."""
    import torch
    from torch import nn

    class MyRNN(nn.Module):
        def __init__(self, d_in, d_h, act=torch.tanh):
            super().__init__()
            self.d_h, self.act = d_h, act
            self.x2h = nn.Linear(d_in, d_h)
            self.h2h = nn.Linear(d_h, d_h)

        def forward(self, x):
            B, T, _ = x.shape
            h = torch.zeros(B, self.d_h)
            outs = []
            for t in range(T):
                h = self.act(self.x2h(x[:, t]) + self.h2h(h))
                outs.append(h)
            return torch.stack(outs, 1), h

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, act in zip(axes, (torch.tanh, torch.relu)):
        for scale, col in ((1, BLUE), (2, AMBER), (4, RED)):
            torch.manual_seed(0)
            x = torch.randn(1, 100, 8, requires_grad=True)
            rnn = MyRNN(8, 32, act)
            with torch.no_grad():
                rnn.h2h.weight *= scale
            _, h = rnn(x)
            h.sum().backward()
            grad = x.grad[0].norm(dim=1)
            ax.semilogy(grad.numpy(), color=col, lw=2.2,
                        label=f"h2h weights × {scale}   (‖h_T‖ = {h.norm().item():.3g})")
        ax.set_title(f"activation: {act.__name__}")
        ax.set_xlabel("time step t (the last state is h_T, T = 100)")
        ax.set_ylim(1e-24, 1e34)
        ax.set_yticks([1e-20, 1e-10, 1, 1e10, 1e20, 1e30])
        ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    axes[0].set_ylabel("‖ d h_T / d x_t ‖  (log scale)")
    save(fig, out, "vanishing_measured")


def _op(ax, xy, sym, color=INK, size=22):
    """A pointwise operation node (×, +, tanh) on a cell diagram, round on any axes."""
    ax.plot(*xy, "o", ms=size, mfc="white", mec=color, mew=1.8, zorder=3)
    ax.text(*xy, sym, ha="center", va="center", fontsize=12 if len(sym) == 1 else 9,
            color=color, zorder=4)


def fig_lstm_intuition(out: Path) -> None:
    """Rewriting the state at every step vs editing a memory that is carried along."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    blank(ax)
    ax.set_title("simple RNN: the state is rewritten at every step", color=RED)
    xs = np.linspace(0.08, 0.92, 4)
    for k, x in enumerate(xs):
        box(ax, (x, 0.55), 0.11, 0.14, rf"$h_{k + 1}$", ec=BLUE, fs=14)
    for a, b in zip(xs, xs[1:]):
        m = (a + b) / 2
        arrow(ax, (a + 0.055, 0.55), (b - 0.055, 0.55), color=INK)
        ax.text(m, 0.68, r"$\times W_{hh}$" "\n" r"$\tanh$", ha="center", fontsize=12,
                color=RED)
    ax.text(0.5, 0.25, "The gradient is multiplied by $W_{hh}$ and $\tanh'$ at every step", ha="center", fontsize=12.5, va="top")

    ax = axes[1]
    blank(ax)
    ax.set_title("LSTM: a memory that is only edited", color=GREEN)
    ax.plot([0.03, 0.97], [0.55, 0.55], color=GREEN, lw=5, solid_capstyle="round",
            zorder=1)
    ax.text(0.03, 0.62, "memory c", fontsize=12.5, color=GREEN)
    for x in (0.25, 0.55, 0.85):
        _op(ax, (x - 0.06, 0.55), "×", AMBER)
        _op(ax, (x + 0.04, 0.55), "+", BLUE)
        ax.text(x - 0.06, 0.43, "erase", ha="center", fontsize=11, color=AMBER)
        ax.text(x + 0.04, 0.43, "write", ha="center", fontsize=11, color=BLUE)
    ax.text(0.5, 0.25, "the memory is carried along"
            , ha="center", fontsize=12.5, va="top")
    save(fig, out, "lstm_intuition")


def fig_gates(out: Path) -> None:
    """A gate is a learned vector of numbers in (0, 1), multiplied element-wise."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    z = np.linspace(-6, 6, 200)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=AMBER, lw=2.6)
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.axhline(1, color=GREY, lw=0.8, ls=":")
    ax.text(-5.8, 0.88, "1: let through", fontsize=12)
    ax.text(1.2, 0.06, "0: block", fontsize=12)
    ax.set_title(r"gate $= \sigma(x_t W + h_{t-1} U + b)$ $\in (0, 1)$")
    ax.set_xlabel("input of the sigmoid")

    ax = axes[1]
    blank(ax)
    vec = [2.0, -1.3, 0.7, 3.1]
    gate = [0.98, 0.02, 0.50, 0.95]
    words = ["keep", "erase", "half", "keep"]
    cols = [("memory", vec, INK, 0.12), ("gate", gate, AMBER, 0.40),
            ("result", [v * g for v, g in zip(vec, gate)], GREEN, 0.68)]
    for name, vals, col, x in cols:
        ax.text(x + 0.06, 0.92, name, ha="center", fontsize=12.5, color=col)
        for k, v in enumerate(vals):
            box(ax, (x + 0.06, 0.76 - 0.17 * k), 0.12, 0.12, f"{v:+.2f}", ec=col, fs=12)
    ax.text(0.33, 0.50, "⊙", fontsize=24, ha="center", va="center")
    ax.text(0.61, 0.50, "=", fontsize=24, ha="center", va="center")
    for k, w in enumerate(words):
        ax.text(0.90, 0.76 - 0.17 * k, w, fontsize=12, va="center", color=GREY)
    ax.text(0.45, 0.0, r"computed from $x_t$ and $h_{t-1}$: the network learns what to keep",
            ha="center", fontsize=12, color=GREY)
    save(fig, out, "gates")


def fig_lstm_cell(out: Path) -> None:
    """The LSTM cell: a memory line, edited by three gates."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.add_patch(FancyBboxPatch((0.12, 0.08), 0.58, 0.84, boxstyle="round,pad=0.01",
                                fc="#f7f8fa", ec=GREY, lw=1.2))
    top, bus, gy = 0.80, 0.18, 0.42
    dx, dy = 0.018, 0.042  # radius of an operation node, in axes units
    # memory line
    ax.text(0.02, top, r"$c_{t-1}$", fontsize=15, va="center", color=GREEN)
    arrow(ax, (0.07, top), (0.28 - dx, top), color=GREEN, lw=2.6)
    _op(ax, (0.28, top), "×", AMBER)
    arrow(ax, (0.28 + dx, top), (0.415 - dx, top), color=GREEN, lw=2.6)
    _op(ax, (0.415, top), "+", BLUE)
    arrow(ax, (0.415 + dx, top), (0.76, top), color=GREEN, lw=2.6)
    ax.text(0.77, top, r"$c_t$", fontsize=15, va="center", color=GREEN)
    # inputs
    ax.text(0.02, bus, r"$h_{t-1}$", fontsize=15, va="center")
    ax.plot([0.07, 0.55], [bus, bus], color=INK, lw=1.6)
    ax.text(0.18, 0.0, r"$x_t$", fontsize=15, ha="center", va="bottom")
    arrow(ax, (0.18, 0.05), (0.18, bus), color=INK)
    # gates
    gates = ((0.28, r"$\sigma$", "f", AMBER), (0.37, r"$\sigma$", "i", AMBER),
             (0.46, "tanh", r"$\tilde c$", BLUE), (0.55, r"$\sigma$", "o", AMBER))
    for x, f, name, col in gates:
        ax.plot([x, x], [bus, gy - 0.06], color=INK, lw=1.6)
        box(ax, (x, gy), 0.055, 0.1, f, ec=col, fs=12)
        ax.text(x + 0.032, gy + 0.065, name, fontsize=13, color=col)
    arrow(ax, (0.28, gy + 0.05), (0.28, top - dy), color=AMBER)
    _op(ax, (0.415, 0.62), "×", AMBER)
    arrow(ax, (0.37, gy + 0.05), (0.415 - dx, 0.62 - dy * 0.6), color=AMBER)
    arrow(ax, (0.46, gy + 0.05), (0.415 + dx, 0.62 - dy * 0.6), color=BLUE)
    arrow(ax, (0.415, 0.62 + dy), (0.415, top - dy), color=BLUE)
    # output
    ax.plot([0.64, 0.64], [top, 0.62 + dy * 1.3], color=GREEN, lw=1.6)
    _op(ax, (0.64, 0.62), "tanh", GREEN, size=30)
    _op(ax, (0.64, gy), "×", AMBER)
    arrow(ax, (0.64, 0.62 - dy * 1.3), (0.64, gy + dy), color=GREEN)
    arrow(ax, (0.5775, gy), (0.64 - dx, gy), color=AMBER)
    arrow(ax, (0.64 + dx, gy), (0.76, gy), color=INK, lw=2.0)
    ax.text(0.77, gy, r"$h_t$", fontsize=15, va="center")
    # legend
    for k, (name, txt, col) in enumerate((
            ("f", "forget gate: what to erase", AMBER),
            ("i", "input gate: what to write", AMBER),
            (r"$\tilde c$", "candidate: what could be written", BLUE),
            ("o", "output gate: what to show", AMBER))):
        ax.text(0.82, 0.30 - 0.07 * k, name, fontsize=13, color=col, va="center")
        ax.text(0.845, 0.30 - 0.07 * k, txt, fontsize=11, va="center")
    save(fig, out, "lstm_cell")


def fig_lstm_equations(out: Path) -> None:
    """The LSTM equations, gate by gate."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    rows = (("forget gate", AMBER, r"$f_t = \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f)$"),
            ("input gate", AMBER, r"$i_t = \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i)$"),
            ("output gate", AMBER, r"$o_t = \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o)$"),
            ("candidate", BLUE, r"$\tilde c_t = \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c)$"),
            ("memory", GREEN, r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$"),
            ("state", GREEN, r"$h_t = o_t \odot \tanh(c_t)$"))
    for k, (name, col, eq) in enumerate(rows):
        y = 0.92 - 0.15 * k
        ax.text(0.02, y, name, fontsize=13, color=col, va="center")
        ax.text(0.22, y, eq, fontsize=18 if k < 4 else 20, va="center")
    ax.text(0.70, 0.92 - 0.15 * 4, "erase & write", fontsize=13, color=GREY,
            va="center")
    ax.text(0.70, 0.92 - 0.15 * 5, "hidden state is\npart of the memory", fontsize=13, color=GREY,
            va="center")
    ax.text(0.70, 0.92 - 0.15 * 1.0, "4 weight blocks",
            fontsize=13, color=GREY, va="center")
    save(fig, out, "lstm_equations")


def fig_lstm_to_gru(out: Path) -> None:
    """The GRU simplifies the LSTM: one state, two gates."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.20, 0.95, "LSTM", ha="center", fontsize=15, color=BLUE)
    ax.text(0.74, 0.95, "GRU", ha="center", fontsize=15, color=GREEN)
    rows = (("memory c  and  state h", "one state h (the memory is shown as is)"),
            ("forget gate f,  input gate i", "one update gate z:\nforget 1 − z, write z"),
            ("output gate o", "removed"),
            (r"candidate from $h_{t-1}$", (r"candidate from $r \odot h_{t-1}$" "\n"
                                           "(reset gate r: how much of the past to use)")))
    for k, (a, b) in enumerate(rows):
        y = 0.78 - 0.2 * k
        box(ax, (0.20, y), 0.30, 0.13, a, ec=BLUE, fs=12.5)
        box(ax, (0.74, y), 0.40, 0.15, b, ec=GREEN, fs=12.5)
        arrow(ax, (0.36, y), (0.53, y), color=INK)
    ax.text(0.47, 0.0, "4 weight blocks → 3: 25% fewer parameters, similar results in "
            "practice (Chung et al., 2014)", ha="center", fontsize=12.5, color=GREY)
    save(fig, out, "lstm_to_gru")


def fig_gru(out: Path) -> None:
    """The GRU: a gate decides how much of the state to keep."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.text(0.02, 0.86, "update gate", fontsize=13, color=AMBER)
    ax.text(0.22, 0.86, r"$z_t = \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z)$  "
            r"$\in (0, 1)$", fontsize=17)
    ax.text(0.02, 0.66, "reset gate", fontsize=13, color=AMBER)
    ax.text(0.22, 0.66, r"$r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r)$", fontsize=17)
    ax.text(0.02, 0.46, "candidate", fontsize=13, color=BLUE)
    ax.text(0.22, 0.46, r"$\tilde h_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} "
            r"+ b_h)$", fontsize=17)
    ax.text(0.02, 0.24, "new state", fontsize=13, color=GREEN)
    ax.text(0.22, 0.24, r"$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$",
            fontsize=19)
    ax.text(0.22, 0.06, r"$z_t \approx 0$: copy the state unchanged $\Rightarrow$ the "
            "gradient flows back without shrinking", fontsize=13, color=GREY)
    save(fig, out, "gru")


# ================================================================ III. next word


def fig_lm_shift(out: Path) -> None:
    """Inputs and targets: the same text, shifted by one word."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    text = ["the", "cat", "sat", "on", "the", "mat"]
    xs = np.linspace(0.1, 0.78, 5)
    for k, x in enumerate(xs):
        box(ax, (x, 0.14), 0.1, 0.11, text[k], ec=GREY, fs=13)
        box(ax, (x, 0.5), 0.1, 0.16, "GRU", ec=BLUE, fs=12)
        box(ax, (x, 0.86), 0.1, 0.11, text[k + 1], ec=GREEN, fs=13)
        arrow(ax, (x, 0.2), (x, 0.42), color=GREY)
        arrow(ax, (x, 0.58), (x, 0.8), color=GREEN)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.05, 0.5), (b - 0.05, 0.5), color=INK)
    ax.text(0.86, 0.14, "input x", fontsize=13, color=GREY, va="center")
    ax.text(0.86, 0.86, "target y\n= x shifted by 1", fontsize=13, color=GREEN,
            va="center")
    ax.text(0.86, 0.5, "cross-entropy\nat every step", fontsize=13, color=RED,
            va="center")
    save(fig, out, "lm_shift")


def fig_onehot_cost(out: Path) -> None:
    """One-hot in and out: a huge first and last layer, and inputs that are all zeros."""
    V, H, d = 10001, 256, 100
    gru = lambda i: 3 * (i * H + H * H + 2 * H)
    one_hot = {"GRU input weights": gru(V), "output layer": H * V + V}
    embedded = {"embedding": V * d, "GRU": gru(d), "to vector": H * d + d,
                "decoder": d * V + V}
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for row, (name, parts) in enumerate((("one-hot in, one-hot out", one_hot),
                                         ("embed, then decode", embedded))):
        left = 0
        cols = (RED, AMBER) if row == 0 else (TEAL, BLUE, PURPLE, TEAL)
        for (part, n), col in zip(parts.items(), cols):
            ax.barh(1 - row, n / 1e6, left=left, color=col, edgecolor="white")
            if row == 0:
                ax.text(left + n / 2e6, 1 - row, f"{part}\n{n / 1e6:.1f}M", ha="center",
                        va="center", color="white", fontsize=10.5)
            left += n / 1e6
        ax.text(left + 0.15, 1 - row, f"{left:.1f}M", va="center", fontsize=13)
    ax.text(0.05, -0.55, "embedding 1.0M + GRU 0.27M + linear 0.03M + decoder 1.0M",
            fontsize=10.5, color=TEAL)
    ax.set_ylim(-0.8, 1.5)
    ax.set_yticks([1, 0], ["one-hot in,\none-hot out", "embed (d = 100),\nthen decode"])
    ax.set_xlabel("parameters (millions), vocabulary of 10 001, GRU of 256")
    ax.set_xlim(0, 12.5)

    ax = axes[1]
    blank(ax)
    ax.text(0.05, 0.80, "a batch of 64 × 32 words", fontsize=13)
    ax.text(0.05, 0.60, "as one-hot vectors: 82 MB", fontsize=13, color=RED)
    ax.text(0.05, 0.47, "99.99% of them zeros", fontsize=13, color=RED)
    ax.text(0.05, 0.25, "as word indices: 16 kB", fontsize=13, color=GREEN)
    save(fig, out, "onehot_cost")


def fig_onehot_select(out: Path) -> None:
    """A one-hot vector times a matrix selects a column: an embedding is a lookup."""
    rng = np.random.default_rng(3)
    W = rng.integers(-9, 10, (4, 6)) / 10
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    x0, y0, c = 0.06, 0.82, 0.055
    for i in range(4):
        for j in range(6):
            hl = j == 2
            ax.add_patch(plt.Rectangle((x0 + j * c, y0 - (i + 1) * c * 2.2), c, c * 2.2,
                                       fc="#dbeafe" if hl else "white", ec=GREY))
            ax.text(x0 + (j + 0.5) * c, y0 - (i + 0.5) * c * 2.2, f"{W[i, j]:+.1f}",
                    ha="center", va="center", fontsize=10.5,
                    color=BLUE if hl else INK)
    ax.text(x0 + 3 * c, 0.9, "W  (d × vocabulary)", ha="center", fontsize=12.5)
    ax.text(0.43, 0.6, "×", fontsize=26, va="center")
    for j in range(6):
        ax.add_patch(plt.Rectangle((0.48, y0 - (j + 1) * 0.11), 0.05, 0.11,
                                   fc="#dbeafe" if j == 2 else "white", ec=GREY))
        ax.text(0.505, y0 - (j + 0.5) * 0.11, "1" if j == 2 else "0", ha="center",
                va="center", fontsize=12, color=BLUE if j == 2 else INK)
    ax.text(0.505, 0.9, "one-hot\n'cat'", ha="center", fontsize=12)
    ax.text(0.57, 0.6, "=", fontsize=26, va="center")
    for i in range(4):
        ax.add_patch(plt.Rectangle((0.62, y0 - (i + 1) * c * 2.2), 0.06, c * 2.2,
                                   fc="#dbeafe", ec=GREY))
        ax.text(0.65, y0 - (i + 0.5) * c * 2.2, f"{W[i, 2]:+.1f}", ha="center",
                va="center", fontsize=11, color=BLUE)
    ax.text(0.65, 0.9, "column 'cat'", ha="center", fontsize=12)
    ax.text(0.73, 0.62, "no multiplication needed:\njust read the column", fontsize=13,
            va="center")
    ax.text(0.73, 0.38, "nn.Embedding(vocab, d)\n= this lookup table", fontsize=13,
            va="center", color=BLUE, family="monospace")
    save(fig, out, "onehot_select")


def fig_cbow_recap(out: Path) -> None:
    """Session 8: CBOW predicts a word from its context; the encoder is the embedding."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ctx = ["the", "cat", "on", "the"]
    for k, w in enumerate(ctx):
        box(ax, (0.08, 0.82 - 0.2 * k), 0.1, 0.12, w, ec=GREY, fs=12)
        arrow(ax, (0.13, 0.82 - 0.2 * k), (0.27, 0.5), color=GREY)
    box(ax, (0.36, 0.5), 0.17, 0.26, "encoder\n(sum of the\nembeddings)", ec=TEAL, fs=12)
    box(ax, (0.62, 0.5), 0.17, 0.26, "decoder\n(one score\nper word)", ec=PURPLE, fs=12)
    box(ax, (0.86, 0.5), 0.12, 0.14, "sat", ec=GREEN, fs=14)
    arrow(ax, (0.45, 0.5), (0.53, 0.5), color=INK)
    ax.text(0.49, 0.56, "100", ha="center", fontsize=11, color=GREY)
    arrow(ax, (0.71, 0.5), (0.8, 0.5), color=INK)
    ax.text(0.36, 0.2, "column j =\nembedding of word j", ha="center", fontsize=12,
            color=TEAL)
    ax.text(0.62, 0.2, "row j =\noutput vector of word j", ha="center", fontsize=12,
            color=PURPLE)
    ax.text(0.5, 0.02, "word2vec (Mikolov et al., 2013): the same idea, plus negative "
            "sampling and subsampling of frequent words", ha="center", fontsize=12,
            color=GREY)
    save(fig, out, "cbow_recap")


def fig_lm_architecture(out: Path) -> None:
    """The model of the recitation, frozen or trained end to end."""
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    y = 0.6
    stages = ((0.07, 0.1, "word\nindex", GREY, "-"),
              (0.25, 0.16, "embedding\n10 001 → 100", TEAL, "--"),
              (0.45, 0.14, "GRU\n100 → 256", BLUE, "-"),
              (0.63, 0.14, "linear\n256 → 100", BLUE, "-"),
              (0.82, 0.16, "decoder\n100 → 10 001", PURPLE, "--"))
    for x, w, txt, col, ls in stages:
        box(ax, (x, y), w, 0.24, txt, ec=col, fs=12.5, ls=ls)
    for (a, wa, *_), (b, wb, *_) in zip(stages, stages[1:]):
        arrow(ax, (a + wa / 2, y), (b - wb / 2, y), color=INK)
    arrow(ax, (0.90, y), (0.97, y), color=INK)
    ax.text(0.975, y, "next\nword", fontsize=11.5, va="center")
    ax.text(0.25, 0.3, "frozen: CBOW or word2vec\nor learned (end to end)", ha="center",
            fontsize=12, color=TEAL)
    ax.text(0.82, 0.3, "frozen: CBOW or word2vec\nor learned (end to end)", ha="center",
            fontsize=12, color=PURPLE)
    ax.text(0.54, 0.3, "always trained", ha="center", fontsize=12, color=BLUE)
    ax.text(0.54, 0.08, "same architecture in every experiment: only what is trained "
            "changes", ha="center", fontsize=12.5, color=GREY)
    save(fig, out, "lm_architecture")


# ================================================================ IV. results
# numbers measured in Course10/rnn_next_word-solutions.ipynb (3000 steps each)

RESULTS = {  # name: (trainable parameters, validation perplexity, top-1 accuracy)
    "unigram\n(no context)": (0, 586.5, 0.102),
    "frozen CBOW\n(Session 8)": (300_644, 182.4, 0.216),
    "frozen\nword2vec": (300_644, 135.3, 0.223),
    "end to end": (2_310_845, 135.9, 0.224),
}

OVERLAPS = {  # share of the 10 nearest neighbours in common (2000 most frequent words)
    "word2vec\nvs random": 0.001,
    "CBOW\nvs word2vec": 0.181,
    "end to end\nvs word2vec": 0.245,
    "word2vec vs\nword2vec (seed 1)": 0.768,
}

ANALOGIES = {"CBOW": 0.029, "word2vec": 0.414, "end to end": 0.056}  # share solved


def fig_results(out: Path) -> None:
    names = list(RESULTS)
    cols = (GREY, TEAL, BLUE, RED)
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    ax = axes[0]
    ppl = [RESULTS[n][1] for n in names]
    ax.bar(names, ppl, color=cols)
    for k, v in enumerate(ppl):
        ax.text(k, v + 10, f"{v:.0f}", ha="center", fontsize=12)
    ax.set_title("validation perplexity (lower is better)")
    ax.tick_params(axis="x", labelsize=10.5)
    ax = axes[1]
    params = [RESULTS[n][0] / 1e6 for n in names]
    ax.bar(names, params, color=cols)
    for k, v in enumerate(params):
        ax.text(k, v + 0.04, f"{v:.2f}M", ha="center", fontsize=12)
    ax.set_title("trainable parameters")
    ax.tick_params(axis="x", labelsize=10.5)
    save(fig, out, "results")


def fig_overlap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    names = list(OVERLAPS)
    vals = [100 * v for v in OVERLAPS.values()]
    ax.bar(names, vals, color=(GREY, TEAL, RED, BLUE), width=0.6)
    for k, v in enumerate(vals):
        ax.text(k, v + 1, f"{v:.0f}%", ha="center", fontsize=13)
    ax.set_ylabel("nearest neighbours in common (%)")
    ax.set_title("of the 10 nearest neighbours of a word, how many do two embeddings "
                 "agree on?")
    ax.set_ylim(0, 100)
    save(fig, out, "overlap")


def fig_analogies(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    names = list(ANALOGIES)
    vals = [100 * v for v in ANALOGIES.values()]
    ax.bar(names, vals, color=(TEAL, BLUE, RED), width=0.6)
    for k, v in enumerate(vals):
        ax.text(k, v + 1, f"{v:.0f}%", ha="center", fontsize=13)
    ax.set_ylabel("analogies solved (%)")
    ax.set_title("france − paris + rome ≈ italy ?")
    ax.set_ylim(0, max(vals) * 1.25 + 1)

    ax = axes[1]
    blank(ax)
    pts = {"paris": (0.10, 0.36), "france": (0.40, 0.70), "rome": (0.50, 0.26),
           "italy": (0.80, 0.60)}
    for w, (x, y) in pts.items():
        ax.plot(x, y, "o", color=BLUE, ms=8)
        ax.text(x + 0.02, y + 0.03, w, fontsize=13)
    arrow(ax, pts["paris"], pts["france"], color=GREEN, lw=2.2)
    arrow(ax, pts["rome"], pts["italy"], color=GREEN, lw=2.2)
    ax.text(0.05, 0.88, "word2vec: a relation (capital of)\nbecomes a direction",
            fontsize=12.5, va="top")
    ax.text(0.30, 0.0, "CBOW and end to end: similar neighbours,\nbut no regular "
            "directions", fontsize=12.5, color=RED, va="bottom")
    save(fig, out, "analogies")


FIGURES = (
    fig_sequences, fig_fixed_window, fig_rnn_cell, fig_rnn_unrolled, fig_rnn_tasks,
    fig_bptt, fig_vanishing, fig_vanishing_measured, fig_lstm_intuition, fig_gates,
    fig_lstm_cell, fig_lstm_equations, fig_lstm_to_gru, fig_gru, fig_lm_shift, fig_onehot_cost,
    fig_onehot_select, fig_cbow_recap, fig_lm_architecture, fig_results, fig_overlap,
    fig_analogies,
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Course10/img")
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(out)
        print(f"{out / (fn.__name__.removeprefix('fig_') + '.png')}")


if __name__ == "__main__":
    main()
