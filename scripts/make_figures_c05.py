#!/usr/bin/env python3
"""Render the framework-share figure used by the Course05 (PyTorch) slides.

Papers with Code shut down in July 2025; its last snapshot of paper <-> code links
survives in the pwc-archive dataset on Hugging Face. Each link carries the detected
framework but no date, so repositories are dated by the arXiv id of their paper
(YYMM.xxxxx). Only official implementations are kept: those are written alongside
the paper, so the paper date is a good proxy for the repository creation date
used by the original Papers with Code "Trends" chart. Shares are computed among
repositories with a detected deep-learning framework.

Usage: make_figures_c05.py [outdir]      (default: Course05/img)
"""

from __future__ import annotations

import collections
import gzip
import json
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

URL = (
    "https://huggingface.co/datasets/pwc-archive/files/resolve/main/"
    "jul-28-links-between-papers-and-code.json.gz"
)
CACHE = Path(__file__).resolve().parent.parent / "data" / "pwc-links.json.gz"

# first full quarter of the original chart, last full quarter of the archive
FIRST, LAST = (2017, 3), (2025, 2)

INK = "#1b1b1b"
GREY = "#9aa0a6"
COLORS = {"TensorFlow": "#ff9900", "PyTorch": "#ee4c2c", "Other": "#4285f4"}


def load_links() -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    with gzip.open(CACHE) as f:
        return json.load(f)


def quarterly_shares(links: list[dict]):
    # one entry per repository, dated by its earliest paper
    repos: dict[str, tuple[tuple[int, int], str]] = {}
    for x in links:
        m = re.match(r"(\d{2})(\d{2})\.", x["paper_arxiv_id"] or "")
        if not m or not x["is_official"] or x["framework"] == "none":
            continue
        q = (2000 + int(m[1]), (int(m[2]) - 1) // 3 + 1)
        r = x["repo_url"].lower().rstrip("/")
        if r not in repos or q < repos[r][0]:
            repos[r] = (q, x["framework"])

    counts = collections.defaultdict(collections.Counter)
    for q, fw in repos.values():
        counts[q][fw] += 1

    quarters = [q for q in sorted(counts) if FIRST <= q <= LAST]
    x, tf, pt = [], [], []
    for q in quarters:
        c = counts[q]
        n = sum(c.values())
        x.append(q[0] + (q[1] - 1) / 4 + 0.125)  # mid-quarter
        tf.append(c["tf"] / n)
        pt.append((c["pytorch"] + c["torch"]) / n)
    tf, pt = np.array(tf), np.array(pt)
    return np.array(x), tf, pt, 1 - tf - pt


def plot(outdir: Path) -> None:
    x, tf, pt, other = quarterly_shares(load_links())

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    layers = [("TensorFlow", tf), ("PyTorch", pt), ("Other", other)]
    bottom = np.zeros_like(x)
    for name, share in layers:
        top = bottom + share
        ax.fill_between(x, bottom, top, color=COLORS[name], alpha=0.5, lw=0)
        ax.plot(x, top, color=COLORS[name], lw=1.8)
        bottom = top

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    years = range(int(np.ceil(x[0])), int(x[-1]) + 1)
    ax.set_xticks(list(years), [f"Jan {y}" for y in years])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(color=GREY, alpha=0.4, lw=0.8)
    ax.set_axisbelow(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK, length=0, labelsize=10)
    ax.set_xlabel("Paper date (arXiv)", color=INK, labelpad=10)
    ax.set_ylabel("Percentage", color=INK)
    ax.set_title(
        "Share of official paper implementations by framework",
        color="#555555", fontsize=14, loc="left", pad=28,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in ("Other", "PyTorch", "TensorFlow")]
    ax.legend(
        handles, ["Other", "PyTorch", "TensorFlow"], ncol=3, frameon=False,
        loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9, handlelength=1, handleheight=1,
    )
    fig.text(0.99, 0.01, "Source: Papers with Code archive (Jul 2025)", ha="right", fontsize=7, color=GREY)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "framework_share.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1] if len(sys.argv) > 1 else "Course05/img"))
