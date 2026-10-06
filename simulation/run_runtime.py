"""Simulation (appendix): computing time.

Seconds per two-sample matched-mode test (both mode searches, covariance,
matching and tests) on the four-mode Gaussian family with D_A = 0:
  * n = m in {400, 800, 1600, 3200, 6400, 12800}, d = 2;
  * d in {2, 4, 6, 8}, n = m = 1600;
  * start multiplier c_start in {5, 10, 25, 50, 100}, n = m = 800, d = 2.
10 replicates each, run one at a time (no parallel workers).  Run this file
alone, with no other computation on the machine, so that the times are not
inflated by other processes.

Tables (results/run_runtime/): raw/, summary.csv.
Figures (Plots/run_runtime/): fig_runtime.pdf.

Usage: python run_runtime.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import (BLUE, RC, build_pair, new_figure, panel_title, plt, region, results_dir,  # noqa: E402
                      run_settings, save, style_axis)
from mt_tools import plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 45
REPS = 10
SAMPLE_SIZES = (400, 800, 1600, 3200, 6400, 12800)
DIMS = (2, 4, 6, 8)
START_MULTIPLIERS = (5, 10, 25, 50, 100)
_PAIRS: dict = {}


def pair(d: int) -> dict:
    if d not in _PAIRS:
        _PAIRS[d] = build_pair("gaussian4", 0.0, extra_dims=d - 2)
    return _PAIRS[d]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["d"])
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    res = modal_transport_test(X, Y, analysis_region=region(s["d"] - 2), start_multiplier=s["c_start"])
    return {"K_x": res["K_x"], "K_y": res["K_y"], "n_starts": res["n_starts_x"],
            "max_residual": max(res["max_residual_x"], res["max_residual_y"])}


def build_settings() -> list[dict]:
    settings = [{"name": f"size_n{n}", "study": "n", "d": 2, "n": n, "c_start": 50.0} for n in SAMPLE_SIZES]
    settings += [{"name": f"dim_d{d}", "study": "d", "d": d, "n": 1600, "c_start": 50.0} for d in DIMS]
    settings += [{"name": f"starts_c{c}", "study": "c_start", "d": 2, "n": 800, "c_start": float(c)}
                 for c in START_MULTIPLIERS]
    return settings


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.assign(correct=(raw["K_x"] == 4) & (raw["K_y"] == 4))
    return (raw.groupby(["study", "d", "n", "c_start"])
            .agg(reps=("seconds", "size"), seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std"),
                 n_starts=("n_starts", "first"), correct_cardinality=("correct", "mean"),
                 max_residual=("max_residual", "max"))
            .reset_index())


def make_figure(s: pd.DataFrame) -> None:
    with plt.rc_context(RC):
        fig, axes = new_figure(1, 3, 9.0, 3.0)
        panels = (("n", "n", "(a) Sample size ($d=2$)", "Sample size $n=m$", True),
                  ("d", "d", "(b) Dimension ($n=m=1600$)", "Dimension $d$", False),
                  ("c_start", "c_start", "(c) Starts ($n=m=800$, $d=2$)", r"Start multiplier $c_{\mathrm{start}}$",
                   True))
        for ax, (study, x, title, xlabel, logx) in zip(axes, panels):
            g = s[s.study == study].sort_values(x)
            ax.errorbar(g[x], g.seconds_mean, yerr=g.seconds_sd, color=BLUE, marker="o", capsize=2)
            if logx:
                ax.set_xscale("log")
                ax.set_xticks(list(g[x]), [f"{v:g}" for v in g[x]])
                ax.minorticks_off()
            else:
                ax.set_xticks(list(g[x]))
            ax.set_ylim(0, None)
            ax.set_xlabel(xlabel)
            panel_title(ax, title)
            style_axis(ax)
        axes[0].set_ylabel("Seconds per test")
        save(fig, FIGDIR / "fig_runtime.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    raw = run_settings(build_settings(), one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw",
                       n_jobs=1)
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
