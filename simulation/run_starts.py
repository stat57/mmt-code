"""Simulation (appendix): sensitivity to the number of starts of the mode search.

Four-mode Gaussian family at the tolerance boundary (D_A = delta = .5),
n = m = 800, 500 replicates per point.  Number of starts
M_n = min{n, ceil(c_start log n)} with c_start in {5, 10, 25, 50, 100}
(default 50).

Reported: applicability (the test returns a result), correct cardinality
(four modes in both samples), and both one-sided boundary rejection rates.
Computing time versus c_start is reported by run_runtime.py.

Tables (results/run_starts/): raw/, summary.csv.
Figures (Plots/run_starts/): fig_starts.pdf.

Usage: python run_starts.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import (ALPHA, BLACK, BLUE, GRAY, RC, REGION2, TOLERANCE, VERMILLION, build_pair, errorbar,  # noqa: E402
                      mt_fields, new_figure, plt, proportion_with_ci, results_dir, run_settings, save, style_axis,
                      top_legend)

from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 42
REPS = 500
N = 800
START_MULTIPLIERS = (5, 10, 25, 50, 100)
_PAIRS: dict = {}


def pair() -> dict:
    if "p" not in _PAIRS:
        _PAIRS["p"] = build_pair("gaussian4", TOLERANCE)
    return _PAIRS["p"]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair()
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    res = modal_transport_test(X, Y, tolerance=TOLERANCE, alpha=ALPHA, analysis_region=REGION2,
                               start_multiplier=s["c_start"])
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == 4)
    row["n_starts"] = int(res["n_starts_x"])
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in raw.groupby("c_start"):
        a = g[g["applicable"]]
        row = {"c_start": c, "n_starts": int(g["n_starts"].iloc[0]), "reps": len(g),
               "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g)}
        for name in ("equivalent", "meaningfully_different"):
            p, lo, hi = proportion_with_ci(a["decision"] == name)
            row.update({f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(s: pd.DataFrame) -> None:
    s = s.sort_values("c_start")
    with plt.rc_context(RC):
        fig, ax = new_figure(1, 1, 4.2, 3.2)
        for name, color, marker, lab in (("equivalent", BLUE, "o", r"$\Pr(Z_{\delta,\mathrm{bc}}<z_\alpha)$"),
                                         ("meaningfully_different", VERMILLION, "^",
                                          r"$\Pr(Z_{\delta,\mathrm{bc}}>z_{1-\alpha})$")):
            errorbar(ax, s.c_start, s[f"p_{name}"], s[f"p_{name}_lo"], s[f"p_{name}_hi"],
                     label=lab, color=color, marker=marker, linestyle="-")
        ax.plot(s.c_start, s.applicability, color=GRAY, linestyle=":", marker="s", markersize=3.5,
                label="Applicability")
        ax.axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.8, label=r"$\alpha=.05$")
        ax.set_xscale("log")
        ax.set_xticks(list(START_MULTIPLIERS), [str(c) for c in START_MULTIPLIERS])
        ax.minorticks_off()
        ax.set_xlabel(r"Start multiplier $c_{\mathrm{start}}$ (boundary $\rho=1$)")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        style_axis(ax)
        top_legend(fig, ax, ncol=2)
        save(fig, FIGDIR / "fig_starts.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    settings = [{"name": f"cstart{c}", "c_start": float(c), "n": N} for c in START_MULTIPLIERS]
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
