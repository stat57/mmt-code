"""Simulation (appendix): dimension -- how large can d be?

The four-mode Gaussian design is embedded in R^d by appending d - 2
independent N(0, 0.5^2) coordinates.  The population modes are (mu_j, 0), so
D_A is unchanged by the embedding.  For each (d, n) we record, at
rho = D_A/delta in {0.75, 1, 1.25}:
  * applicability (the test returns a result) and correct cardinality (both
    samples recover exactly four modes),
  * equivalence power at rho = .75, both one-sided rejection rates at the
    boundary rho = 1, and change power at rho = 1.25,
Design: d in {2, 4, 6, 8}, n = m in {400, 1600}, 200 replicates per point.


Tables (results/run_dimension/): raw/, summary.csv.
Figures (Plots/run_dimension/): fig_dimension.pdf.
Computing time versus d is reported by run_runtime.py.

Usage: python run_dimension.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from mt_tools import (ALPHA, GRAY, BLACK, BLUE, RC, TOLERANCE, VERMILLION, build_pair, errorbar, mt_fields,  # noqa: E402
                      new_figure, panel_title, plt, proportion_with_ci, region, results_dir, run_settings, save,
                      style_axis)

from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 3
REPS = 200
DIMS = (2, 4, 6, 8)
SAMPLE_SIZES = (400, 1600)
RHOS = (0.75, 1.0, 1.25)
_PAIRS: dict = {}


def pair(d: int, rho: float) -> dict:
    if (d, rho) not in _PAIRS:
        _PAIRS[(d, rho)] = build_pair("gaussian4", rho * TOLERANCE, extra_dims=d - 2)
    return _PAIRS[(d, rho)]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["d"], s["rho"])
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    res = modal_transport_test(X, Y, tolerance=TOLERANCE, alpha=ALPHA, analysis_region=region(s["d"] - 2))
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == 4)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, n, rho), g in raw.groupby(["d", "n", "rho"]):
        a = g[g["applicable"]]
        row = {"d": d, "n": n, "rho": rho, "reps": len(g), "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g)}
        for name in ("equivalent", "meaningfully_different"):
            p, lo, hi = proportion_with_ci(a["decision"] == name)
            row.update({f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(s: pd.DataFrame) -> None:
    n_style = {400: (BLUE, "o", -0.08), 1600: (VERMILLION, "s", 0.08)}
    with plt.rc_context(RC):
        fig, axes = new_figure(1, 3, 9.0, 3.1)
        for n, (color, marker, dx) in n_style.items():
            g = s[s.n == n].sort_values("d").copy()
            g["d"] = g["d"] + dx  # small horizontal offset so the error bars do not overlap
            b, e, c = (g[np.isclose(g.rho, r)] for r in (1.0, 0.75, 1.25))
            axes[0].plot(b.d, b.applicability, color=color, marker=marker, linestyle="-", label=f"$n=m={n}$")
            errorbar(axes[1], b.d, b.p_equivalent, b.p_equivalent_lo, b.p_equivalent_hi,
                     label=f"$n=m={n}$", color=color, marker=marker, linestyle="-")
            errorbar(axes[1], b.d, b.p_meaningfully_different, b.p_meaningfully_different_lo,
                     b.p_meaningfully_different_hi, label="_", color=color, marker=marker, linestyle="--")
            errorbar(axes[2], e.d, e.p_equivalent, e.p_equivalent_lo, e.p_equivalent_hi,
                     label="_", color=color, marker=marker, linestyle="-")
            errorbar(axes[2], c.d, c.p_meaningfully_different, c.p_meaningfully_different_lo,
                     c.p_meaningfully_different_hi, label="_", color=color, marker=marker, linestyle="--")
        axes[1].axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.9, label=r"$\alpha=.05$")
        panel_title(axes[0], "(a) Applicability")
        panel_title(axes[1], r"(b) Boundary $\rho=1$")
        panel_title(axes[2], r"(c) Power at $\rho=0.75$ and $\rho=1.25$")
        axes[0].set_ylabel("Probability")
        axes[1].set_ylim(-0.01, 0.2)
        for ax in (axes[0], axes[2]):
            ax.set_ylim(-0.02, 1.02)
        for ax in axes:
            ax.set_xlabel(r"Dimension $d$")
            ax.set_xticks(DIMS)
            style_axis(ax)
        handles = [Line2D([], [], color=BLUE, marker="o", label="$n=m=400$"),
                   Line2D([], [], color=VERMILLION, marker="s", label="$n=m=1600$"),
                   Line2D([], [], color=GRAY, linestyle="-", label=r"$\Pr(Z_{\delta,\mathrm{bc}}<z_\alpha)$"),
                   Line2D([], [], color=GRAY, linestyle="--", label=r"$\Pr(Z_{\delta,\mathrm{bc}}>z_{1-\alpha})$"),
                   Line2D([], [], color=BLACK, linestyle=":", label=r"$\alpha=.05$")]
        fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
        save(fig, FIGDIR / "fig_dimension.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    settings = [{"name": f"d{d}_n{n}_rho{rho:.2f}", "d": d, "n": n, "rho": rho}
                for d in DIMS for n in SAMPLE_SIZES for rho in RHOS]
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))

if __name__ == "__main__":
    main()
