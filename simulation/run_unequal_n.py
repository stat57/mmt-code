"""Simulation (appendix): unequal sample sizes.

Four-mode Gaussian family, D_A = rho * delta with rho in {.75, 1, 1.25},
delta = .5, (n, m) in {(400, 1600), (1600, 400)}, with (800, 800) as the
equal-size benchmark; 500 replicates per point.

Applicability: the test returns a result; correct cardinality (four modes in
both samples) is reported separately.

Tables (results/run_unequal_n/): raw/, summary.csv.
Figures (Plots/run_unequal_n/): fig_unequal_n.pdf.

Usage: python run_unequal_n.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import (ALPHA, BLACK, BLUE, GRAY, RC, REGION2, TOLERANCE, VERMILLION, build_pair,  # noqa: E402
                      errorbar, mt_fields, new_figure, panel_title, plt, proportion_with_ci, results_dir,
                      run_settings, save, style_axis, top_legend)

from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 43
REPS = 500
RHOS = (0.75, 1.0, 1.25)
SIZES = ((800, 800), (400, 1600), (1600, 400))
_PAIRS: dict = {}


def pair(D: float) -> dict:
    if D not in _PAIRS:
        _PAIRS[D] = build_pair("gaussian4", D)
    return _PAIRS[D]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["rho"] * TOLERANCE)
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["m"], rng)
    res = modal_transport_test(X, Y, tolerance=TOLERANCE, alpha=ALPHA, analysis_region=REGION2)
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == 4)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, m, rho), g in raw.groupby(["n", "m", "rho"]):
        a = g[g["applicable"]]
        D = rho * TOLERANCE
        row = {"n": n, "m": m, "rho": rho, "reps": len(g), "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g),
               "bias_D": a["D_hat"].mean() - D}
        for name in ("equivalent", "inconclusive", "meaningfully_different"):
            p, lo, hi = proportion_with_ci(a["decision"] == name)
            row.update({f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi})
        p, lo, hi = proportion_with_ci(a["U_D"] >= D)
        row.update({"coverage_UD": p, "coverage_UD_lo": lo, "coverage_UD_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(s: pd.DataFrame) -> None:
    styles = {(800, 800): (GRAY, "s", ":", 0.0), (400, 1600): (BLUE, "o", "-", -0.012),
              (1600, 400): (VERMILLION, "^", "--", 0.012)}
    with plt.rc_context(RC):
        fig, axes = new_figure(1, 3, 9.0, 3.1)
        for (n, m), (color, marker, ls, dx) in styles.items():
            g = s[(s.n == n) & (s.m == m)].sort_values("rho")
            x = g.rho + dx  # small horizontal offset so the error bars do not overlap
            kw = dict(label=f"$(n,m)=({n},{m})$", color=color, marker=marker, linestyle=ls)
            errorbar(axes[0], x, g.p_equivalent, g.p_equivalent_lo, g.p_equivalent_hi, **kw)
            errorbar(axes[1], x, g.p_meaningfully_different, g.p_meaningfully_different_lo,
                     g.p_meaningfully_different_hi, **kw)
            errorbar(axes[2], x, g.coverage_UD, g.coverage_UD_lo, g.coverage_UD_hi, **kw)
        panel_title(axes[0], r"(a) $\Pr(Z_{\delta,\mathrm{bc}}<z_\alpha)$")
        panel_title(axes[1], r"(b) $\Pr(Z_{\delta,\mathrm{bc}}>z_{1-\alpha})$")
        panel_title(axes[2], r"(c) Coverage of $U_D$")
        for ax in axes[:2]:
            ax.axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.8)
            ax.set_ylim(-0.02, 1.02)
        axes[2].axhline(1 - ALPHA, color=BLACK, linestyle=":", linewidth=0.8)
        axes[2].set_ylim(0.85, 1.0)
        for ax in axes:
            ax.set_xlabel(r"$\rho=D_A/\delta$")
            ax.set_xticks(list(RHOS))
            style_axis(ax)
        axes[0].set_ylabel("Probability")
        top_legend(fig, axes[0], ncol=3)
        save(fig, FIGDIR / "fig_unequal_n.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    settings = [{"name": f"n{n}_m{m}_rho{r:.2f}", "n": n, "m": m, "rho": r} for n, m in SIZES for r in RHOS]
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
