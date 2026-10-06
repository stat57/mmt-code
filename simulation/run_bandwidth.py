"""Simulation (appendix): sensitivity to the bandwidth and to the kernel order.

Four-mode Gaussian family, n = m = 800, D_A = rho * delta with rho in
{.75, 1, 1.25}, delta = .5, 500 replicates per point.
  * bandwidth h = c * n^{-1/(d+8)} with c in {.5, .75, 1, 1.5, 2} (kernel order 6);
  * kernel order r in {2, 4, 6, 8, 10} (r = 2 is the Gaussian kernel) with its own
    bandwidth h = n^{-1/(d+r+2)}.

Applicability: the test returns a result (equal cardinalities, positive
definite covariance); correct cardinality (four modes in both samples) is
reported separately.

Tables (results/run_bandwidth/): raw/, summary.csv.
Figures (Plots/run_bandwidth/): fig_bandwidth.pdf, fig_kernel_order.pdf.

Usage: python run_bandwidth.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import default_bandwidth, modal_transport_test  # noqa: E402
from mt_tools import (ALPHA, BLACK, BLUE, GRAY, RC, REGION2, TOLERANCE, VERMILLION, build_pair, errorbar,  # noqa: E402
                      mt_fields, new_figure, panel_title, plt, proportion_with_ci, results_dir, run_settings, save,
                      style_axis, top_legend)

from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
MIN_APPLICABLE = 50  # decision probabilities are plotted only when at least this many replicates are applicable
EXPERIMENT_ID = 41
REPS = 500
N = 800
RHOS = (0.75, 1.0, 1.25)
MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)
ORDERS = (2, 4, 6, 8, 10)
_PAIRS: dict = {}


def pair(D: float) -> dict:
    if D not in _PAIRS:
        _PAIRS[D] = build_pair("gaussian4", D)
    return _PAIRS[D]


def build_settings() -> list[dict]:
    # Bandwidth multipliers (order 6), then the other kernel orders (c = 1; order 6 is the c = 1 row).
    settings = [{"name": f"c{c:.2f}_beta6_rho{r:.2f}", "c_h": c, "order": 6, "rho": r, "n": N}
                for c in MULTIPLIERS for r in RHOS]
    settings += [{"name": f"order{k}_rho{r:.2f}", "c_h": 1.0, "order": k, "rho": r, "n": N}
                 for k in ORDERS if k != 6 for r in RHOS]
    return settings


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["rho"] * TOLERANCE)
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    h = s["c_h"] * default_bandwidth(s["n"], 2, s["order"])
    res = modal_transport_test(X, Y, tolerance=TOLERANCE, alpha=ALPHA, analysis_region=REGION2,
                               smoothness=float(s["order"]), bandwidth_x=h, bandwidth_y=h)
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == 4)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, k, rho), g in raw.groupby(["c_h", "order", "rho"]):
        a = g[g["applicable"]]
        row = {"c_h": c, "order": k, "rho": rho, "reps": len(g), "n_applicable": len(a),
               "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g),
               "mean_D_hat": a["D_hat"].mean(), "D_A": rho * TOLERANCE}
        for name in ("equivalent", "inconclusive", "meaningfully_different"):
            p, lo, hi = proportion_with_ci(a["decision"] == name)
            row.update({f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


EQ = r"$\Pr(Z_{\delta,\mathrm{bc}}<z_\alpha)$"
CHG = r"$\Pr(Z_{\delta,\mathrm{bc}}>z_{1-\alpha})$"


def _panels(s: pd.DataFrame, x: str, xlabel: str, xticks, fname: str) -> None:
    with plt.rc_context(RC):
        fig, axes = new_figure(1, 3, 9.0, 3.1, sharey=True)
        for ax, rho, tag in zip(axes, RHOS, "abc"):
            g = s[np.isclose(s.rho, rho)].sort_values(x)
            ok = g[g.n_applicable >= MIN_APPLICABLE]
            for name, color, marker, lab in (("equivalent", BLUE, "o", EQ),
                                             ("meaningfully_different", VERMILLION, "^", CHG)):
                errorbar(ax, ok[x], ok[f"p_{name}"], ok[f"p_{name}_lo"], ok[f"p_{name}_hi"],
                         label=lab, color=color, marker=marker, linestyle="-")
            ax.plot(g[x], g.applicability, color=GRAY, linestyle=":", marker="s", markersize=3.5,
                    label="Applicability")
            ax.axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.8, label=r"$\alpha=.05$")
            panel_title(ax, f"({tag}) $\\rho={rho:g}$")
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(xticks))
            ax.set_ylim(-0.02, 1.02)
            style_axis(ax)
        axes[0].set_ylabel("Probability")
        top_legend(fig, axes[0], ncol=4)
        save(fig, FIGDIR / fname)


def make_figure(s: pd.DataFrame) -> None:
    _panels(s[s.order == 6], "c_h", r"Bandwidth multiplier $c$ (kernel order 6)", MULTIPLIERS, "fig_bandwidth.pdf")
    _panels(s[np.isclose(s.c_h, 1.0)], "order", r"Kernel order $r$ ($h=n^{-1/(d+r+2)}$)", ORDERS,
            "fig_kernel_order.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    raw = run_settings(build_settings(), one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
