"""Simulation (appendix): a state carrying little probability mass.

Four-mode Gaussian family with the weight of one state w in
{.25, .15, .10, .05, .025} and the other three equal; the population modes
are held fixed as w varies.  f and g share the weights; under the null g = f,
under the alternative g moves the mode of the low-mass state by .3 (D_A = .15).
n = m = 800, 500 replicates per point.

Methods: matched-mode exact-equality test (non-applicable replicates count as
non-rejections), MMD, Energy, GEC (5-MST); level .05. Also reported:
applicability (the test returns a result), correct cardinality (four modes in
both samples), and the RMSE of the estimated mode of the low-mass state.

Tables (results/run_low_prevalence/): raw/, summary.csv.
Figures (Plots/run_low_prevalence/): fig_low_prevalence.pdf.

Usage: python run_low_prevalence.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import (ALPHA, BLACK, BLUE, GRAY, METHOD_STYLE, RC, REGION2, by_component, errorbar,  # noqa: E402
                      family, mt_fields, new_figure, panel_title, plt, population_modes, proportion_with_ci,
                      results_dir, run_omnibus, run_settings, save, style_axis, top_legend, with_modes)

from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 44
REPS = 500
N = 800
LOW_WEIGHTS = (0.25, 0.15, 0.10, 0.05, 0.025)
SHIFT = 0.30
METHODS = ("mt", "mmd", "energy", "gec")
_PAIRS: dict = {}


def pair(w: float, shifted: bool) -> dict:
    if (w, shifted) not in _PAIRS:
        tmpl = family("gaussian4")
        modes = by_component(population_modes(tmpl, REGION2), tmpl.centers)
        weights = np.r_[w, np.full(3, (1 - w) / 3)]
        f = with_modes(tmpl, modes, weights)
        moved = modes.copy()
        if shifted:
            moved[0] += np.array([SHIFT, 0.0])
        g = with_modes(tmpl, moved, weights)
        for mix in (f, g):
            if len(population_modes(mix, REGION2)) != 4:
                raise RuntimeError(f"Low-mass design lost a mode at w = {w}.")
        _PAIRS[(w, shifted)] = {"f": f, "g": g, "mode0": modes[0]}
    return _PAIRS[(w, shifted)]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["w"], s["shifted"])
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    res = modal_transport_test(X, Y, analysis_region=REGION2, alpha=ALPHA)
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == 4)
    mx = res.get("modes_x", np.empty((0, 2)))
    row["low_mode_error"] = float(np.min(np.linalg.norm(mx - p["mode0"], axis=1))) if len(mx) else np.nan
    row.update(run_omnibus(X, Y, ("mmd", "energy", "gec"), seed=int(rng.integers(2**31))))
    for m in ("mmd", "energy", "gec"):
        row[f"{m}_reject"] = bool(row[f"{m}_p"] < ALPHA)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (w, shifted), g in raw.groupby(["w", "shifted"]):
        row = {"w": w, "shifted": shifted, "reps": len(g), "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g),
               "low_mode_rmse": float(np.sqrt(np.nanmean(g["low_mode_error"] ** 2)))}
        for m in METHODS:
            p, lo, hi = proportion_with_ci(g[f"{m}_reject"])
            row.update({f"{m}_rej": p, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def _weight_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(LOW_WEIGHTS), [f"{w:g}" for w in LOW_WEIGHTS])
    ax.minorticks_off()


def make_figure(s: pd.DataFrame) -> None:
    with plt.rc_context(RC):
        fig, axes = new_figure(1, 3, 9.6, 3.1)
        for shifted, ax in ((False, axes[0]), (True, axes[1])):
            g = s[s.shifted == shifted].sort_values("w")
            for m in METHODS:
                label, color, marker, ls = METHOD_STYLE[m]
                errorbar(ax, g.w, g[f"{m}_rej"], g[f"{m}_lo"], g[f"{m}_hi"],
                         label=label, color=color, marker=marker, linestyle=ls)
            ax.axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.8, label=r"$\alpha=.05$")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mass $w$ of the moved state")
            _weight_axis(ax)
            style_axis(ax)
        panel_title(axes[0], "(a) Null: no displacement")
        panel_title(axes[1], r"(b) Mode of the low-mass state moved ($D_A=0.15$)")
        axes[0].set_ylabel("Rejection probability")
        g = s[~s.shifted].sort_values("w")
        ax = axes[2]
        ax.plot(g.w, g.low_mode_rmse, color=BLUE, marker="o", label="_")
        ref = g.low_mode_rmse.iloc[-1] * np.sqrt(g.w.iloc[-1] / g.w)
        ax.plot(g.w, ref, color=GRAY, linestyle="--", label="_")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Mass $w$ of the state")
        ax.text(0.97, 0.95, r"dashed: $\propto w^{-1/2}$", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=GRAY)
        panel_title(ax, "(c) RMSE of the low-mass mode")
        _weight_axis(ax)
        ax.set_ylim(0, None)
        style_axis(ax)
        top_legend(fig, axes[0], ncol=5)
        save(fig, FIGDIR / "fig_low_prevalence.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figure(pd.read_csv(OUTDIR / "summary.csv"))
        return
    settings = [{"name": f"w{w:.3f}_{'shift' if sh else 'null'}", "w": w, "shifted": sh, "n": N}
                for w in LOW_WEIGHTS for sh in (False, True)]
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    s = summarize(raw)
    s.to_csv(OUTDIR / "summary.csv", index=False)
    make_figure(s)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
