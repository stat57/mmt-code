"""Simulation: tolerance inference, three-way decisions, and calibration (main text + appendix).

Designs: four-mode Gaussian and five-mode Student-t_5 families, equal weights,
geometry deformed so that D_A = rho * delta (delta = 0.5) exactly;
n = m in {400, 800}; rho in {0, .5, .75, .9, 1, 1.1, 1.25, 1.5};
1000 replicates per point.
Tail study: Student-t_nu, nu in {3, 5, 8, 15, 30}, n = 800, rho in {.75, 1, 1.25}.

Reported (conditional on applicability unless stated):
  * three-way decision probabilities (equivalent / inconclusive / meaningful);
  * coverage of the one-sided upper bound U_D for rho > 0;
  * exact-equality Wald test size at rho = 0 and power for rho > 0;
  * bias and RMSE of the plug-in distance, and the ratio of the mean
    estimated standard error s_Q to the Monte Carlo SD of Q_bc;
  * applicability (all replicates).

Tables (results/run_tolerance/): raw/, calibration.csv, summary.csv.
Figures (Plots/run_tolerance/):
  fig_three_way_col.pdf, fig_one_sided_col.pdf (main text);
  fig_three_way.pdf, fig_population.pdf, fig_one_sided.pdf, fig_tails.pdf,
  fig_calibration.pdf (appendix).

Usage: python run_tolerance.py [--plot-only]
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

from mt_tools import (ALPHA, BLACK, BLUE, GRAY, LABELS, RC, REGION2, TOLERANCE, VERMILLION,  # noqa: E402
                      build_pair, errorbar, mt_fields, new_figure, panel_title, plt, proportion_with_ci,
                      results_dir, run_settings, save, style_axis, top_legend)
from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 2
REPS = 1000
MAIN_FAMILIES = ("gaussian4", "t5")
RHOS = (0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
SAMPLE_SIZES = (400, 800)
TAIL_DFS = (3, 5, 8, 15, 30)
TAIL_RHOS = (0.75, 1.0, 1.25)


def build_settings() -> list[dict]:
    specs = {(k, n, r) for k in MAIN_FAMILIES for n in SAMPLE_SIZES for r in RHOS}
    specs |= {(f"t{nu}", 800, r) for nu in TAIL_DFS for r in TAIL_RHOS}
    return [{"name": f"{k}_rho{r:.2f}_n{n}", "family": k, "n": n, "rho": r, "D_A": r * TOLERANCE}
            for k, n, r in sorted(specs)]


_PAIRS: dict = {}


def pair(key: str, D: float) -> dict:
    if (key, D) not in _PAIRS:
        _PAIRS[(key, D)] = build_pair(key, D)
    return _PAIRS[(key, D)]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["family"], s["D_A"])
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    res = modal_transport_test(X, Y, tolerance=TOLERANCE, alpha=ALPHA, analysis_region=REGION2)
    row = mt_fields(res)
    row["correct_cardinality"] = bool(res["K_x"] == res["K_y"] == p["f"].K)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, n, rho), g in raw.groupby(["family", "n", "rho"]):
        a = g[g["applicable"]]
        D = rho * TOLERANCE
        row = {"family": key, "n": n, "rho": rho, "D_A": D, "reps": len(g),
               "applicability": g["applicable"].mean(), "correct_cardinality": g["correct_cardinality"].mean()}
        for name in ("equivalent", "inconclusive", "meaningfully_different"):
            p, lo, hi = proportion_with_ci(a["decision"] == name)
            row.update({f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi})
        p, lo, hi = proportion_with_ci(a["p_exact"] < ALPHA)
        row.update({"p_exact_reject": p, "p_exact_reject_lo": lo, "p_exact_reject_hi": hi})
        if rho > 0:
            p, lo, hi = proportion_with_ci(a["U_D"] >= D)
            row.update({"coverage_UD": p, "coverage_UD_lo": lo, "coverage_UD_hi": hi})
        row.update({
            "bias_D": a["D_hat"].mean() - D,
            "rmse_D": float(np.sqrt(np.mean((a["D_hat"] - D) ** 2))),
            "se_ratio": a["s_Q"].mean() / a["Q_bc"].std(ddof=1) if len(a) > 1 else np.nan,
            "mean_success": 0.5 * (g["success_x"].mean() + g["success_y"].mean()),
            "max_residual": max_residual(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


EQ = r"$\Pr(Z_{\delta,\mathrm{bc}}<z_\alpha)$"
CHG = r"$\Pr(Z_{\delta,\mathrm{bc}}>z_{1-\alpha})$"
RHO = r"$\rho=D_A/\delta$"
N_STYLE = {400: (BLUE, "o", "-"), 800: (VERMILLION, "s", "--")}


def _prob_axis(ax, ylabel: bool = False) -> None:
    ax.axvline(1.0, color=BLACK, linestyle=":", linewidth=0.9)
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)
    if ylabel:
        ax.set_ylabel("Probability")


def make_figures(summary: pd.DataFrame) -> None:
    main = summary[summary.family.isin(MAIN_FAMILIES)]
    with plt.rc_context(RC):
        # Population geometry at the tolerance boundary (appendix).
        fig, axes = new_figure(1, 2, 7.0, 3.6)
        gx = np.linspace(-3.4, 3.4, 240)
        XX, YY = np.meshgrid(gx, gx)
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        for ax, key, tag in zip(axes, MAIN_FAMILIES, "ab"):
            p0, p1 = pair(key, 0.0), pair(key, TOLERANCE)
            Z0, Z1 = p0["f"].density2(pts).reshape(XX.shape), p1["g"].density2(pts).reshape(XX.shape)
            lev = max(Z0.max(), Z1.max()) * np.array([0.35, 0.6, 0.85])
            ax.contour(XX, YY, Z0, levels=lev, colors=[BLUE], linewidths=1.3)
            ax.contour(XX, YY, Z1, levels=lev, colors=[BLACK], linewidths=1.3, linestyles="dashed")
            ax.scatter(*p0["modes_f"].T, s=34, facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=5)
            ax.scatter(*p1["modes_g"].T, s=28, color=BLACK, zorder=6)
            ax.set_aspect("equal")
            panel_title(ax, f"({tag}) {LABELS[key]}")
            ax.set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")
        handles = [Line2D([], [], color=BLUE, linewidth=1.3, label=r"Baseline density ($D_A=0$)"),
                   Line2D([], [], color=BLACK, linewidth=1.3, linestyle="dashed",
                          label=rf"Boundary density ($D_A=\delta={TOLERANCE:.2f}$)"),
                   Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=BLUE,
                          label="Baseline population mode"),
                   Line2D([], [], marker="o", linestyle="none", color=BLACK, label="Matched boundary mode")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
        save(fig, FIGDIR / "fig_population.pdf")

        # One-sided rejection probabilities (appendix).
        fig, axes = new_figure(2, 2, 6.8, 5.2, sharex=True, sharey=True)
        for r, key in enumerate(MAIN_FAMILIES):
            for c, (prob, title) in enumerate((("equivalent", EQ), ("meaningfully_different", CHG))):
                ax = axes[r, c]
                for n, (color, marker, ls) in N_STYLE.items():
                    pnl = main[(main.family == key) & (main.n == n)].sort_values("rho")
                    errorbar(ax, pnl["rho"], pnl[f"p_{prob}"], pnl[f"p_{prob}_lo"], pnl[f"p_{prob}_hi"],
                             label=f"$n=m={n}$", color=color, marker=marker, linestyle=ls)
                ax.axhline(ALPHA, color=GRAY, linestyle="--", linewidth=0.9, label=r"$\alpha=.05$")
                _prob_axis(ax)
                panel_title(ax, f"({'abcd'[2 * r + c]}) {LABELS[key]}: {title}")
                if r == 1:
                    ax.set_xlabel(RHO)
            axes[r, 0].set_ylabel("Probability")
        top_legend(fig, axes[0, 0], ncol=3)
        save(fig, FIGDIR / "fig_one_sided.pdf")

        # Three-way decisions at n = 800 (main text).
        styles = {"equivalent": ("Equivalent", BLUE, "o", "-"),
                  "inconclusive": ("Inconclusive", BLACK, "s", "--"),
                  "meaningfully_different": ("Meaningful change", VERMILLION, "^", "-.")}
        fig, axes = new_figure(1, 2, 7.0, 3.0, sharey=True)
        for ax, key, tag in zip(axes, MAIN_FAMILIES, "ab"):
            pnl = main[(main.family == key) & (main.n == 800)].sort_values("rho")
            for name, (label, color, marker, ls) in styles.items():
                errorbar(ax, pnl["rho"], pnl[f"p_{name}"], pnl[f"p_{name}_lo"], pnl[f"p_{name}_hi"],
                         label=label, color=color, marker=marker, linestyle=ls)
            _prob_axis(ax)
            ax.set_xlabel(RHO)
            panel_title(ax, f"({tag}) {LABELS[key]} ($n=m=800$)")
        axes[0].set_ylabel("Decision probability")
        top_legend(fig, axes[0], ncol=3)
        save(fig, FIGDIR / "fig_three_way.pdf")

        # Tail robustness (appendix).
        tails = summary[summary.family.isin([f"t{nu}" for nu in TAIL_DFS]) & (summary.n == 800)].copy()
        tails["nu"] = tails.family.str[1:].astype(int)
        fig, axes = new_figure(1, 3, 8.0, 3.0, sharey=True)
        for ax, rho, tag in zip(axes, TAIL_RHOS, "abc"):
            pnl = tails[np.isclose(tails.rho, rho)].sort_values("nu")
            x = np.arange(len(pnl))
            for prob, color, marker, lab in (("equivalent", BLUE, "o", EQ),
                                             ("meaningfully_different", VERMILLION, "^", CHG)):
                errorbar(ax, x, pnl[f"p_{prob}"], pnl[f"p_{prob}_lo"], pnl[f"p_{prob}_hi"],
                         label=lab, color=color, marker=marker, linestyle="-")
            ax.axhline(ALPHA, color=GRAY, linestyle="--", linewidth=0.9, label=r"$\alpha=.05$")
            ax.set_xticks(x, pnl["nu"].astype(str))
            ax.set_xlabel(r"Degrees of freedom $\nu$")
            panel_title(ax, f"({tag}) $\\rho={rho:g}$")
            ax.set_ylim(-0.02, 1.02)
            style_axis(ax)
        axes[0].set_ylabel("Probability")
        top_legend(fig, axes[0], ncol=3)
        save(fig, FIGDIR / "fig_tails.pdf")

        # Calibration diagnostics (appendix): U_D coverage, exact-test rejection, SE ratio.
        fam_style = {"gaussian4": (BLUE, "o"), "t5": (VERMILLION, "s")}
        fig, axes = new_figure(1, 3, 9.0, 3.1)
        for key, (color, marker) in fam_style.items():
            for n, ls in ((400, "-"), (800, "--")):
                pnl = main[(main.family == key) & (main.n == n)].sort_values("rho")
                lab = f"{LABELS[key]}, $n=m={n}$"
                cov = pnl[pnl.rho > 0]
                errorbar(axes[0], cov["rho"], cov["coverage_UD"], cov["coverage_UD_lo"], cov["coverage_UD_hi"],
                         label=lab, color=color, marker=marker, linestyle=ls)
                pos = pnl[pnl.rho > 0]
                axes[1].plot(pos["rho"], pos["se_ratio"], color=color, marker=marker, linestyle=ls, label=lab)
                axes[2].plot(pnl["rho"], pnl["bias_D"], color=color, marker=marker, linestyle=ls, label=lab)
        axes[0].axhline(1 - ALPHA, color=GRAY, linestyle="--", linewidth=0.9)
        axes[0].set_ylim(0.85, 1.0)
        axes[0].set_ylabel("Probability")
        panel_title(axes[0], r"(a) Coverage of $U_D$")
        axes[1].axhline(1.0, color=GRAY, linestyle="--", linewidth=0.9)
        axes[1].set_ylim(0.7, 1.3)
        axes[1].set_ylabel("Ratio")
        panel_title(axes[1], r"(b) Mean $\widehat s_Q$ / Monte Carlo SD of $\widehat Q_{\mathrm{bc}}$")
        axes[2].axhline(0.0, color=GRAY, linestyle="--", linewidth=0.9)
        axes[2].set_ylabel(r"Bias of $\widehat D_A$")
        panel_title(axes[2], r"(c) Bias of $\widehat D_A$")
        for ax in axes:
            ax.set_xlabel(RHO)
            style_axis(ax)
        top_legend(fig, axes[0], ncol=2)
        save(fig, FIGDIR / "fig_calibration.pdf")

        # Main text, Figure 3: three-way decisions, Gaussian family, n = 800.
        legend_kw = dict(fontsize=6.5, columnspacing=0.9, handlelength=1.8, handletextpad=0.4)
        styles = {"equivalent": ("Equivalent", BLUE, "o", "-"),
                  "inconclusive": ("Inconclusive", BLACK, "s", "--"),
                  "meaningfully_different": ("Meaningful change", VERMILLION, "^", "-.")}
        pnl = main[(main.family == "gaussian4") & (main.n == 800)].sort_values("rho")
        fig, ax = new_figure(1, 1, 3.25, 1.8)
        for name, (label, color, marker, ls) in styles.items():
            errorbar(ax, pnl["rho"], pnl[f"p_{name}"], pnl[f"p_{name}_lo"], pnl[f"p_{name}_hi"],
                     label=label, color=color, marker=marker, linestyle=ls)
        _prob_axis(ax)
        ax.set_xlabel(RHO)
        ax.set_ylabel("Decision probability")
        top_legend(fig, ax, ncol=3, **legend_kw)
        save(fig, FIGDIR / "fig_three_way_col.pdf")

        # Main text, Figure 4: boundary calibration, Student-t_5 family.
        fig, axes = new_figure(1, 2, 3.25, 2.0, sharey=True)
        fig.get_layout_engine().set(wspace=0.08)
        for ax, (name, title), tag in zip(axes, (("equivalent", "Equivalence"),
                                                 ("meaningfully_different", "Change")), "ab"):
            for n, (color, marker, ls) in N_STYLE.items():
                pnl = main[(main.family == "t5") & (main.n == n)].sort_values("rho")
                errorbar(ax, pnl["rho"], pnl[f"p_{name}"], pnl[f"p_{name}_lo"], pnl[f"p_{name}_hi"],
                         label=f"$n={n}$", color=color, marker=marker, linestyle=ls)
            ax.axhline(ALPHA, color=GRAY, linestyle="--", linewidth=0.9, label=r"$\alpha$")
            _prob_axis(ax)
            ax.set_xticks([0, 0.5, 1, 1.5], ["0", ".5", "1", "1.5"])
            ax.set_xlabel(RHO)
            panel_title(ax, f"({tag}) {title}")
        axes[0].set_ylabel("Rejection probability")
        top_legend(fig, axes[0], ncol=3, **legend_kw)
        save(fig, FIGDIR / "fig_one_sided_col.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figures(pd.read_csv(OUTDIR / "summary.csv"))
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    settings = build_settings()
    cal = []
    for s in settings:
        p = pair(s["family"], s["D_A"])
        cal.append({"family": s["family"], "rho": s["rho"], "D_A_target": s["D_A"], "D_A_population": p["D_A"],
                    "deformation_a": p["a"], "matching_gap": p["matching_gap"]})
    pd.DataFrame(cal).drop_duplicates().to_csv(OUTDIR / "calibration.csv", index=False)
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    summary = summarize(raw)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    make_figures(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
