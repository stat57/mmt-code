"""Simulation: mass redistribution versus mode displacement (main text, Figures 1-2).

Families: four-mode Gaussian and five-mode Student-t_5 mixtures.  f is the
equal-weight mixture; g has weights
    w(lambda) = (1 - lambda) * uniform + lambda * skewed,
and component locations solved so that the modes of g are exactly those of f
displaced by a calibrated deformation with modal distance D_A = D.
lambda redistributes mass among the states; D moves the modes.

Design (500 replicates per point), for n = m = 1600 (main text) and
n = m = 400 (appendix; the displacement grid is doubled because the standard
error doubles):
  * mass redistribution only:                    D = 0, lambda in {0, .25, .5, .75, 1}
  * mass redistribution + mode displacement:      lambda = 1, D in the displacement grid
  * mode displacement only:                       lambda = 0, D in the displacement grid

Methods: matched-mode exact-equality test (non-applicable replicates count as
non-rejections), MMD, Energy, GEC (5-MST); level .05.

Tables (results/run_prevalence/): raw/, calibration.csv, summary.csv.
Figures (Plots/run_prevalence/):
  fig_contours_col.pdf           main text, Figure 1
  fig_prevalence_power_col.pdf   main text, Figure 2 (Student-t_5, n = 1600)
  fig_simulation_main.pdf        appendix: full version, Student-t_5
  fig_prevalence_gaussian.pdf    appendix: Gaussian family, n = 1600
  fig_power_lambda0.pdf          appendix: displacement only, n = 1600
  fig_prevalence_n400.pdf        appendix: both families at n = 400

Usage: python run_prevalence.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import (ALPHA, BLACK, BLUE, LABELS, METHOD_STYLE, RC, REGION2, build_pair, errorbar,  # noqa: E402
                      family, mt_fields, new_figure, panel_title, plt, proportion_with_ci, results_dir,
                      run_omnibus, run_settings, save, style_axis, top_legend)
from mt_tools import max_residual, plots_dir  # noqa: E402

OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
EXPERIMENT_ID = 1
REPS = 500
N_MAIN = 1600
FAMILIES = ("t5", "gaussian4")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DISPLACEMENTS = {400: (0.04, 0.08, 0.12, 0.16, 0.20), 1600: (0.02, 0.04, 0.06, 0.08, 0.10)}
METHODS = ("mt", "mmd", "energy", "gec")


def build_settings() -> list[dict]:
    settings = []
    for n, grid in DISPLACEMENTS.items():
        points = [(lam, 0.0) for lam in LAMBDAS]
        points += [(1.0, D) for D in grid] + [(0.0, D) for D in grid]
        settings += [{"name": f"{key}_lam{lam:.2f}_D{D:.2f}_n{n}", "family": key, "lam": lam, "D": D, "n": n}
                     for key in FAMILIES for lam, D in points]
    return settings


_PAIRS: dict = {}


def pair(key: str, D: float, lam: float) -> dict:
    if (key, D, lam) not in _PAIRS:
        _PAIRS[(key, D, lam)] = build_pair(key, D, lam)
    return _PAIRS[(key, D, lam)]


def one_rep(s: dict, rep: int, rng: np.random.Generator) -> dict:
    p = pair(s["family"], s["D"], s["lam"])
    X, Y = p["f"].sample(s["n"], rng), p["g"].sample(s["n"], rng)
    row = mt_fields(modal_transport_test(X, Y, analysis_region=REGION2, alpha=ALPHA))
    row["correct_cardinality"] = bool(row["K_x"] == row["K_y"] == p["f"].K)
    row.update(run_omnibus(X, Y, ("mmd", "energy", "gec"), seed=int(rng.integers(2**31))))
    for m in ("mmd", "energy", "gec"):
        row[f"{m}_reject"] = bool(row[f"{m}_p"] < ALPHA)
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, key, lam, D), g in raw.groupby(["n", "family", "lam", "D"]):
        row = {"n": n, "family": key, "lam": lam, "D": D, "reps": len(g), "applicability": g["applicable"].mean(),
               "correct_cardinality": g["correct_cardinality"].mean(), "max_residual": max_residual(g)}
        for m in METHODS:
            p, lo, hi = proportion_with_ci(g[f"{m}_reject"])
            row.update({f"{m}_rej": p, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def _curve(ax, panel: pd.DataFrame, x: str) -> None:
    for m in METHODS:
        label, color, marker, linestyle = METHOD_STYLE[m]
        errorbar(ax, panel[x], panel[f"{m}_rej"], panel[f"{m}_lo"], panel[f"{m}_hi"],
                 label=label, color=color, marker=marker, linestyle=linestyle)
    ax.axhline(ALPHA, color=BLACK, linestyle=":", linewidth=0.9, label=r"$\alpha=.05$")
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)


def _contours(ax, key: str, title: bool = True) -> None:
    f, g = pair(key, 0.0, 0.0)["f"], pair(key, 0.0, 1.0)["g"]
    gx = np.linspace(-3.4, 3.4, 240)
    XX, YY = np.meshgrid(gx, gx)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    Zf, Zg = f.density2(pts).reshape(XX.shape), g.density2(pts).reshape(XX.shape)
    levels = max(Zf.max(), Zg.max()) * np.array([0.12, 0.35, 0.65])
    ax.contour(XX, YY, Zf, levels=levels, colors=[BLUE], linewidths=1.2)
    ax.contour(XX, YY, Zg, levels=levels, colors=[BLACK], linewidths=1.2, linestyles="dashed")
    m = pair(key, 0.0, 0.0)["modes_f"]
    ax.scatter(m[:, 0], m[:, 1], s=22, facecolor="white", edgecolor=BLUE, linewidth=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if title:
        panel_title(ax, r"(a) $f$ (solid), $g$ (dashed)")


def make_figures(summary: pd.DataFrame) -> None:
    main = summary[summary.n == N_MAIN]
    # Figure 1 (t5, main text) and its Gaussian counterpart (appendix), n = 1600.
    for key, fname in (("t5", "fig_simulation_main.pdf"), ("gaussian4", "fig_prevalence_gaussian.pdf")):
        s = main[main.family == key]
        with plt.rc_context(RC):
            fig, axes = new_figure(1, 3, 8.6, 3.0, gridspec_kw={"width_ratios": [0.85, 1, 1]})
            _contours(axes[0], key)
            _curve(axes[1], s[s.D == 0].sort_values("lam"), "lam")
            axes[1].set_xlabel(r"Mass redistribution $\lambda$ ($D_A=0$)")
            axes[1].set_ylabel("Rejection probability")
            panel_title(axes[1], "(b) Mass redistributed, modes fixed")
            _curve(axes[2], s[s.lam == 1.0].sort_values("D"), "D")
            axes[2].set_xlabel(r"Modal displacement $D_A$ ($\lambda=1$)")
            panel_title(axes[2], "(c) Mass redistributed and modes moved")
            top_legend(fig, axes[1], ncol=5)
            save(fig, FIGDIR / fname)
    with plt.rc_context(RC):
        # Appendix: geometry only (lambda = 0), n = 1600.
        fig, axes = new_figure(1, 2, 7.2, 3.0, sharey=True)
        for ax, key, tag in zip(axes, FAMILIES, "ab"):
            s = main[(main.family == key) & (main.lam == 0.0)].sort_values("D")
            _curve(ax, s, "D")
            panel_title(ax, f"({tag}) " + LABELS[key] + r", $\lambda=0$")
            ax.set_xlabel(r"Modal displacement $D_A$")
        axes[0].set_ylabel("Rejection probability")
        top_legend(fig, axes[0], ncol=5)
        save(fig, FIGDIR / "fig_power_lambda0.pdf")

        # Appendix: both families at n = 400.
        small = summary[summary.n == 400]
        fig, axes = new_figure(2, 2, 7.2, 5.4, sharey=True)
        for r, key in enumerate(FAMILIES):
            s = small[small.family == key]
            _curve(axes[r, 0], s[s.D == 0].sort_values("lam"), "lam")
            _curve(axes[r, 1], s[s.lam == 1.0].sort_values("D"), "D")
            panel_title(axes[r, 0], f"({'ac'[r]}) {LABELS[key]}:\nmass redistributed, modes fixed")
            panel_title(axes[r, 1], f"({'bd'[r]}) {LABELS[key]}:\nmass redistributed and modes moved")
            axes[r, 0].set_ylabel("Rejection probability")
        axes[1, 0].set_xlabel(r"Mass redistribution $\lambda$ ($D_A=0$)")
        axes[1, 1].set_xlabel(r"Modal displacement $D_A$ ($\lambda=1$)")
        top_legend(fig, axes[0, 0], ncol=5)
        save(fig, FIGDIR / "fig_prevalence_n400.pdf")

        # Main text, Figure 2: Student-t_5, n = 1600, single column.
        t5 = main[main.family == "t5"]
        fig, axes = new_figure(1, 2, 3.3, 1.95, sharey=True)
        fig.get_layout_engine().set(wspace=0.08)
        _curve(axes[0], t5[t5.D == 0].sort_values("lam"), "lam")
        axes[0].set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        axes[0].set_xlabel(r"$\lambda$ ($D_A=0$)")
        axes[0].set_ylabel("Rejection probability")
        axes[0].set_title("(a) Mass shifted", loc="left", fontsize=8)
        _curve(axes[1], t5[t5.lam == 0.0].sort_values("D"), "D")
        axes[1].set_xlabel(r"$D_A$ ($\lambda=0$)")
        axes[1].set_xticks([0, 0.05, 0.1], ["0", ".05", ".1"])
        axes[1].set_title("(b) Modes moved", loc="left", fontsize=8)
        top_legend(fig, axes[0], ncol=3, fontsize=6.5, columnspacing=0.9, handlelength=1.8, handletextpad=0.4)
        save(fig, FIGDIR / "fig_prevalence_power_col.pdf")

        # Main text, Figure 1: contours of f and of g with identical modes.
        fig, ax = new_figure(1, 1, 1.9, 1.9)
        _contours(ax, "t5", title=False)
        save(fig, FIGDIR / "fig_contours_col.pdf")


def main() -> None:
    if "--plot-only" in sys.argv:
        make_figures(pd.read_csv(OUTDIR / "summary.csv"))
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    settings = build_settings()
    cal = [{"n": s["n"], "family": s["family"], "lam": s["lam"], "D": s["D"],
            "D_A_population": pair(s["family"], s["D"], s["lam"])["D_A"], "K": family(s["family"]).K}
           for s in settings]
    pd.DataFrame(cal).to_csv(OUTDIR / "calibration.csv", index=False)
    raw = run_settings(settings, one_rep, experiment=EXPERIMENT_ID, reps=REPS, raw_dir=OUTDIR / "raw")
    summary = summarize(raw)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    make_figures(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
