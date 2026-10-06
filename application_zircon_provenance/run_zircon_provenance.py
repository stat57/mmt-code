"""Application (appendix): detrital-zircon provenance (Smith et al., 2023, GSA Bulletin).

Data: supplemental U-Pb ages, workbook B36285_SuppData2.xlsx inside the
supplement archive 21554754.zip (placed next to this script); one sheet per
sample, ages in column ``best_age``.

Analysis window [1300, 1850] Ma: the KDE is fitted to each full sample and
modes are searched inside the window.  Bandwidth 20 Ma, tolerance
delta = 20 Ma, alpha = .05.  Reference SJMT3, comparison SJMT7.

Tables (results/run_zircon_provenance/):
  modal_comparison.csv       SJMT3 vs SJMT7: modes, D_hat, U_D, decision, younger-mode shares
  omnibus.csv                KS, Kuiper, MMD, Energy, W2 permutation on the window, SJMT3 vs each target
  targets.csv                window W2 and recovered modes of each target
  bandwidth_sensitivity.csv  h in {15, 20, 25, 30, 35, 40} Ma
  tolerance_sensitivity.csv  delta in {10, 15, 20, 25, 30} Ma
Figures (Plots/run_zircon_provenance/): fig_zircon.pdf, fig_zircon_bandwidth.pdf.

Usage: python run_zircon_provenance.py [--plot-only]
"""

from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from modal_transport import fit_modal_configuration, modal_transport_test  # noqa: E402
from mt_tools import (BLACK, BLUE, GRAY, GRID, RC, VERMILLION, kde_curve, new_figure, panel_title, plt,  # noqa: E402
                      results_dir, run_omnibus, save, style_axis, top_legend, wasserstein_distance_1d)
from mt_tools import plots_dir  # noqa: E402

ARCHIVE = HERE / "21554754.zip"
WORKBOOK = "B36285_SuppData2.xlsx"
OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
REFERENCE, COMPARISON, OTHERS = "SJMT3", "SJMT7", ()
WINDOW = (1300.0, 1850.0)
BANDWIDTH = 20.0
TOLERANCE = 20.0
ALPHA = 0.05
BANDWIDTHS = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
TOLERANCES = (10.0, 15.0, 20.0, 25.0, 30.0)
TESTS = ("ks", "kuiper", "mmd", "energy", "w2")
N_PERM = 2000
REGION = np.array([WINDOW])


def load_ages(sheet: str) -> np.ndarray:
    with zipfile.ZipFile(ARCHIVE) as z:
        name = next(n for n in z.namelist() if Path(n).name == WORKBOOK)
        df = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=sheet, header=1)
    return pd.to_numeric(df["best_age"], errors="coerce").dropna().to_numpy(float)


def window(x: np.ndarray) -> np.ndarray:
    return x[(x >= WINDOW[0]) & (x <= WINDOW[1])]


def test(x: np.ndarray, y: np.ndarray, h: float, delta: float) -> dict:
    return modal_transport_test(x[:, None], y[:, None], bandwidth_x=h, bandwidth_y=h, analysis_region=REGION,
                                tolerance=delta, alpha=ALPHA)


def main() -> None:
    ages = {s: load_ages(s) for s in (REFERENCE, COMPARISON, *OTHERS)}
    if "--plot-only" in sys.argv:
        comparison = pd.read_csv(OUTDIR / "modal_comparison.csv")
        mx = np.array(ast.literal_eval(comparison.modes_ref[0]))
        my = np.array(ast.literal_eval(comparison.modes_cmp[0]))
        sens = pd.read_csv(OUTDIR / "bandwidth_sensitivity.csv")
        sens["two_mode"] = sens["two_mode"].astype(str).str.lower().eq("true")
        make_figures(ages, mx, my, sens)
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    x, y = ages[REFERENCE], ages[COMPARISON]

    # Main comparison.
    res = test(x, y, BANDWIDTH, TOLERANCE)
    if not (res["applicable"] and res["K"] == 2):
        raise RuntimeError(f"Expected a two-mode comparison, got K = {res['K_x']}/{res['K_y']}.")
    mx, my = res["modes_x"].ravel(), res["modes_y"].ravel()[res["matching"]]
    cut = float(np.mean((mx + my) / 2))
    share = {s: float(np.mean(window(ages[s]) <= cut)) for s in (REFERENCE, COMPARISON)}
    comparison = pd.DataFrame([{
        "reference": REFERENCE, "comparison": COMPARISON, "n_ref": len(x), "n_cmp": len(y),
        "n_ref_window": len(window(x)), "n_cmp_window": len(window(y)),
        "modes_ref": np.round(mx, 2).tolist(), "modes_cmp": np.round(my, 2).tolist(),
        "D_hat": res["D_hat"], "U_D": res["U_D"], "Z": res["Z_tolerance"], "decision": res["tolerance_decision"],
        "cut_ma": cut,
        "younger_share_ref": share[REFERENCE], "younger_share_cmp": share[COMPARISON],
    }])
    comparison.to_csv(OUTDIR / "modal_comparison.csv", index=False)

    # Omnibus tests and recovered modes of each target.
    omni_rows, target_rows = [], []
    for target in (COMPARISON, *OTHERS):
        xr, yr = window(x), window(ages[target])
        omni_rows.append({"reference": REFERENCE, "target": target, "n_ref": len(xr), "n_target": len(yr),
                          **run_omnibus(xr, yr, TESTS, seed=0, n_perm=N_PERM)})
        fit = fit_modal_configuration(ages[target][:, None], bandwidth=BANDWIDTH, analysis_region=REGION)
        target_rows.append({"target": target, "W2_window": wasserstein_distance_1d(xr, yr, 2),
                            "K_target": len(fit.modes), "modes_target": np.round(fit.modes.ravel(), 2).tolist()})
    omni = pd.DataFrame(omni_rows)
    omni.to_csv(OUTDIR / "omnibus.csv", index=False)
    targets = pd.DataFrame(target_rows)
    targets.to_csv(OUTDIR / "targets.csv", index=False)

    # Bandwidth sensitivity.
    sens = []
    for h in BANDWIDTHS:
        r = test(x, y, h, TOLERANCE)
        two = bool(r["applicable"] and r["K"] == 2)
        sens.append({"bandwidth": h, "K_ref": r["K_x"], "K_cmp": r["K_y"], "two_mode": two,
                     "D_hat": r["D_hat"] if two else np.nan, "U_D": r["U_D"] if two else np.nan,
                     "decision": r["tolerance_decision"] if two else "not_applicable"})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUTDIR / "bandwidth_sensitivity.csv", index=False)

    # Tolerance sensitivity (same fit, tau = K delta^2).
    tol = []
    for d in TOLERANCES:
        z = (res["Q_bc"] - res["K"] * d * d) / res["s_Q"]
        tol.append({"tolerance": d, "Z": z,
                    "decision": "equivalent" if z < norm.ppf(ALPHA) else
                    "meaningfully_different" if z > norm.ppf(1 - ALPHA) else "inconclusive"})
    pd.DataFrame(tol).to_csv(OUTDIR / "tolerance_sensitivity.csv", index=False)

    make_figures(ages, mx, my, sens)
    print(comparison.T.to_string())
    print(omni.to_string(index=False))
    print(targets.to_string(index=False))
    print(sens.to_string(index=False))


def make_figures(ages: dict, mx: np.ndarray, my: np.ndarray, sens: pd.DataFrame) -> None:
    grid = np.linspace(*WINDOW, 700)
    with plt.rc_context(RC):
        fig, ax = new_figure(1, 1, 4.2, 3.0)
        for s, color, m in ((REFERENCE, BLUE, mx), (COMPARISON, VERMILLION, my)):
            ax.plot(grid, kde_curve(ages[s], grid, BANDWIDTH), color=color, linewidth=2.0, label=s)
            for v in m:
                ax.axvline(v, ymax=0.06, color=color, linewidth=1.6)
        ax.set_xlabel("Detrital-zircon age (Ma)")
        ax.set_ylabel("Estimated density")
        ax.legend(frameon=False, loc="upper center")
        style_axis(ax, ygrid=False)
        save(fig, FIGDIR / "fig_zircon.pdf")

        fig, axes = new_figure(1, 2, 7.25, 3.0)
        axes[0].plot(sens.bandwidth, sens.K_ref, "o-", color=BLUE, label=REFERENCE)
        axes[0].plot(sens.bandwidth, sens.K_cmp, "s--", color=VERMILLION, label=COMPARISON)
        axes[0].set_ylabel("Recovered number of modes")
        ks = sorted(set(sens.K_ref) | set(sens.K_cmp))
        axes[0].set_yticks(ks)
        axes[0].set_ylim(min(ks) - 0.3, max(ks) + 0.3)
        panel_title(axes[0], "(a) Modal cardinality")
        good = sens[sens.two_mode]
        axes[1].errorbar(good.bandwidth, good.D_hat, yerr=[np.zeros(len(good)), good.U_D - good.D_hat],
                         fmt="o-", color=BLUE, capsize=2, label=r"$\widehat D_A$ with upper bound $U_D$")
        axes[1].axhline(TOLERANCE, color=BLACK, linestyle="--", linewidth=1.0, label=r"$\delta=20$ Ma")
        for h in sens.bandwidth[~sens.two_mode]:
            axes[1].axvspan(h - 2.5, h + 2.5, color=GRAY, alpha=0.15, linewidth=0,
                            label="Cardinalities differ (not applicable)")
        axes[1].set_ylabel(r"$\widehat D_A$ (Ma)")
        axes[1].set_ylim(0, 1.2 * max(TOLERANCE, float(good.U_D.max()) if len(good) else TOLERANCE))
        panel_title(axes[1], "(b) Modal distance")
        for ax in axes:
            ax.set_xlabel("Bandwidth (Ma)")
            ax.set_xticks(list(BANDWIDTHS))
            style_axis(ax)
        top_legend(fig, axes, ncol=3)
        save(fig, FIGDIR / "fig_zircon_bandwidth.pdf")


if __name__ == "__main__":
    main()