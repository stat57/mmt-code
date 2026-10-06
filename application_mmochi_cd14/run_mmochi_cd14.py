"""Application: CITE-seq landmark registration under depletion of CD14-high cells.

Data: 10x Genomics PBMC CITE-seq filtered feature matrices (1k and 5k protein
v3), placed next to this script.  Counts over the ADTs shared by both runs are
normalized to counts per thousand and log1p-transformed; CD14 from the 5k run
is analyzed, excluding zero counts.

Design: each repetition splits the cells at random into two halves.  The
*ordinary split* compares the halves; the *95% depletion* comparison retains
each CD14-high cell (log(CP1k+1) > 2) of the second half with probability .05.
The matched-mode tolerance test (X = [0, 10], delta = .20) is compared with KS,
MMD, Energy and a W1 permutation test on the same pairs; 200 repetitions.

Tables (results/run_mmochi_cd14/):
  replicates__<key>.csv  per repetition and comparison: modes, D_hat, U_D, decision, exact-equality
                         T0 and p-value, omnibus p-values
  summary.csv            recovery, decisions, median D_hat and U_D, exact-equality rejections
                         (among two-mode comparisons), omnibus rejection rates
  tolerance_grid.csv     three-way decisions for delta in {.10, .15, .20, .25, .30}
Figures (Plots/run_mmochi_cd14/):
  fig_citeseq_col.pdf (main text), fig_citeseq.pdf, fig_citeseq_tolerance.pdf (appendix)
<key> hashes the code and settings; a replicates file is reused only if its key
matches.

Usage: python run_mmochi_cd14.py [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402
from scipy.stats import norm  # noqa: E402

from modal_transport import modal_transport_test  # noqa: E402
from mt_tools import _cache_key, plots_dir  # noqa: E402
from mt_tools import (BLACK, BLUE, GRAY, GREEN, N_JOBS, RC, VERMILLION, kde_curve, new_figure,  # noqa: E402
                      panel_title, plt, results_dir, run_omnibus, save, style_axis, top_legend)

FILE_1K = HERE / "pbmc_1k_protein_v3_filtered_feature_bc_matrix.h5"
FILE_5K = HERE / "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5"
OUTDIR = results_dir(__file__)
FIGDIR = plots_dir(__file__)
REPS = 200
SEED = 20260907
THRESHOLD = 2.0     # CD14-high: log(CP1k + 1) > 2
RETAIN = 0.05       # 95% depletion of CD14-high cells in the second half
TOLERANCE = 0.20
TOLERANCES = (0.10, 0.15, 0.20, 0.25, 0.30)
ALPHA = 0.05
N_PERM = 500
REGION = np.array([[0.0, 10.0]])
TESTS = ("ks", "mmd", "energy", "w1")


def load_adt(path: Path) -> tuple[np.ndarray, list[str]]:
    import h5py

    with h5py.File(path, "r") as h:
        g = h["matrix"]
        mat = csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        names = np.asarray(g["features/name"]).astype(str)
        types = np.asarray(g["features/feature_type"]).astype(str)
    idx = np.flatnonzero(types == "Antibody Capture")
    return mat[idx, :].toarray().astype(float).T, [names[i].split("_TotalSeq")[0] for i in idx]


def cd14_log_cp1k(file_1k: Path, file_5k: Path) -> np.ndarray:
    (x1, n1), (x5, n5) = load_adt(file_1k), load_adt(file_5k)
    shared = sorted(set(n1) & set(n5))
    x = x5[:, [n5.index(m) for m in shared]]
    total = x.sum(axis=1, keepdims=True)
    cp1k = np.divide(1000.0 * x, total, out=np.zeros_like(x), where=total > 0)
    cd14 = np.log1p(cp1k)[:, shared.index("CD14")]
    return cd14[cd14 > 0]


def split(cd14: np.ndarray, rep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    """Random halves (x, y) and the depleted second half for repetition ``rep``."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, rep]))
    perm = rng.permutation(len(cd14))
    x, y = cd14[perm[: len(cd14) // 2]], cd14[perm[len(cd14) // 2:]]
    y_dep = y[(y <= THRESHOLD) | (rng.random(len(y)) < RETAIN)]
    return x, y, y_dep, rng


def one_rep(cd14: np.ndarray, rep: int) -> list[dict]:
    x, y, y_dep, rng = split(cd14, rep)
    rows = []
    for kind, yy in (("ordinary", y), ("depletion", y_dep)):
        res = modal_transport_test(x[:, None], yy[:, None], analysis_region=REGION, tolerance=TOLERANCE, alpha=ALPHA)
        ok = bool(res["applicable"] and res["K"] == 2)
        row = {"rep": rep, "kind": kind, "n_x": len(x), "n_y": len(yy),
               "high_frac_x": float(np.mean(x > THRESHOLD)), "high_frac_y": float(np.mean(yy > THRESHOLD)),
               "K_x": res["K_x"], "K_y": res["K_y"], "two_mode": ok,
               "modes_x": np.round(res["modes_x"].ravel(), 4).tolist() if ok else [],
               "modes_y": np.round(res["modes_y"].ravel(), 4).tolist() if ok else []}
        for k, key in (("D_hat", "D_hat"), ("U_D", "U_D"), ("Q_bc", "Q_bc"), ("s_Q", "s_Q"),
                       ("Z_tolerance", "Z_tolerance")):
            row[k] = float(res[key]) if ok else np.nan
        row["T0"] = float(res["T0"]) if ok else np.nan
        row["p_exact"] = float(res["p_value"]) if ok else np.nan
        row["decision"] = res["tolerance_decision"] if ok else "not_applicable"
        row.update(run_omnibus(x, yy, TESTS, seed=int(rng.integers(2**31)), n_perm=N_PERM))
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, g in df.groupby("kind", sort=False):
        a = g[g.two_mode]
        row = {"comparison": kind, "reps": len(g), "median_high_frac_y": g.high_frac_y.median(),
               "two_mode": int(g.two_mode.sum()), "equivalent": int((a.decision == "equivalent").sum()),
               "inconclusive": int((a.decision == "inconclusive").sum()),
               "meaningful": int((a.decision == "meaningfully_different").sum()),
               "median_D_hat": a.D_hat.median(), "median_U_D": a.U_D.median(),
               "exact_reject": int((a.p_exact < ALPHA).sum())}
        for t in TESTS:
            row[f"{t}_reject_rate"] = float((g[f"{t}_p"] < ALPHA).mean())
        out.append(row)
    return pd.DataFrame(out)


def tolerance_grid(df: pd.DataFrame) -> pd.DataFrame:
    a = df[(df.kind == "depletion") & df.two_mode]
    rows = []
    for d in TOLERANCES:
        z = (a.Q_bc - 2 * d * d) / a.s_Q
        rows.append({"tolerance": d, "n": len(a), "equivalent": float(np.mean(z < norm.ppf(ALPHA))),
                     "meaningful": float(np.mean(z > norm.ppf(1 - ALPHA))),
                     "inconclusive": float(np.mean((z >= norm.ppf(ALPHA)) & (z <= norm.ppf(1 - ALPHA))))})
    return pd.DataFrame(rows)


def figures(cd14, df, grid) -> None:
    dep = df[(df.kind == "depletion") & df.two_mode]
    rep = int(dep.iloc[(dep.D_hat - dep.D_hat.median()).abs().argmin()].rep)
    x, _, y, _ = split(cd14, rep)
    res = modal_transport_test(x[:, None], y[:, None], analysis_region=REGION, tolerance=TOLERANCE)
    grid_x = np.linspace(0.02, 7.5, 500)
    with plt.rc_context(RC):
        # Main figure: (a) representative split, (b) upper bounds U_D.
        fig, axes = new_figure(1, 2, 7.0, 3.0, gridspec_kw={"width_ratios": [1.35, 1]})
        ax = axes[0]
        for sample, color, lab, modes, h in ((x, BLUE, "Reference half", res["modes_x"], res["bandwidth_x"]),
                                             (y, VERMILLION, "95% depletion", res["modes_y"], res["bandwidth_y"])):
            ax.plot(grid_x, kde_curve(sample, grid_x, h), color=color, linewidth=1.9, label=lab)
            for m in modes.ravel():
                ax.axvline(m, ymax=0.06, color=color, linewidth=1.6)
        ax.set_xlabel(r"CD14 $\log(\mathrm{CP1k}+1)$")
        ax.set_ylabel("Estimated density")
        panel_title(ax, "(a) Depletion of CD14-high cells")
        ax.legend(frameon=False, loc="upper right")
        style_axis(ax, ygrid=False)
        ax = axes[1]
        vals = dep.U_D.to_numpy()
        jit = np.random.default_rng(1).normal(0, 0.06, len(vals))
        ax.scatter(1 + jit, vals, s=10, color=VERMILLION, alpha=0.25, edgecolors="none")
        q = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.vlines(1, q[0], q[2], color=BLACK, linewidth=3.5, zorder=5)
        ax.plot(1, q[1], "o", color="white", markeredgecolor=BLACK, markersize=5, zorder=6)
        ax.axhline(TOLERANCE, color=BLACK, linestyle="--", linewidth=1.0)
        ax.text(0.63, TOLERANCE + 0.004, r"$\delta=0.20$", va="bottom", ha="left", fontsize=8)
        ax.set_xlim(0.6, 1.6)
        ax.set_xticks([1], [f"95% depletion ({len(vals)} of {df.rep.nunique()})"])
        ax.set_ylim(0, TOLERANCE * 1.2)
        ax.set_ylabel(r"One-sided 95% upper bound $U_D$")
        panel_title(ax, "(b) Equivalence inference")
        style_axis(ax)
        save(fig, FIGDIR / "fig_citeseq.pdf")

        # Appendix: three-way decision as a function of the tolerance.
        fig, ax = new_figure(1, 1, 4.2, 3.2)
        for col, color, marker, lab in (("equivalent", GREEN, "o", "Equivalent"),
                                        ("inconclusive", GRAY, "s", "Inconclusive"),
                                        ("meaningful", VERMILLION, "^", "Meaningful change")):
            ax.plot(grid.tolerance, grid[col], color=color, marker=marker, label=lab)
        ax.axvline(TOLERANCE, color=BLACK, linestyle="--", linewidth=1.0, label=r"$\delta=0.20$ (main analysis)")
        ax.set_xlabel(r"Tolerance $\delta$")
        ax.set_ylabel("Decision probability")
        ax.set_xticks(list(TOLERANCES))
        ax.set_ylim(-0.03, 1.03)
        style_axis(ax)
        top_legend(fig, ax, ncol=2)
        save(fig, FIGDIR / "fig_citeseq_tolerance.pdf")

        # Main text: representative split only, single column.
        fig, ax = new_figure(1, 1, 3.25, 1.9)
        for sample, color, lab, modes, h in ((x, BLUE, "Reference half", res["modes_x"], res["bandwidth_x"]),
                                             (y, VERMILLION, "95% depletion", res["modes_y"], res["bandwidth_y"])):
            ax.plot(grid_x, kde_curve(sample, grid_x, h), color=color, linewidth=1.9, label=lab)
            for m in modes.ravel():
                ax.axvline(m, ymax=0.06, color=color, linewidth=1.6)
        ax.set_xlabel(r"CD14 $\log(\mathrm{CP1k}+1)$")
        ax.set_ylabel("Estimated density")
        ax.legend(frameon=False, loc="upper right")
        style_axis(ax, ygrid=False)
        save(fig, FIGDIR / "fig_citeseq_col.pdf")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cd14 = cd14_log_cp1k(FILE_1K, FILE_5K)
    print(f"{len(cd14)} cells with nonzero CD14", flush=True)
    key = _cache_key({"name": "replicates", "reps": REPS, "seed": SEED, "threshold": THRESHOLD,
                      "retain": RETAIN, "tolerance": TOLERANCE, "n_perm": N_PERM}, one_rep)
    path = OUTDIR / f"replicates__{key}.csv"
    if "--plot-only" in sys.argv:
        saved = sorted(OUTDIR.glob("replicates__*.csv"), key=lambda f: f.stat().st_mtime)
        if not saved:
            raise FileNotFoundError(f"No replicates file in {OUTDIR}; run without --plot-only first.")
        path = saved[-1]
    if path.exists():
        df = pd.read_csv(path)
        print(f"reusing {path}", flush=True)
    else:
        rows = Parallel(n_jobs=N_JOBS)(delayed(one_rep)(cd14, r) for r in range(REPS))
        df = pd.DataFrame([row for rr in rows for row in rr])
        df.to_csv(path, index=False)
    df["two_mode"] = df["two_mode"].astype(str).str.lower().eq("true")
    summary = summarize(df)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    grid = tolerance_grid(df)
    grid.to_csv(OUTDIR / "tolerance_grid.csv", index=False)
    figures(cd14, df, grid)
    print(summary.T.to_string())
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()