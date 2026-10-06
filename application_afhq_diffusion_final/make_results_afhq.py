# Figures and tables for the AFHQv2 diffusion application.
# Inputs (application_afhq_diffusion_final/inputs/): results_core.zip, results_images.zip, analysis.zip
# Outputs: results/application_afhq_diffusion_final/{figures,tables}/
# Tolerance rule: delta = 0.10 * (minimum separation of the calibration modes), RMS scale.

import os, sys, json, csv, glob, zipfile, argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import modal_transport as mt
import mt_tools as mtt

ENCODERS = ["clip", "inception"]
ENC_NAME = {"clip": "CLIP", "inception": "Inception"}
MODELS = ["vp", "ve"]
CLASS = ["cat", "dog", "wild"]
ABBR = {"cat": "C", "dog": "D", "wild": "W"}
RULE = "rule_rms"
FIG_DIMS = [2, 3]
TABLE_DIMS = list(range(2, 9))
COMPACT_FROM = 10
IMG_STEPS = list(range(0, 41, 5))
SIGMA_TICKS = [0, 10, 20, 30, 40]
SEED = 20271001
ALPHA = 0.05
C_DEC = {"equivalent": mtt.GREEN, "meaningful change": mtt.VERMILLION, "inconclusive": mtt.GRAY}
LETTER = {"equivalent": "E", "meaningful change": "C", "inconclusive": "I", "unavailable": "-"}
C_MOD = {"vp": mtt.BLUE, "ve": mtt.VERMILLION}
LS_MOD = {"vp": "-", "ve": "--"}
C_CLS = {0: mtt.BLUE, 1: mtt.VERMILLION, 2: mtt.GREEN}


# ---------------- inputs ----------------

def unzip_inputs(inp):
    out = inp / "unzipped"
    for name in ["results_core", "results_images", "analysis"]:
        dst = out / name
        if not dst.exists():
            with zipfile.ZipFile(inp / (name + ".zip")) as z:
                z.extractall(dst)
    find = lambda pat: Path(sorted(glob.glob(str(out / "**" / pat), recursive=True))[0]).parent
    return find("splits.npz"), find("images_gen_vp.npz"), find("modal_tests_all.csv")


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if v is None or v == "":
                r[k] = None
                continue
            if v[:1] in "[{":
                try:
                    r[k] = json.loads(v)
                    continue
                except ValueError:
                    pass
            if v in ("True", "False"):
                r[k] = v == "True"
                continue
            try:
                r[k] = float(v) if any(c in v for c in ".eEn") else int(v)
            except ValueError:
                pass
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def load_core(core):
    sp = np.load(core / "splits.npz")
    data = {"y_A": sp["y_A"], "y_ref": np.concatenate([sp["y_B"], sp["y_test"]])}
    for e in ENCODERS:
        R = np.load(core / ("real_%s.npz" % e))
        sd = R["S_A"][:, :8].std(axis=0, ddof=1)
        ent = {"A": R["S_A"][:, :8] / sd, "ref": np.concatenate([R["S_B"], R["S_test"]])[:, :8] / sd, "gen": {}}
        for m in MODELS:
            G = np.load(core / ("gen_%s_%s.npz" % (m, e)))
            ent["gen"][m] = {"S": G["S"][:, :, :8] / sd, "sigma": G["sigma"], "seeds": G["seeds"]}
        data[e] = ent
    return data


# ---------------- refits (identical settings to the Kaggle analysis) ----------------

def box_for(A_d):
    h = mt.default_bandwidth(1000, A_d.shape[1])
    return np.stack([A_d.min(axis=0) - 2 * h, A_d.max(axis=0) + 2 * h], axis=1)


def fit_job(args):
    X, box, seed = args
    f = mt.fit_modal_configuration(np.ascontiguousarray(X, dtype=float), analysis_region=box, start_seed=seed)
    return np.asarray(f.modes), np.asarray(f.covariance)


COVS = {}


def refit_modes(data, workers):
    jobs, keys = [], []
    for e in ENCODERS:
        for d in FIG_DIMS:
            box = box_for(data[e]["A"][:, :d])
            keys.append((e, d, "ref"))
            jobs.append((data[e]["ref"][:, :d], box, SEED + 1))
            keys.append((e, d, "cal"))
            jobs.append((data[e]["A"][:, :d], box, SEED))
            for m in MODELS:
                for j in range(data[e]["gen"][m]["S"].shape[0]):
                    keys.append((e, d, m, j))
                    jobs.append((data[e]["gen"][m]["S"][j][:, :d], box, SEED + 7 * j + (0 if m == "vp" else 3)))
    print("refitting %d modal configurations on %d workers" % (len(jobs), workers), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(fit_job, jobs, chunksize=4))
    COVS.update({k: r[1] for k, r in zip(keys, res)})
    return {k: r[0] for k, r in zip(keys, res)}


def nearest(X, M):
    return np.argmin(((X[:, None, :] - M[None, :, :]) ** 2).sum(-1), axis=1)


# ---------------- Holm decisions ----------------

def apply_holm(tests, omni):
    """Decisions use the Holm-adjusted p-values computed in the Kaggle analysis: Holm over each (encoder, d)
    family of equivalence and change tests, and over each (encoder, space) family of MMD and energy tests."""
    for r in tests:
        pe = r.get("p_eq_holm_" + RULE)
        pc = r.get("p_chg_holm_" + RULE)
        pe = 1.0 if pe is None else float(pe)
        pc = 1.0 if pc is None else float(pc)
        r["decision_" + RULE] = ("unavailable" if not r.get("available") else "equivalent" if pe < ALPHA
                                 else "meaningful change" if pc < ALPHA else "inconclusive")
    for o in omni:
        o["mmd_adj"], o["energy_adj"] = float(o["mmd_holm"]), float(o["energy_holm"])


# ---------------- tables ----------------

def make_tables(an, data, modes, summ, tdir):
    tests = read_csv(an / "modal_tests_all.csv")
    omni = read_csv(an / "omnibus_tests_all.csv")
    apply_holm(tests, omni)
    cal = summ["calibration"]

    rows = []
    for e in ENCODERS:
        for d in TABLE_DIMS:
            c = cal["%s_d%d" % (e, d)]
            rows.append({"encoder": e, "d": d, "K_calibration": c["K_cal"], "K_reference": c["K_ref"],
                         "min_separation": c.get("s"), "delta": c["delta"][RULE] if c.get("delta") else None,
                         "mode_majority_class": c.get("majority"), "mode_purity": c.get("purity"),
                         "modes_calibration": c.get("modes_cal"), "modes_reference": c.get("modes_ref")})
    write_csv(rows, tdir / "calibration.csv")

    keep = ["encoder", "d", "model", "step", "sigma", "K_gen", "K_ref", "available", "D_hat", "U_D",
            "Z_" + RULE, "p_eq_" + RULE, "p_chg_" + RULE, "p_eq_holm_" + RULE, "p_chg_holm_" + RULE,
            "decision_" + RULE, "mode_labels"]
    mm = [{k.replace("_" + RULE, ""): r.get(k) for k in keep} for r in tests]
    write_csv(mm, tdir / "matched_mode_tests.csv")

    write_csv([o for o in omni if str(o["space"]).startswith("pca")], tdir / "omnibus_tests_pca.csv")
    write_csv([o for o in omni if o["space"] == "full"], tdir / "omnibus_tests_full_features_not_in_figures.csv")

    for e in ENCODERS:
        steps = sorted({r["step"] for r in mm if r["encoder"] == e})
        wide = []
        for s in steps:
            row = {"step": s, "sigma": [r["sigma"] for r in mm if r["encoder"] == e and r["step"] == s][0]}
            for d in TABLE_DIMS:
                for m in MODELS:
                    r = [x for x in mm if x["encoder"] == e and x["d"] == d and x["model"] == m and x["step"] == s][0]
                    row["d%d_%s" % (d, m)] = LETTER[r["decision"]]
            wide.append(row)
        write_csv(wide, tdir / ("verdicts_%s.csv" % e))
        cols = ["d%d_%s" % (d, m) for d in TABLE_DIMS if d >= 4 for m in MODELS]
        if not cols:
            continue
        with open(tdir / ("verdicts_%s_d4to8.tex" % e), "w") as fh:
            fh.write("\\begin{tabular}{rr" + "c" * len(cols) + "}\n\\toprule\n")
            fh.write("Step & $\\sigma$ & " + " & ".join("$d=%s$ %s" % (c[1], c.split("_")[1].upper()) for c in cols)
                     + " \\\\\n\\midrule\n")
            for row in wide:
                sg = "final" if row["sigma"] == 0 else "%.3g" % row["sigma"]
                fh.write("%d & %s & %s \\\\\n" % (row["step"], sg, " & ".join(row[c] for c in cols)))
            fh.write("\\bottomrule\n\\end{tabular}\n% E equivalent, C meaningful change, I inconclusive, - unavailable\n")

    desc = []
    for e in ENCODERS:
        D = summ["descriptive"][e]
        for m in MODELS:
            g = D["models"][m]
            for j, s in enumerate(g["sigma"]):
                desc.append({"encoder": e, "model": m, "step": j, "sigma": s,
                             "prop_cat": g["proportions"][j][0], "prop_dog": g["proportions"][j][1],
                             "prop_wild": g["proportions"][j][2], "agreement_with_final_class": g["commitment"][j],
                             "mass_between_modes": g["gap_fraction"][j],
                             "real_prop_cat": D["real_proportions"][0], "real_prop_dog": D["real_proportions"][1],
                             "real_prop_wild": D["real_proportions"][2],
                             "real_mass_between_modes": D["gap_real_ref"]})
    write_csv(desc, tdir / "descriptives.csv")

    pm = []
    for e in ENCODERS:
        for d in FIG_DIMS:
            ref, calm = modes[(e, d, "ref")], modes[(e, d, "cal")]
            maj = [CLASS.index(c) for c in cal["%s_d%d" % (e, d)]["majority"]] if len(calm) else []
            for m in MODELS:
                for j in range(data[e]["gen"][m]["S"].shape[0]):
                    g = modes[(e, d, m, j)]
                    row = {"encoder": e, "d": d, "model": m, "step": j, "sigma": float(data[e]["gen"][m]["sigma"][j]),
                           "K_gen": len(g), "K_ref": len(ref)}
                    if len(g) == len(ref) and len(g) > 0:
                        from scipy.optimize import linear_sum_assignment
                        cost = ((g[:, None] - ref[None]) ** 2).sum(-1)
                        r_, c_ = linear_sum_assignment(cost)
                        cls_ref = [maj[k] for k in nearest(ref, calm)] if len(calm) else [None] * len(ref)
                        for a, b in zip(r_, c_):
                            row["disp_%s" % CLASS[cls_ref[b]] if cls_ref[b] is not None else "disp_%d" % b] = \
                                float(np.sqrt(cost[a, b]))
                    pm.append(row)
    write_csv(pm, tdir / "per_mode_displacement.csv")
    return mm, omni


# ---------------- figures ----------------

def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", dpi=220, facecolor="white")
    mtt.save(fig, Path(str(path) + ".pdf"))


def columns(steps, compact):
    if not compact:
        return [[s] for s in steps]
    return [list(range(steps[0], COMPACT_FROM))] + [[s] for s in steps if s >= COMPACT_FROM]


def col_label(group):
    return "%d-%d" % (group[0], group[-1]) if len(group) > 1 else str(group[0])


def step_axes(ax, groups, sig_of):
    ticks = [k for k, g in enumerate(groups) if (len(g) > 1) or g[0] in SIGMA_TICKS or g[0] % 5 == 0]
    ax.set_xticks(ticks, [col_label(groups[k]) for k in ticks])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel(r"Sampler step $i$ (time $t_i = \sigma_i$)")
    top = ax.secondary_xaxis("top")
    tt = [k for k, g in enumerate(groups) if len(g) == 1 and g[0] in SIGMA_TICKS]
    top.set_xticks(tt, ["final" if sig_of[groups[k][0]] == 0 else "%.2g" % sig_of[groups[k][0]] for k in tt])
    top.set_xlabel(r"Noise level $\sigma$", fontsize=8)
    top.tick_params(labelsize=7.5)


def fig_decisions(mm, omni, e, d, compact, path):
    from matplotlib.patches import Patch
    rows = [r for r in mm if r["encoder"] == e and r["d"] == d]
    steps = sorted({r["step"] for r in rows})
    sig_of = {r["step"]: r["sigma"] for r in rows}
    groups = columns(steps, compact)
    with mtt.plt.rc_context(mtt.RC):
        fig, ax = mtt.new_figure(1, 1, 7.0 if not compact else 6.0, 2.9)
        yt, yl, y = [], [], 0
        for m in MODELS:
            R = {r["step"]: r for r in rows if r["model"] == m}
            O = {o["step"]: o for o in omni if o["encoder"] == e and o["model"] == m and o["space"] == "pca_d%d" % d}
            for k, (key, nm) in enumerate([("mmd_adj", "MMD"), ("energy_adj", "Energy")]):
                for x, g in enumerate(groups):
                    rej = [O[s][key] < ALPHA for s in g]
                    col = mtt.VERMILLION if all(rej) else mtt.GREEN if not any(rej) else mtt.GRAY
                    ax.add_patch(mtt.plt.Rectangle((x - 0.45, y + k - 0.4), 0.9, 0.8, lw=0, color=col))
                yt.append(y + k)
                yl.append("EDM-%s: %s" % (m.upper(), nm))
            ky = y + 2
            labs = []
            for x, g in enumerate(groups):
                decs = {R[s]["decision"] for s in g}
                dec = decs.pop() if len(decs) == 1 else "mixed"
                ks = sorted({R[s]["K_gen"] for s in g})
                lab = " | ".join(R[g[-1]]["mode_labels"] or [])
                labs.append((dec, lab))
                if dec in ("unavailable", "mixed"):
                    ax.add_patch(mtt.plt.Rectangle((x - 0.45, ky - 0.4), 0.9, 0.8, facecolor="white",
                                                   edgecolor=mtt.GRAY, lw=0.6))
                    ax.text(x, ky, "/".join(str(v) for v in ks), ha="center", va="center", fontsize=5)
                else:
                    ax.add_patch(mtt.plt.Rectangle((x - 0.45, ky - 0.4), 0.9, 0.8, lw=0, color=C_DEC[dec]))
            x = 0
            while x < len(groups):
                if labs[x][0] not in ("unavailable", "mixed"):
                    x += 1
                    continue
                s0 = x
                while x + 1 < len(groups) and labs[x + 1][0] in ("unavailable", "mixed") and labs[x + 1][1] == labs[s0][1]:
                    x += 1
                ax.annotate("", xy=(s0 - 0.45, ky + 0.75), xytext=(x + 0.45, ky + 0.75),
                            arrowprops=dict(arrowstyle="|-|", lw=0.5, color=mtt.BLACK, shrinkA=0, shrinkB=0,
                                            mutation_scale=2))
                if labs[s0][1].count("|") <= 3:
                    ax.text((s0 + x) / 2, ky + 1.05, labs[s0][1], ha="center", va="top", fontsize=5.5)
                x += 1
            yt.append(ky)
            yl.append("EDM-%s: Matched-mode test" % m.upper())
            y = ky + 2.1
        ax.set_yticks(yt, yl, fontsize=7.5)
        ax.set_ylim(y - 0.7, -0.8)
        step_axes(ax, groups, sig_of)
        for s in ("left", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        handles = [Patch(color=mtt.VERMILLION, label="Reject equality / meaningful change"),
                   Patch(color=mtt.GREEN, label="No rejection / practically equivalent"),
                   Patch(color=mtt.GRAY, label="Inconclusive"),
                   Patch(facecolor="white", edgecolor=mtt.GRAY,
                         label="Unavailable (number of generated modes; C cat, D dog, W wild)")]
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
        save(fig, path)


def fig_distance(mm, cal, e, d, compact, path):
    rows = [r for r in mm if r["encoder"] == e and r["d"] == d]
    steps = sorted({r["step"] for r in rows})
    sig_of = {r["step"]: r["sigma"] for r in rows}
    groups = columns(steps, compact)
    xpos = {g[0]: k for k, g in enumerate(groups) if len(g) == 1}
    delta = cal["%s_d%d" % (e, d)]["delta"][RULE]
    with mtt.plt.rc_context(mtt.RC):
        fig, ax = mtt.new_figure(1, 1, 3.4, 2.3)
        for m in MODELS:
            rr = sorted([r for r in rows if r["model"] == m and r["available"] and r["step"] in xpos],
                        key=lambda r: r["step"])
            if not rr:
                continue
            x = np.array([xpos[r["step"]] for r in rr])
            dh = np.array([r["D_hat"] for r in rr], float)
            ax.plot(x, dh, LS_MOD[m], color=C_MOD[m], lw=1.1, label="EDM-%s" % m.upper())
            for xi, di, r in zip(x, dh, rr):
                ax.scatter(xi, di, s=10, color=C_DEC.get(r["decision"], "white"), edgecolor=C_MOD[m], lw=0.5,
                           zorder=4)
        ax.axhline(delta, ls=":", color=mtt.BLACK, lw=0.9, label=r"$\delta = %.3f$" % delta)
        ax.set_ylabel(r"$\widehat D_A$")
        ax.set_ylim(0, None)
        step_axes(ax, groups, sig_of)
        mtt.style_axis(ax)
        mtt.top_legend(fig, ax, ncol=3)
        save(fig, path)


def fig_descriptives(summ, e, fdir):
    D = summ["descriptive"][e]
    from matplotlib.lines import Line2D
    specs = [("proportions", "Class proportion"), ("commitment", "Agreement with final class"),
             ("gap_fraction", "Mass between modes")]
    for key, ylab in specs:
        with mtt.plt.rc_context(mtt.RC):
            fig, ax = mtt.new_figure(1, 1, 3.4, 2.3)
            sig = D["models"]["vp"]["sigma"]
            x = np.arange(len(sig))
            for m in MODELS:
                g = D["models"][m]
                if key == "proportions":
                    P = np.array(g["proportions"])
                    for k in range(3):
                        ax.plot(x, P[:, k], LS_MOD[m], color=C_CLS[k], lw=1.1)
                else:
                    ax.plot(x, g[key], LS_MOD[m], color=C_MOD[m], lw=1.2, label="EDM-%s" % m.upper())
            if key == "proportions":
                for k in range(3):
                    ax.axhline(D["real_proportions"][k], color=C_CLS[k], lw=0.7, ls=":")
                hs = [Line2D([], [], color=C_CLS[k], lw=1.5, label=CLASS[k].capitalize()) for k in range(3)]
                hs += [Line2D([], [], color=mtt.BLACK, ls=LS_MOD[m], label="EDM-%s" % m.upper()) for m in MODELS]
                hs += [Line2D([], [], color=mtt.BLACK, ls=":", lw=0.8, label="Real")]
                fig.legend(handles=hs, loc="outside upper center", ncol=3, frameon=False)
            else:
                if key == "gap_fraction":
                    ax.axhline(D["gap_real_ref"], color=mtt.BLACK, lw=0.8, ls=":", label="Real")
                mtt.top_legend(fig, ax, ncol=3)
            ax.set_ylabel(ylab)
            step_axes(ax, [[s] for s in range(len(sig))], {s: v for s, v in enumerate(sig)})
            mtt.style_axis(ax)
            save(fig, fdir / ("%s_%s" % (key.replace("gap_fraction", "mass_between_modes")
                                          .replace("commitment", "agreement_with_final_class")
                                          .replace("proportions", "class_proportions"), e)))


def fig_embedding(data, summ, modes, e, path):
    from matplotlib.lines import Line2D
    A2 = data[e]["A"][:, :2]
    with mtt.plt.rc_context(mtt.RC):
        fig, ax = mtt.new_figure(1, 1, 3.4, 3.0)
        for k in range(3):
            s = data["y_A"] == k
            ax.scatter(A2[s, 0], A2[s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0, rasterized=True)
        mc = modes[(e, 2, "cal")]
        ax.scatter(mc[:, 0], mc[:, 1], marker="*", s=130, color="white", edgecolor=mtt.BLACK, zorder=5)
        ax.set_xlabel("PC1 (standardized)")
        ax.set_ylabel("PC2 (standardized)")
        mtt.style_axis(ax, ygrid=False)
        hs = [Line2D([], [], marker="o", ls="", color=C_CLS[k], label=CLASS[k].capitalize()) for k in range(3)]
        hs.append(Line2D([], [], marker="*", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=10,
                         label="Mode"))
        fig.legend(handles=hs, loc="outside upper center", ncol=4, frameon=False)
        save(fig, path)


def fig_overlay(data, summ, modes, mm, e, path):
    from matplotlib.lines import Line2D
    cal = summ["calibration"]["%s_d2" % e]
    maj = np.array([CLASS.index(c) for c in cal["majority"]])
    mcal, mref = modes[(e, 2, "cal")], modes[(e, 2, "ref")]
    rng = np.random.default_rng(SEED)
    ref2 = data[e]["ref"][:, :2]
    iref = rng.choice(len(ref2), 2000, replace=False)
    stages = {}
    for m in MODELS:
        ks = [r["step"] for r in sorted([r for r in mm if r["encoder"] == e and r["d"] == 2 and r["model"] == m],
                                         key=lambda r: r["step"]) if r["K_gen"] == len(mref)]
        stages[m] = [ks[0] if ks else 40, data[e]["gen"][m]["S"].shape[0] - 1]
    with mtt.plt.rc_context(mtt.RC):
        fig, axes = mtt.new_figure(2, 3, 7.0, 4.6)
        for i in range(2):
            ax = axes[i, 0]
            for k in range(3):
                s = data["y_ref"][iref] == k
                ax.scatter(ref2[iref][s, 0], ref2[iref][s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0, rasterized=True)
            ax.scatter(mref[:, 0], mref[:, 1], marker="*", s=110, color="white", edgecolor=mtt.BLACK, zorder=5)
            mtt.panel_title(ax, "Real reference")
            for jm, m in enumerate(MODELS):
                ax = axes[i, jm + 1]
                st = stages[m][i]
                X = data[e]["gen"][m]["S"][st][:, :2]
                ig = rng.choice(len(X), 2000, replace=False)
                cls = maj[nearest(X[ig], mcal)]
                for k in range(3):
                    s = cls == k
                    ax.scatter(X[ig][s, 0], X[ig][s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0, rasterized=True)
                gm = modes[(e, 2, m, st)]
                ax.scatter(mref[:, 0], mref[:, 1], marker="*", s=110, facecolor="none", edgecolor=mtt.BLACK,
                           lw=0.8, zorder=5)
                if len(gm):
                    ax.scatter(gm[:, 0], gm[:, 1], marker="P", s=55, color="white", edgecolor=mtt.BLACK, zorder=6)
                sg = data[e]["gen"][m]["sigma"][st]
                mtt.panel_title(ax, "EDM-%s, step %d (%s)" % (m.upper(), st,
                                                               "final" if sg == 0 else r"$\sigma$=%.2g" % sg))
        xl = [min(a.get_xlim()[0] for a in axes.ravel()), max(a.get_xlim()[1] for a in axes.ravel())]
        yl = [min(a.get_ylim()[0] for a in axes.ravel()), max(a.get_ylim()[1] for a in axes.ravel())]
        for ax in axes.ravel():
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            mtt.style_axis(ax, ygrid=False)
        hs = [Line2D([], [], marker="o", ls="", color=C_CLS[k], label=CLASS[k].capitalize()) for k in range(3)]
        hs += [Line2D([], [], marker="*", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=9,
                      label="Real modes"),
               Line2D([], [], marker="P", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=7,
                      label="Generated modes")]
        fig.legend(handles=hs, loc="outside upper center", ncol=5, frameon=False)
        save(fig, path)
    return stages


def image_strip(rows_img, names, steps, sig, path, bottom_text, forward=False):
    """Image grid whose height is fitted to the images (no gaps between rows)."""
    from matplotlib.patches import FancyArrowPatch
    plt = mtt.plt
    W, left, right, gap = 7.0, 0.55, 0.05, 0.04
    nr, nc = len(rows_img), len(steps)
    cell = (W - left - right - gap * (nc - 1)) / nc
    top_pad = 0.40 if forward else 0.05
    bottom = 0.42 + (0.32 if bottom_text else 0.0)
    H = top_pad + nr * cell + gap * (nr - 1) + bottom
    with plt.rc_context(mtt.RC):
        fig = plt.figure(figsize=(W, H))
        for i, rr in enumerate(rows_img):
            for j, im in enumerate(rr):
                x0, y0 = left + j * (cell + gap), H - top_pad - (i + 1) * cell - i * gap
                ax = fig.add_axes([x0 / W, y0 / H, cell / W, cell / H])
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=7.5)
                if i == nr - 1:
                    ax.set_xlabel("step %d\n%s" % (steps[j], "final" if sig[steps[j]] == 0 else
                                                    r"$\sigma$=%.2g" % sig[steps[j]]), fontsize=7.5)
        l, r = left / W, (W - right) / W
        if forward:
            ya = (H - 0.18) / H
            fig.add_artist(FancyArrowPatch((r, ya), (l, ya), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=12, color=mtt.BLACK, lw=1.0))
            fig.text((l + r) / 2, ya + 0.04 / H, r"Forward process: image $\rightarrow$ noise", ha="center",
                     va="bottom", fontsize=8.5)
        if bottom_text:
            yb = 0.30 / H
            fig.add_artist(FancyArrowPatch((l, yb), (r, yb), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=12, color=mtt.BLACK, lw=1.0))
            fig.text((l + r) / 2, yb - 0.04 / H, bottom_text, ha="center", va="top", fontsize=8.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path) + ".png", dpi=220, facecolor="white")
        mtt.save(fig, Path(str(path) + ".pdf"))


def image_figures(data, summ, modes, imgdir, fdir, tdir):
    e = "clip"
    cal = summ["calibration"]["clip_d2"]
    maj = np.array([CLASS.index(c) for c in cal["majority"]])
    mcal = modes[(e, 2, "cal")]
    prov = []
    for m in MODELS:
        G = np.load(imgdir / ("images_gen_%s.npz" % m))
        traj, sig = G["trajectories"], G["sigma"]
        S = data[e]["gen"][m]["S"]
        seeds = data[e]["gen"][m]["seeds"]
        nT = traj.shape[1]
        fin = maj[nearest(S[-1][:nT, :2], mcal)]
        pick = [int(np.where(fin == k)[0][0]) for k in (2, 0, 1)]
        rng = np.random.default_rng(3)
        x0 = traj[-1, pick[0]].astype(float) / 127.5 - 1
        eps = rng.standard_normal(x0.shape)
        fwd = [((np.clip((x0 + sig[s] * eps) / np.sqrt(1 + sig[s] ** 2), -1, 1) + 1) * 127.5).astype(np.uint8)
               for s in IMG_STEPS]
        rows_img = [fwd] + [[traj[s, p] for s in IMG_STEPS] for p in pick]
        image_strip(rows_img, ["Forward\n(noise added)", "Reverse\n(wild)", "Reverse\n(cat)", "Reverse\n(dog)"],
                    IMG_STEPS, sig, fdir / "trajectories" / ("trajectories_%s" % m),
                    r"Reverse process (generation, EDM-%s): noise $\rightarrow$ image" % m.upper(), forward=True)
        for k, p in zip((2, 0, 1), pick):
            prov.append({"figure": "trajectories_%s" % m, "row": CLASS[k], "trajectory_index": p,
                         "seed": int(seeds[p])})
        proto = []
        for k in (2, 0, 1):
            row = []
            target = mcal[np.where(maj == k)[0][0]]
            for s in IMG_STEPS:
                gm = modes[(e, 2, m, s)]
                if len(gm) == 0:
                    row.append(np.full_like(traj[0, 0], 255))
                    continue
                mk = gm[np.argmin(((gm - target) ** 2).sum(1))]
                idx = int(np.argmin(((S[s][:nT, :2] - mk) ** 2).sum(1)))
                row.append(traj[s, idx])
                prov.append({"figure": "prototypes_%s" % m, "row": CLASS[k], "step": s, "sigma": float(sig[s]),
                             "generated_mode": mk.tolist(), "trajectory_index": idx, "seed": int(seeds[idx])})
            proto.append(row)
        image_strip(proto, ["Mode\n(wild)", "Mode\n(cat)", "Mode\n(dog)"], IMG_STEPS, sig,
                    fdir / "prototypes" / ("prototypes_%s" % m), None)
    write_csv(prov, tdir / "image_provenance.csv")


# ---------------- one-column main-text figures ----------------

MAIN_DIR = mtt.PLOTS / "run_afhq_diffusion"
MAIN_WIDTH = 3.4
MAIN_PROTO_STEPS = [0, 10, 15, 20, 25, 30, 40]


def fig_decisions_main(mm, omni, e, d, path):
    """One-column decision grid: steps 0-9 collapsed, MMD and energy in one row per model."""
    from matplotlib.patches import Patch
    plt = mtt.plt
    rows = [r for r in mm if r["encoder"] == e and r["d"] == d]
    steps = sorted({r["step"] for r in rows})
    sig_of = {r["step"]: r["sigma"] for r in rows}
    groups = columns(steps, True)
    disagree_seen = False
    with plt.rc_context(mtt.RC):
        fig, ax = mtt.new_figure(1, 1, MAIN_WIDTH, 2.05)
        yt, yl, y = [], [], 0.0
        for m in MODELS:
            R = {r["step"]: r for r in rows if r["model"] == m}
            O = {o["step"]: o for o in omni if o["encoder"] == e and o["model"] == m and o["space"] == "pca_d%d" % d}
            for x, g in enumerate(groups):
                rej = [O[s]["mmd_adj"] < ALPHA for s in g] + [O[s]["energy_adj"] < ALPHA for s in g]
                if all(rej):
                    ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, lw=0, color=mtt.VERMILLION))
                elif not any(rej):
                    ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, lw=0, color=mtt.GREEN))
                else:
                    disagree_seen = True
                    ax.add_patch(plt.Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor="white",
                                               edgecolor=mtt.VERMILLION, hatch="//////", lw=0.4))
            yt.append(y)
            yl.append("%s: MMD, energy" % m.upper())
            ky = y + 1.0
            for x, g in enumerate(groups):
                decs = {R[s]["decision"] for s in g}
                dec = decs.pop() if len(decs) == 1 else "mixed"
                if dec in ("unavailable", "mixed"):
                    ks = sorted({int(R[s]["K_gen"]) for s in g})
                    ax.add_patch(plt.Rectangle((x - 0.45, ky - 0.4), 0.9, 0.8, facecolor="white",
                                               edgecolor=mtt.GRAY, lw=0.5))
                    ax.text(x, ky, "/".join(str(v) for v in ks), ha="center", va="center", fontsize=4.8)
                else:
                    ax.add_patch(plt.Rectangle((x - 0.45, ky - 0.4), 0.9, 0.8, lw=0, color=C_DEC[dec]))
            yt.append(ky)
            yl.append("%s: matched-mode" % m.upper())
            y = ky + 1.6
        ax.set_yticks(yt, yl, fontsize=7)
        ax.set_ylim(y - 1.1, -0.7)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        bt = [0] + [k for k, g in enumerate(groups) if len(g) == 1 and g[0] >= 15 and g[0] % 5 == 0]
        ax.set_xticks(bt, [col_label(groups[k]).replace("-", "–") for k in bt], fontsize=7)
        ax.set_xlabel(r"Sampler step $i$", fontsize=8)
        top = ax.secondary_xaxis("top")
        tt = [k for k, g in enumerate(groups) if len(g) == 1 and g[0] in (10, 20, 30, 40)]
        top.set_xticks(tt, ["final" if sig_of[groups[k][0]] == 0 else "%.2g" % sig_of[groups[k][0]] for k in tt])
        top.set_xlabel(r"Noise level $\sigma$", fontsize=8)
        top.tick_params(labelsize=7)
        for s in ("left", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        handles = [Patch(color=mtt.VERMILLION, label="Reject / meaningful change"),
                   Patch(color=mtt.GREEN, label="No rejection / equivalent"),
                   Patch(color=mtt.GRAY, label="Inconclusive"),
                   Patch(facecolor="white", edgecolor=mtt.GRAY, lw=0.5, label="Modes differ (count)")]
        if disagree_seen:
            handles.append(Patch(facecolor="white", edgecolor=mtt.VERMILLION, hatch="//////",
                                 label="MMD and energy disagree"))
        fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=6.5,
                   handlelength=1.2, columnspacing=1.0)
        save(fig, path)


def fig_prototypes_main(data, summ, modes, imgdir, m, path):
    """One-column image grid: generated image closest to each mode at MAIN_PROTO_STEPS."""
    plt = mtt.plt
    e = "clip"
    cal = summ["calibration"]["clip_d2"]
    maj = np.array([CLASS.index(c) for c in cal["majority"]])
    mcal = modes[(e, 2, "cal")]
    G = np.load(imgdir / ("images_gen_%s.npz" % m))
    traj, sig = G["trajectories"], G["sigma"]
    S = data[e]["gen"][m]["S"]
    nT = traj.shape[1]
    grid = []
    for k in (2, 0, 1):
        target = mcal[np.where(maj == k)[0][0]]
        row = []
        for s in MAIN_PROTO_STEPS:
            gm = modes[(e, 2, m, s)]
            if len(gm) == 0:
                row.append(np.full_like(traj[0, 0], 255))
                continue
            mk = gm[np.argmin(((gm - target) ** 2).sum(1))]
            row.append(traj[s, int(np.argmin(((S[s][:nT, :2] - mk) ** 2).sum(1)))])
        grid.append(row)
    nr, nc = len(grid), len(MAIN_PROTO_STEPS)
    left, right, gap, top_pad, bottom = 0.22, 0.02, 0.025, 0.02, 0.30
    cell = (MAIN_WIDTH - left - right - gap * (nc - 1)) / nc
    height = top_pad + nr * cell + gap * (nr - 1) + bottom
    with plt.rc_context(mtt.RC):
        fig = plt.figure(figsize=(MAIN_WIDTH, height))
        for i, row in enumerate(grid):
            for j, im in enumerate(row):
                x0 = left + j * (cell + gap)
                y0 = height - top_pad - (i + 1) * cell - i * gap
                ax = fig.add_axes([x0 / MAIN_WIDTH, y0 / height, cell / MAIN_WIDTH, cell / height])
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(["Wild", "Cat", "Dog"][i], fontsize=7, labelpad=2)
                if i == nr - 1:
                    s = MAIN_PROTO_STEPS[j]
                    ax.set_xlabel("%d\n%s" % (s, "final" if sig[s] == 0 else r"$\sigma$=%.2g" % sig[s]),
                                  fontsize=6, labelpad=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path) + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(str(path) + ".png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main_figures(an, data, summ, modes, imgdir):
    tests = read_csv(an / "modal_tests_all.csv")
    omni = read_csv(an / "omnibus_tests_all.csv")
    apply_holm(tests, omni)
    keep = ["encoder", "d", "model", "step", "sigma", "K_gen", "decision_" + RULE]
    mm = [{k.replace("_" + RULE, ""): r.get(k) for k in keep} for r in tests]
    pca = [o for o in omni if str(o["space"]).startswith("pca")]
    fig_decisions_main(mm, pca, "clip", 2, MAIN_DIR / "decisions_clip_d2_main")
    fig_prototypes_main(data, summ, modes, imgdir, "vp", MAIN_DIR / "prototypes_vp_main")
    print("main figures written to %s" % MAIN_DIR, flush=True)


def refit_main(data, workers):
    """Only the fits the main-text figures need: CLIP, d = 2, calibration and EDM-VP at MAIN_PROTO_STEPS."""
    e, d = "clip", 2
    box = box_for(data[e]["A"][:, :d])
    keys = [(e, d, "cal")] + [(e, d, "vp", j) for j in MAIN_PROTO_STEPS]
    jobs = [(data[e]["A"][:, :d], box, SEED)] + \
           [(data[e]["gen"]["vp"]["S"][j][:, :d], box, SEED + 7 * j) for j in MAIN_PROTO_STEPS]
    print("refitting %d modal configurations on %d workers" % (len(jobs), workers), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(fit_job, jobs))
    return {k: r[0] for k, r in zip(keys, res)}


def enc_dir(fdir, e):
    return fdir / ("clip" if e == "clip" else "inception_appendix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(HERE / "inputs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "application_afhq_diffusion_final"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--main-only", action="store_true",
                    help="only the two one-column main-text figures (8 refits instead of all)")
    ap.add_argument("--distance-only", action="store_true",
                    help="only the distance figures (no refits; reads the Kaggle test table)")
    args = ap.parse_args()
    out = Path(args.out)
    fdir, tdir = out / "figures", out / "tables"
    core, imgdir, an = unzip_inputs(Path(args.inputs))
    summ = json.load(open(an / "summary.json"))
    if args.distance_only:
        tests = read_csv(an / "modal_tests_all.csv")
        apply_holm(tests, [])
        keep = ["encoder", "d", "model", "step", "sigma", "available", "D_hat", "decision_" + RULE]
        mm = [{k.replace("_" + RULE, ""): r.get(k) for k in keep} for r in tests]
        for e in ENCODERS:
            for d in FIG_DIMS:
                for compact in (False, True):
                    tag = "compact" if compact else "full"
                    fig_distance(mm, summ["calibration"], e, d, compact,
                                 enc_dir(fdir, e) / "distance" / ("distance_%s_d%d_%s" % (e, d, tag)))
        print("distance figures written to %s" % fdir, flush=True)
        return
    data = load_core(core)
    if args.main_only:
        main_figures(an, data, summ, refit_main(data, args.workers), imgdir)
        return
    modes = refit_modes(data, args.workers)
    main_figures(an, data, summ, modes, imgdir)
    print("tables", flush=True)
    mm, omni = make_tables(an, data, modes, summ, tdir)
    print("figures", flush=True)
    for e in ENCODERS:
        for d in FIG_DIMS:
            for compact in (False, True):
                tag = "compact" if compact else "full"
                ed = enc_dir(fdir, e)
                fig_decisions(mm, omni, e, d, compact, ed / "decisions" / ("decisions_%s_d%d_%s" % (e, d, tag)))
                fig_distance(mm, summ["calibration"], e, d, compact,
                             ed / "distance" / ("distance_%s_d%d_%s" % (e, d, tag)))
        ed = enc_dir(fdir, e)
        st = fig_overlay(data, summ, modes, mm, e, ed / "overlay" / ("overlay_%s_d2" % e))
        print("  %s done (overlay steps %s)" % (e, st), flush=True)
    image_figures(data, summ, modes, imgdir, enc_dir(fdir, "clip"), tdir)
    print("done; outputs in %s" % out, flush=True)


if __name__ == "__main__":
    main()