# Matched-mode analysis of the AFHQv2 EDM experiment (v3 outputs).
# Kaggle: GPU T4 x2. Inputs: the output of the v3 notebook, and a dataset containing
# modal_transport.py and mt_tools.py. Output: /kaggle/working/analysis.zip

import os, sys, glob, json, time, zipfile, traceback, csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

ENCODERS = ["clip", "inception"]
MODELS = ["vp", "ve"]
DIMS = list(range(2, 9))
DELTA_RULES = {"rule_k": lambda s, K: 0.10 * s / np.sqrt(K), "rule_rms": lambda s, K: 0.10 * s}
DESCRIPTIVE_D = 2
ALPHA = 0.05
N_PERM = 4999
PURITY_K = 50
DISPLAY_STEPS = list(range(0, 41, 5))
SEED = 20271001
CLASS = ["cat", "dog", "wild"]
ABBR = {"cat": "C", "dog": "D", "wild": "W"}
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "analysis")
PREFLIGHT = os.environ.get("MMT_PREFLIGHT_ONLY", "0") == "1"


def find_dir(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return os.path.dirname(hits[0])


IN_RES = os.environ.get("MMT_RESULTS") or find_dir("/kaggle/input/**/splits.npz")
CODE = os.environ.get("MMT_CODE") or find_dir("/kaggle/input/**/modal_transport.py")
sys.path.insert(0, CODE)
import modal_transport as mt


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
        self.t0 = time.time()

    def __call__(self, msg):
        line = "[%7.1f min] %s" % ((time.time() - self.t0) / 60, msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def write_csv(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ---------------- data ----------------

def load_all(n_gen=None, stages=None):
    sp = np.load(os.path.join(IN_RES, "splits.npz"))
    data = {"y_A": sp["y_A"], "y_ref": np.concatenate([sp["y_B"], sp["y_test"]])}
    for e in ENCODERS:
        R = np.load(os.path.join(IN_RES, "real_%s.npz" % e))
        sd = R["S_A"][:, :max(DIMS)].std(axis=0, ddof=1)
        ent = {"A": R["S_A"][:, :max(DIMS)] / sd,
               "ref": np.concatenate([R["S_B"], R["S_test"]])[:, :max(DIMS)] / sd,
               "F_ref": np.concatenate([R["F_B"], R["F_test"]]).astype(np.float32), "gen": {}}
        for m in MODELS:
            G = np.load(os.path.join(IN_RES, "gen_%s_%s.npz" % (m, e)))
            S, F1, sig = G["S"], G["F_first1000"], G["sigma"]
            Ff = G["F_final"] if "F_final" in G.files else None
            ids = np.arange(S.shape[0])
            if stages is not None:
                S, F1, sig, ids = S[stages], F1[stages], sig[stages], ids[stages]
            if n_gen is not None:
                S, F1 = S[:, :n_gen], F1[:, :min(n_gen, F1.shape[1])]
                Ff = None if Ff is None else Ff[:n_gen]
            ent["gen"][m] = {"S": S[:, :, :max(DIMS)] / sd, "F1000": F1.astype(np.float32),
                             "F_final": None if Ff is None else Ff.astype(np.float32), "sigma": sig,
                             "steps": ids}
        data[e] = ent
    return data


# ---------------- modes and tests ----------------

def box_for(A_d):
    h = mt.default_bandwidth(1000, A_d.shape[1])
    return np.stack([A_d.min(axis=0) - 2 * h, A_d.max(axis=0) + 2 * h], axis=1)


WORKER_SRC = """import sys
import numpy as np


def fit_job(args):
    X, box, seed, code = args
    if code not in sys.path:
        sys.path.insert(0, code)
    import modal_transport as mt
    f = mt.fit_modal_configuration(np.ascontiguousarray(X, dtype=float), analysis_region=np.asarray(box),
                                   start_seed=seed)
    return {"modes": np.asarray(f.modes), "cov": np.asarray(f.covariance), "dens": np.asarray(f.densities),
            "h": float(f.bandwidth)}
"""
os.makedirs(WORK, exist_ok=True)
with open(os.path.join(WORK, "mmt_worker.py"), "w") as _fh:
    _fh.write(WORKER_SRC)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
from mmt_worker import fit_job


def mmt_from_fits(fx, fy, tols):
    """Matched-mode tolerance test, identical to modal_transport.modal_transport_test with A = I."""
    out = {"K_gen": len(fx["modes"]), "K_ref": len(fy["modes"]), "available": False, "D_hat": np.nan,
           "U_D": np.nan}
    for t in tols:
        out.update({"Z_" + t: np.nan, "p_eq_" + t: 1.0, "p_chg_" + t: 1.0})
    K = len(fx["modes"])
    if K == 0 or K != len(fy["modes"]):
        return out
    d = fx["modes"].shape[1]
    cost = ((fx["modes"][:, None, :] - fy["modes"][None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    pi = np.empty(K, dtype=int)
    pi[rows] = cols
    delta = (fx["modes"] - fy["modes"][pi]).reshape(K * d)
    idx = np.concatenate([np.arange(j * d, (j + 1) * d) for j in pi])
    Om = fx["cov"] + fy["cov"][np.ix_(idx, idx)]
    Om = 0.5 * (Om + Om.T)
    Q = float(delta @ delta)
    Qbc = Q - float(np.trace(Om))
    s2 = float(4.0 * delta @ Om @ delta + 2.0 * np.trace(Om @ Om))
    if s2 < 0 and abs(s2) < 1e-12 * max(1.0, abs(Q)):
        s2 = 0.0
    sQ = np.sqrt(s2) if s2 >= 0 else np.nan
    eig = np.linalg.eigvalsh(Om)
    pd = eig[0] > 1e-12 * max(1.0, float(np.max(np.abs(eig))))
    out["D_hat"] = float(np.sqrt(max(Q / K, 0.0)))
    out["U_D"] = float(np.sqrt(max(Qbc + norm.ppf(1 - ALPHA) * sQ, 0.0) / K)) if np.isfinite(sQ) else np.nan
    if not (pd and np.isfinite(sQ) and sQ > 0):
        return out
    out["available"] = True
    for t, tol in tols.items():
        z = (Qbc - K * tol ** 2) / sQ
        out.update({"Z_" + t: float(z), "p_eq_" + t: float(norm.cdf(z)), "p_chg_" + t: float(norm.sf(z))})
    return out


def nearest(X, M):
    return np.argmin(((X[:, None, :] - M[None, :, :]) ** 2).sum(-1), axis=1)


def labels_for(X, modes, final_cls):
    labs = []
    if len(modes) == 0:
        return labs
    a = nearest(X, modes)
    for k in range(len(modes)):
        f = np.bincount(final_cls[a == k], minlength=3) / max(1, np.sum(a == k))
        labs.append("+".join(ABBR[CLASS[c]] for c in range(3) if f[c] >= 0.2) or "-")
    return labs


def kde_values(data, pts, h, order=6, chunk=400):
    out = []
    for i in range(0, len(pts), chunk):
        u = (pts[i:i + chunk, None, :] - data[None, :, :]) / h
        k0, _, _ = mt._k1_all(u, order)
        out.append(np.prod(k0, axis=2).mean(axis=1) / h ** data.shape[1])
    return np.concatenate(out)


# ---------------- permutation tests on GPU ----------------

class PermGPU:
    def __init__(self, log):
        import torch
        self.torch = torch
        self.devs = ["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"]
        log("permutation tests on %s" % self.devs)

    def run(self, X, Y, n_perm, seed, dev):
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        Z = torch.from_numpy(np.vstack([X, Y]).astype(np.float32)).to(dev)
        n, N = len(X), len(X) + len(Y)
        sq = (Z * Z).sum(1)
        D2 = (sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T).clamp_min_(0.0)
        iu = torch.randint(0, N, (2, 2_000_000), generator=g).to(dev)
        keep = iu[0] != iu[1]
        bw = D2[iu[0][keep], iu[1][keep]].median()
        Kg = torch.exp(-D2 / bw)
        Dm = D2.sqrt_()
        del D2
        lab = torch.cat([torch.full((n,), 1.0 / n), torch.full((N - n,), -1.0 / (N - n))]).to(dev)
        sk, se = [], []
        done = 0
        while done < n_perm + 1:
            b = min(250, n_perm + 1 - done)
            P = torch.empty((N, b), device=dev)
            for j in range(b):
                P[:, j] = lab if done + j == 0 else lab[torch.randperm(N, generator=g).to(dev)]
            sk.append(((Kg @ P) * P).sum(0))
            se.append(-((Dm @ P) * P).sum(0))
            done += b
        sk, se = torch.cat(sk).cpu().numpy(), torch.cat(se).cpu().numpy()
        del Kg, Dm
        torch.cuda.empty_cache() if dev != "cpu" else None
        pv = lambda s: float((1 + np.sum(s[1:] >= s[0])) / (n_perm + 1))
        return pv(sk), pv(se)

    def run_many(self, jobs, log, label):
        from concurrent.futures import ThreadPoolExecutor
        out = [None] * len(jobs)

        def worker(k):
            dev = self.devs[k % len(self.devs)]
            for i in range(k, len(jobs), len(self.devs)):
                X, Y, n_perm, seed = jobs[i]
                out[i] = self.run(X, Y, n_perm, seed, dev)
                if i % 25 == 0:
                    log("  %s: %d/%d" % (label, i + 1, len(jobs)))

        with ThreadPoolExecutor(max_workers=len(self.devs)) as ex:
            list(ex.map(worker, range(len(self.devs))))
        return out


class PermCPU(PermGPU):
    def __init__(self, log):
        self.devs = ["cpu"]
        log("permutation tests on CPU (numpy)")

    def run(self, X, Y, n_perm, seed, dev):
        rng = np.random.default_rng(int(seed))
        Z = np.vstack([X, Y]).astype(np.float64)
        n, N = len(X), len(X) + len(Y)
        sq = (Z ** 2).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
        i, j = rng.integers(0, N, (2, 200000))
        bw = np.median(D2[i[i != j], j[i != j]])
        Kg, Dm = np.exp(-D2 / bw), np.sqrt(D2)
        lab = np.r_[np.full(n, 1.0 / n), np.full(N - n, -1.0 / (N - n))]
        P = np.stack([lab] + [rng.permutation(lab) for _ in range(n_perm)], axis=1)
        sk = np.einsum("ij,ij->j", P, Kg @ P)
        se = -np.einsum("ij,ij->j", P, Dm @ P)
        pv = lambda s: float((1 + np.sum(s[1:] >= s[0])) / (n_perm + 1))
        return pv(sk), pv(se)


# ---------------- main computation ----------------

def compute(data, log, workers, n_perm):
    res = {"calibration": {}, "tests": [], "omnibus": [], "descriptive": {}, "gen_modes": {}}
    fit_jobs, fit_keys = [], []
    cal = {}
    for e in ENCODERS:
        A, yA = data[e]["A"], data["y_A"]
        for d in DIMS:
            box = box_for(A[:, :d])
            cal[(e, d)] = {"box": box}
            fit_keys.append(("cal", e, d))
            fit_jobs.append((A[:, :d], box, SEED, CODE))
            fit_keys.append(("ref", e, d))
            fit_jobs.append((data[e]["ref"][:, :d], box, SEED + 1, CODE))
            for m in MODELS:
                S = data[e]["gen"][m]["S"]
                for j in range(S.shape[0]):
                    fit_keys.append(("gen", e, d, m, j))
                    fit_jobs.append((S[j][:, :d], box, SEED + 7 * j + (0 if m == "vp" else 3), CODE))
    log("fitting %d modal configurations on %d CPU workers" % (len(fit_jobs), workers))
    fits = {}
    import multiprocessing as mpc
    with ProcessPoolExecutor(max_workers=workers, mp_context=mpc.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(fit_job, fit_jobs, chunksize=1)):
            fits[fit_keys[i]] = r
            if i % 50 == 0:
                log("  fits %d/%d" % (i + 1, len(fit_jobs)))

    for e in ENCODERS:
        A, yA = data[e]["A"], data["y_A"]
        f2 = fits[("cal", e, DESCRIPTIVE_D)]
        maj2 = []
        for mu in f2["modes"]:
            nn = np.argsort(((A[:, :DESCRIPTIVE_D] - mu) ** 2).sum(1))[:PURITY_K]
            maj2.append(int(np.bincount(yA[nn], minlength=3).argmax()))
        maj2 = np.array(maj2)
        final_cls = {m: maj2[nearest(data[e]["gen"][m]["S"][-1][:, :DESCRIPTIVE_D], f2["modes"])] for m in MODELS}
        for d in DIMS:
            fc, fr = fits[("cal", e, d)], fits[("ref", e, d)]
            K = len(fc["modes"])
            ent = {"K_cal": K, "K_ref": len(fr["modes"]), "box": cal[(e, d)]["box"], "modes_cal": fc["modes"],
                   "modes_ref": fr["modes"], "h_cal": fc["h"]}
            if K >= 2:
                D = np.linalg.norm(fc["modes"][:, None] - fc["modes"][None], axis=-1)
                s = float(D[np.triu_indices(K, 1)].min())
                pur, maj = [], []
                for mu in fc["modes"]:
                    nn = np.argsort(((A[:, :d] - mu) ** 2).sum(1))[:PURITY_K]
                    c = np.bincount(yA[nn], minlength=3)
                    pur.append(float(c.max() / PURITY_K))
                    maj.append(CLASS[int(c.argmax())])
                ent.update(s=s, purity=pur, majority=maj,
                           delta={t: float(f(s, K)) for t, f in DELTA_RULES.items()})
            else:
                ent.update(s=np.nan, purity=[], majority=[], delta={t: np.nan for t in DELTA_RULES})
            res["calibration"]["%s_d%d" % (e, d)] = ent
            for m in MODELS:
                S, sig, ids = data[e]["gen"][m]["S"], data[e]["gen"][m]["sigma"], data[e]["gen"][m]["steps"]
                for j in range(S.shape[0]):
                    fg = fits[("gen", e, d, m, j)]
                    res["gen_modes"][(e, d, m, j)] = fg["modes"]
                    r = {"encoder": e, "d": d, "model": m, "pos": j, "step": int(ids[j]), "sigma": float(sig[j])}
                    if K >= 2:
                        r.update(mmt_from_fits(fg, fr, ent["delta"]))
                    else:
                        r.update({"K_gen": len(fg["modes"]), "K_ref": len(fr["modes"]), "available": False})
                    r["mode_labels"] = labels_for(S[j][:, :d], fg["modes"], final_cls[m])
                    res["tests"].append(r)
        h2 = fits[("cal", e, DESCRIPTIVE_D)]["h"]
        A2 = A[:, :DESCRIPTIVE_D]
        thr = float(np.quantile(kde_values(A2, A2, h2), 0.05))
        ref2 = data[e]["ref"][:, :DESCRIPTIVE_D]
        desc = {"gap_real_ref": float(np.mean(kde_values(A2, ref2, h2) < thr)),
                "real_proportions": (np.bincount(data["y_ref"], minlength=3) / len(data["y_ref"])).tolist(),
                "majority_d2": [CLASS[c] for c in maj2], "models": {}}
        for m in MODELS:
            S = data[e]["gen"][m]["S"]
            cls = np.stack([maj2[nearest(S[j][:, :DESCRIPTIVE_D], f2["modes"])] for j in range(S.shape[0])])
            desc["models"][m] = {
                "sigma": data[e]["gen"][m]["sigma"].tolist(),
                "proportions": [(np.bincount(cls[j], minlength=3) / cls.shape[1]).tolist() for j in range(len(cls))],
                "commitment": [float(np.mean(cls[j] == cls[-1])) for j in range(len(cls))],
                "gap_fraction": [float(np.mean(kde_values(A2, S[j][:, :DESCRIPTIVE_D], h2) < thr))
                                 for j in range(len(cls))]}
        res["descriptive"][e] = desc
        log("encoder %s: tests and descriptives done" % e)

    try:
        perm = PermGPU(log)
    except ImportError:
        perm = PermCPU(log)
    ojobs, ometa = [], []
    for e in ENCODERS:
        ref, Fref = data[e]["ref"], data[e]["F_ref"]
        for m in MODELS:
            G = data[e]["gen"][m]
            for j in range(G["S"].shape[0]):
                for d in DIMS:
                    ojobs.append((G["S"][j][:, :d], ref[:, :d], n_perm, SEED + 100000 * d + 1000 * j + (0 if m == "vp" else 7)))
                    ometa.append({"encoder": e, "model": m, "pos": j, "step": int(G["steps"][j]),
                                  "sigma": float(G["sigma"][j]), "space": "pca_d%d" % d})
                Xf = G["F1000"][j]
                ojobs.append((Xf, Fref, n_perm, SEED + 999 * j + (11 if m == "vp" else 13)))
                ometa.append({"encoder": e, "model": m, "pos": j, "step": int(G["steps"][j]),
                              "sigma": float(G["sigma"][j]), "space": "full", "n_gen": len(Xf)})
    log("running %d permutation tests (%d permutations each)" % (len(ojobs), n_perm))
    ores = perm.run_many(ojobs, log, "permutation tests")
    for mm, (pm, pe) in zip(ometa, ores):
        res["omnibus"].append({**mm, "mmd_p": pm, "energy_p": pe})
    return res


def decide(res):
    for e in ENCODERS:
        for d in DIMS:
            rows = [r for r in res["tests"] if r["encoder"] == e and r["d"] == d]
            for t in DELTA_RULES:
                pe = [r.get("p_eq_" + t, 1.0) for r in rows]
                pc = [r.get("p_chg_" + t, 1.0) for r in rows]
                adj = holm(pe + pc)
                for i, r in enumerate(rows):
                    a_eq, a_ch = float(adj[i]), float(adj[len(rows) + i])
                    r["p_eq_holm_" + t], r["p_chg_holm_" + t] = a_eq, a_ch
                    r["decision_" + t] = ("unavailable" if not r.get("available") else
                                          "equivalent" if a_eq < ALPHA else
                                          "meaningful change" if a_ch < ALPHA else "inconclusive")
        for space in ["full"] + ["pca_d%d" % d for d in DIMS]:
            rows = [o for o in res["omnibus"] if o["encoder"] == e and o["space"] == space]
            if not rows:
                continue
            adj = holm([o["mmd_p"] for o in rows] + [o["energy_p"] for o in rows])
            for i, o in enumerate(rows):
                o["mmd_holm"], o["energy_holm"] = float(adj[i]), float(adj[len(rows) + i])


# ---------------- figures ----------------

def figures(res, data, outdir, log):
    sys.path.insert(0, CODE)
    import mt_tools as mtt
    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt = mtt.plt
    C_DEC = {"equivalent": mtt.GREEN, "meaningful change": mtt.VERMILLION, "inconclusive": mtt.GRAY}
    C_MOD = {"vp": mtt.BLUE, "ve": mtt.VERMILLION}
    C_CLS = {0: mtt.BLUE, 1: mtt.VERMILLION, 2: mtt.GREEN}

    def save(fig, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path + ".png", dpi=200, facecolor="white")
        mtt.save(fig, Path(path + ".pdf"))

    def step_axis(ax, sig, xlabel=True, steps=None):
        n = len(sig)
        steps = list(range(n)) if steps is None else [int(s) for s in steps]
        ticks = [p for p, st in enumerate(steps) if st in DISPLAY_STEPS] or list(range(n))
        ax.set_xticks(ticks, [str(steps[p]) for p in ticks])
        ax.set_xlim(-0.6, n - 0.4)
        if xlabel:
            ax.set_xlabel("Sampler step (noise to image)")
        top = ax.secondary_xaxis("top")
        top.set_xticks(ticks, ["final" if sig[j] == 0 else "%.2g" % sig[j] for j in ticks])
        top.set_xlabel(r"Noise level $\sigma$", fontsize=8)
        top.tick_params(labelsize=7)

    with plt.rc_context(mtt.RC):
        for e in ENCODERS:
            sig = np.asarray(data[e]["gen"]["vp"]["sigma"])
            steps = data[e]["gen"]["vp"]["steps"]
            n = len(sig)
            for d in DIMS:
                cal = res["calibration"]["%s_d%d" % (e, d)]
                for t in DELTA_RULES:
                    od = os.path.join(outdir, e, "d%d" % d, t)
                    rows = [r for r in res["tests"] if r["encoder"] == e and r["d"] == d]
                    fig, ax = mtt.new_figure(1, 1, 7.0, 3.3)
                    yt, yl, y = [], [], 0
                    for m in MODELS:
                        rr = sorted([r for r in rows if r["model"] == m], key=lambda r: r["pos"])
                        spaces = [("pca_d%d" % d, "MMD", "mmd_holm"), ("pca_d%d" % d, "Energy", "energy_holm"),
                                  ("full", "MMD (full)", "mmd_holm"), ("full", "Energy (full)", "energy_holm")]
                        for k, (sp, nm, key) in enumerate(spaces):
                            oo = {o["pos"]: o for o in res["omnibus"] if o["encoder"] == e and o["model"] == m
                                  and o["space"] == sp}
                            for j in range(n):
                                ax.add_patch(plt.Rectangle((j - 0.45, y + k - 0.4), 0.9, 0.8, lw=0,
                                             color=mtt.VERMILLION if oo[j][key] < ALPHA else mtt.GREEN))
                            yt.append(y + k)
                            yl.append("EDM-%s: %s" % (m.upper(), nm))
                        ky = y + len(spaces)
                        for r in rr:
                            dec = r.get("decision_" + t, "unavailable")
                            j = r["pos"]
                            if dec == "unavailable":
                                ax.add_patch(plt.Rectangle((j - 0.45, ky - 0.4), 0.9, 0.8, facecolor="white",
                                                           edgecolor=mtt.GRAY, lw=0.6))
                                ax.text(j, ky, str(r["K_gen"]), ha="center", va="center", fontsize=5)
                            else:
                                ax.add_patch(plt.Rectangle((j - 0.45, ky - 0.4), 0.9, 0.8, lw=0, color=C_DEC[dec]))
                        unav = [r.get("decision_" + t, "unavailable") == "unavailable" for r in rr]
                        labs = [" | ".join(r["mode_labels"]) if len(r["mode_labels"]) <= 4
                                else "K=%d" % len(r["mode_labels"]) for r in rr]
                        j = 0
                        while j < n:
                            if not unav[j]:
                                j += 1
                                continue
                            start = j
                            while j + 1 < n and unav[j + 1] and labs[j + 1] == labs[start]:
                                j += 1
                            ax.annotate("", xy=(start - 0.45, ky + 0.75), xytext=(j + 0.45, ky + 0.75),
                                        arrowprops=dict(arrowstyle="|-|", lw=0.5, color=mtt.BLACK,
                                                        shrinkA=0, shrinkB=0, mutation_scale=2))
                            if j - start + 1 >= 2 or len(labs[start]) <= 3:
                                ax.text((start + j) / 2, ky + 1.05, labs[start], ha="center", va="top", fontsize=5)
                            j += 1
                        yt.append(ky)
                        yl.append("EDM-%s: Matched-mode test" % m.upper())
                        y = ky + 2.0
                    ax.set_yticks(yt, yl, fontsize=7)
                    ax.set_ylim(y - 0.6, -0.8)
                    step_axis(ax, sig, steps=steps)
                    for spn in ("left", "right", "bottom"):
                        ax.spines[spn].set_visible(False)
                    ax.tick_params(axis="y", length=0)
                    dl = cal["delta"][t]
                    mtt.panel_title(ax, "%s, d = %d: calibration K = %d, delta = %.3f" % (
                        "CLIP" if e == "clip" else "Inception", d, cal["K_cal"], dl if np.isfinite(dl) else float("nan")))
                    handles = [Patch(color=mtt.VERMILLION, label="Reject equality / meaningful change"),
                               Patch(color=mtt.GREEN, label="No rejection / practically equivalent"),
                               Patch(color=mtt.GRAY, label="Inconclusive"),
                               Patch(facecolor="white", edgecolor=mtt.GRAY,
                                     label="Unavailable (number = generated modes; C cat, D dog, W wild)")]
                    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
                    save(fig, os.path.join(od, "fig_decisions"))
                    plt.close(fig)

                    fig, ax = mtt.new_figure(1, 1, 3.4, 2.2)
                    for m in MODELS:
                        rr = sorted([r for r in rows if r["model"] == m and r.get("available")], key=lambda r: r["pos"])
                        if rr:
                            x = [r["pos"] for r in rr]
                            dh = np.array([r["D_hat"] for r in rr])
                            ud = np.array([r["U_D"] for r in rr])
                            ax.plot(x, dh, "-o", ms=2.5, color=C_MOD[m], label="EDM-" + m.upper())
                            ax.fill_between(x, dh, ud, color=C_MOD[m], alpha=0.2, lw=0)
                    if np.isfinite(dl):
                        ax.axhline(dl, ls="--", color=mtt.BLACK, lw=0.9, label=r"$\delta = %.3f$" % dl)
                    ax.set_ylabel(r"$\widehat D_A$ (band to $U_D$)")
                    ax.set_ylim(0, None)
                    step_axis(ax, sig, steps=steps)
                    mtt.style_axis(ax)
                    mtt.top_legend(fig, ax, ncol=3)
                    save(fig, os.path.join(od, "fig_distance"))
                    plt.close(fig)
            log("figures for %s (decisions, distance) done" % e)

            desc = res["descriptive"][e]
            fig, axes = mtt.new_figure(1, 3, 7.0, 2.2)
            for m in MODELS:
                g = desc["models"][m]
                x = np.arange(len(g["sigma"]))
                P = np.array(g["proportions"])
                for k in range(3):
                    axes[0].plot(x, P[:, k], ["-", "--"][MODELS.index(m)], color=C_CLS[k], lw=1.1,
                                 label="%s, EDM-%s" % (CLASS[k], m.upper()))
                axes[1].plot(x, g["commitment"], "-o", ms=2, color=C_MOD[m], label="EDM-" + m.upper())
                axes[2].plot(x, g["gap_fraction"], "-o", ms=2, color=C_MOD[m], label="EDM-" + m.upper())
            for k in range(3):
                axes[0].axhline(desc["real_proportions"][k], color=C_CLS[k], lw=0.7, ls=":")
            axes[2].axhline(desc["gap_real_ref"], color=mtt.BLACK, lw=0.8, ls=":", label="Real reference")
            for ax, ttl, yl_ in zip(axes, ["(a) Class proportions", "(b) Agreement with final class",
                                            "(c) Mass between modes"],
                                    ["Proportion", "Fraction", "Fraction"]):
                step_axis(ax, sig, steps=steps)
                ax.set_ylabel(yl_)
                mtt.panel_title(ax, ttl)
                mtt.style_axis(ax)
            axes[1].legend(fontsize=7, frameon=False)
            axes[2].legend(fontsize=7, frameon=False)
            save(fig, os.path.join(outdir, e, "fig_descriptive_d2"))
            plt.close(fig)

            cal2 = res["calibration"]["%s_d2" % e]
            A2 = data[e]["A"][:, :2]
            rng = np.random.default_rng(SEED)
            ref2 = data[e]["ref"][:, :2]
            first3 = {}
            for m in MODELS:
                rr = sorted([r for r in res["tests"] if r["encoder"] == e and r["d"] == 2 and r["model"] == m],
                            key=lambda r: r["pos"])
                k3 = [r["pos"] for r in rr if r["K_gen"] == cal2["K_cal"]]
                first3[m] = k3[0] if k3 else len(rr) - 1
            stages = [("first", first3), ("final", {m: n - 1 for m in MODELS})]
            fig, axes = mtt.new_figure(2, 3, 7.0, 4.6)
            mref = np.asarray(cal2["modes_ref"])
            iref = rng.choice(len(ref2), min(2000, len(ref2)), replace=False)
            maj2 = np.array([CLASS.index(c) for c in desc["majority_d2"]])
            mcal = np.asarray(cal2["modes_cal"])
            for i, (nm, stg) in enumerate(stages):
                ax = axes[i, 0]
                for k in range(3):
                    s = data["y_ref"][iref] == k
                    ax.scatter(ref2[iref][s, 0], ref2[iref][s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0,
                               rasterized=True)
                ax.scatter(mref[:, 0], mref[:, 1], marker="*", s=110, color="white", edgecolor=mtt.BLACK, zorder=5)
                mtt.panel_title(ax, "Real reference")
                for jm, m in enumerate(MODELS):
                    ax = axes[i, jm + 1]
                    st = stg[m]
                    X = data[e]["gen"][m]["S"][st][:, :2]
                    ig = rng.choice(len(X), min(2000, len(X)), replace=False)
                    cls = maj2[nearest(X[ig], mcal)]
                    for k in range(3):
                        s = cls == k
                        ax.scatter(X[ig][s, 0], X[ig][s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0, rasterized=True)
                    gm = np.asarray(fits_modes(res, e, 2, m, st))
                    ax.scatter(mref[:, 0], mref[:, 1], marker="*", s=110, facecolor="none", edgecolor=mtt.BLACK,
                               lw=0.8, zorder=5)
                    if len(gm):
                        ax.scatter(gm[:, 0], gm[:, 1], marker="P", s=55, color="white", edgecolor=mtt.BLACK, zorder=6)
                    s_ = data[e]["gen"][m]["sigma"][st]
                    mtt.panel_title(ax, "EDM-%s, step %d (%s)" % (m.upper(), int(steps[st]), "final" if s_ == 0 else
                                                                   "sigma %.2g" % s_))
            for ax in axes.ravel():
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                mtt.style_axis(ax, ygrid=False)
            hs = [Line2D([], [], marker="o", ls="", color=C_CLS[k], label=CLASS[k].capitalize()) for k in range(3)]
            hs += [Line2D([], [], marker="*", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=9,
                          label="Real modes"),
                   Line2D([], [], marker="P", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=7,
                          label="Generated modes")]
            fig.legend(handles=hs, loc="outside upper center", ncol=5, frameon=False)
            save(fig, os.path.join(outdir, e, "fig_overlay_d2"))
            plt.close(fig)
        log("descriptive and overlay figures done")

        for e in ENCODERS:
            fig, ax = mtt.new_figure(1, 1, 3.4, 3.0)
            A2 = data[e]["A"][:, :2]
            for k in range(3):
                s = data["y_A"] == k
                ax.scatter(A2[s, 0], A2[s, 1], s=1.2, alpha=0.35, color=C_CLS[k], lw=0, rasterized=True)
            mc = np.asarray(res["calibration"]["%s_d2" % e]["modes_cal"])
            ax.scatter(mc[:, 0], mc[:, 1], marker="*", s=130, color="white", edgecolor=mtt.BLACK, zorder=5)
            ax.set_xlabel("PC1 (standardized)")
            ax.set_ylabel("PC2 (standardized)")
            mtt.style_axis(ax, ygrid=False)
            hs = [Line2D([], [], marker="o", ls="", color=C_CLS[k], label=CLASS[k].capitalize()) for k in range(3)]
            hs.append(Line2D([], [], marker="*", ls="", markerfacecolor="white", markeredgecolor=mtt.BLACK, ms=10,
                             label="Estimated mode"))
            fig.legend(handles=hs, loc="outside upper center", ncol=4, frameon=False)
            save(fig, os.path.join(outdir, e, "fig_embedding_d2"))
            plt.close(fig)

        try:
            img_figures(res, data, outdir, mtt, save, log)
        except Exception:
            log("image figures failed:\n" + traceback.format_exc())


def fits_modes(res, e, d, m, pos):
    return res["gen_modes"][(e, d, m, pos)]


def img_figures(res, data, outdir, mtt, save, log):
    from matplotlib.patches import FancyArrowPatch
    plt = mtt.plt
    G = np.load(os.path.join(IN_RES, "images_gen_vp.npz"))
    traj = G["trajectories"]
    sig = G["sigma"]
    e = "clip"
    steps = list(data[e]["gen"]["vp"]["steps"])
    cols = [j for j in DISPLAY_STEPS if j in steps and j < traj.shape[0]]
    posof = {st: p for p, st in enumerate(steps)}
    desc = res["descriptive"][e]
    maj2 = np.array([CLASS.index(c) for c in desc["majority_d2"]])
    mcal = np.asarray(res["calibration"]["%s_d2" % e]["modes_cal"])
    S = data[e]["gen"]["vp"]["S"]
    nT = min(traj.shape[1], S.shape[1])
    fin = maj2[nearest(S[-1][:nT, :2], mcal)]
    pick = []
    for k in [2, 0, 1]:
        w = np.where(fin == k)[0]
        pick.append(int(w[0]) if len(w) else 0)
    rng = np.random.default_rng(3)
    x0 = traj[-1, pick[0]].astype(float) / 127.5 - 1
    eps = rng.standard_normal(x0.shape)
    fwd = [((np.clip((x0 + s * eps) / np.sqrt(1 + s * s), -1, 1) + 1) * 127.5).astype(np.uint8) for s in sig[cols]]
    rows_img = [fwd] + [[traj[j, p] for j in cols] for p in pick]
    names = ["Forward\n(noise added)", "Reverse\n(wild)", "Reverse\n(cat)", "Reverse\n(dog)"]

    def strip(rows_img, names, path, title_bottom):
        fig, axes = plt.subplots(len(rows_img), len(cols), figsize=(7.0, 1.05 * len(rows_img) + 0.9))
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.17, wspace=0.05, hspace=0.08)
        for i, rr in enumerate(rows_img):
            for j, im in enumerate(rr):
                ax = axes[i, j]
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                for spn in ax.spines.values():
                    spn.set_visible(False)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=7)
                if i == len(rows_img) - 1:
                    ax.set_xlabel("step %d\n%s" % (cols[j], "final" if sig[cols[j]] == 0 else
                                                    r"$\sigma$=%.2g" % sig[cols[j]]), fontsize=7)
        l, r = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
        top, bot = axes[0, 0].get_position().y1, axes[-1, 0].get_position().y0
        if names[0].startswith("Forward"):
            fig.add_artist(FancyArrowPatch((r, top + 0.03), (l, top + 0.03), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=12, color=mtt.BLACK, lw=1.0))
            fig.text((l + r) / 2, top + 0.045, r"Forward process: image $\rightarrow$ noise", ha="center",
                     va="bottom", fontsize=8)
        fig.add_artist(FancyArrowPatch((l, bot - 0.1), (r, bot - 0.1), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=12, color=mtt.BLACK, lw=1.0))
        fig.text((l + r) / 2, bot - 0.115, title_bottom, ha="center", va="top", fontsize=8)
        fig.savefig(path + ".png", dpi=200, facecolor="white")
        from pathlib import Path
        mtt.save(fig, Path(path + ".pdf"))
        plt.close(fig)

    strip(rows_img, names, os.path.join(outdir, "fig_trajectories"),
          r"Reverse process (generation): noise $\rightarrow$ image")

    proto = []
    for k in [2, 0, 1]:
        row = []
        for j in cols:
            gm = np.asarray(res["gen_modes"][(e, 2, "vp", posof[j])])
            Xj = S[posof[j]][:nT, :2]
            if len(gm) == 0:
                row.append(np.full_like(traj[0, 0], 255))
                continue
            mk = gm[np.argmin(((gm - mcal[maj2 == k][0]) ** 2).sum(1))] if np.any(maj2 == k) else gm[0]
            row.append(traj[j, int(np.argmin(((Xj - mk) ** 2).sum(1)))])
        proto.append(row)
    strip(proto, ["Mode\n(wild)", "Mode\n(cat)", "Mode\n(dog)"], os.path.join(outdir, "fig_mode_prototypes"),
          r"Generated image closest to each mode (EDM-VP, CLIP, d = 2)")
    log("image figures done")


# ---------------- run ----------------

def zip_dir(src, dst):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for f in files:
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, os.path.dirname(src)))


def run(outdir, n_gen, stages, n_perm, log):
    data = load_all(n_gen=n_gen, stages=stages)
    workers = max(1, (os.cpu_count() or 2))
    res = compute(data, log, workers, n_perm)
    decide(res)
    write_csv(res["tests"], os.path.join(outdir, "modal_tests_all.csv"))
    write_csv(res["omnibus"], os.path.join(outdir, "omnibus_tests_all.csv"))
    for e in ENCODERS:
        for d in DIMS:
            for t in DELTA_RULES:
                od = os.path.join(outdir, e, "d%d" % d, t)
                rows = [{k: v for k, v in r.items() if not (k.endswith(("_rule_k", "_rule_rms")) and t not in k)}
                        for r in res["tests"] if r["encoder"] == e and r["d"] == d]
                write_csv(rows, os.path.join(od, "modal_tests.csv"))
                write_csv([o for o in res["omnibus"] if o["encoder"] == e and o["space"] in ("pca_d%d" % d, "full")],
                          os.path.join(od, "omnibus_tests.csv"))
    jdump({"calibration": res["calibration"], "descriptive": res["descriptive"]}, os.path.join(outdir, "summary.json"))
    return res, data


def main():
    log = Log(os.path.join(OUT, "log.txt"))
    log("inputs: %s | code: %s" % (IN_RES, CODE))
    pre = os.path.join(WORK, "preflight")
    try:
        global DIMS
        dims_full = DIMS
        DIMS = [2, 3]
        res, data = run(pre, 300, [0, 20, 40], 99, log)
        figures(res, data, pre, log)
        DIMS = dims_full
        log("PREFLIGHT PASSED")
    except Exception:
        log("PREFLIGHT FAILED:\n" + traceback.format_exc())
        zip_dir(OUT, os.path.join(WORK, "analysis.zip"))
        return
    if PREFLIGHT:
        zip_dir(pre, os.path.join(WORK, "preflight.zip"))
        return
    res, data = run(OUT, None, None, N_PERM, log)
    try:
        figures(res, data, OUT, log)
    except Exception:
        log("figures failed:\n" + traceback.format_exc())
    zip_dir(OUT, os.path.join(WORK, "analysis.zip"))
    log("ALL DONE: download /kaggle/working/analysis.zip")


if __name__ == "__main__":
    main()