"""Shared helpers for the matched-mode testing experiments (imported, not run).

Sections
  1. Comparison (omnibus) tests: MMD, Energy, GEC, KS, Kuiper, Wasserstein permutation.
  2. Gaussian / Student-t mixtures with exact population modes.
  3. Simulation families (four-mode Gaussian, five-mode Student-t).
  4. Monte Carlo runner (4 worker processes, SeedSequence per replicate).
  5. Figure style.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402
from scipy.optimize import linear_sum_assignment, minimize, root  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from scipy.stats import chi2, ks_2samp  # noqa: E402

from modal_transport import ho_density  # noqa: E402

N_JOBS = 4  # worker processes used by every script
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"  # tables: results/<script>/
PLOTS = ROOT / "Plots"  # figures: Plots/<script>/, the layout the LaTeX sources expect


def results_dir(script_file: str) -> Path:
    """results/<name of the calling script>/."""
    return RESULTS / Path(script_file).stem


def plots_dir(script_file: str) -> Path:
    """Plots/<name of the calling script>/."""
    return PLOTS / Path(script_file).stem


# ========================================================================
# 1. comparison tests (omnibus)
# ========================================================================
# Full-distribution two-sample tests used as comparators.
#
# All functions return ``(statistic, p_value)``.
#
# * ``mmd_test``, ``energy_test``: hyppo implementations with the fast
#   chi-square calibration (``auto=True``).
# * ``gec_test``: Chen--Friedman generalized edge-count test on the union of
#   ``k`` sequential Euclidean minimum spanning trees, asymptotic chi-square(2).
# * ``ks_test``, ``kuiper_test``: univariate CDF tests.
# * ``wasserstein_permutation_test``: univariate W_p distance with a
#   permutation p-value.

def _2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def _hyppo():
    try:
        from hyppo.ksample import MMD, Energy
    except ImportError as err:  # pragma: no cover
        raise ImportError("MMD/Energy tests require hyppo (pip install hyppo).") from err
    return MMD, Energy


def mmd_test(X, Y, seed: int = 0) -> tuple[float, float]:
    MMD, _ = _hyppo()
    stat, p = MMD().test(_2d(X), _2d(Y), auto=True, workers=1, random_state=seed)
    return float(stat), float(p)


def energy_test(X, Y, seed: int = 0) -> tuple[float, float]:
    _, Energy = _hyppo()
    stat, p = Energy().test(_2d(X), _2d(Y), auto=True, workers=1, random_state=seed)
    return float(stat), float(p)


def _k_mst_edges(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Union of k sequential Euclidean MSTs (each built after removing previous edges)."""
    N = len(Z)
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    us, vs = [], []
    for _ in range(k):
        in_tree = np.zeros(N, dtype=bool)
        in_tree[N - 1] = True
        best = D[:, N - 1].copy()
        best[N - 1] = np.inf
        parent = np.full(N, N - 1)
        tu, tv = np.empty(N - 1, dtype=int), np.empty(N - 1, dtype=int)
        for e in range(N - 1):
            v = int(np.argmin(best))
            if not np.isfinite(best[v]):
                raise RuntimeError("k-MST construction became disconnected.")
            tu[e], tv[e] = parent[v], v
            in_tree[v] = True
            best[v] = np.inf
            improve = (~in_tree) & (D[:, v] < best)
            best[improve] = D[improve, v]
            parent[improve] = v
        D[tu, tv] = D[tv, tu] = np.inf
        us.append(tu)
        vs.append(tv)
    return np.concatenate(us), np.concatenate(vs)


def gec_test(X, Y, k: int = 5) -> tuple[float, float]:
    """Generalized edge-count test (Chen and Friedman, 2017) on a k-MST graph."""
    X, Y = _2d(X), _2d(Y)
    n, m = len(X), len(Y)
    N = n + m
    u, v = _k_mst_edges(np.vstack([X, Y]), k)
    R1 = float(np.sum((u < n) & (v < n)))
    R2 = float(np.sum((u >= n) & (v >= n)))
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(float)
    nE, nEi = float(len(u)), float(np.sum(deg * (deg - 1.0)))
    d3 = N * (N - 1.0) * (N - 2.0)
    d4 = d3 * (N - 3.0)
    mu1 = nE * n * (n - 1.0) / (N * (N - 1.0))
    mu2 = nE * m * (m - 1.0) / (N * (N - 1.0))
    common = nE * (nE - 1.0) - nEi
    V1 = nEi * n * (n - 1) * (n - 2) / d3 + common * n * (n - 1) * (n - 2) * (n - 3) / d4 + mu1 - mu1**2
    V2 = nEi * m * (m - 1) * (m - 2) / d3 + common * m * (m - 1) * (m - 2) * (m - 3) / d4 + mu2 - mu2**2
    V12 = common * m * n * (m - 1) * (n - 1) / d4 - mu1 * mu2
    c = np.array([R1 - mu1, R2 - mu2])
    stat = float(c @ np.linalg.solve(np.array([[V1, V12], [V12, V2]]), c))
    return stat, float(chi2.sf(stat, df=2))


def ks_test(x, y) -> tuple[float, float]:
    r = ks_2samp(np.ravel(x), np.ravel(y))
    return float(r.statistic), float(r.pvalue)


def _kuiper_sf(lam: float) -> float:
    """Asymptotic Kuiper tail probability Q_KP(lambda)."""
    if lam < 0.4:
        return 1.0
    j = np.arange(1, 101)
    val = 2.0 * np.sum((4.0 * j**2 * lam**2 - 1.0) * np.exp(-2.0 * j**2 * lam**2))
    return float(min(max(val, 0.0), 1.0))


def kuiper_test(x, y) -> tuple[float, float]:
    """Two-sample Kuiper test with the Stephens small-sample correction."""
    x, y = np.sort(np.ravel(x)), np.sort(np.ravel(y))
    grid = np.concatenate([x, y])
    Fx = np.searchsorted(x, grid, side="right") / len(x)
    Fy = np.searchsorted(y, grid, side="right") / len(y)
    V = float(np.max(Fx - Fy) + np.max(Fy - Fx))
    ne = len(x) * len(y) / (len(x) + len(y))
    lam = (np.sqrt(ne) + 0.155 + 0.24 / np.sqrt(ne)) * V
    return V, _kuiper_sf(lam)


def wasserstein_distance_1d(x, y, p: int = 2) -> float:
    """Exact W_p between the empirical distributions of x and y.

    Both empirical quantile functions are step functions with jumps at k/n and
    l/m; on each interval between consecutive jumps both are constant.
    """
    x, y = np.sort(np.ravel(x)), np.sort(np.ravel(y))
    n, m = len(x), len(y)
    u = np.unique(np.concatenate([np.arange(1, n + 1) / n, np.arange(1, m + 1) / m]))
    du = np.diff(u, prepend=0.0)
    qx = x[np.minimum(np.searchsorted(np.arange(1, n + 1) / n, u - 1e-12), n - 1)]
    qy = y[np.minimum(np.searchsorted(np.arange(1, m + 1) / m, u - 1e-12), m - 1)]
    return float(np.sum(du * np.abs(qx - qy) ** p) ** (1.0 / p))


def wasserstein_permutation_test(x, y, p: int = 2, n_perm: int = 500, seed: int = 0) -> tuple[float, float]:
    x, y = np.ravel(x), np.ravel(y)
    rng = np.random.default_rng(seed)
    obs = wasserstein_distance_1d(x, y, p)
    z = np.concatenate([x, y])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        count += wasserstein_distance_1d(z[: len(x)], z[len(x):], p) >= obs
    return obs, (1 + count) / (1 + n_perm)


def run_omnibus(X, Y, tests: tuple[str, ...], seed: int = 0, n_perm: int = 500) -> dict:
    """Run the named comparator tests and return ``{name}_stat`` and ``{name}_p`` entries."""
    funcs = {
        "mmd": lambda: mmd_test(X, Y, seed),
        "energy": lambda: energy_test(X, Y, seed),
        "gec": lambda: gec_test(X, Y),
        "ks": lambda: ks_test(X, Y),
        "kuiper": lambda: kuiper_test(X, Y),
        "w1": lambda: wasserstein_permutation_test(X, Y, 1, n_perm, seed),
        "w2": lambda: wasserstein_permutation_test(X, Y, 2, n_perm, seed),
    }
    out = {}
    for name in tests:
        stat, p = funcs[name]()
        out[f"{name}_stat"], out[f"{name}_p"] = stat, p
    return out


# ========================================================================
# 2. mixtures with exact population modes
# ========================================================================
# Gaussian and Student-t location mixtures with exact population modes.
#
# Used to construct simulation designs whose population modal configurations
# are known to numerical precision.  A mixture has component locations
# ``centers`` (K, 2), weights, a common scale matrix, and optional degrees of
# freedom (Student-t).  ``extra_dims`` appends independent N(0, extra_sd^2)
# coordinates, which leaves the modes at ``(mu_j, 0)``.

@dataclass(frozen=True)
class Mixture:
    centers: np.ndarray
    scale: np.ndarray
    weights: np.ndarray | None = None
    df: float | None = None
    extra_dims: int = 0
    extra_sd: float = 0.5

    @property
    def K(self) -> int:
        return len(self.centers)

    @property
    def w(self) -> np.ndarray:
        return np.full(self.K, 1.0 / self.K) if self.weights is None else np.asarray(self.weights, float)

    @property
    def dim(self) -> int:
        return 2 + self.extra_dims

    # -- planar component terms -------------------------------------------------
    def _terms2(self, x: np.ndarray):
        """Weighted density, gradient, Hessian of the planar mixture at x (2,)."""
        inv = np.linalg.inv(self.scale)
        _, logdet = np.linalg.slogdet(self.scale)
        v = x[None, :] - self.centers
        r = v @ inv.T
        q = np.einsum("ki,ki->k", v, r)
        if self.df is None:
            p = np.exp(-0.5 * (2 * math.log(2 * math.pi) + logdet + q))
            a = np.ones(self.K)
            c2 = np.ones(self.K)
        else:
            nu = float(self.df)
            logc = gammaln((nu + 2) / 2) - gammaln(nu / 2) - 0.5 * (2 * math.log(nu * math.pi) + logdet)
            p = np.exp(logc - 0.5 * (nu + 2) * np.log1p(q / nu))
            a = (nu + 2) / (nu + q)
            c2 = a * a + 2.0 * a / (nu + q)
        w = self.w
        f = float(np.sum(w * p))
        g = -np.sum((w * p * a)[:, None] * r, axis=0)
        H = np.einsum("k,ki,kj->ij", w * p * c2, r, r) - np.sum(w * p * a) * inv
        return f, g, H

    def density2(self, pts: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.scale)
        _, logdet = np.linalg.slogdet(self.scale)
        v = pts[:, None, :] - self.centers[None]
        q = np.einsum("nki,ij,nkj->nk", v, inv, v)
        if self.df is None:
            logp = -0.5 * (2 * math.log(2 * math.pi) + logdet + q)
        else:
            nu = float(self.df)
            logp = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - 0.5 * (2 * math.log(nu * math.pi) + logdet)
                    - 0.5 * (nu + 2) * np.log1p(q / nu))
        return np.exp(logp) @ self.w

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.choice(self.K, size=n, p=self.w)
        Z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(self.scale).T
        if self.df is not None:
            Z = Z / np.sqrt(rng.chisquare(self.df, size=n) / self.df)[:, None]
        X = self.centers[labels] + Z
        if self.extra_dims:
            X = np.hstack([X, self.extra_sd * rng.standard_normal((n, self.extra_dims))])
        return X

    def population_modes(self, region2: np.ndarray, audit: bool = True) -> np.ndarray:
        """Regular modes (in the full dimension) inside the planar box ``region2``."""
        modes2 = population_modes(self, region2, audit=audit)
        return np.hstack([modes2, np.zeros((len(modes2), self.extra_dims))])


def population_modes(mix: Mixture, region2: np.ndarray, audit: bool = True) -> np.ndarray:
    """Planar regular modes of ``mix``: L-BFGS-B from the centers (and grid maxima if audit)."""
    lo, hi = region2[:, 0], region2[:, 1]
    starts = [c.copy() for c in mix.centers]
    if audit:
        g = [np.linspace(lo[j], hi[j], 181) for j in range(2)]
        XX, YY = np.meshgrid(*g)
        Z = mix.density2(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        loc = (Z >= maximum_filter(Z, size=3, mode="nearest") - 1e-15) & (Z >= 1e-3 * Z.max())
        loc[[0, -1], :] = loc[:, [0, -1]] = False
        iy, ix = np.where(loc)
        starts += list(np.column_stack([g[0][ix], g[1][iy]]))
    found: list[tuple[np.ndarray, float]] = []
    for s in starts:
        res = minimize(lambda z: -mix._terms2(z)[0], s, jac=lambda z: -mix._terms2(z)[1],
                       method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 600})
        f, gr, H = mix._terms2(res.x)
        if (np.linalg.norm(gr) <= 3e-7 and np.linalg.eigvalsh(H)[-1] < -1e-8
                and np.all(res.x > lo + 1e-6) and np.all(res.x < hi - 1e-6)):
            found.append((res.x, f))
    if not found:
        return np.empty((0, 2))
    found.sort(key=lambda t: -t[1])
    kept = []
    for z, f in found:
        if f >= 1e-3 * found[0][1] and (not kept or np.min(np.linalg.norm(np.vstack(kept) - z, axis=1)) > 5e-4):
            kept.append(z)
    modes = np.vstack(kept)
    return modes[np.lexsort((modes[:, 1], modes[:, 0]))]


def match_modes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Permutation pi minimizing sum_j |a_j - b_pi(j)|^2."""
    rows, cols = linear_sum_assignment(np.sum((a[:, None] - b[None]) ** 2, axis=2))
    pi = np.empty(len(a), dtype=int)
    pi[rows] = cols
    return pi


def modal_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("Configurations must have the same positive cardinality.")
    return math.sqrt(float(np.sum((a - b[match_modes(a, b)]) ** 2)) / len(a))


def matching_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Cost gap between the optimal and the second-best matching."""
    costs = sorted(float(np.sum((a - b[list(p)]) ** 2)) for p in itertools.permutations(range(len(a))))
    return costs[1] - costs[0] if len(costs) > 1 else float("inf")


def with_modes(template: Mixture, target_modes: np.ndarray, weights: np.ndarray) -> Mixture:
    """Mixture with the given weights whose planar modes equal ``target_modes``.

    ``target_modes[j]`` is the mode generated by component j.  The component
    centers are obtained by solving grad f(target_j) = 0 for all j (a square
    system of 2K equations), starting from ``target_modes``.
    """
    K = template.K
    base = replace(template, weights=np.asarray(weights, float))

    def equations(cflat):
        mix = replace(base, centers=cflat.reshape(K, 2))
        return np.concatenate([mix._terms2(m)[1] for m in target_modes])

    sol = root(equations, target_modes.ravel(), method="hybr", tol=1e-14)
    mix = replace(base, centers=sol.x.reshape(K, 2))
    if np.max(np.abs(equations(sol.x))) > 1e-10:
        raise RuntimeError("Could not place the modes at the requested locations.")
    return mix


def calibrate_deformation(template: Mixture, deform, target_D: float, region2: np.ndarray,
                          a_max: float = 0.8) -> tuple[float, np.ndarray]:
    """Bisection for the deformation parameter a with D_A(template, deform(a)) = target_D.

    ``deform(a)`` returns deformed component centers.  Returns ``(a, modes_g)``.
    """
    base = population_modes(template, region2)
    if target_D == 0.0:
        return 0.0, base

    def D(a):
        g = population_modes(replace(template, centers=deform(a)), region2, audit=False)
        if len(g) != template.K:
            raise RuntimeError(f"Deformation a={a:.6f} changed the modal cardinality.")
        return modal_distance(base, g)

    lo, hi = 0.0, 0.05
    while D(hi) < target_D:
        lo, hi = hi, min(a_max, 1.45 * hi + 0.01)
        if hi >= a_max and D(hi) < target_D:
            raise RuntimeError("Could not bracket the target modal distance.")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if D(mid) < target_D else (lo, mid)
    a = 0.5 * (lo + hi)
    return a, population_modes(replace(template, centers=deform(a)), region2)


def gaussian_mixture_modes(centers: np.ndarray, weights: np.ndarray, variances: np.ndarray,
                           region: np.ndarray, n_random_starts: int = 400, seed: int = 0) -> np.ndarray:
    """Exact regular modes of a Gaussian mixture with diagonal covariance.

    Uses the Gaussian mean-shift fixed-point iteration (Carreira-Perpinan, 2000),
    which is monotone and converges to stationary points, started from every
    component center and from random points in ``region``; stationary points
    are kept if the Hessian is negative definite.
    """
    rng = np.random.default_rng(seed)
    lo, hi = region[:, 0], region[:, 1]
    starts = np.vstack([centers, rng.uniform(lo, hi, size=(n_random_starts, len(lo)))])
    found: list[np.ndarray] = []
    for x in starts:
        for _ in range(50_000):
            k = weights * np.exp(-0.5 * np.sum((x - centers) ** 2 / variances, axis=1))
            xn = (k[:, None] * centers).sum(0) / k.sum()
            if np.linalg.norm(xn - x) < 1e-12:
                x = xn
                break
            x = xn
        k = weights * np.exp(-0.5 * np.sum((x - centers) ** 2 / variances, axis=1))
        r = (x - centers) / variances
        H = np.einsum("k,ki,kj->ij", k, r, r) - np.diag(k.sum() / variances)
        if np.linalg.eigvalsh(H)[-1] < 0 and np.all(x > lo) and np.all(x < hi):
            if not found or np.min(np.linalg.norm(np.vstack(found) - x, axis=1)) > 1e-5:
                found.append(x)
    return np.vstack(found) if found else np.empty((0, len(lo)))


# ========================================================================
# 3. simulation families
# ========================================================================
# Two planar location mixtures (the families of the tolerance experiment):
#
# * ``gaussian4``: four Gaussian components, geometry changed by an anisotropic
#   scaling of the centers about their centroid, M(a) = diag(1 + a, 1 - a);
# * ``t5`` (and ``t{nu}``): five Student-t components, geometry changed by a
#   symmetric shear M(a) = [[1, a], [a, 1]].
#
# Population modes are computed numerically (section 2); all modal
# distances refer to the actual mixture modes, not the component centers.

REGION2 = np.array([[-3.0, 3.0], [-3.0, 3.0]])
ALPHA = 0.05
TOLERANCE = 0.50

GAUSSIAN4 = Mixture(
    centers=np.array([[-1.8, -1.5], [-1.6, 1.6], [1.6, 1.5], [1.8, -1.6]]),
    scale=np.array([[0.30, 0.06], [0.06, 0.24]]),
)
STUDENT5_CENTERS = np.array([[-2.2, -0.4], [-1.1, 2.0], [1.3, 1.9], [2.3, -0.5], [0.0, -2.2]])
STUDENT5_SCALE = np.array([[0.20, -0.02], [-0.02, 0.18]])

LABELS = {"gaussian4": "Four-mode Gaussian", "t5": r"Five-mode Student-$t_5$"}

# Masses of the states at full redistribution (lambda = 1).
SKEWED_WEIGHTS = {
    "gaussian4": np.array([0.40, 0.30, 0.18, 0.12]),
    "t5": np.array([0.34, 0.26, 0.18, 0.13, 0.09]),
}


def family(key: str, extra_dims: int = 0) -> Mixture:
    if key == "gaussian4":
        return replace(GAUSSIAN4, extra_dims=extra_dims)
    if key.startswith("t"):
        return Mixture(centers=STUDENT5_CENTERS, scale=STUDENT5_SCALE, df=float(key[1:]),
                       extra_dims=extra_dims)
    raise KeyError(key)


def deformation(key: str):
    """Map a -> deformed centers for the family ``key``."""
    base = family(key).centers
    c = base.mean(axis=0)
    if key == "gaussian4":
        return lambda a: c + (base - c) @ np.array([[1 + a, 0.0], [0.0, 1 - a]]).T
    return lambda a: c + (base - c) @ np.array([[1.0, a], [a, 1.0]]).T


def redistributed_masses(key: str, lam: float) -> np.ndarray:
    K = family(key).K
    return (1.0 - lam) * np.full(K, 1.0 / K) + lam * SKEWED_WEIGHTS["gaussian4" if key == "gaussian4" else "t5"]


def by_component(modes: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Reorder modes so that row j is the mode generated by component j."""
    pi = match_modes(centers, modes)
    return modes[pi]


def build_pair(key: str, D: float, lam: float = 0.0, extra_dims: int = 0) -> dict:
    """Population pair (f, g) with D_A(f, g) = D and g with mass redistributed by lam, modes held fixed.

    f is the equal-weight family.  The modes of g are the modes of the
    equal-weight family deformed so that D_A = D; g has weights w(lam) and
    centers solved so that its modes are exactly at those locations.
    """
    tmpl = family(key)
    modes_f = by_component(population_modes(tmpl, REGION2), tmpl.centers)
    a, modes_def = calibrate_deformation(tmpl, deformation(key), D, REGION2)
    target = by_component(modes_def, deformation(key)(a))
    g = tmpl if (D == 0 and lam == 0) else with_modes(tmpl, target, redistributed_masses(key, lam))
    modes_g = population_modes(g, REGION2)
    if len(modes_g) != tmpl.K:
        raise RuntimeError(f"{key}: population audit found {len(modes_g)} modes (D={D}, lam={lam}).")
    D_check = modal_distance(modes_f, modes_g)
    if abs(D_check - D) > 1e-6:
        raise RuntimeError(f"{key}: calibration error |{D_check} - {D}| (lam={lam}).")
    f = replace(tmpl, extra_dims=extra_dims)
    g = replace(g, extra_dims=extra_dims)
    return {"f": f, "g": g, "modes_f": modes_f, "modes_g": modes_g, "a": a,
            "D_A": D_check, "matching_gap": matching_gap(modes_f, modes_g) if D > 0 else np.inf}


def region(extra_dims: int, extra_sd: float = 0.5) -> np.ndarray:
    return np.vstack([REGION2, np.tile([-3 * extra_sd, 3 * extra_sd], (extra_dims, 1))])


def mt_fields(res: dict, alpha: float = ALPHA) -> dict:
    """Scalar outputs of ``modal_transport_test`` stored for every replicate.

    ``mt_reject``: exact-equality Wald test rejects (non-applicable replicates
    count as non-rejections).
    """
    ok = bool(res["applicable"])
    get = lambda k: float(res.get(k, np.nan)) if ok else np.nan  # noqa: E731
    return {
        "applicable": ok,
        "K_x": int(res["K_x"]), "K_y": int(res["K_y"]),
        "D_hat": get("D_hat"), "Q_hat": get("Q_hat"), "Q_bc": get("Q_bc"), "s_Q": get("s_Q"),
        "Z": get("Z_tolerance"), "U_D": get("U_D"), "T0": get("T0"), "p_exact": get("p_value"),
        "decision": res.get("tolerance_decision") if ok else "not_applicable",
        "mt_reject": bool(ok and res["p_value"] < alpha),
        "success_x": float(res["optimizer_success_fraction_x"]),
        "success_y": float(res["optimizer_success_fraction_y"]),
        "max_residual_x": float(res["max_residual_x"]),
        "max_residual_y": float(res["max_residual_y"]),
    }


def max_residual(df: pd.DataFrame) -> float:
    """Largest relative stationarity h |grad f| / f at any accepted mode in df."""
    vals = df[["max_residual_x", "max_residual_y"]].to_numpy(dtype=float)
    return float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan


# ========================================================================
# 4. Monte Carlo runner
# ========================================================================
# Replicate r of design point s of experiment e uses
# numpy.random.SeedSequence([BASE_SEED, e, s, r]), so results do not depend on
# the number of workers. Each design point is saved to raw/<name>__<key>.csv as
# soon as it finishes, where <key> hashes the setting, modal_transport.py,
# mt_tools.py and the calling script. A saved file is reused only if its key
# matches, so changed code or settings are always recomputed.

BASE_SEED = 20260925


def rng_for(experiment: int, setting: int, rep: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, experiment, setting, rep]))


def _timed_rep(one_rep, setting, rep, rng) -> dict:
    t0 = time.time()
    row = one_rep(setting, rep, rng)
    row["rep"] = rep
    row["seconds"] = time.time() - t0
    return row


def _cache_key(setting: dict, one_rep: Callable) -> str:
    files = [ROOT / "modal_transport.py", ROOT / "mt_tools.py", Path(inspect.getsourcefile(one_rep))]
    h = hashlib.sha1()
    for f in files:
        h.update(f.read_bytes())
    scalars = {k: v for k, v in setting.items() if np.isscalar(v)}
    h.update(json.dumps(scalars, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def run_settings(
    settings: Sequence[dict],
    one_rep: Callable[[dict, int, np.random.Generator], dict],
    *,
    experiment: int,
    reps: int,
    raw_dir: Path,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    """Run ``one_rep(setting, rep, rng)`` for every setting and replicate."""
    names = [s["name"] for s in settings]
    if len(set(names)) != len(names):
        raise ValueError("Setting names must be unique.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    t_start = time.time()
    for s_idx, setting in enumerate(settings):
        path = raw_dir / f"{setting['name']}__{_cache_key(setting, one_rep)}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if len(df) >= reps:
                df = df[df["rep"] < reps].copy()
                for key, value in setting.items():
                    if np.isscalar(value):
                        df[key] = value
                frames.append(df)
                continue
        t0 = time.time()
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_timed_rep)(one_rep, setting, r, rng_for(experiment, s_idx, r)) for r in range(reps))
        df = pd.DataFrame(rows)
        for key, value in setting.items():
            if np.isscalar(value) and key not in df:
                df[key] = value
        df.to_csv(path, index=False)
        frames.append(df)
        print(f"[{s_idx + 1}/{len(settings)}] {setting['name']}: {reps} reps in "
              f"{(time.time() - t0) / 60:.1f} min (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
    return pd.concat(frames, ignore_index=True)


def kde_curve(x: np.ndarray, grid: np.ndarray, h: float, kernel_order: int = 6) -> np.ndarray:
    """Univariate KDE of ``x`` on ``grid`` (for figures)."""
    X = np.asarray(x, float).reshape(-1, 1)
    return np.array([ho_density(X, np.array([g]), h, kernel_order) for g in grid])


# ========================================================================
# 5. figure style
# ========================================================================
RC = {
    "font.family": "sans-serif",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "legend.fontsize": 8.4,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}

BLUE, VERMILLION, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
BLACK, GRAY, GRID = "#252525", "#777777", "#D9D9D9"

METHOD_STYLE = {
    "mt": ("Matched-mode test", BLUE, "o", "-"),
    "mmd": ("MMD", VERMILLION, "s", "--"),
    "energy": ("Energy", GREEN, "^", "-."),
    "gec": ("GEC (5-MST)", PURPLE, "D", ":"),
    "ks": ("KS", VERMILLION, "s", "--"),
    "kuiper": ("Kuiper", PURPLE, "D", ":"),
    "w1": (r"$W_1$ perm.", GRAY, "v", "--"),
    "w2": (r"$W_2$ perm.", GRAY, "v", "--"),
}


def style_axis(ax, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out", pad=3)


def new_figure(nrows: int, ncols: int, width: float, height: float, **kwargs):
    """Create a figure with constrained layout."""
    return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kwargs)


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9)


def top_legend(fig, axes, ncol: int, **kwargs) -> None:
    """One legend for the whole figure, placed above all panels."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels and not lab.startswith("_"):
                handles.append(h)
                labels.append(lab)
    # Place reference lines last.
    order = sorted(range(len(labels)), key=lambda k: labels[k].lstrip("$\\").startswith(("alpha", "delta")))
    handles, labels = [handles[k] for k in order], [labels[k] for k in order]
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False, **kwargs)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix.lstrip(".") or "pdf", facecolor="white")
    plt.close(fig)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, c - half)), (1.0 if k == n else min(1.0, c + half))


def proportion_with_ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=bool)
    k, n = int(x.sum()), len(x)
    lo, hi = wilson_interval(k, n)
    return (k / n if n else float("nan")), lo, hi


def errorbar(ax, x, p, lo, hi, *, label, color, marker, linestyle) -> None:
    p, lo, hi = map(np.asarray, (p, lo, hi))
    ax.errorbar(x, p, yerr=np.vstack([np.maximum(p - lo, 0), np.maximum(hi - p, 0)]),
                marker=marker, linestyle=linestyle, color=color, linewidth=1.6,
                markersize=4.4, capsize=2.0, label=label)


