"""
Matched-mode testing: two-sample inference for the locations of density modes.

For assumed Holder smoothness beta >= 2, the KDE uses the smallest even kernel
order r >= beta and the bandwidth h = n^{-1/(d+beta+2)} (data units). The
order-r kernel is the Gaussian scale mixture
    k_r(u) = sum_{j<r/2} w_j phi(u / 8^j) / 8^j,
with weights chosen so that the moments of order 2, 4, ..., r-2 vanish; r = 2
is the Gaussian kernel. The default beta = 6 gives the sixth-order kernel.

Mode search (find_regular_modes): from min{n, ceil(c_start log n)} sample
points, mean-shift-type gradient ascent x <- x + a h^2 grad f(x) / f(x) with
Armijo backtracking; stop when h |grad f| / f <= 1e-6 (or on entering the
merge radius of a mode already found); keep points with a negative definite
Hessian; merge within 0.04 h.

Inference (modal_transport_test): optimal matching of the two modal
configurations, sandwich covariance, bias-corrected tolerance test Z_{delta,bc},
one-sided upper bound U_D, and the exact-equality Wald test. The test is not
applicable when the cardinalities differ or the covariance is not positive
definite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2, norm


@dataclass
class ModalFit:
    modes: np.ndarray
    hessians: np.ndarray
    densities: np.ndarray
    bandwidth: float
    covariance: np.ndarray
    optimizer_success_fraction: float
    smoothness: float
    kernel_order: int
    n_starts: int
    start_multiplier: float
    max_residual: float


DEFAULT_SMOOTHNESS = 6.0
DEFAULT_START_MULTIPLIER = 50.0
DEFAULT_STATIONARITY_TOL = 1e-6
DEFAULT_MERGE_MULTIPLIER = 0.04
DEFAULT_START_SEED = 20260921


def _validate_smoothness(beta: float) -> float:
    beta = float(beta)
    if not np.isfinite(beta) or beta < 2.0:
        raise ValueError("smoothness beta must be finite and >= 2.")
    return beta


def kernel_order_for_smoothness(beta: float) -> int:
    beta = _validate_smoothness(beta)
    return 2 * int(math.ceil(beta / 2.0))


def default_bandwidth(n: int, d: int, smoothness: float = DEFAULT_SMOOTHNESS) -> float:
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if d < 1:
        raise ValueError("Dimension must be positive.")
    beta = _validate_smoothness(smoothness)
    return float(n ** (-1.0 / (d + beta + 2.0)))


def _validate_start_multiplier(start_multiplier: float) -> float:
    c = float(start_multiplier)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("start_multiplier must be a positive finite number.")
    return c


def default_n_starts(
    n: int,
    start_multiplier: float = DEFAULT_START_MULTIPLIER,
) -> int:
    """Default multistart count: min(n, ceil(c_start * log n))."""
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    c = _validate_start_multiplier(start_multiplier)
    return min(int(n), max(1, int(math.ceil(c * math.log(n)))))


def _validate_sample(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Sample must have shape (n, d).")
    if X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("Need n >= 3 and d >= 1.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Sample contains non-finite values.")
    return X


def _validate_analysis_region(analysis_region: Optional[np.ndarray], d: int) -> Optional[np.ndarray]:
    if analysis_region is None:
        return None
    region = np.asarray(analysis_region, dtype=float)
    if region.shape != (d, 2):
        raise ValueError(f"analysis_region must have shape {(d, 2)}.")
    if not np.all(np.isfinite(region)) or np.any(region[:, 0] >= region[:, 1]):
        raise ValueError("Invalid analysis_region.")
    return region


def _gaussian_phi_1d(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@lru_cache(maxsize=None)
def _kernel_spec(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights and scales of the order-``order`` Gaussian scale-mixture kernel.

    Scales s_j = 8^j, j = 0, ..., order/2 - 1; the weights solve
    sum_j w_j s_j^(2m) = 1{m = 0} for m = 0, ..., order/2 - 1 (exact rational
    arithmetic), so the kernel integrates to one and its moments of order
    2, ..., order - 2 vanish.  Order 2 is the Gaussian kernel.
    """
    order = int(order)
    if order < 2 or order % 2 != 0:
        raise ValueError("kernel order must be an even integer >= 2.")
    p = order // 2
    s2 = [Fraction(64) ** j for j in range(p)]
    aug = [[s2[j] ** m for j in range(p)] + [Fraction(int(m == 0))] for m in range(p)]
    for col in range(p):
        pivot = next(r for r in range(col, p) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        aug[col] = [v / aug[col][col] for v in aug[col]]
        for r in range(p):
            if r != col and aug[r][col] != 0:
                aug[r] = [a - aug[r][col] * b for a, b in zip(aug[r], aug[col])]
    weights = np.array([float(aug[j][-1]) for j in range(p)])
    scales = np.array([8.0 ** j for j in range(p)])
    return weights, scales


def _k1_all(u: np.ndarray, kernel_order: int):
    weights, scales = _kernel_spec(int(kernel_order))
    k0 = np.zeros_like(u, dtype=float)
    k1 = np.zeros_like(u, dtype=float)
    k2 = np.zeros_like(u, dtype=float)
    for w, s in zip(weights, scales):
        z = u / s
        phi = _gaussian_phi_1d(z)
        k0 += w * phi / s
        k1 += -w * u * phi / (s ** 3)
        k2 += w * (u * u / (s ** 5) - 1.0 / (s ** 3)) * phi
    return k0, k1, k2


def _ho_contributions(X: np.ndarray, x: np.ndarray, h: float, kernel_order: int, need_hessian: bool = True):
    n, d = X.shape
    u = (x[None, :] - X) / h
    k0, k1, k2 = _k1_all(u, kernel_order)

    prod0 = np.prod(k0, axis=1)
    density_contrib = prod0 / (h ** d)

    grad = np.empty((n, d), dtype=float)
    for a in range(d):
        if d == 1:
            other = np.ones(n)
        else:
            other = np.prod(np.delete(k0, a, axis=1), axis=1)
        grad[:, a] = k1[:, a] * other / (h ** (d + 1))

    if not need_hessian:
        return density_contrib, grad, None

    hess = np.empty((n, d, d), dtype=float)
    for a in range(d):
        if d == 1:
            other = np.ones(n)
        else:
            other = np.prod(np.delete(k0, a, axis=1), axis=1)
        hess[:, a, a] = k2[:, a] * other / (h ** (d + 2))
        for b in range(a + 1, d):
            keep = [j for j in range(d) if j not in (a, b)]
            other2 = np.prod(k0[:, keep], axis=1) if keep else np.ones(n)
            val = k1[:, a] * k1[:, b] * other2 / (h ** (d + 2))
            hess[:, a, b] = val
            hess[:, b, a] = val
    return density_contrib, grad, hess


def ho_density(X: np.ndarray, x: np.ndarray, h: float, kernel_order: int = 6) -> float:
    c, _, _ = _ho_contributions(X, np.asarray(x, dtype=float), h, kernel_order, False)
    return float(np.mean(c))


def ho_gradient(X: np.ndarray, x: np.ndarray, h: float, kernel_order: int = 6) -> np.ndarray:
    _, g, _ = _ho_contributions(X, np.asarray(x, dtype=float), h, kernel_order, False)
    return np.mean(g, axis=0)


def _ho_density_gradient(
    X: np.ndarray, x: np.ndarray, h: float, kernel_order: int
) -> tuple[float, np.ndarray]:
    """Evaluate KDE density and gradient in one pass."""
    c, g, _ = _ho_contributions(
        X, np.asarray(x, dtype=float), h, kernel_order, False
    )
    return float(np.mean(c)), np.mean(g, axis=0)


def _ho_density_gradient_hessian(
    X: np.ndarray, x: np.ndarray, h: float, kernel_order: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate KDE density, gradient, and Hessian in one pass."""
    c, g, H = _ho_contributions(
        X, np.asarray(x, dtype=float), h, kernel_order, True
    )
    Hbar = np.mean(H, axis=0)
    Hbar = 0.5 * (Hbar + Hbar.T)
    return float(np.mean(c)), np.mean(g, axis=0), Hbar


def ho_hessian(X: np.ndarray, x: np.ndarray, h: float, kernel_order: int = 6) -> np.ndarray:
    _, _, H = _ho_contributions(X, np.asarray(x, dtype=float), h, kernel_order, True)
    out = np.mean(H, axis=0)
    return 0.5 * (out + out.T)


def _sample_start_points(
    X: np.ndarray,
    n_starts: Optional[int] = None,
    start_multiplier: float = DEFAULT_START_MULTIPLIER,
    start_seed: int = DEFAULT_START_SEED,
) -> np.ndarray:
    """Choose distinct sample observations uniformly without replacement."""
    n = len(X)
    m = default_n_starts(n, start_multiplier) if n_starts is None else int(n_starts)
    if m < 1:
        raise ValueError("n_starts must be positive.")
    m = min(m, n)
    if m == n:
        return np.asarray(X, dtype=float).copy()
    rng = np.random.default_rng(int(start_seed))
    idx = rng.choice(n, size=m, replace=False)
    return np.asarray(X[idx], dtype=float).copy()


def _merge_points(points: np.ndarray, values: np.ndarray, tol: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, points.shape[1] if points.ndim == 2 else 0))
    order = np.argsort(-np.asarray(values))
    kept: list[np.ndarray] = []
    for idx in order:
        z = np.asarray(points[idx], dtype=float)
        if not kept or np.min(np.linalg.norm(np.vstack(kept) - z[None, :], axis=1)) > tol:
            kept.append(z.copy())
    return np.vstack(kept)


def find_regular_modes(
    X: np.ndarray,
    h: float,
    *,
    kernel_order: int = 6,
    analysis_region: Optional[np.ndarray] = None,
    n_starts: Optional[int] = None,
    start_multiplier: float = DEFAULT_START_MULTIPLIER,
    start_seed: int = DEFAULT_START_SEED,
    merge_tol: Optional[float] = None,
    merge_multiplier: float = DEFAULT_MERGE_MULTIPLIER,
    stationarity_tol: float = DEFAULT_STATIONARITY_TOL,
    armijo: float = 1e-4,
    max_iter: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    """Regular modes of the KDE by mean-shift-type gradient ascent from sample points.

    1. Starts: min(n, ceil(c_start log n)) distinct sample points, chosen at
       random independently of their values (all points when n is small).
    2. Ascent: x <- x + a h^2 grad f(x) / f(x), with Armijo backtracking
       from a = 1 (the step is in bandwidth units, hence unit free).
    3. Stop when h |grad f| / f <= stationarity_tol; keep the point if it is
       inside the analysis box and the KDE Hessian is negative definite.
    4. Merge kept points within merge_multiplier * h.

    Also returns the largest relative stationarity h |grad f| / f at the
    returned modes.
    """
    n, d = X.shape
    if analysis_region is None:
        lo = np.min(X, axis=0) - 2.0 * h
        hi = np.max(X, axis=0) + 2.0 * h
    else:
        lo, hi = analysis_region[:, 0], analysis_region[:, 1]
    starts = _sample_start_points(X, n_starts, start_multiplier, start_seed)
    m = len(starts)
    h2 = h * h
    mtol = merge_multiplier * h if merge_tol is None else float(merge_tol)
    candidates, values, n_success = [], [], 0
    for x in starts:
        x = np.clip(np.asarray(x, dtype=float), lo, hi)
        f, g = _ho_density_gradient(X, x, h, kernel_order)
        reached = joined = False
        for _ in range(max_iter):
            if not (f > 0 and np.all(np.isfinite(g))):
                break
            if h * np.linalg.norm(g) / f <= stationarity_tol:
                reached = True
                break
            if candidates and np.min(np.linalg.norm(np.asarray(candidates) - x, axis=1)) <= mtol:
                reached = joined = True  # entered the merge radius of an existing mode
                break
            step = h2 * g / f
            slope = float(g @ step)
            a = 1.0
            while a >= 2.0 ** -30:
                trial = x + a * step
                if np.all(trial > lo) and np.all(trial < hi):
                    ft, gt = _ho_density_gradient(X, trial, h, kernel_order)
                    if ft >= f + armijo * a * slope:
                        x, f, g = trial, ft, gt
                        break
                a *= 0.5
            else:
                break
        n_success += int(reached)
        if not reached or joined:
            continue
        dens, grad, H = _ho_density_gradient_hessian(X, x, h, kernel_order)
        if dens > 0 and np.linalg.eigvalsh(H)[-1] < 0:
            candidates.append(x)
            values.append(dens)
    success_fraction = n_success / m
    if not candidates:
        return np.empty((0, d)), np.empty((0, d, d)), np.empty(0), success_fraction, m, np.nan
    merged = _merge_points(np.vstack(candidates), np.asarray(values), mtol)
    modes, hessians, densities, residuals = [], [], [], []
    for x in merged:
        dens, grad, H = _ho_density_gradient_hessian(X, x, h, kernel_order)
        modes.append(x); hessians.append(H); densities.append(dens)
        residuals.append(h * np.linalg.norm(grad) / dens)
    modes = np.asarray(modes)
    order = np.lexsort(tuple(modes[:, j] for j in range(d - 1, -1, -1)))
    return (modes[order], np.asarray(hessians)[order], np.asarray(densities)[order], success_fraction, m,
            float(np.max(residuals)))


def empirical_modal_covariance(X: np.ndarray, modes: np.ndarray, hessians: np.ndarray, h: float, kernel_order: int) -> np.ndarray:
    n, d = X.shape
    K = len(modes)
    Psi = np.empty((n, K * d), dtype=float)
    for j, mu in enumerate(modes):
        _, grad, _ = _ho_contributions(X, mu, h, kernel_order, False)
        Psi[:, j * d:(j + 1) * d] = grad
    Psi -= np.mean(Psi, axis=0, keepdims=True)
    S = (Psi.T @ Psi) / (n - 1)
    B = block_diag(*[np.linalg.inv(H) for H in hessians])
    C = (B @ S @ B.T) / n
    return 0.5 * (C + C.T)


def fit_modal_configuration(
    X: np.ndarray,
    *,
    smoothness: float = DEFAULT_SMOOTHNESS,
    kernel_order: Optional[int] = None,
    bandwidth: Optional[float] = None,
    analysis_region: Optional[np.ndarray] = None,
    n_starts: Optional[int] = None,
    start_multiplier: float = DEFAULT_START_MULTIPLIER,
    start_seed: int = DEFAULT_START_SEED,
    merge_tol: Optional[float] = None,
    merge_multiplier: float = DEFAULT_MERGE_MULTIPLIER,
) -> ModalFit:
    X = _validate_sample(X)
    n, d = X.shape
    analysis_region = _validate_analysis_region(analysis_region, d)
    beta = _validate_smoothness(smoothness)
    order = kernel_order_for_smoothness(beta) if kernel_order is None else int(kernel_order)
    if order < 2 or order % 2 != 0 or order + 1e-12 < beta:
        raise ValueError("kernel_order must be an even integer >= smoothness beta.")
    h = default_bandwidth(n, d, beta) if bandwidth is None else float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Bandwidth must be a positive finite number.")
    modes, hessians, densities, success_fraction, m, max_residual = find_regular_modes(
        X, h, kernel_order=order, analysis_region=analysis_region,
        n_starts=n_starts, start_multiplier=start_multiplier, start_seed=start_seed, merge_tol=merge_tol,
        merge_multiplier=merge_multiplier,
    )
    C = np.empty((0, 0)) if len(modes) == 0 else empirical_modal_covariance(X, modes, hessians, h, order)
    return ModalFit(
        modes=modes, hessians=hessians, densities=densities, bandwidth=h,
        covariance=C, optimizer_success_fraction=success_fraction,
        smoothness=beta, kernel_order=order, n_starts=m,
        start_multiplier=float(start_multiplier), max_residual=max_residual,
    )


def _validate_metric(A: Optional[np.ndarray], d: int) -> np.ndarray:
    if A is None:
        return np.eye(d)
    A = np.asarray(A, dtype=float)
    if A.shape != (d, d) or not np.allclose(A, A.T, atol=1e-12, rtol=1e-12):
        raise ValueError("A must be symmetric with shape (d,d).")
    np.linalg.cholesky(A)
    return A


def _wald(delta: np.ndarray, Omega: np.ndarray):
    Omega = 0.5 * (Omega + Omega.T)
    eig = np.linalg.eigvalsh(Omega)
    scale = max(1.0, float(np.max(np.abs(eig))))
    if float(eig[0]) <= 1e-12 * scale:
        raise np.linalg.LinAlgError("Covariance is not sufficiently positive definite.")
    cf = cho_factor(Omega, lower=True, check_finite=True)
    solved = cho_solve(cf, delta, check_finite=True)
    T = float(delta @ solved)
    return T, float(chi2.sf(T, df=len(delta))), float(np.linalg.cond(Omega)), float(eig[0])


def modal_transport_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    A: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    smoothness: float = DEFAULT_SMOOTHNESS,
    kernel_order: Optional[int] = None,
    bandwidth_x: Optional[float] = None,
    bandwidth_y: Optional[float] = None,
    n_starts_x: Optional[int] = None,
    n_starts_y: Optional[int] = None,
    start_multiplier: float = DEFAULT_START_MULTIPLIER,
    start_seed: int = DEFAULT_START_SEED,
    merge_tol: Optional[float] = None,
    merge_multiplier: float = DEFAULT_MERGE_MULTIPLIER,
    tolerance: Optional[float] = None,
    direction: str = "both",
    analysis_region: Optional[np.ndarray] = None,
) -> dict:
    """Two-sample matched-mode test.

    direction: "both" (default) reports the three-way decision; "equivalence"
    runs only the equivalence test (decision "equivalent" or "not_equivalent");
    "change" runs only the meaningful-change test (decision
    "meaningfully_different" or "not_different"). Each test has level alpha.
    """
    if direction not in ("both", "equivalence", "change"):
        raise ValueError('direction must be "both", "equivalence", or "change".')
    X = _validate_sample(X)
    Y = _validate_sample(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same dimension.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1).")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0.0):
        raise ValueError("tolerance must be positive.")

    d = X.shape[1]
    A = _validate_metric(A, d)
    region = _validate_analysis_region(analysis_region, d)
    fx = fit_modal_configuration(
        X, smoothness=smoothness, kernel_order=kernel_order,
        bandwidth=bandwidth_x, analysis_region=region, n_starts=n_starts_x,
        start_multiplier=start_multiplier, start_seed=start_seed, merge_tol=merge_tol,
        merge_multiplier=merge_multiplier,
    )
    fy = fit_modal_configuration(
        Y, smoothness=smoothness, kernel_order=kernel_order,
        bandwidth=bandwidth_y, analysis_region=region, n_starts=n_starts_y,
        start_multiplier=start_multiplier, start_seed=int(start_seed) + 1, merge_tol=merge_tol,
        merge_multiplier=merge_multiplier,
    )
    Kx, Ky = len(fx.modes), len(fy.modes)
    base = {
        "applicable": False, "reason": None, "K_x": Kx, "K_y": Ky,
        "bandwidth_x": fx.bandwidth, "bandwidth_y": fy.bandwidth,
        "n_starts_x": fx.n_starts, "n_starts_y": fy.n_starts,
        "start_multiplier": float(start_multiplier), "start_seed": int(start_seed),
        "merge_multiplier": float(merge_multiplier),
        "optimizer_success_fraction_x": fx.optimizer_success_fraction,
        "optimizer_success_fraction_y": fy.optimizer_success_fraction,
        "max_residual_x": fx.max_residual, "max_residual_y": fy.max_residual,
        "smoothness": fx.smoothness, "kernel_order_x": fx.kernel_order, "kernel_order_y": fy.kernel_order,
    }
    if Kx == 0 or Ky == 0 or Kx != Ky:
        base["reason"] = "modal_cardinality_mismatch_or_empty"
        return base

    K = Kx
    diff = fx.modes[:, None, :] - fy.modes[None, :, :]
    cost = np.einsum("jka,ab,jkb->jk", diff, A, diff)
    rows, cols = linear_sum_assignment(cost)
    pi = np.empty(K, dtype=int); pi[rows] = cols
    matched_y = fy.modes[pi]
    delta = (fx.modes - matched_y).reshape(K * d)
    idx = np.concatenate([np.arange(j * d, (j + 1) * d) for j in pi])
    Omega = fx.covariance + fy.covariance[np.ix_(idx, idx)]
    Omega = 0.5 * (Omega + Omega.T)

    A_K = np.kron(np.eye(K), A)
    Q_hat = float(delta @ A_K @ delta)
    D_hat = math.sqrt(max(Q_hat / K, 0.0))
    trace_A_omega = float(np.trace(A_K @ Omega))
    Q_bc = float(Q_hat - trace_A_omega)
    v = A_K @ delta
    var_Q_linear = float(4.0 * v @ Omega @ v)
    AOmega = A_K @ Omega
    var_Q_quadratic = float(2.0 * np.trace(AOmega @ AOmega))
    s_Q_sq = var_Q_linear + var_Q_quadratic
    if s_Q_sq < 0 and abs(s_Q_sq) < 1e-12 * max(1.0, abs(Q_hat), abs(trace_A_omega)):
        s_Q_sq = 0.0
    s_Q = math.sqrt(s_Q_sq) if s_Q_sq >= 0 else float("nan")

    # One-sided (1 - alpha) upper confidence bound for D_A.
    U_D = math.sqrt(max(Q_bc + float(norm.ppf(1 - alpha)) * s_Q, 0.0) / K) if np.isfinite(s_Q) else float("nan")

    tol_result = {
        "U_D": U_D,
        "tolerance": tolerance, "tau_tolerance": np.nan, "Z_tolerance": np.nan,
        "equivalence_reject": False, "meaningful_change_reject": False,
        "direction": direction,
        "tolerance_decision": None if tolerance is None else
        {"both": "inconclusive", "equivalence": "not_equivalent", "change": "not_different"}[direction],
    }
    if tolerance is not None:
        tau = float(K * tolerance * tolerance)
        tol_result["tau_tolerance"] = tau
        if np.isfinite(s_Q) and s_Q > 0:
            z = float((Q_bc - tau) / s_Q)
            eq = bool(z < norm.ppf(alpha)) and direction != "change"
            chg = bool(z > norm.ppf(1 - alpha)) and direction != "equivalence"
            tol_result.update({
                "Z_tolerance": z, "equivalence_reject": eq,
                "meaningful_change_reject": chg,
                "tolerance_decision": "equivalent" if eq else "meaningfully_different" if chg
                else tol_result["tolerance_decision"],
            })

    try:
        T0, p0, cond, mineig = _wald(delta, Omega)
    except np.linalg.LinAlgError:
        base.update({
            "reason": "covariance_not_positive_definite", "modes_x": fx.modes, "modes_y": fy.modes,
            "matching": pi, "delta": delta, "D_hat": D_hat, "Q_hat": Q_hat, "Q_bc": Q_bc,
            "trace_A_omega": trace_A_omega, "var_Q_linear": var_Q_linear,
            "var_Q_quadratic": var_Q_quadratic, "s_Q": s_Q, **tol_result,
        })
        return base

    base.update({
        "applicable": True, "K": K, "df": K * d, "modes_x": fx.modes, "modes_y": fy.modes,
        "matching": pi, "delta": delta, "D_hat": D_hat, "Q_hat": Q_hat, "Q_bc": Q_bc,
        "trace_A_omega": trace_A_omega, "var_Q_linear": var_Q_linear,
        "var_Q_quadratic": var_Q_quadratic, "s_Q": s_Q, **tol_result,
        "T0": T0, "p_value": p0, "reject": bool(p0 < alpha), "omega": Omega,
        "omega_min_eig": mineig, "omega_condition": cond,
    })
    return base
