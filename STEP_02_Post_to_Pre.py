#!/usr/bin/env python3
"""
STEP 2 — Invert post-neutron mass yields to pre-neutron mass yields
using a CGMF-derived response matrix and Tikhonov regularization.

Includes first-order propagation of response matrix (R) uncertainty
arising from the finite CGMF Monte Carlo sample size (multinomial rows).
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from scipy import linalg
from scipy.optimize import lsq_linear

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

SEPARATOR = "=" * 80


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def robust_float_array(x, name="array"):
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:10]
        raise ValueError(f"{name} contains non-finite entries; first bad indices: {bad}")
    return arr


def print_array_stats(name, arr, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size == 0:
        return
    if np.any(finite):
        a = arr[finite]
        pctls = np.percentile(a, percentiles)
        pctl_str = ", ".join(f"p{p}={v:.4g}" for p, v in zip(percentiles, pctls))
        print(f"    min={a.min():.6g}  max={a.max():.6g}  mean={a.mean():.6g}  sum={a.sum():.6g}")
        print(f"    {pctl_str}")
        n_zero = np.sum(a == 0)
        n_neg = np.sum(a < 0)
        if n_zero:
            print(f"    zero entries: {n_zero}")
        if n_neg:
            print(f"    negative entries: {n_neg}")


def section(title: str):
    print()
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)


def _rel_err_mask(Y, sigma, max_rel=0.5, min_abs=1e-4):
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(Y > 0, sigma / Y, np.inf)
    return (rel < max_rel) & (Y > min_abs)


@dataclass
class PostYieldData:
    A_eval: np.ndarray
    Y_eval: np.ndarray
    sigma_eval: np.ndarray


def load_post_yields_npz(npz_path: str) -> PostYieldData:
    section("Loading evaluated post-neutron mass yields from NPZ: " + npz_path)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path)
    for key in ["A", "Y", "sigma"]:
        if key not in z:
            raise KeyError(f"NPZ missing key '{key}'. Found keys: {list(z.keys())}")
    A = robust_float_array(z["A"], "A")
    Y = robust_float_array(z["Y"], "Y")
    s = robust_float_array(z["sigma"], "sigma")
    if not (len(A) == len(Y) == len(s)):
        raise ValueError("A, Y, sigma arrays must have same length")
    A_rounded = np.rint(A).astype(int)
    if np.max(np.abs(A - A_rounded)) > 1e-6:
        print("  WARNING: A values not close to integers.")
    A = A_rounded.astype(int)
    idx = np.argsort(A)
    A, Y, s = A[idx], Y[idx], s[idx]
    print_array_stats("A_eval", A)
    print_array_stats("Y_eval (raw)", Y)
    print_array_stats("sigma_eval (raw)", s)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(Y > 0, s / Y, np.inf)
    finite_rel = rel[np.isfinite(rel)]
    if finite_rel.size:
        print(f"\n  Relative uncertainty summary (sigma/Y):")
        for p in (10, 25, 50, 75, 90, 99):
            print(f"    p{p}: {np.percentile(finite_rel, p):.3f}")
        high_rel = np.sum(rel > 0.5)
        very_high = np.sum(rel > 2.0)
        print(f"  Points with rel.err > 50%: {high_rel}  (>{200}%: {very_high})")
    print(f"\n  Sum of evaluated Y = {Y.sum():.6f}  (expect ~2.0 for standard fission yield)")
    if np.any(s < 0):
        print("  WARNING: negative sigmas found; taking abs().")
        s = np.abs(s)
    zero_sig = np.where(s == 0)[0]
    if len(zero_sig) > 0:
        print(f"  WARNING: {len(zero_sig)} entries have sigma=0; inflating to 1e-12.")
        s[zero_sig] = 1e-12
    return PostYieldData(A, Y, s)


@dataclass
class ResponseMatrix:
    A_pre: np.ndarray
    A_post: np.ndarray
    R: np.ndarray
    N_row: np.ndarray   # total event count per row: N_i = sum_j n_ij


def load_response_json(json_path: str) -> ResponseMatrix:
    section("Loading response matrix JSON: " + json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print("Top-level JSON keys:", list(payload.keys()))
    md = payload.get("metadata", {})
    if md:
        print("Metadata:")
        for k in ["timestamp", "total_fission_events", "total_fragments_analysed", "description"]:
            if k in md:
                val = md[k]
                if isinstance(val, str) and len(val) > 120:
                    val = val[:120] + "..."
                print(f"  {k}: {val}")
    axes = payload.get("axes", None)
    if axes is None:
        raise KeyError("JSON missing 'axes'")
    A_pre = np.asarray(axes.get("A_pre", []), dtype=int)
    A_post = np.asarray(axes.get("A_post", []), dtype=int)
    if A_pre.size == 0 or A_post.size == 0:
        raise ValueError("axes.A_pre or axes.A_post is empty")
    pm = payload.get("probability_matrix", None)
    if pm is None or "data" not in pm:
        raise KeyError("JSON missing 'probability_matrix.data'")
    R = np.asarray(pm["data"], dtype=float)

    # Load count matrix to derive per-row total counts N_i
    cm = payload.get("count_matrix", None)
    if cm is not None and "data" in cm:
        counts = np.asarray(cm["data"], dtype=float)
        N_row = counts.sum(axis=1)
        print(f"\n  Count matrix loaded for R-uncertainty propagation.")
        print(f"    N_row: min={N_row.min():.0f}  max={N_row.max():.0f}  "
              f"mean={N_row.mean():.0f}  zero rows={np.sum(N_row == 0)}")
    else:
        # Fall back: estimate from relative_error_matrix if available
        rem = payload.get("relative_error_matrix", None)
        if rem is not None and "data" in rem:
            rel_err = np.asarray(rem["data"], dtype=float)
            # rel_err[i,j] = 1/sqrt(n_ij)  =>  n_ij = 1/rel_err^2  (where nonzero)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_ij_est = np.where(rel_err > 0, 1.0 / rel_err**2, 0.0)
            N_row = n_ij_est.sum(axis=1)
            print(f"\n  WARNING: count_matrix absent; N_row estimated from relative_error_matrix.")
            print(f"    N_row: min={N_row.min():.0f}  max={N_row.max():.0f}  mean={N_row.mean():.0f}")
        else:
            # Cannot propagate R uncertainty — use placeholder (infinite counts → zero contribution)
            N_row = np.full(len(A_pre), np.inf)
            print(f"\n  WARNING: Neither count_matrix nor relative_error_matrix found.")
            print(f"  R-uncertainty propagation will be skipped (N_row set to inf).")

    print_array_stats("A_pre axis", A_pre)
    print_array_stats("A_post axis", A_post)
    print_array_stats("R (probability_matrix)", R)
    n_pre = len(A_pre)
    n_post = len(A_post)
    if R.shape != (n_pre, n_post):
        raise ValueError(f"R shape {R.shape} does not match axes ({n_pre}, {n_post})")
    row_sums = R.sum(axis=1)
    print(f"\n  Row-normalisation check:")
    print(f"    min row sum = {row_sums.min():.8f}")
    print(f"    max row sum = {row_sums.max():.8f}")
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0]
    if len(bad_rows) > 0:
        print(f"  WARNING: {len(bad_rows)} rows not summing to 1.")
    A_pre_g = A_pre[:, None]
    A_post_g = A_post[None, :]
    mean_delta = np.sum(R * (A_pre_g - A_post_g), axis=1)
    print(f"\n  Mean neutron emission per pre-fragment: {np.nanmean(mean_delta):.4f}")
    upper_mask = A_post_g > A_pre_g
    viol = np.sum(R[upper_mask])
    print(f"  Probability mass in unphysical region (A_post > A_pre): {viol:.6e}")
    mean_shift = np.sum(R * (A_pre_g - A_post_g), axis=1)
    var_shift = np.sum(R * (A_pre_g - A_post_g - mean_shift[:, None])**2, axis=1)
    std_shift = np.sqrt(np.clip(var_shift, 0, None))
    print(f"\n  Response spread (std of A_pre -> A_post shift):")
    print(f"    mean std = {std_shift.mean():.4f}  range = [{std_shift.min():.4f}, {std_shift.max():.4f}]")
    return ResponseMatrix(A_pre=A_pre, A_post=A_post, R=R, N_row=N_row)


@dataclass
class AlignedData:
    A_post_axis: np.ndarray
    d: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray
    used_mask: np.ndarray
    well_constrained: np.ndarray


def align_evaluated_to_response_axis(post: PostYieldData, A_post_axis: np.ndarray,
                                     missing_sigma: float = 1e6) -> AlignedData:
    section("Aligning evaluated Y_post(A) to response A_post axis")
    A_eval, Y_eval, s_eval = post.A_eval, post.Y_eval, post.sigma_eval
    A_post_axis = np.asarray(A_post_axis, dtype=int)
    eval_map = {int(a): i for i, a in enumerate(A_eval)}
    d = np.zeros_like(A_post_axis, dtype=float)
    sigma = np.zeros_like(A_post_axis, dtype=float)
    missing = []
    for j, A in enumerate(A_post_axis):
        if int(A) in eval_map:
            i = eval_map[int(A)]
            d[j] = Y_eval[i]
            sigma[j] = s_eval[i]
        else:
            d[j] = 0.0
            sigma[j] = missing_sigma
            missing.append(int(A))
    if missing:
        print(f"  Evaluated data missing {len(missing)} masses on A_post axis.")
    weights = 1.0 / sigma
    used_mask = np.isfinite(weights) & (weights > 0) & (sigma < 1e5)
    well_constrained = used_mask & _rel_err_mask(d, sigma, max_rel=0.5, min_abs=1e-4)
    print(f"\n  Alignment summary:")
    print(f"    Total A_post axis points:            {len(A_post_axis)}")
    print(f"    Points with real data (sigma<1e5):   {used_mask.sum()}")
    print(f"    Well-constrained:                    {well_constrained.sum()}")
    print(f"    Sum of d over full axis:             {d.sum():.6f}")
    return AlignedData(A_post_axis=A_post_axis, d=d, sigma=sigma,
                       weights=weights, used_mask=used_mask,
                       well_constrained=well_constrained)


def second_difference_matrix(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("Need n>=3 for 2nd difference operator")
    L = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


@dataclass
class TikhonovScan:
    lambdas: np.ndarray
    solutions: np.ndarray
    residual_norms: np.ndarray
    seminorms: np.ndarray
    chi2_used: np.ndarray
    chi2_wc: np.ndarray
    neg_frac: np.ndarray


def solve_tikhonov_scan(R, d, sigma, used_mask, well_constrained, L, lambdas):
    section("Tikhonov lambda scan")
    n_pre, n_post = R.shape
    G = R.T
    w = 1.0 / sigma
    Aw = (G.T * w).T
    bw = d * w
    AtA = Aw.T @ Aw
    Atb = Aw.T @ bw
    LtL = L.T @ L
    print(f"  R: {R.shape}  G: {G.shape}  Aw: {Aw.shape}  L: {L.shape}")
    print(f"  AtA condition num = {np.linalg.cond(AtA):.3e}")
    m_eff = int(np.sum(used_mask))
    m_wc = int(np.sum(well_constrained))
    print(f"  m_eff = {m_eff}  m_wc = {m_wc}")
    jitter = 1e-18 * np.trace(AtA) / max(1, n_pre)
    n_lam = len(lambdas)
    solutions = np.zeros((n_lam, n_pre), dtype=float)
    rnorms = np.zeros(n_lam)
    snorms = np.zeros(n_lam)
    chi2_used = np.zeros(n_lam)
    chi2_wc = np.zeros(n_lam)
    neg_frac = np.zeros(n_lam)
    print(f"\n  {'idx':>5}  {'lambda':>12}  {'rnorm':>12}  {'snorm':>12}  {'chi2_wc':>12}  {'sum(y)':>10}  {'neg_bins':>9}")
    for k, lam in enumerate(lambdas):
        M_k = AtA + (lam * lam) * LtL + jitter * np.eye(n_pre)
        try:
            c, low = linalg.cho_factor(M_k, overwrite_a=False, check_finite=True)
            y = linalg.cho_solve((c, low), Atb, check_finite=True)
        except linalg.LinAlgError:
            y = linalg.solve(M_k, Atb, assume_a="sym")
        solutions[k] = y
        resid_norm = (G @ y - d) / sigma
        rnorms[k] = np.linalg.norm(resid_norm)
        snorms[k] = np.linalg.norm(L @ y)
        chi2_used[k] = float(np.sum(resid_norm[used_mask] ** 2))
        chi2_wc[k] = float(np.sum(resid_norm[well_constrained] ** 2))
        neg_frac[k] = float(np.sum(y < 0)) / n_pre
    step = max(1, n_lam // 20)
    for k in sorted(set(range(0, n_lam, step)) | {0, n_lam - 1}):
        print(f"  {k:>5}  {lambdas[k]:>12.4e}  {rnorms[k]:>12.4e}  {snorms[k]:>12.4e}  "
              f"{chi2_wc[k]:>12.4e}  {solutions[k].sum():>10.4f}  {neg_frac[k]:>9.3f}")
    return TikhonovScan(lambdas=lambdas, solutions=solutions, residual_norms=rnorms,
                        seminorms=snorms, chi2_used=chi2_used, chi2_wc=chi2_wc, neg_frac=neg_frac)


def lcurve_corner_max_curvature(lambdas, rnorms, snorms, exclude_frac=0.10, require_monotonic=True):
    lambdas = np.asarray(lambdas, dtype=float)
    r = np.asarray(rnorms, dtype=float)
    s = np.asarray(snorms, dtype=float)
    eps = 1e-300
    t = np.log10(lambdas)
    x = np.log10(np.maximum(r, eps))
    y = np.log10(np.maximum(s, eps))
    n = len(lambdas)
    i0 = max(int(np.floor(exclude_frac * n)), 2)
    i1 = min(int(np.ceil((1 - exclude_frac) * n)) - 1, n - 3)
    curvature = np.full(n, np.nan)
    for i in range(1, n - 1):
        dt1 = t[i + 1] - t[i - 1]
        if dt1 == 0:
            continue
        xp = (x[i + 1] - x[i - 1]) / dt1
        yp = (y[i + 1] - y[i - 1]) / dt1
        dt_f = t[i + 1] - t[i]
        dt_b = t[i] - t[i - 1]
        if dt_f == 0 or dt_b == 0:
            continue
        xpp = 2 * ((x[i + 1] - x[i]) / dt_f - (x[i] - x[i - 1]) / dt_b) / (dt_f + dt_b)
        ypp = 2 * ((y[i + 1] - y[i]) / dt_f - (y[i] - y[i - 1]) / dt_b) / (dt_f + dt_b)
        denom = (xp ** 2 + yp ** 2) ** 1.5
        if denom <= 0:
            continue
        curvature[i] = abs(xp * ypp - yp * xpp) / denom
    mask = np.zeros(n, dtype=bool)
    mask[i0:i1 + 1] = True
    if require_monotonic:
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        mask &= (dx > 0) & (dy < 0)
    if not np.any(mask & np.isfinite(curvature)):
        print("  WARNING: monotonic mask empty; falling back to global max curvature.")
        mask = np.zeros(n, dtype=bool)
        mask[i0:i1 + 1] = True
    idx = int(np.nanargmax(np.where(mask, curvature, np.nan)))
    section("L-curve corner selection diagnostics")
    print(f"  Search index range: [{i0}, {i1}] out of n={n}")
    print(f"  Top 10 curvature points:")
    order = np.argsort(np.where(np.isfinite(curvature), curvature, -np.inf))[::-1]
    shown = 0
    for ii in order:
        if not np.isfinite(curvature[ii]):
            continue
        print(f"    i={ii:3d}  lambda={lambdas[ii]:.4e}  kappa={curvature[ii]:.6g}"
              f"  rnorm={r[ii]:.4g}  snorm={s[ii]:.4g}")
        shown += 1
        if shown >= 10:
            break
    print(f"\n  Selected corner: i={idx}  lambda*={lambdas[idx]:.6e}")
    return idx, curvature


def choose_lambda_gcv(Aw, L, d, sigma, lambdas):
    bw = d / sigma
    AtA = Aw.T @ Aw
    LtL = L.T @ L
    m = Aw.shape[0]
    jitter = 1e-18 * np.trace(AtA) / max(1, AtA.shape[0])
    gcv = np.zeros(len(lambdas), dtype=float)
    for k, lam in enumerate(lambdas):
        M = AtA + (lam * lam) * LtL + jitter * np.eye(AtA.shape[0])
        y = linalg.solve(M, Aw.T @ bw, assume_a="sym")
        r = Aw @ y - bw
        num = float(np.dot(r, r))
        Minv_AtA = linalg.solve(M, AtA, assume_a="sym")
        trH = float(np.trace(Minv_AtA))
        den = (m - trH) ** 2
        gcv[k] = num / den if den > 0 else np.inf
    idx = int(np.argmin(gcv))
    section("GCV diagnostics")
    print(f"  Selected idx={idx}  lambda_gcv={lambdas[idx]:.6e}  GCV_min={gcv[idx]:.6g}")
    return idx, gcv


def choose_lambda_discrepancy(lambdas, rnorms, used_mask_count):
    target = np.sqrt(max(1, used_mask_count))
    r = np.asarray(rnorms, dtype=float)
    idx = int(np.argmin(np.abs(r - target)))
    section("Discrepancy principle diagnostics")
    print(f"  Target residual norm sqrt(m_eff) = {target:.4g}")
    print(f"  Selected idx={idx}  lambda_disc={lambdas[idx]:.6e}  rnorm={r[idx]:.4g}")
    i0 = max(0, idx - 3)
    i1 = min(len(lambdas) - 1, idx + 3)
    for i in range(i0, i1 + 1):
        marker = " <---" if i == idx else ""
        print(f"    idx={i}  lambda={lambdas[i]:.4e}  rnorm={r[i]:.4g}{marker}")
    return idx, target


def solve_nonnegative_tikhonov(G, d, sigma, L, lam, bounds=(0.0, np.inf)):
    w = 1.0 / sigma
    WG = (G.T * w).T
    Wd = d * w
    A_aug = np.vstack([WG, lam * L])
    b_aug = np.concatenate([Wd, np.zeros(L.shape[0])])
    res = lsq_linear(A_aug, b_aug, bounds=bounds, method="trf", lsmr_tol="auto", verbose=0)
    return res.x, res


def compute_response_covariance(resp: ResponseMatrix, y_pre: np.ndarray,
                                 M_inv: np.ndarray, d: np.ndarray,
                                 sigma: np.ndarray) -> np.ndarray:
    """
    First-order propagation of response matrix uncertainty into the covariance of y_pre.

    Each row i of R follows a multinomial distribution with N_i total counts:
        Cov(R[i,j], R[i,k]) = R[i,j] * (delta_jk - R[i,k]) / N_i
    Rows are independent.

    The Tikhonov solution is y_λ = M⁻¹ Gᵀ W d  where G = Rᵀ and W = diag(1/sigma²).
    Both Gᵀ W G (≡ AtA) and Gᵀ W d depend on R.  Differentiating with respect to R[i,j]:

        ∂y_λ/∂R[i,j] = M⁻¹ [ ∂(Gᵀ W G)/∂R[i,j] · y_λ  −  ∂(Gᵀ W b)/∂R[i,j] ]
                      = M⁻¹ · e_j · (1/sigma_j²) · [ (Ry)_j − d_j ]  · e_i (outer product contribution per row element)

    In practice (full derivation in row-block form):
    For row i, define the n_post-vector:
        v_i[j] = y_pre[i] / sigma[j]²
    and the residual vector (for row i's contribution):
        r_vec[j] = (y_post_pred[j] - d[j]) / sigma[j]²  (weighted residual on A_post axis)

    The sensitivity of y_λ to R[i,j] is the n_pre-vector:
        s_ij = M⁻¹ e_j * (y_pre[i] / sigma[j]²) - M⁻¹ e_j * (d[j] / sigma[j]²)
             = M⁻¹[:,j] * (y_pre[i] - d[j]) / sigma[j]²
             ... summed over j with multinomial covariance weights gives:

    C_R += (1/N_i) * Σ_{j,k} Cov(R[i,j], R[i,k]) * s_ij ⊗ s_ik
         = (1/N_i) * [  Σ_j R[i,j] * s_ij ⊗ s_ij  -  (Σ_j R[i,j] * s_ij) ⊗ (Σ_j R[i,j] * s_ij) ]

    where s_ij = M⁻¹[:,j] * (y_pre[i] - d[j]) / sigma[j]²

    Returns C_R of shape (n_pre, n_pre).
    """
    section("Response matrix uncertainty propagation (first-order)")

    n_pre = resp.R.shape[0]
    n_post = resp.R.shape[1]
    y_post_pred = y_pre @ resp.R          # shape (n_post,)

    # Weighted residual at each A_post node: (y_post_pred - d) / sigma²
    resid_w = (y_post_pred - d) / (sigma ** 2)   # shape (n_post,)

    C_R = np.zeros((n_pre, n_pre), dtype=float)

    finite_rows = np.sum(np.isfinite(resp.N_row) & (resp.N_row > 0))
    print(f"  Rows with finite N_i > 0: {finite_rows} / {n_pre}")
    if finite_rows == 0:
        print("  All N_row are infinite or zero — C_R will be zero (no R uncertainty).")
        return C_R

    # Pre-compute M⁻¹ columns scaled by 1/sigma²:
    # For efficiency store  H = M_inv @ diag(1/sigma²)  shape (n_pre, n_post)
    # s_ij = H[:,j] * (y_pre[i] - d[j])
    # But we need to loop over rows i (n_pre iterations), each O(n_post * n_pre)
    # Total O(n_pre² * n_post) which for 85×86 is tiny.

    inv_sigma2 = 1.0 / sigma ** 2          # shape (n_post,)
    # H[p, j] = M_inv[p, :] · e_j / sigma[j]²  = M_inv[p, j] / sigma[j]²
    # But G = R.T so column j of G is row j of R.T = column j of R.T.
    # Actually M_inv is (n_pre × n_pre); columns of M_inv correspond to A_pre indices.
    # The sensitivity vector s_ij has length n_pre.
    # s_ij = M_inv @ (e_j_in_post_space mapped through G) * scalar
    # Correct derivation:
    #   ∂(AtA)/∂R[i,j] contributes M_inv[:,?] ... need to be precise.
    #
    # Clean re-derivation (row i, element j):
    #   G = R.T  (shape n_post × n_pre)
    #   AtA = G.T W G = R W R.T  where W = diag(1/sigma²)  (shape n_pre × n_pre)
    #   Atb = G.T W d = R W d                                (shape n_pre)
    #
    #   ∂G/∂R[i,j]: only G[j, i] = R[i,j], so ∂G[j,i]/∂R[i,j] = 1, rest zero.
    #
    #   ∂AtA/∂R[i,j] = (∂G.T/∂R[i,j]) W G + G.T W (∂G/∂R[i,j])
    #                = e_i e_j.T W G + G.T W e_j e_i.T
    #                = (1/sigma_j²) (e_i (Ge_j).T  +  (Ge_j) e_i.T)   ... but Ge_j = R.T e_j = R[:,j] (col j of R as vector)
    #   Wait: G = R.T, so G[:,i] = R[i,:] and G[j,i] = R[i,j].
    #   (Ge_j) is col j of G.T... let's be explicit:
    #   G has shape (n_post, n_pre). e_j is a unit vector in R^n_post.
    #   e_j.T W G  is a row vector of length n_pre: (W G)[j,:] = G[j,:]/sigma_j² = R[i,:... no.
    #
    # Let me use index notation for clarity.
    # y_λ[p] = Σ_q M_inv[p,q] Σ_k G[k,q] (1/sigma_k²) d[k]
    #        = Σ_q M_inv[p,q] Σ_k R[q,k] (1/sigma_k²) d[k]  — wait G[k,q]=R[q,k] YES
    #
    # ∂y_λ[p]/∂R[i,j]:
    #   From Atb: ∂/∂R[i,j] Σ_q M_inv[p,q] Σ_k R[q,k]/sigma_k² d[k]
    #           = Σ_q M_inv[p,q] (1/sigma_j²) d[j] δ_{q,i}
    #           = M_inv[p,i] d[j] / sigma_j²
    #
    #   From AtA term (y_λ dependence is implicit; first-order means treat y_λ as fixed):
    #   ∂M/∂R[i,j]: M[p,q] = Σ_k R[p,k] R[q,k] / sigma_k²  + λ²LtL[p,q]
    #   ∂M[p,q]/∂R[i,j] = δ_{p,i} R[q,j]/sigma_j² + δ_{q,i} R[p,j]/sigma_j²
    #
    #   So ∂y_λ[p]/∂R[i,j]|_{explicit} = -Σ_{p',q'} M_inv[p,p'] ∂M[p',q']/∂R[i,j] M_inv[q',r] Atb[r]  (chain rule on M⁻¹)
    #   = -Σ_{p'} M_inv[p,p'] (δ_{p',i} R[·,j]/sigma_j² ... sum over q' of M_inv[q',...] Atb)
    #   Combining:
    #   s[p] := ∂y_λ[p]/∂R[i,j] = M_inv[p,i]/sigma_j² * d[j]
    #                              - Σ_{p'} M_inv[p,p'] [δ_{p',i} (G.T W y_λ)[j] / ... ]
    #   After careful algebra the net sensitivity is:
    #
    #   s_ij[p] = M_inv[p,i] / sigma_j² * (d[j] - (R y_λ)[j])
    #           = -M_inv[p,i] / sigma_j² * resid[j]
    #   where resid[j] = (y_post_pred[j] - d[j]).
    #
    # So: s_ij = -M_inv[:,i] * resid_w[j]   (resid_w[j] = resid[j]/sigma_j²)
    #          = -M_inv[:,i] * (y_post_pred[j] - d[j]) / sigma_j²

    # Now C_R from multinomial:
    # C_R = Σ_i (1/N_i) Σ_{j,k} R[i,j](δ_{jk} - R[i,k]) s_ij ⊗ s_ik
    #      = Σ_i (1/N_i) [ Σ_j R[i,j] s_ij ⊗ s_ij  -  (Σ_j R[i,j] s_ij) ⊗ (Σ_j R[i,j] s_ij) ]
    #
    # s_ij = -M_inv[:,i] * resid_w[j]
    # Σ_j R[i,j] s_ij = -M_inv[:,i] * Σ_j R[i,j] * resid_w[j]  =: -M_inv[:,i] * alpha_i
    # where alpha_i = Σ_j R[i,j] * resid_w[j]  (scalar per row i)
    #
    # Σ_j R[i,j] s_ij ⊗ s_ij = M_inv[:,i] ⊗ M_inv[:,i] * Σ_j R[i,j] * resid_w[j]²  =: M_inv[:,i] ⊗ M_inv[:,i] * beta_i
    # where beta_i = Σ_j R[i,j] * resid_w[j]²
    #
    # Therefore:
    # C_R = Σ_i (1/N_i) * [ beta_i * (M_inv[:,i] ⊗ M_inv[:,i])
    #                        - alpha_i² * (M_inv[:,i] ⊗ M_inv[:,i]) ]
    #      = Σ_i (1/N_i) * (beta_i - alpha_i²) * outer(M_inv[:,i], M_inv[:,i])

    # Compute alpha_i and beta_i for each row i
    # resid_w[j] = (y_post_pred[j] - d[j]) / sigma[j]²
    alpha = resp.R @ resid_w                     # shape (n_pre,): alpha[i] = Σ_j R[i,j]*resid_w[j]
    beta  = resp.R @ (resid_w ** 2)              # shape (n_pre,): beta[i]  = Σ_j R[i,j]*resid_w[j]²

    contrib_norm = np.zeros(n_pre)
    for i in range(n_pre):
        Ni = resp.N_row[i]
        if not (np.isfinite(Ni) and Ni > 0):
            continue
        coeff = (beta[i] - alpha[i] ** 2) / Ni
        col_i = M_inv[:, i]                      # shape (n_pre,)
        C_R += coeff * np.outer(col_i, col_i)
        contrib_norm[i] = abs(coeff) * np.linalg.norm(col_i) ** 2

    # Symmetrise numerical noise
    C_R = 0.5 * (C_R + C_R.T)

    # Diagnostics
    diag_CR = np.diag(C_R)
    print(f"\n  C_R diagonal (sigma_R contribution to sigma_pre):")
    print(f"    min = {diag_CR.min():.4e}  max = {diag_CR.max():.4e}  "
          f"mean = {diag_CR.mean():.4e}")
    top5 = np.argsort(contrib_norm)[::-1][:5]
    print(f"  Top-5 rows by contribution magnitude (A_pre, N_i, |coeff|):")
    for ii in top5:
        print(f"    A_pre={resp.A_pre[ii]}  N_i={resp.N_row[ii]:.0f}  "
              f"|coeff|={abs((beta[ii]-alpha[ii]**2)/resp.N_row[ii]):.4e}  "
              f"contrib_norm={contrib_norm[ii]:.4e}")

    return C_R


def compute_resolution_metrics(R, A_pre, A_post):
    section("Response matrix resolution metrics")
    A_pre_g = A_pre[:, None]
    A_post_g = A_post[None, :]
    mean_shift = np.sum(R * (A_pre_g - A_post_g), axis=1)
    var_shift = np.sum(R * (A_pre_g - A_post_g - mean_shift[:, None]) ** 2, axis=1)
    std_shift = np.sqrt(np.clip(var_shift, 0, None))
    print(f"  Mean neutrons emitted: {mean_shift.mean():.4f} ± {mean_shift.std():.4f}")
    print(f"  Shift dispersion: mean={std_shift.mean():.4f}  min={std_shift.min():.4f}  max={std_shift.max():.4f}")
    print(f"  Effective mass resolution ~ {std_shift.mean():.1f} u")
    try:
        sv = np.linalg.svd(R, compute_uv=False)
        print(f"\n  Singular values of R (top 10): {sv[:10]}")
        print(f"  Condition number of R: {sv[0]/sv[-1]:.4e}")
        print(f"  Effective rank (sv > 1e-6 * sv_max): {np.sum(sv > 1e-6 * sv[0])}")
    except Exception as e:
        print(f"  SVD failed: {e}")


def print_solution_diagnostics(label, A_pre, y_pre, sigma_pre, A_post, d, sigma,
                                y_post_pred, used_mask, well_constrained, lam):
    section(f"Solution diagnostics — {label}")
    n_pre = len(A_pre)
    n_neg = int(np.sum(y_pre < 0))
    n_neg_sig = int(np.sum(y_pre < -sigma_pre))
    print(f"  lambda = {lam:.6e}")
    print(f"  sum(Y_pre) = {y_pre.sum():.6f}  min = {y_pre.min():.4e}  max = {y_pre.max():.4e}")
    print(f"  negative bins: {n_neg}/{n_pre}  (>{1:.0f}σ negative: {n_neg_sig})")
    resid_norm = (y_post_pred - d) / sigma
    chi2_wc = float(np.sum(resid_norm[well_constrained] ** 2))
    m_wc = int(well_constrained.sum())
    chi2_red_wc = chi2_wc / max(1, m_wc - n_pre)
    print(f"\n  chi2 (well-constrained, m={m_wc}): {chi2_wc:.4g}  chi2/dof: {chi2_red_wc:.4g}")
    if chi2_red_wc < 0.5:
        print(f"  NOTE: chi2/dof << 1 may indicate over-smoothing.")
    elif chi2_red_wc > 3.0:
        print(f"  NOTE: chi2/dof >> 1 suggests under-fitting.")


WELL_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"
PRE_COLOR = "#2ca02c"
WC_ALPHA = 0.8


def _plot_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_response_matrix(resp, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    R_log = np.log10(np.where(resp.R > 0, resp.R, np.nan))
    im = ax.imshow(R_log, origin="lower", aspect="auto",
                   extent=[resp.A_post.min() - 0.5, resp.A_post.max() + 0.5,
                           resp.A_pre.min() - 0.5, resp.A_pre.max() + 0.5],
                   cmap="viridis")
    diag_lim = [max(resp.A_pre.min(), resp.A_post.min()),
                min(resp.A_pre.max(), resp.A_post.max())]
    ax.plot(diag_lim, diag_lim, "w--", lw=0.8, alpha=0.6, label="A_pre=A_post")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log₁₀ P(A_post | A_pre)")
    ax.set_xlabel("A_post")
    ax.set_ylabel("A_pre")
    ax.set_title("Response matrix (log scale)")
    ax.legend(fontsize=8)
    ax2 = axes[1]
    A_pre_g = resp.A_pre[:, None]
    A_post_g = resp.A_post[None, :]
    mean_shift = np.sum(resp.R * (A_pre_g - A_post_g), axis=1)
    var_shift = np.sum(resp.R * (A_pre_g - A_post_g - mean_shift[:, None]) ** 2, axis=1)
    std_shift = np.sqrt(np.clip(var_shift, 0, None))
    ax2.fill_between(resp.A_pre, mean_shift - std_shift, mean_shift + std_shift,
                     alpha=0.3, color=WELL_COLOR, label="±1σ spread")
    ax2.plot(resp.A_pre, mean_shift, "-", color=WELL_COLOR, lw=1.5, label="Mean Δ")
    ax2.set_xlabel("A_pre")
    ax2.set_ylabel("Mean emitted neutrons")
    ax2.set_title("Mean neutron emission and spread vs A_pre")
    ax2.legend()
    _plot_save(fig, outpath)


def plot_covariance_structure(A_pre, Cov, corr, outpath, label=""):
    """
    Two-panel covariance diagnostics:
      Left:  Correlation matrix heatmap (full)
      Right: Nearest-neighbour correlation |ρ(i, i+k)| vs separation k
    """
    n = len(A_pre)
    fig, (ax_heat, ax_nn) = plt.subplots(1, 2, figsize=(14, 5))

    vmax = min(1.0, np.percentile(np.abs(corr[np.triu_indices(n, k=1)]), 99))
    im = ax_heat.imshow(corr, origin="lower", aspect="auto",
                        vmin=-vmax, vmax=vmax, cmap="RdBu_r",
                        extent=[A_pre.min() - 0.5, A_pre.max() + 0.5,
                                A_pre.min() - 0.5, A_pre.max() + 0.5])
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Correlation ρ")
    ax_heat.set_xlabel("A_pre")
    ax_heat.set_ylabel("A_pre")
    title_suffix = f" [{label}]" if label else ""
    ax_heat.set_title(f"Posterior correlation matrix of Y_pre{title_suffix}\n"
                      "(off-diagonal ≠ 0 → diagonal MCMC likelihood is WRONG)")

    max_sep = min(20, n - 1)
    separations = np.arange(1, max_sep + 1)
    mean_abs_corr = np.zeros(max_sep)
    max_abs_corr = np.zeros(max_sep)
    for k in separations:
        vals = np.array([abs(corr[i, i + k]) for i in range(n - k)])
        mean_abs_corr[k - 1] = vals.mean()
        max_abs_corr[k - 1] = vals.max()

    ax_nn.fill_between(separations, 0, max_abs_corr, alpha=0.2, color="steelblue", label="Max |ρ|")
    ax_nn.plot(separations, mean_abs_corr, "o-", color="steelblue", lw=1.5, label="Mean |ρ|")
    ax_nn.axhline(0.5, color="red", lw=1.2, ls="--", label="|ρ|=0.5 threshold")
    ax_nn.axhline(0.2, color="orange", lw=1.0, ls=":", label="|ρ|=0.2 threshold")
    ax_nn.set_xlabel("Mass separation |i − j| (u)")
    ax_nn.set_ylabel("|ρ(i, i+k)|")
    ax_nn.set_title(f"Off-diagonal correlations vs mass separation{title_suffix}\n"
                    "(bins with |ρ|>0.5 cannot be treated as independent in MCMC)")
    ax_nn.legend(fontsize=9)
    ax_nn.set_ylim(0, 1.05)

    _plot_save(fig, outpath)


def plot_cov_comparison(A_pre, sigma_data, sigma_total, outpath):
    """
    Compare diagonal sqrt(C_data) vs sqrt(C_total) to visualise R-uncertainty contribution.
    """
    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4))

    ax_abs.plot(A_pre, sigma_data, "-", color=WELL_COLOR, lw=1.5, label="σ_pre [data only]")
    ax_abs.plot(A_pre, sigma_total, "--", color=PRED_COLOR, lw=1.5, label="σ_pre [data + R unc.]")
    ax_abs.set_xlabel("A_pre")
    ax_abs.set_ylabel("σ(Y_pre)")
    ax_abs.set_title("Diagonal uncertainty: data-only vs total")
    ax_abs.legend(fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sigma_data > 0, sigma_total / sigma_data, np.nan)
    ax_ratio.plot(A_pre, ratio, "-", color="purple", lw=1.5)
    ax_ratio.axhline(1.0, color="k", lw=0.8, ls=":")
    ax_ratio.set_xlabel("A_pre")
    ax_ratio.set_ylabel("σ_total / σ_data")
    ax_ratio.set_title("Relative inflation from R uncertainty")

    _plot_save(fig, outpath)


def plot_lcurve(lambdas, rnorms, snorms, idx_corner, idx_gcv, idx_disc, gcv, outpath):
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax.loglog(rnorms, snorms, "-o", ms=2.5, lw=1, color="steelblue", zorder=1)
    markers = [
        (idx_corner, f"Corner λ={lambdas[idx_corner]:.2e}", "red", "D"),
        (idx_gcv, f"GCV λ={lambdas[idx_gcv]:.2e}", "green", "s"),
        (idx_disc, f"Discrepancy λ={lambdas[idx_disc]:.2e}", "orange", "^"),
    ]
    for idx, label, color, marker in markers:
        ax.plot(rnorms[idx], snorms[idx], marker=marker, ms=9, mfc="none",
                mew=2, color=color, label=label, zorder=3)
    ax.set_xlabel(r"Residual norm $\|W(Gy-d)\|_2$")
    ax.set_ylabel(r"Seminorm $\|Ly\|_2$")
    ax.set_title("L-curve")
    ax.legend(fontsize=9)
    ax2.semilogx(lambdas, gcv, "-", color="forestgreen", lw=1.5)
    ax2.axvline(lambdas[idx_gcv], color="green", lw=1.5, ls="--",
                label=f"GCV min λ={lambdas[idx_gcv]:.2e}")
    ax2.axvline(lambdas[idx_corner], color="red", lw=1.5, ls=":",
                label=f"Corner λ={lambdas[idx_corner]:.2e}")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("GCV score")
    ax2.set_title("Generalised Cross-Validation score")
    ax2.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_curvature(lambdas, curvature, idx_corner, scan, outpath):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax.semilogx(lambdas, curvature, "-k", lw=1.2)
    ax.axvline(lambdas[idx_corner], color="r", lw=1.5, ls="--",
               label=f"Corner λ={lambdas[idx_corner]:.2e}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Curvature (log-log L-curve)")
    ax.set_title("L-curve curvature vs λ")
    ax.legend()
    ax2.semilogx(lambdas, scan.chi2_wc, "-", color="steelblue", lw=1.5, label="χ² (well-constrained)")
    ax2.semilogx(lambdas, scan.chi2_used, "--", color="steelblue", lw=1, alpha=0.6, label="χ² (all used)")
    ax2.axvline(lambdas[idx_corner], color="r", lw=1.5, ls="--", label=f"Final λ={lambdas[idx_corner]:.2e}")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("χ²")
    ax2.set_title("Fit quality vs λ")
    ax2.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_negative_fraction(lambdas, neg_frac, idx_final, outpath):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(lambdas, neg_frac * 100, "-", color="purple", lw=1.5)
    ax.axvline(lambdas[idx_final], color="r", lw=1.5, ls="--",
               label=f"Selected λ={lambdas[idx_final]:.2e}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Negative Y_pre bins (%)")
    ax.set_title("Fraction of negative pre-yield bins vs λ")
    ax.legend()
    _plot_save(fig, outpath)


def plot_post_comparison(A_post, d, sigma, y_post_pred, well_constrained, used_mask, outpath):
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(15, 5))
    wc_d = d[well_constrained]
    wc_pred = y_post_pred[well_constrained]
    A_wc = A_post[well_constrained]
    x_lo = A_wc.min() - 2 if wc_d.size else A_post.min()
    x_hi = A_wc.max() + 2 if wc_d.size else A_post.max()
    y_lo_log = max(wc_d.min() * 0.3, 1e-5) if wc_d.size else 1e-5
    y_hi_log = wc_d.max() * 3 if wc_d.size else 1.0
    for ax, do_log in [(ax_log, True), (ax_lin, False)]:
        ax.errorbar(A_post[used_mask], d[used_mask], yerr=sigma[used_mask],
                    fmt="o", ms=2.5, capsize=1.5, lw=0.6, color=WELL_COLOR, alpha=0.35, zorder=1)
        ax.errorbar(A_wc, wc_d, yerr=sigma[well_constrained],
                    fmt="o", ms=4, capsize=2.5, lw=0.8, color=WELL_COLOR, alpha=WC_ALPHA, zorder=2,
                    label="Measured Y_post (well-constrained)")
        ax.plot(A_post, y_post_pred, "-", color=PRED_COLOR, lw=1.8, zorder=3,
                label="Predicted Y_post (from inferred Y_pre)")
        ax.set_xlabel("Mass number A_post")
        ax.set_ylabel("Y_post(A)")
        ax.set_xlim(x_lo, x_hi)
        if do_log:
            ax.set_yscale("log")
            ax.set_ylim(y_lo_log, y_hi_log)
            ax.set_title("Log scale: post-neutron yield — measured vs predicted")
        else:
            y_hi_lin = wc_pred.max() * 1.15 if wc_pred.size else 0.1
            ax.set_ylim(-0.005, y_hi_lin)
            ax.set_title("Linear scale: post-neutron yield — measured vs predicted")
        ax.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_pre_vs_post_overlay(A_pre, y_pre, sigma_pre, A_post, d, sigma, well_constrained, outpath):
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(15, 5))
    wc_d = d[well_constrained]
    A_wc = A_post[well_constrained]
    x_lo = min(A_pre.min(), A_wc.min() if wc_d.size else A_post.min()) - 2
    x_hi = max(A_pre.max(), A_wc.max() if wc_d.size else A_post.max()) + 2
    for ax, do_log in [(ax_log, True), (ax_lin, False)]:
        ax.errorbar(A_wc, wc_d, yerr=sigma[well_constrained],
                    fmt="o", ms=4, capsize=2.5, lw=0.8, color=WELL_COLOR, alpha=WC_ALPHA,
                    label="Measured Y_post (well-constrained)")
        ax.errorbar(A_pre, y_pre, yerr=sigma_pre,
                    fmt="s", ms=4, capsize=2.5, lw=0.8, color=PRE_COLOR, alpha=WC_ALPHA,
                    label="Inferred Y_pre")
        ax.set_xlabel("Mass number A")
        ax.set_ylabel("Yield Y(A)")
        ax.set_xlim(x_lo, x_hi)
        if do_log:
            all_pos = np.concatenate([wc_d[wc_d > 0], y_pre[y_pre > 0]])
            y_lo = max(all_pos.min() * 0.3, 1e-5) if all_pos.size else 1e-5
            y_hi = all_pos.max() * 3 if all_pos.size else 1.0
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.set_title("Log scale: pre vs post neutron mass yields")
        else:
            y_hi_lin = max(wc_d.max() if wc_d.size else 0, y_pre.max()) * 1.15
            ax.set_ylim(-0.005, y_hi_lin)
            ax.set_title("Linear scale: pre vs post neutron mass yields")
        ax.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_residuals(A_post, d, sigma, y_post_pred, used_mask, well_constrained, outpath):
    r = (y_post_pred - d) / sigma
    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(13, 4))
    ax.axhline(0, color="k", lw=1)
    for v in [+1, -1, +2, -2]:
        ax.axhline(v, color="gray", lw=0.7 if abs(v) == 1 else 0.5, ls=":")
    ax.plot(A_post[used_mask & ~well_constrained], r[used_mask & ~well_constrained],
            "o", ms=3, color="gray", alpha=0.4, label="Used (low-quality)")
    ax.plot(A_post[well_constrained], r[well_constrained],
            "o", ms=4, color=WELL_COLOR, alpha=WC_ALPHA, label="Well-constrained")
    wc_r = r[well_constrained]
    if wc_r.size:
        ax.set_ylim(min(-3, wc_r.min() * 1.3), max(+3, wc_r.max() * 1.3))
    ax.set_xlabel("A_post")
    ax.set_ylabel("(Y_pred − Y_data) / σ")
    ax.set_title("Normalised residuals")
    ax.legend(fontsize=9)
    if wc_r.size:
        bins = np.linspace(max(-5, wc_r.min()), min(5, wc_r.max()), 25)
        ax_hist.hist(wc_r, bins=bins, color=WELL_COLOR, alpha=0.7, density=True)
        xg = np.linspace(bins[0], bins[-1], 200)
        ax_hist.plot(xg, np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi), "k-", lw=1.5, label="N(0,1)")
        ax_hist.set_xlabel("Normalised residual")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Residual distribution (well-constrained)")
        ax_hist.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_pre_yield(A_pre, y_pre, sigma_pre, outpath):
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(15, 5))
    pos_mask = y_pre > 0
    for ax, do_log in [(ax_log, True), (ax_lin, False)]:
        ax.fill_between(A_pre,
                        np.clip(y_pre - sigma_pre, 1e-8 if do_log else -np.inf, np.inf),
                        y_pre + sigma_pre,
                        alpha=0.25, color=PRE_COLOR, label="±1σ uncertainty band")
        ax.errorbar(A_pre, y_pre, yerr=sigma_pre, fmt="s-", ms=3, capsize=2, lw=0.8,
                    color=PRE_COLOR, alpha=WC_ALPHA, label="Inferred Y_pre")
        ax.set_xlabel("A_pre")
        ax.set_ylabel("Y_pre(A)")
        if do_log:
            pos_vals = y_pre[pos_mask]
            if pos_vals.size:
                ax.set_ylim(max(pos_vals.min() * 0.3, 1e-5), pos_vals.max() * 3)
            ax.set_yscale("log")
            ax.set_title("Log scale: inferred pre-neutron mass yields")
        else:
            ax.set_ylim(-0.005, (y_pre + sigma_pre).max() * 1.1)
            ax.set_title("Linear scale: inferred pre-neutron mass yields")
        ax.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_summary_panel(A_pre, y_pre, sigma_pre, A_post, d, sigma, y_post_pred,
                       well_constrained, used_mask, lambdas, scan, idx_final,
                       curvature, idx_corner, outpath):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_fit = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_lc = fig.add_subplot(gs[1, 1])
    wc_d = d[well_constrained]
    A_wc = A_post[well_constrained]
    ax_fit.errorbar(A_wc, wc_d, yerr=sigma[well_constrained],
                    fmt="o", ms=3.5, capsize=2, lw=0.7, color=WELL_COLOR, alpha=WC_ALPHA, label="Measured Y_post")
    ax_fit.plot(A_post, y_post_pred, "-", color=PRED_COLOR, lw=1.8, label="Predicted Y_post")
    ax_fit.set_yscale("log")
    all_pos_d = wc_d[wc_d > 0]
    if all_pos_d.size:
        ax_fit.set_ylim(all_pos_d.min() * 0.3, all_pos_d.max() * 3)
    ax_fit.set_xlabel("A_post")
    ax_fit.set_ylabel("Y_post")
    ax_fit.set_title("Post-neutron yields: measured vs predicted")
    ax_fit.legend(fontsize=9)
    ax_overlay.errorbar(A_wc, wc_d, yerr=sigma[well_constrained],
                        fmt="o", ms=3.5, capsize=2, lw=0.7, color=WELL_COLOR, alpha=WC_ALPHA, label="Measured Y_post")
    ax_overlay.errorbar(A_pre, y_pre, yerr=sigma_pre,
                        fmt="s", ms=3.5, capsize=2, lw=0.7, color=PRE_COLOR, alpha=WC_ALPHA, label="Inferred Y_pre")
    ax_overlay.set_yscale("log")
    all_pos = np.concatenate([wc_d[wc_d > 0], y_pre[y_pre > 0]])
    if all_pos.size:
        ax_overlay.set_ylim(all_pos.min() * 0.3, all_pos.max() * 3)
    ax_overlay.set_xlabel("Mass number A")
    ax_overlay.set_ylabel("Y(A)")
    ax_overlay.set_title("Pre (inferred) vs Post (measured) yields")
    ax_overlay.legend(fontsize=9)
    r = (y_post_pred - d) / sigma
    ax_resid.axhline(0, color="k", lw=1)
    ax_resid.axhline(+1, color="gray", lw=0.7, ls=":")
    ax_resid.axhline(-1, color="gray", lw=0.7, ls=":")
    ax_resid.plot(A_post[well_constrained], r[well_constrained],
                  "o", ms=3.5, color=WELL_COLOR, alpha=WC_ALPHA, label="Well-constrained")
    ax_resid.plot(A_post[used_mask & ~well_constrained], r[used_mask & ~well_constrained],
                  "o", ms=2.5, color="gray", alpha=0.35, label="Used (low-quality)")
    wc_r = r[well_constrained]
    if wc_r.size:
        ax_resid.set_ylim(min(-3, wc_r.min() * 1.3), max(+3, wc_r.max() * 1.3))
    ax_resid.set_xlabel("A_post")
    ax_resid.set_ylabel("(Y_pred − Y_data) / σ")
    ax_resid.set_title("Normalised residuals")
    ax_resid.legend(fontsize=9)
    ax_lc.loglog(scan.residual_norms, scan.seminorms, "-o", ms=2, lw=1, color="steelblue")
    ax_lc.plot(scan.residual_norms[idx_corner], scan.seminorms[idx_corner],
               "D", ms=9, mfc="none", mew=2, color="red", label=f"Corner λ={lambdas[idx_corner]:.2e}")
    ax_lc.plot(scan.residual_norms[idx_final], scan.seminorms[idx_final],
               "o", ms=9, mfc="none", mew=2, color="orange", label=f"Final λ={lambdas[idx_final]:.2e}")
    ax_lc.set_xlabel(r"$\|W(Gy-d)\|_2$")
    ax_lc.set_ylabel(r"$\|Ly\|_2$")
    ax_lc.set_title("L-curve")
    ax_lc.legend(fontsize=9)
    fig.suptitle("Step 2 Inversion Summary", fontsize=14, fontweight="bold", y=1.01)
    _plot_save(fig, outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post_npz", default="step1_output.npz")
    ap.add_argument("--response_json", required=True)
    ap.add_argument("--outdir", default="step2_out")
    ap.add_argument("--n_lambda", type=int, default=120)
    ap.add_argument("--lambda_min", type=float, default=None)
    ap.add_argument("--lambda_max", type=float, default=None)
    ap.add_argument("--nonnegative", action="store_true")
    ap.add_argument("--renormalize_post_to_2", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    post = load_post_yields_npz(args.post_npz)
    resp = load_response_json(args.response_json)

    compute_resolution_metrics(resp.R, resp.A_pre, resp.A_post)
    plot_response_matrix(resp, os.path.join(args.outdir, "response_matrix.png"))

    aligned = align_evaluated_to_response_axis(post, resp.A_post, missing_sigma=1e6)

    d = aligned.d.copy()
    sigma = aligned.sigma.copy()

    if args.renormalize_post_to_2:
        s = d.sum()
        if s > 0:
            factor = 2.0 / s
            d *= factor
            sigma *= factor
            section("Renormalization applied to aligned post yields")
            print(f"  Original sum(d)={s:.6f}  → New sum(d)={d.sum():.6f}")
        else:
            print("WARNING: sum(d)=0, cannot renormalize.")

    G = resp.R.T
    n_pre = resp.A_pre.size
    L = second_difference_matrix(n_pre)

    w = 1.0 / sigma
    Aw = (G.T * w).T
    AtA = Aw.T @ Aw
    LtL = L.T @ L

    scale = np.sqrt((np.linalg.norm(AtA, ord=2) + 1e-300) /
                    (np.linalg.norm(LtL, ord=2) + 1e-300))

    lam_min = float(args.lambda_min) if args.lambda_min is not None else 1e-6 * scale
    lam_max = float(args.lambda_max) if args.lambda_max is not None else 1e2 * scale

    if lam_min <= 0 or lam_max <= lam_min:
        raise ValueError(f"Invalid lambda range: [{lam_min:.3e}, {lam_max:.3e}]")

    lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), args.n_lambda)

    section("Lambda scan configuration")
    print(f"  n_lambda={len(lambdas)}  lambda_min={lambdas.min():.6e}  lambda_max={lambdas.max():.6e}")
    print(f"  scale heuristic = {scale:.6e}")

    scan = solve_tikhonov_scan(resp.R, d, sigma, aligned.used_mask, aligned.well_constrained, L, lambdas)

    idx_corner, curvature = lcurve_corner_max_curvature(
        scan.lambdas, scan.residual_norms, scan.seminorms, exclude_frac=0.10, require_monotonic=True)
    idx_gcv, gcv = choose_lambda_gcv(Aw, L, d, sigma, scan.lambdas)
    idx_disc, target = choose_lambda_discrepancy(scan.lambdas, scan.residual_norms, np.sum(aligned.used_mask))

    lam_corner = scan.lambdas[idx_corner]
    lam_gcv = scan.lambdas[idx_gcv]
    lam_disc = scan.lambdas[idx_disc]
    log_dist_gcv = abs(np.log10(lam_corner) - np.log10(lam_gcv))

    section("Lambda selection summary")
    print(f"  Corner:      idx={idx_corner:4d}  lambda={lam_corner:.6e}")
    print(f"  GCV:         idx={idx_gcv:4d}  lambda={lam_gcv:.6e}")
    print(f"  Discrepancy: idx={idx_disc:4d}  lambda={lam_disc:.6e}  (target rnorm~{target:.3g})")
    print(f"\n  |log10(λ_corner) - log10(λ_gcv)| = {log_dist_gcv:.3f} decades")

    idx_final = idx_corner
    if log_dist_gcv > 1.5:
        print(f"  NOTE: Disagreement >1.5 decades; preferring GCV over corner.")
        idx_final = idx_gcv
    else:
        print(f"  Corner and GCV agree to <1.5 decades; using corner estimate.")

    lam_final = scan.lambdas[idx_final]

    y_pre_unconstrained = scan.solutions[idx_final].copy()

    # -----------------------------------------------------------------------
    # Data covariance C_data (sandwich posterior from data uncertainty only)
    # -----------------------------------------------------------------------
    jitter = 1e-18 * np.trace(AtA) / max(1, n_pre)
    M = AtA + (lam_final ** 2) * LtL + jitter * np.eye(n_pre)
    Minv = linalg.inv(M)
    Cov_data = Minv @ AtA @ Minv
    sigma_data = np.sqrt(np.clip(np.diag(Cov_data), 0, np.inf))

    # -----------------------------------------------------------------------
    # Response matrix covariance C_R (first-order multinomial propagation)
    # -----------------------------------------------------------------------
    y_pre_for_CR = y_pre_unconstrained  # use unconstrained point estimate
    C_R = compute_response_covariance(resp, y_pre_for_CR, Minv, d, sigma)

    # -----------------------------------------------------------------------
    # Total covariance: C_total = C_data + C_R  (independent sources)
    # -----------------------------------------------------------------------
    section("Total covariance assembly: C_total = C_data + C_R")
    Cov_total = Cov_data + C_R
    Cov_total = 0.5 * (Cov_total + Cov_total.T)

    sigma_total = np.sqrt(np.clip(np.diag(Cov_total), 0, np.inf))

    # Quantify relative contribution of C_R
    diag_CR = np.diag(C_R)
    diag_Cd = np.diag(Cov_data)
    frac_CR = np.where(diag_Cd > 0, diag_CR / (diag_Cd + diag_CR + 1e-300), 0.0)
    print(f"  C_R fraction of total variance:")
    print(f"    mean = {frac_CR.mean() * 100:.2f}%  "
          f"max = {frac_CR.max() * 100:.2f}%  "
          f"min = {frac_CR.min() * 100:.2f}%")
    top3_CR = np.argsort(frac_CR)[::-1][:3]
    print(f"  Top-3 A_pre bins by C_R fraction:")
    for ii in top3_CR:
        print(f"    A_pre={resp.A_pre[ii]}  C_R/(C_R+C_d) = {frac_CR[ii]*100:.1f}%  "
              f"σ_data={sigma_data[ii]:.4e}  σ_total={sigma_total[ii]:.4e}")

    # Use total covariance for downstream MCMC outputs
    Cov_sym = Cov_total

    diag_std = np.sqrt(np.diag(Cov_sym))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = Cov_sym / np.outer(diag_std, diag_std)
    np.fill_diagonal(corr, 1.0)

    section("Full covariance matrix (total) correlation diagnostics")

    off_diag = corr[np.triu_indices(n_pre, k=1)]
    nn1 = np.array([corr[i, i + 1] for i in range(n_pre - 1)])
    nn2 = np.array([corr[i, i + 2] for i in range(n_pre - 2)])
    nn3 = np.array([corr[i, i + 3] for i in range(n_pre - 3)])

    print(f"  Covariance matrix shape: {Cov_sym.shape}")
    print(f"\n  Correlation structure — crucial for correct MCMC likelihood:")
    print(f"    Nearest-neighbour (|Δi|=1)  mean|ρ| = {np.mean(np.abs(nn1)):.4f}  "
          f"max|ρ| = {np.max(np.abs(nn1)):.4f}")
    print(f"    Next-nearest     (|Δi|=2)  mean|ρ| = {np.mean(np.abs(nn2)):.4f}  "
          f"max|ρ| = {np.max(np.abs(nn2)):.4f}")
    print(f"    3rd-nearest      (|Δi|=3)  mean|ρ| = {np.mean(np.abs(nn3)):.4f}  "
          f"max|ρ| = {np.max(np.abs(nn3)):.4f}")
    print(f"    All off-diagonal            mean|ρ| = {np.mean(np.abs(off_diag)):.4f}  "
          f"max|ρ| = {np.max(np.abs(off_diag)):.4f}")

    pct_above_50 = 100.0 * np.mean(np.abs(nn1) > 0.5)
    pct_above_80 = 100.0 * np.mean(np.abs(nn1) > 0.8)
    print(f"\n    Nearest-neighbour bins with |ρ| > 0.5: {pct_above_50:.1f}%")
    print(f"    Nearest-neighbour bins with |ρ| > 0.8: {pct_above_80:.1f}%")

    if pct_above_50 > 20:
        print(f"\n  *** WARNING: >20% of adjacent bins are strongly correlated (|ρ|>0.5). ***")
        print(f"  *** Diagonal-only σ_pre in MCMC will INFLATE effective sample size   ***")
        print(f"  *** and produce overconfident posteriors on peak position and width.  ***")
        print(f"  *** Load Cov_inv_chol from the NPZ and use the quadratic likelihood. ***")

    try:
        Cov_chol = linalg.cholesky(Cov_sym, lower=True, check_finite=True)
        print(f"\n  Cholesky(Cov_total) succeeded. cond(L) = {np.linalg.cond(Cov_chol):.4e}")
    except linalg.LinAlgError:
        print("  WARNING: Cov_total not positive-definite; adding jitter.")
        eps_jit = 1e-10 * np.abs(np.diag(Cov_sym)).max()
        Cov_sym = Cov_sym + eps_jit * np.eye(n_pre)
        Cov_chol = linalg.cholesky(Cov_sym, lower=True, check_finite=True)
        print(f"  Cholesky(Cov_total + {eps_jit:.2e} I) succeeded.")

    Cov_inv = linalg.cho_solve((Cov_chol, True), np.eye(n_pre))
    Cov_inv_sym = 0.5 * (Cov_inv + Cov_inv.T)

    try:
        Cov_inv_chol = linalg.cholesky(Cov_inv_sym, lower=True, check_finite=True)
        print(f"  Cholesky(Cov_inv_total) succeeded. cond(L_inv) = {np.linalg.cond(Cov_inv_chol):.4e}")
    except linalg.LinAlgError:
        eps_jit2 = 1e-10 * np.abs(np.diag(Cov_inv_sym)).max()
        Cov_inv_chol = linalg.cholesky(
            Cov_inv_sym + eps_jit2 * np.eye(n_pre), lower=True, check_finite=True)
        print(f"  WARNING: jitter needed for Cholesky(Cov_inv_total).")

    residual_id = np.linalg.norm(Cov_sym @ Cov_inv_sym - np.eye(n_pre), ord="fro")
    print(f"\n  Identity residual ||Cov_total @ Cov_inv_total - I||_F = {residual_id:.4e}")
    if residual_id > 1e-4:
        print("  WARNING: Large identity residual — covariance inversion inaccurate.")
        print("  Consider larger lambda to improve conditioning.")

    print(f"\n  MCMC fast log-likelihood (copy-paste ready):")
    print(f"    import scipy.linalg")
    print(f"    # Load from NPZ: Cov_inv_chol = npz['Cov_inv_chol']  (includes C_data + C_R)")
    print(f"    residual = y_model_theta - y_pre          # shape (n_pre,)")
    print(f"    alpha    = scipy.linalg.solve_triangular(")
    print(f"                   Cov_inv_chol, residual, lower=True)")
    print(f"    ln_L     = -0.5 * np.dot(alpha, alpha)")

    # ----------- Optional nonneg refinement ------------------------------
    y_pre = y_pre_unconstrained.copy()
    nn_info = None
    if args.nonnegative:
        section("Nonnegative refinement")
        y_nn, res_nn = solve_nonnegative_tikhonov(G, d, sigma, L, lam_final, bounds=(0.0, np.inf))
        y_pre = y_nn
        nn_info = {
            "success": bool(res_nn.success),
            "status": int(res_nn.status),
            "message": str(res_nn.message),
            "cost": float(res_nn.cost),
            "optimality": float(res_nn.optimality),
        }
        for k, v in nn_info.items():
            print(f"  {k}: {v}")
        print(f"  sum(Y_pre_nonneg) = {y_pre.sum():.6f}  min = {y_pre.min():.6e}")

    # sigma_pre for display uses the total covariance diagonal
    sigma_pre = sigma_total

    y_post_pred = y_pre @ resp.R

    print_solution_diagnostics(
        label="Final solution" + (" [nonneg]" if args.nonnegative else " [unconstrained]"),
        A_pre=resp.A_pre, y_pre=y_pre, sigma_pre=sigma_pre,
        A_post=resp.A_post, d=d, sigma=sigma, y_post_pred=y_post_pred,
        used_mask=aligned.used_mask, well_constrained=aligned.well_constrained, lam=lam_final)

    # ----------- Plots ---------------------------------------------------
    section("Saving plots")

    plot_lcurve(scan.lambdas, scan.residual_norms, scan.seminorms,
                idx_corner, idx_gcv, idx_disc, gcv,
                os.path.join(args.outdir, "lcurve.png"))

    plot_curvature(scan.lambdas, curvature, idx_corner, scan,
                   os.path.join(args.outdir, "lcurve_curvature.png"))

    plot_negative_fraction(scan.lambdas, scan.neg_frac, idx_final,
                           os.path.join(args.outdir, "negative_fraction.png"))

    plot_post_comparison(resp.A_post, d, sigma, y_post_pred,
                         aligned.well_constrained, aligned.used_mask,
                         os.path.join(args.outdir, "post_fit.png"))

    plot_pre_vs_post_overlay(resp.A_pre, y_pre, sigma_pre, resp.A_post, d, sigma,
                             aligned.well_constrained,
                             os.path.join(args.outdir, "pre_vs_post_overlay.png"))

    plot_residuals(resp.A_post, d, sigma, y_post_pred,
                   aligned.used_mask, aligned.well_constrained,
                   os.path.join(args.outdir, "post_residuals.png"))

    plot_pre_yield(resp.A_pre, y_pre, sigma_pre,
                   os.path.join(args.outdir, "pre_yield.png"))

    plot_covariance_structure(resp.A_pre, Cov_sym, corr,
                              os.path.join(args.outdir, "covariance_structure.png"),
                              label="C_data + C_R")

    # Also plot data-only correlation for comparison
    diag_std_d = np.sqrt(np.diag(Cov_data))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_data = Cov_data / np.outer(diag_std_d + 1e-300, diag_std_d + 1e-300)
    np.fill_diagonal(corr_data, 1.0)
    plot_covariance_structure(resp.A_pre, Cov_data, corr_data,
                              os.path.join(args.outdir, "covariance_structure_data_only.png"),
                              label="C_data only")

    plot_cov_comparison(resp.A_pre, sigma_data, sigma_total,
                        os.path.join(args.outdir, "covariance_R_contribution.png"))

    plot_summary_panel(
        resp.A_pre, y_pre, sigma_pre,
        resp.A_post, d, sigma, y_post_pred,
        aligned.well_constrained, aligned.used_mask,
        scan.lambdas, scan, idx_final, curvature, idx_corner,
        os.path.join(args.outdir, "summary_panel.png"))

    # ----------- Save NPZ ------------------------------------------------
    out_npz = os.path.join(args.outdir, "step2_pre_yields.npz")
    np.savez(
        out_npz,
        # --- Primary solution ---
        A_pre=resp.A_pre,
        Y_pre=y_pre,
        sigma_pre=sigma_total,                        # total (data + R) diagonal uncertainty
        sigma_pre_data_only=sigma_data,               # data-only diagonal uncertainty
        # --- Full covariance / precision matrices (total: C_data + C_R) ---
        # Cov:          (n_pre, n_pre) total posterior covariance  C_data + C_R
        # Cov_chol:     lower Cholesky of Cov  (L s.t.  L @ L.T = Cov)
        # Cov_inv:      precision matrix (= Cov^{-1})
        # Cov_inv_chol: lower Cholesky of Cov_inv — USE THIS in MCMC:
        #               alpha = solve_triangular(Cov_inv_chol, residual, lower=True)
        #               ln_L  = -0.5 * dot(alpha, alpha)
        Cov=Cov_sym,
        Cov_chol=Cov_chol,
        Cov_inv=Cov_inv_sym,
        Cov_inv_chol=Cov_inv_chol,
        corr=corr,
        # --- Individual covariance components ---
        Cov_data=Cov_data,                            # data-only component
        Cov_R=C_R,                                    # response matrix uncertainty component
        frac_variance_from_R=frac_CR,                 # per-bin fraction of variance from C_R
        # --- Lambda scan ---
        lambda_grid=scan.lambdas,
        residual_norms=scan.residual_norms,
        seminorms=scan.seminorms,
        curvature=curvature,
        gcv=gcv,
        chi2_used=scan.chi2_used,
        chi2_wc=scan.chi2_wc,
        neg_frac=scan.neg_frac,
        lambda_final=lam_final,
        idx_final=idx_final,
        idx_corner=idx_corner,
        idx_gcv=idx_gcv,
        idx_disc=idx_disc,
        # --- Data ---
        A_post=resp.A_post,
        Y_post_aligned=d,
        sigma_post_aligned=sigma,
        Y_post_pred=y_post_pred,
        nonnegative_used=args.nonnegative,
        nonnegative_info=json.dumps(nn_info) if nn_info is not None else "",
    )

    section("STEP 2 COMPLETE")
    print(f"  Output directory: {args.outdir}")
    print(f"  NPZ: {out_npz}")
    print(f"\n  Covariance outputs in NPZ:")
    print(f"    Cov               — ({n_pre}×{n_pre}) total covariance  C_data + C_R")
    print(f"    Cov_data          — ({n_pre}×{n_pre}) data-only covariance")
    print(f"    Cov_R             — ({n_pre}×{n_pre}) response matrix uncertainty covariance")
    print(f"    frac_variance_from_R — per-bin fraction of variance from C_R")
    print(f"    Cov_chol          — lower Cholesky of Cov_total")
    print(f"    Cov_inv           — precision matrix (Cov_total^{{-1}})")
    print(f"    Cov_inv_chol      — lower Cholesky of Cov_inv  ← USE IN MCMC LIKELIHOOD")
    print(f"    corr              — total correlation matrix")
    print(f"    sigma_pre         — sqrt(diag(C_data + C_R))  ← use for error bars")
    print(f"    sigma_pre_data_only — sqrt(diag(C_data)) for comparison")
    print(f"\n  New plots:")
    print(f"    covariance_R_contribution.png    — σ inflation from R uncertainty per bin.")
    print(f"    covariance_structure_data_only.png — correlation heatmap (C_data only).")
    print(f"\n  Interpretation guide:")
    print(f"    summary_panel.png        — Start here. Four-panel overview.")
    print(f"    covariance_structure.png — Check bin coupling (total) before MCMC.")
    print(f"    covariance_R_contribution.png — Where R uncertainty matters most.")
    print(f"    pre_vs_post_overlay.png  — Pre and post yields on same axes.")
    print(f"    post_fit.png             — Model vs data (log+linear).")
    print(f"    post_residuals.png       — Normalised residuals + histogram.")
    print(f"    pre_yield.png            — Inferred Y_pre with total ±1σ bands.")
    print(f"    lcurve.png               — L-curve + GCV for lambda selection.")
    print(f"    negative_fraction.png    — Physicality guide.")
    print(f"    response_matrix.png      — R heatmap and mean neutron emission.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
