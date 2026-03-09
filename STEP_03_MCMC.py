#!/usr/bin/env python3
"""
STEP 3 — MCMC inference of the 3-Gaussian CGMF pre-neutron mass yield parameters.

Model
-----
Y(A; θ, E) is a sum of 5 Gaussians (3 shapes, two mirrored across A0/2):

  G(A, w, mu, sig) = w / sqrt(2π sig²) * exp[-(A-mu)²/(2 sig²)]

  Y(A) = G(A; w1, mu1,    sig1)   ← heavy fragment peak 1
       + G(A; w1, A0-mu1, sig1)   ← mirror of peak 1 (light fragment)
       + G(A; w2, mu2,    sig2)   ← heavy fragment peak 2
       + G(A; w2, A0-mu2, sig2)   ← mirror of peak 2
       + G(A; w3, A0/2,   sig3)   ← symmetric valley peak

  Total integrated yield = 2w1 + 2w2 + w3 = 2  (standard fission convention)

Energy-dependent parameters:
  w_i(E)   = 1 / (1 + exp[(E − w_a_i) / w_b_i])   ← Fermi function
  mu_i(E)  = mu_a_i + mu_b_i * E
  sig_i(E) = sig_a_i + sig_b_i * E

CRITICAL: w_b CAN be negative (CGMF nominal w_b2 = −6.14).
  A negative w_b inverts the logistic (weight *decreases* with E).
  The old version had bounds [0.05, 15] for w_b2 — this completely
  excluded the physical solution and caused walkers to be stuck at
  an unphysical single-Gaussian centred on the valley.

Constraints (enforced analytically):
  w3(E)  = 2 − 2·w1(E) − 2·w2(E)    → requires w1+w2 < 1
  mu3(E) = A0 / 2                     → fixed by symmetry
  sig3(E) = sig_a3 + sig_b3 * E      → 2 free params

Total free parameters: 14
  θ = [w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
       w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
       sig_a3, sig_b3]

CGMF Nominal (U235+nth → A0=236):
  θ = [-6.856, 6.082, 133.79, -0.28, 3.029, 0.0,
       -6.864, -6.144, 140.97, -0.27, 4.694, 0.185,
        9.885, 0.032]
  At thermal En≈0:  w1≈0.245, w2≈0.753, w3≈0.004
  Heavy peaks at A≈133.8, 141.0  (light mirrors: A≈102.2, 95.0)

Usage
-----
  python step3_mcmc_3gauss.py \\
      --npz step2_output/step2_pre_yields.npz \\
      --A0 236 --En 2.53e-8 \\
      --outdir step3_out \\
      --nwalkers 96 --nsteps 8000 --burnin 3000 --thin 10 \\
      [--peak_weight 0.0]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import emcee
except ImportError:
    sys.exit("emcee not found.  Install:  pip install emcee")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("  [warn] corner not installed — corner plot skipped.  pip install corner")

plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})

SEPARATOR = "=" * 80
C_DATA = "#1f77b4"; C_MODEL = "#d62728"; C_SAMPLE = "#2ca02c"; C_PRIOR = "#ff7f0e"

# ── CGMF nominal parameters (U235+n → A0=236) ──────────────────────────────
# These serve as prior centres. The two primary peaks are at A≈133.8 and
# A≈141.0 in the heavy fragment region; their mirrors are the light fragment
# peaks at A≈102.2 and A≈95.0 respectively.
THETA_NOMINAL = np.array([
    -6.8560,   # [0]  w_a_1
     6.0824,   # [1]  w_b_1    (positive: weight increases with E)
   133.79,     # [2]  mu_a_1   (heavy fragment, lower of the two peaks)
    -0.28,     # [3]  mu_b_1
     3.0288,   # [4]  sigma_a_1
     0.0,      # [5]  sigma_b_1
    -6.8637,   # [6]  w_a_2
    -6.1438,   # [7]  w_b_2    *** NEGATIVE: inverted logistic ***
   140.97,     # [8]  mu_a_2   (heavy fragment, upper peak)
    -0.27,     # [9]  mu_b_2
     4.6942,   # [10] sigma_a_2
     0.1853,   # [11] sigma_b_2
     9.8854,   # [12] sigma_a_3 (symmetric valley — broad)
     0.0322,   # [13] sigma_b_3
])

PARAM_NAMES = [
    r"$w_{a,1}$", r"$w_{b,1}$",
    r"$\mu_{a,1}$", r"$\mu_{b,1}$",
    r"$\sigma_{a,1}$", r"$\sigma_{b,1}$",
    r"$w_{a,2}$", r"$w_{b,2}$",
    r"$\mu_{a,2}$", r"$\mu_{b,2}$",
    r"$\sigma_{a,2}$", r"$\sigma_{b,2}$",
    r"$\sigma_{a,3}$", r"$\sigma_{b,3}$",
]
PARAM_LABELS = [
    "w_a1", "w_b1", "mu_a1", "mu_b1", "sig_a1", "sig_b1",
    "w_a2", "w_b2", "mu_a2", "mu_b2", "sig_a2", "sig_b2",
    "sig_a3", "sig_b3",
]
N_PARAMS = 14


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def section(title):
    print(); print(SEPARATOR); print(f"  {title}"); print(SEPARATOR)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _plot_save(fig, path):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

def _gauss(A, w, mu, sig):
    """Normalised Gaussian: integrates to w over all A."""
    return w / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-0.5 * ((A - mu) / sig) ** 2)


def model_yield(theta, A, En, A0):
    """
    Evaluate 5-Gaussian pre-neutron mass yield Y(A; θ, En).

    Parameters
    ----------
    theta : array (14,)
    A     : array (n,)  integer mass numbers
    En    : float       incident neutron energy [MeV]
    A0    : int         compound nucleus mass

    Returns
    -------
    Y : array (n,)  with sum(Y) ≈ 2 when integrated over all A.
    """
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta

    # Fermi-function weights.  w_b can be negative (inverted logistic).
    # Clip argument to prevent overflow in exp().
    arg1 = np.clip((En - w_a1) / w_b1, -500.0, 500.0)
    arg2 = np.clip((En - w_a2) / w_b2, -500.0, 500.0)
    w1 = 1.0 / (1.0 + np.exp(arg1))
    w2 = 1.0 / (1.0 + np.exp(arg2))
    # Derived — normalisation constraint
    w3 = 2.0 - 2.0 * w1 - 2.0 * w2

    mu1  = mu_a1 + mu_b1 * En
    mu2  = mu_a2 + mu_b2 * En
    mu3  = A0 / 2.0           # fixed by symmetry
    mu1m = A0 - mu1           # mirror of peak 1
    mu2m = A0 - mu2           # mirror of peak 2

    sig1 = sig_a1 + sig_b1 * En
    sig2 = sig_a2 + sig_b2 * En
    sig3 = sig_a3 + sig_b3 * En

    # Five Gaussians; each integrates to its weight w_i.
    # Sum of integrated weights = 2*w1 + 2*w2 + w3 = 2. ✓
    return (_gauss(A, w1, mu1,  sig1) +
            _gauss(A, w1, mu1m, sig1) +
            _gauss(A, w2, mu2,  sig2) +
            _gauss(A, w2, mu2m, sig2) +
            _gauss(A, w3, mu3,  sig3))


def derived_quantities(theta, En, A0):
    """Return dict of physically meaningful derived values."""
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta
    a1 = np.clip((En - w_a1) / w_b1, -500, 500)
    a2 = np.clip((En - w_a2) / w_b2, -500, 500)
    w1 = 1.0 / (1.0 + np.exp(a1))
    w2 = 1.0 / (1.0 + np.exp(a2))
    w3 = 2.0 - 2.0*w1 - 2.0*w2
    return dict(
        w1=w1, w2=w2, w3=w3,
        mu1=mu_a1+mu_b1*En, mu2=mu_a2+mu_b2*En, mu3=A0/2.0,
        sig1=sig_a1+sig_b1*En, sig2=sig_a2+sig_b2*En, sig3=sig_a3+sig_b3*En,
    )


def check_physicality(theta, En):
    """
    Return True iff parameter vector is physically valid:
      - |w_b1|, |w_b2| > 0.05  (avoid division by zero)
      - sig_1, sig_2, sig_3 > 0  at energy En
      - w3 > 0  at energy En  (i.e., w1 + w2 < 1)

    Note: w1, w2 ∈ (0,1) always (logistic), so w1>0 and w2>0 are guaranteed.
    """
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta

    if abs(w_b1) < 0.05 or abs(w_b2) < 0.05:
        return False

    sig1 = sig_a1 + sig_b1 * En
    sig2 = sig_a2 + sig_b2 * En
    sig3 = sig_a3 + sig_b3 * En
    if sig1 <= 0.0 or sig2 <= 0.0 or sig3 <= 0.0:
        return False

    a1 = np.clip((En - w_a1) / w_b1, -500, 500)
    a2 = np.clip((En - w_a2) / w_b2, -500, 500)
    w1 = 1.0 / (1.0 + np.exp(a1))
    w2 = 1.0 / (1.0 + np.exp(a2))
    if (2.0 - 2.0*w1 - 2.0*w2) <= 0.0:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# PRIOR
# ═══════════════════════════════════════════════════════════════════════════

class Prior:
    """
    Gaussian prior centred on CGMF nominal values, with hard box bounds.

    Design philosophy
    -----------------
    1. Centre on CGMF nominal — physically validated for U235(nth,f).
    2. Prior widths reflect physical uncertainty:
         - mu_a: ±3 u covers reasonable peak position uncertainty.
         - sig_a: ±1 u covers width uncertainty.
         - w_a: ±2 in logistic midpoint covers ~1 unit in weight.
         - 'b' (slope) params: tight (±0.3) at thermal — these are
           nearly unidentifiable from single-energy data; the prior
           prevents runaway while allowing small deviations.
    3. Hard bounds prevent clearly unphysical solutions:
         - w_b bounds allow BOTH signs (critical fix vs previous version).
         - mu ordering (mu_a1 < mu_a2) breaks the peak-swap degeneracy.
         - Both mu_a1, mu_a2 confined to heavy-fragment half (> A0/2).

    For non-U235 systems pass theta_nominal as a (14,) array.
    """

    def __init__(self, A0, En, theta_nominal=None):
        self.A0  = float(A0)
        self.En  = float(En)
        self.nom = (theta_nominal.copy() if theta_nominal is not None
                    else THETA_NOMINAL.copy())

        mu_half = self.A0 / 2.0

        # Prior widths (1σ Gaussian).  These reflect confidence in the
        # CGMF nominal: at thermal energies the 'b' slopes are essentially
        # unconstrained by data so we keep them tight.
        self._prior_sigma = np.array([
            0.025,   # w_a1   — TIGHTENED 3→1; prevents logistic midpoint drift
            0.025,   # w_b1   — TIGHTENED 2→0.8; resists steepness runaway
            2.0,   # mu_a1  — peak 1 position; ±4u around 133.8u
            2.0,   # mu_b1  — slope; tight at thermal
            0.5,   # sig_a1 — peak 1 width; ±1.5u around 3.0u
            0.5,   # sig_b1 — width slope; tight
            0.025,   # w_a2   — TIGHTENED 3→1
            0.025,   # w_b2   — TIGHTENED 2→0.8; CRITICAL: resists sign flip
            2.0,   # mu_a2  — peak 2 position; ±4u around 141.0u
            2.0,   # mu_b2  — slope; tight
            0.5,   # sig_a2 — peak 2 width; ±1.5u around 4.7u
            0.5,   # sig_b2 — width slope; tight
            1.5,   # sig_a3 — symmetric peak width; ±3u around 9.9u
            1.5,   # sig_b3 — width slope; tight
        ])
        
        
        # Hard box bounds.  Critically, w_b2 bounds allow negative values.
        frag_lo = mu_half + 2.0     # peaks must be in heavy-fragment half
        frag_hi = self.A0 - 50.0   # complement must be ≥ 50 u

        self.bounds = np.array([
            [-20.0,  20.0],   # w_a1
            [2.0,  15.0],   # w_b1  — BOTH SIGNS; physicality enforces |w_b|>0.05
            [frag_lo, frag_hi],  # mu_a1
            [-3.0,    3.0],   # mu_b1
            [0.3,    15.0],   # sig_a1  — must be positive
            [-1.5,    1.5],   # sig_b1
            [-20.0,  20.0],   # w_a2
            [-15.0,  -2.0],   # w_b2  — BOTH SIGNS CRITICAL (nominal = -6.14)
            [frag_lo, frag_hi],  # mu_a2
            [-3.0,    3.0],   # mu_b2
            [0.3,    15.0],   # sig_a2  — must be positive
            [-1.5,    1.5],   # sig_b2
            [0.3,    25.0],   # sig_a3  — symmetric peak; broad allowed
            [-1.5,    1.5],   # sig_b3
        ])

    def ln_prior(self, theta):
        """
        Log-prior: Gaussian centred on nominal + box bounds + physicality.

        Returns −∞ if any of:
          - parameter outside box bound
          - physicality check fails  (w3>0, sig>0, |w_b|>0.05)
          - ordering constraint violated (mu_a1 < mu_a2)
        """
        theta = np.asarray(theta, dtype=float)

        # Box bounds
        for k in range(N_PARAMS):
            lo, hi = self.bounds[k]
            if not (lo < theta[k] < hi):
                return -np.inf

        # Physicality
        if not check_physicality(theta, self.En):
            return -np.inf

        # Label ordering: enforce mu_a1 < mu_a2 to prevent peak-swap degeneracy.
        # Peak 1 is the lower of the two heavy-fragment peaks (closer to A0/2).
        if theta[2] >= theta[8]:
            return -np.inf

        # Gaussian prior
        diff = (theta - self.nom) / self._prior_sigma
        return -0.5 * float(np.dot(diff, diff))

    def sample_prior(self, n=1, rng=None):
        """Draw n valid samples from the prior by rejection sampling."""
        rng = np.random.default_rng(rng)
        out = np.empty((n, N_PARAMS))
        filled = 0
        attempts = 0
        while filled < n:
            attempts += 1
            if attempts > 500 * n:
                raise RuntimeError(
                    "Prior sampling failed after many attempts. "
                    "Check that THETA_NOMINAL satisfies all constraints.")
            cand = self.nom + self._prior_sigma * rng.standard_normal(N_PARAMS)
            if np.isfinite(self.ln_prior(cand)):
                out[filled] = cand
                filled += 1
        return out


# ═══════════════════════════════════════════════════════════════════════════
# LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════════

class Likelihood:
    """
    Full-covariance Gaussian likelihood using Cov_inv_chol from step2.

    ln L = −½ ‖ L_inv (Y_model − Y_pre) ‖²

    The step2 sigma_pre values are very small at the peaks (~1e-4) and
    large in the valley/tails.  The Cov_inv_chol naturally encodes this:
    peak bins dominate the likelihood.  No reweighting is needed in the
    likelihood itself — the prior on theta already steers walkers to the
    correct region of parameter space when centred on CGMF nominal.

    Optional peak_weight term (default 0): additive boost to peak bins.
    Only needed if walkers cannot find the peaks during burn-in.
    """

    def __init__(self, A_pre, Y_pre, sigma_pre, Cov_inv_chol, En, A0,
                 peak_weight=0.0):
        self.A_pre        = A_pre
        self.Y_pre        = Y_pre
        self.sigma_pre    = sigma_pre
        self.Cov_inv_chol = Cov_inv_chol
        self.En           = float(En)
        self.A0           = int(A0)
        self.peak_weight  = float(peak_weight)
        # Pre-compute peak weight vector (yield-normalised)
        Ysafe = np.clip(Y_pre, 1e-15, None)
        self._peak_w = Ysafe / Ysafe.mean()

    def ln_likelihood(self, theta):
        """Compute log-likelihood for parameter vector theta."""
        try:
            Y_model = model_yield(theta, self.A_pre, self.En, self.A0)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(Y_model)):
            return -np.inf

        residual = Y_model - self.Y_pre

        # Full-covariance term
        try:
            alpha = linalg.solve_triangular(
                self.Cov_inv_chol, residual, lower=True, check_finite=False)
        except Exception:
            return -np.inf
        ll = -0.5 * float(np.dot(alpha, alpha))

        # Optional peak-focus additive term
        if self.peak_weight > 0.0:
            pull2 = (residual / self.sigma_pre) ** 2
            ll -= 0.5 * self.peak_weight * float(np.dot(self._peak_w, pull2))

        return ll if np.isfinite(ll) else -np.inf


# ═══════════════════════════════════════════════════════════════════════════
# LOG-POSTERIOR (emcee target)
# ═══════════════════════════════════════════════════════════════════════════

def make_log_posterior(likelihood, prior):
    def ln_posterior(theta):
        lp = prior.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = likelihood.ln_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    return ln_posterior


# ═══════════════════════════════════════════════════════════════════════════
# MAP SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def find_map_estimate(ln_posterior, theta0, prior, max_iter=6000):
    """
    MAP estimate via two-pass Nelder-Mead.

    Starting from CGMF nominal (theta0) ensures we begin in the physical
    double-hump region, not some local minimum.
    """
    section("MAP search (Nelder-Mead, two passes from CGMF nominal)")

    best_val = np.inf
    best_x   = theta0.copy()

    rng = np.random.default_rng(0)
    starts = [theta0,
              theta0 + 0.02 * np.abs(theta0) * rng.standard_normal(N_PARAMS)]

    for i_pass, t0 in enumerate(starts):
        # Ensure starting point satisfies prior
        if not np.isfinite(prior.ln_prior(t0)):
            t0 = prior.sample_prior(rng=rng)[0]
            print(f"  Pass {i_pass+1}: start not in prior support, resampled.")

        result = minimize(lambda t: -ln_posterior(t), t0,
                          method="Nelder-Mead",
                          options={"maxiter": max_iter, "xatol": 1e-6,
                                   "fatol": 1e-6, "adaptive": True})
        print(f"  Pass {i_pass+1}: success={result.success}  "
              f"-ln_post={result.fun:.6g}  msg={result.message}")
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()

    print(f"\n  Best MAP  -ln_post = {best_val:.6g}")
    phys = check_physicality(best_x, prior.En)
    print(f"  MAP physicality:  {'OK' if phys else '*** FAILED — check prior/data ***'}")

    dq = derived_quantities(best_x, prior.En, prior.A0)
    print(f"\n  MAP parameter estimates (vs CGMF nominal):")
    print(f"  {'Param':10s}  {'MAP':>10s}  {'Nominal':>10s}  {'Δ':>8s}  {'Δ/σ':>7s}")
    print(f"  {'-'*54}")
    for k in range(N_PARAMS):
        lo, hi = prior.bounds[k]
        delta  = best_x[k] - prior.nom[k]
        dsig   = delta / prior._prior_sigma[k]
        near   = " *** BOUND ***" if (abs(best_x[k]-lo) < 0.05*(hi-lo) or
                                       abs(best_x[k]-hi) < 0.05*(hi-lo)) else ""
        print(f"  {PARAM_LABELS[k]:10s}  {best_x[k]:>10.4f}  "
              f"{prior.nom[k]:>10.4f}  {delta:>8.4f}  {dsig:>7.2f}{near}")

    print(f"\n  Derived quantities at MAP (E={prior.En:.3e} MeV):")
    print(f"    Peak 1:  w1={dq['w1']:.5f}  mu1={dq['mu1']:.3f}  sig1={dq['sig1']:.3f}")
    print(f"    Mirror1: mu={prior.A0-dq['mu1']:.3f}")
    print(f"    Peak 2:  w2={dq['w2']:.5f}  mu2={dq['mu2']:.3f}  sig2={dq['sig2']:.3f}")
    print(f"    Mirror2: mu={prior.A0-dq['mu2']:.3f}")
    print(f"    Sym pk:  w3={dq['w3']:.5f}  mu3={dq['mu3']:.1f}  sig3={dq['sig3']:.3f}")

    return best_x, best_val


# ═══════════════════════════════════════════════════════════════════════════
# WALKER INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════

def initialise_walkers(theta_map, nwalkers, prior, ln_post,
                        scale=0.005, rng=None):
    """
    Initialise walkers in a tight ball around theta_map.

    scale=0.005 means each parameter is perturbed by 0.5% of |theta_map|.
    This is deliberately small so all walkers start at nearly the same,
    physically correct (double-hump) configuration.  The ensemble then
    expands to fill the posterior during burn-in.
    """
    rng = np.random.default_rng(rng)
    p0  = np.zeros((nwalkers, N_PARAMS))
    n_fallback = 0

    for k in range(nwalkers):
        for attempt in range(3000):
            # Small random perturbation + a tiny jitter to avoid identical starts
            trial = (theta_map
                     + scale * np.abs(theta_map) * rng.standard_normal(N_PARAMS)
                     + 1e-8 * rng.standard_normal(N_PARAMS))
            if np.isfinite(ln_post(trial)):
                p0[k] = trial
                break
        else:
            p0[k]    = prior.sample_prior(rng=rng)[0]
            n_fallback += 1

    if n_fallback:
        print(f"  WARNING: {n_fallback}/{nwalkers} walkers initialised from prior "
              f"(MAP ball rejected).")
    else:
        print(f"  All {nwalkers} walkers initialised in tight ball around MAP.")
    return p0


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def gelman_rubin(chain):
    """Gelman-Rubin R̂.  chain: (nwalkers, nsteps, ndim)."""
    nw, ns, nd = chain.shape
    half = ns // 2
    split = np.concatenate([chain[:, :half, :], chain[:, half:, :]], axis=0)
    m, n = split.shape[0], split.shape[1]
    chain_mean = split.mean(axis=1)
    grand_mean = chain_mean.mean(axis=0)
    B = n * np.var(chain_mean, axis=0, ddof=1)
    W = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    var_plus = (n-1)/n * W + B/n
    with np.errstate(divide="ignore", invalid="ignore"):
        Rhat = np.sqrt(np.where(W > 0, var_plus/W, np.nan))
    return Rhat


def effective_sample_size(flat_chain):
    """ESS per parameter via FFT autocorrelation (Sokal's method)."""
    ns, nd = flat_chain.shape
    ess = np.zeros(nd)
    for k in range(nd):
        x = flat_chain[:, k] - flat_chain[:, k].mean()
        f = np.fft.fft(x, n=2*ns)
        acf = np.fft.ifft(f * np.conj(f)).real[:ns]
        if acf[0] <= 0:
            ess[k] = ns; continue
        acf /= acf[0]
        tau = 1.0
        for lag in range(1, ns//2):
            tau += 2.0 * acf[lag]
            if acf[lag] < 0 or tau < 0:
                break
        ess[k] = min(ns, ns / max(tau, 1.0))
    return ess


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_chains(chain, outpath):
    """Trace plots. chain: (nwalkers, nsteps, ndim)."""
    nw, ns, nd = chain.shape
    fig, axes = plt.subplots((nd+1)//2, 2, figsize=(14, 2.8*((nd+1)//2)))
    axes = axes.flatten()
    steps = np.arange(ns)
    for k in range(nd):
        ax = axes[k]
        for w in range(min(nw, 50)):
            ax.plot(steps, chain[w, :, k], lw=0.25, alpha=0.30, color="steelblue")
        ax.plot(steps, np.median(chain[:, :, k], axis=0), lw=1.3, color="k",
                label="Median")
        ax.plot(steps, np.percentile(chain[:, :, k], 16, axis=0),
                lw=0.7, color="k", ls=":")
        ax.plot(steps, np.percentile(chain[:, :, k], 84, axis=0),
                lw=0.7, color="k", ls=":")
        ax.axhline(THETA_NOMINAL[k], color="red", lw=1.2, ls="--",
                   alpha=0.8, label="CGMF nominal")
        ax.set_ylabel(PARAM_NAMES[k], fontsize=9)
        ax.set_xlabel("Step")
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(nd, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("MCMC Trace Plots (all walkers; red dashed = CGMF nominal)",
                 fontsize=12, fontweight="bold")
    _plot_save(fig, outpath)


def plot_acceptance(sampler, outpath):
    frac = sampler.acceptance_fraction
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(np.arange(len(frac)), frac, color="steelblue", alpha=0.8)
    ax.axhline(0.2, color="red",   ls="--", lw=1, label="0.20 (low end)")
    ax.axhline(0.5, color="green", ls="--", lw=1, label="0.50 (target)")
    ax.set_xlabel("Walker"); ax.set_ylabel("Acceptance fraction")
    ax.set_title("Per-walker acceptance fraction"); ax.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_autocorr(sampler, outpath):
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception as e:
        print(f"  [warn] autocorr failed: {e}"); return
    nsteps = sampler.get_chain().shape[0]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(np.arange(N_PARAMS), tau, color=C_DATA, alpha=0.8)
    ax.axhline(nsteps/50, color="red", ls="--", lw=1.2,
               label=f"N/50 = {nsteps/50:.0f}  (want τ < this)")
    ax.set_xticks(np.arange(N_PARAMS))
    ax.set_xticklabels(PARAM_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("τ (integrated autocorrelation)")
    ax.set_title("Autocorrelation time per parameter"); ax.legend(fontsize=9)
    for k, t in enumerate(tau):
        ax.text(k, t+0.3, f"{t:.0f}", ha="center", va="bottom", fontsize=7)
    _plot_save(fig, outpath)


def plot_yield_posterior(A_pre, Y_pre, sigma_pre, flat_chain, En, A0,
                          theta_nominal, n_posterior=300, outpath=None):
    """Posterior predictive overlay. Shows data, nominal, median, and samples."""
    rng  = np.random.default_rng(42)
    idx  = rng.integers(0, flat_chain.shape[0], size=n_posterior)
    samp = flat_chain[idx]

    Y_nom = model_yield(theta_nominal, A_pre, En, A0)
    Y_med = model_yield(np.median(flat_chain, axis=0), A_pre, En, A0)

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(15, 5))
    for ax, do_log in [(ax_log, True), (ax_lin, False)]:
        for i, th in enumerate(samp):
            Y_s = model_yield(th, A_pre, En, A0)
            ax.plot(A_pre, Y_s, lw=0.35, alpha=0.12, color=C_SAMPLE,
                    label="Posterior samples" if i == 0 else None)
        ax.plot(A_pre, Y_nom, lw=1.8, color=C_PRIOR, ls="--", zorder=7,
                label="CGMF nominal")
        ax.plot(A_pre, Y_med, lw=2.2, color=C_MODEL, zorder=8,
                label="Posterior median")
        ax.errorbar(A_pre, Y_pre, yerr=sigma_pre, fmt="o", ms=3.5, capsize=2,
                    lw=0.8, color=C_DATA, alpha=0.85, zorder=9,
                    label=r"Step-2 $Y_{\rm pre}$ ± σ")
        ax.set_xlabel("$A_{\\rm pre}$"); ax.set_ylabel("$Y_{\\rm pre}(A)$")
        if do_log:
            pos = Y_pre[Y_pre > 0]
            ax.set_ylim(max(pos.min()*0.3, 1e-6), pos.max()*3)
            ax.set_yscale("log")
            ax.set_title("Log scale: posterior predictive vs step-2 data")
        else:
            yhi = max(Y_pre.max(), Y_med.max()) * 1.2
            ax.set_ylim(-0.002, yhi)
            ax.set_title("Linear scale: posterior predictive vs step-2 data")
        ax.legend(fontsize=8)
    _plot_save(fig, outpath)


def plot_marginals(flat_chain, prior, outpath):
    """1-D marginal posteriors vs Gaussian priors."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    axes = axes.flatten()
    for k in range(N_PARAMS):
        ax   = axes[k]
        vals = flat_chain[:, k]
        lo, hi = prior.bounds[k]
        # Posterior histogram
        ax.hist(vals, bins=50, density=True, color=C_DATA, alpha=0.65,
                label="Posterior")
        # Prior (Gaussian)
        c, s = prior.nom[k], prior._prior_sigma[k]
        xg   = np.linspace(max(lo, c-5*s), min(hi, c+5*s), 300)
        pg   = np.exp(-0.5*((xg-c)/s)**2) / (np.sqrt(2*np.pi)*s)
        ax.plot(xg, pg, lw=1.5, color=C_PRIOR, ls="--", label="Prior")
        # CGMF nominal line
        ax.axvline(prior.nom[k], color="red",  lw=1.5, ls=":",  label="CGMF nom")
        # Posterior median and 68% CI
        ax.axvline(np.median(vals),          color="k", lw=1.2)
        ax.axvline(np.percentile(vals,  16), color="k", lw=0.7, ls=":")
        ax.axvline(np.percentile(vals,  84), color="k", lw=0.7, ls=":")
        ax.set_title(PARAM_NAMES[k], fontsize=10)
        ax.set_xlabel(PARAM_LABELS[k], fontsize=8)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(N_PARAMS, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("1-D Marginal Posteriors\n"
                 "(orange dashed = prior, red dotted = CGMF nominal, "
                 "black = posterior median ± 68%CI)",
                 fontsize=11, fontweight="bold")
    _plot_save(fig, outpath)


def plot_parameter_summary(flat_chain, prior, outpath):
    """Horizontal bar chart: raw values + normalised pull from nominal."""
    median = np.median(flat_chain, axis=0)
    p16    = np.percentile(flat_chain, 16, axis=0)
    p84    = np.percentile(flat_chain, 84, axis=0)
    lo_err = median - p16
    hi_err = p84 - median

    pull      = (median - prior.nom) / prior._prior_sigma
    pull_lo   = lo_err / prior._prior_sigma
    pull_hi   = hi_err / prior._prior_sigma

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(14, 7))
    y = np.arange(N_PARAMS)

    ax_r.barh(y, median, xerr=[lo_err, hi_err], color=C_DATA, alpha=0.7,
              error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "k"})
    for k in range(N_PARAMS):
        ax_r.plot(prior.nom[k], k, "D", ms=7, color="red",
                  label="CGMF nominal" if k == 0 else None)
    ax_r.set_yticks(y); ax_r.set_yticklabels(PARAM_LABELS, fontsize=10)
    ax_r.set_xlabel("Parameter value")
    ax_r.set_title("Posterior median ± 68%CI\n(red diamonds = CGMF nominal)")
    ax_r.legend(fontsize=9)

    colours = [C_MODEL if abs(p)>2 else C_DATA for p in pull]
    ax_p.barh(y, pull, xerr=[pull_lo, pull_hi], color=colours, alpha=0.7,
              error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "k"})
    ax_p.axvline(0,  color="red", lw=1.5, ls="--", label="CGMF nominal (0)")
    ax_p.axvline(-2, color="gray", lw=0.8, ls=":")
    ax_p.axvline(+2, color="gray", lw=0.8, ls=":")
    ax_p.set_yticks(y); ax_p.set_yticklabels(PARAM_LABELS, fontsize=10)
    ax_p.set_xlabel("(Posterior median − CGMF nominal) / σ_prior")
    ax_p.set_title("Data pull from CGMF nominal\n"
                   "|pull|>2 (red) = data strongly prefers different value")
    ax_p.legend(fontsize=9)
    _plot_save(fig, outpath)


def plot_posterior_correlation(flat_chain, outpath):
    """Full posterior parameter correlation heatmap."""
    corr = np.corrcoef(flat_chain.T)
    n    = N_PARAMS
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label("ρ")
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(PARAM_LABELS, rotation=45,
                                                      ha="right", fontsize=8)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(PARAM_LABELS, fontsize=8)
    ax.set_title("Posterior parameter correlation matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="k" if abs(corr[i,j])<0.7 else "w")
    _plot_save(fig, outpath)


def plot_residuals_posterior(A_pre, Y_pre, sigma_pre, flat_chain, En, A0, outpath):
    """Pull distribution coloured by log-yield magnitude."""
    theta_med = np.median(flat_chain, axis=0)
    Y_med     = model_yield(theta_med, A_pre, En, A0)
    pull      = (Y_med - Y_pre) / sigma_pre

    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(13, 4))
    ax.axhline(0, color="k", lw=1)
    for v in [-2, -1, 1, 2]:
        ax.axhline(v, color="gray", lw=0.7, ls=":")
    sc = ax.scatter(A_pre, pull,
                    c=np.log10(np.clip(Y_pre, 1e-12, None)),
                    cmap="viridis", s=25, zorder=5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log₁₀ Y_pre  (yellow = peaks, purple = valley)")
    ax.set_xlabel("$A_{\\rm pre}$")
    ax.set_ylabel(r"$(Y_{\rm model} - Y_{\rm data})/\sigma$")
    ax.set_title("Normalised residuals (posterior median model)\n"
                 "Good fit = points within ±2 bands, especially in peaks")
    ylim = max(3.5, np.percentile(np.abs(pull), 99) * 1.2)
    ax.set_ylim(-ylim, ylim)

    bins = np.linspace(-min(ylim, 10), min(ylim, 10), 30)
    ax_h.hist(pull, bins=bins, density=True, color=C_DATA, alpha=0.7)
    xg = np.linspace(bins[0], bins[-1], 200)
    ax_h.plot(xg, np.exp(-0.5*xg**2)/np.sqrt(2*np.pi), "k-", lw=1.5, label="N(0,1)")
    ax_h.set_xlabel("Pull"); ax_h.set_ylabel("Density")
    ax_h.set_title(f"Residual distribution\nmean={pull.mean():.2f}  std={pull.std():.2f}")
    ax_h.legend()
    _plot_save(fig, outpath)


def plot_gaussian_fit_check(flat_chain, mu_g, Sigma_g, outpath, n_draw=3000):
    """Validate N(μ,Σ) approximation by comparing marginals."""
    rng  = np.random.default_rng(0)
    draw = rng.multivariate_normal(mu_g, Sigma_g, size=n_draw)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for k in range(N_PARAMS):
        ax = axes[k]
        lo = min(flat_chain[:,k].min(), draw[:,k].min())
        hi = max(flat_chain[:,k].max(), draw[:,k].max())
        be = np.linspace(lo, hi, 40)
        ax.hist(flat_chain[:,k], bins=be, density=True, color=C_DATA,
                alpha=0.60, label="MCMC")
        ax.hist(draw[:,k], bins=be, density=True, color=C_MODEL,
                alpha=0.0, histtype="step", lw=2.0, label="N(μ,Σ)")
        ax.set_title(PARAM_NAMES[k], fontsize=10)
        ax.set_xlabel(PARAM_LABELS[k], fontsize=8)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(N_PARAMS, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("Gaussian Approximation Check: MCMC (blue filled) vs N(μ,Σ) (red outline)\n"
                 "Good match means the offline Gaussian is valid for online HPC sampling",
                 fontsize=11, fontweight="bold")
    _plot_save(fig, outpath)


def plot_summary_panel(A_pre, Y_pre, sigma_pre, flat_chain, En, A0,
                        mu_g, Sigma_g, theta_nominal, outpath, n_samp=300):
    """4-panel overview: yield, residuals, correlation, Gaussian check."""
    rng = np.random.default_rng(7)
    idx = rng.integers(0, flat_chain.shape[0], size=n_samp)

    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.33)
    ax_y = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[1, 1])

    # Yield
    for i, th in enumerate(flat_chain[idx]):
        ax_y.plot(A_pre, model_yield(th, A_pre, En, A0),
                  lw=0.3, alpha=0.10, color=C_SAMPLE)
    Y_nom = model_yield(theta_nominal, A_pre, En, A0)
    Y_med = model_yield(np.median(flat_chain, axis=0), A_pre, En, A0)
    ax_y.errorbar(A_pre, Y_pre, yerr=sigma_pre, fmt="o", ms=2.5,
                  capsize=1.5, lw=0.6, color=C_DATA, alpha=0.8, zorder=8,
                  label="Data")
    ax_y.plot(A_pre, Y_nom, lw=1.5, color=C_PRIOR, ls="--", zorder=9,
              label="CGMF nominal")
    ax_y.plot(A_pre, Y_med, lw=2.0, color=C_MODEL, zorder=10,
              label="Posterior median")
    pos = Y_pre[Y_pre > 0]
    ax_y.set_ylim(max(pos.min()*0.3, 1e-6), pos.max()*3)
    ax_y.set_yscale("log")
    ax_y.set_xlabel("$A$"); ax_y.set_ylabel("$Y(A)$")
    ax_y.set_title("Posterior predictive (log scale)")
    ax_y.legend(fontsize=8)

    # Residuals
    pull = (Y_med - Y_pre) / sigma_pre
    sc   = ax_p.scatter(A_pre, pull,
                        c=np.log10(np.clip(Y_pre, 1e-12, None)),
                        cmap="viridis", s=18, zorder=5)
    fig.colorbar(sc, ax=ax_p).set_label("log₁₀ Y_pre")
    ax_p.axhline(0, color="k", lw=1)
    for v in [-2, -1, 1, 2]:
        ax_p.axhline(v, color="gray", lw=0.6, ls=":")
    ylim = max(3.5, np.percentile(np.abs(pull), 99)*1.2)
    ax_p.set_ylim(-ylim, ylim)
    ax_p.set_xlabel("$A$"); ax_p.set_ylabel("Pull")
    ax_p.set_title("Normalised residuals (median model)")

    # Correlation matrix
    corr = np.corrcoef(flat_chain.T)
    im   = ax_c.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax_c, fraction=0.04, pad=0.03)
    ax_c.set_xticks(np.arange(N_PARAMS))
    ax_c.set_xticklabels(PARAM_LABELS, rotation=45, ha="right", fontsize=7)
    ax_c.set_yticks(np.arange(N_PARAMS))
    ax_c.set_yticklabels(PARAM_LABELS, fontsize=7)
    ax_c.set_title("Posterior correlation matrix")

    # Gaussian check (selected params)
    rng2 = np.random.default_rng(99)
    draw = rng2.multivariate_normal(mu_g, Sigma_g, size=3000)
    colours = [C_DATA, C_SAMPLE, C_MODEL, C_PRIOR]
    for ki, k in enumerate([2, 8, 4, 10]):  # mu_a1, mu_a2, sig_a1, sig_a2
        lo = min(flat_chain[:,k].min(), draw[:,k].min())
        hi = max(flat_chain[:,k].max(), draw[:,k].max())
        be = np.linspace(lo, hi, 30)
        ax_g.hist(flat_chain[:,k], bins=be, density=True, color=colours[ki],
                  alpha=0.40, label=PARAM_LABELS[k])
        ax_g.hist(draw[:,k], bins=be, density=True, color=colours[ki],
                  alpha=0.0, histtype="step", lw=2.0)
    ax_g.set_xlabel("Value"); ax_g.set_ylabel("Density")
    ax_g.set_title("Gauss approx vs MCMC (selected params)")
    ax_g.legend(fontsize=8)

    fig.suptitle("Step-3 MCMC Summary Panel", fontsize=14, fontweight="bold", y=1.01)
    _plot_save(fig, outpath)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Step-3: MCMC for 3-Gaussian fission yield parameters.")
    ap.add_argument("--npz",          required=True)
    ap.add_argument("--A0",           type=int,   required=True,
                    help="Compound nucleus mass (e.g. 236)")
    ap.add_argument("--En",           type=float, default=2.53e-8,
                    help="Neutron energy [MeV]")
    ap.add_argument("--outdir",       default="step3_out")
    ap.add_argument("--nwalkers",     type=int,   default=96)
    ap.add_argument("--nsteps",       type=int,   default=8000)
    ap.add_argument("--burnin",       type=int,   default=3000)
    ap.add_argument("--thin",         type=int,   default=10)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--peak_weight",  type=float, default=0.0,
                    help="Extra weighting on peak bins (0=off). Try 1–5 if "
                         "walkers cannot find the double hump.")
    ap.add_argument("--progress",     action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.outdir)

    # ── 1. Load ──────────────────────────────────────────────────────────
    section("Loading step-2 pre-neutron yield data")
    if not os.path.exists(args.npz):
        sys.exit(f"ERROR: file not found: {args.npz}")

    z = np.load(args.npz, allow_pickle=True)
    print(f"  Keys: {list(z.keys())}")

    A_pre        = z["A_pre"].astype(int)
    Y_pre        = z["Y_pre"].astype(float)
    sigma_pre    = z["sigma_pre"].astype(float)
    Cov_inv_chol = z["Cov_inv_chol"].astype(float)

    print(f"\n  A_pre     : {A_pre.min()}–{A_pre.max()}  ({len(A_pre)} bins)")
    print(f"  Y_pre     : min={Y_pre.min():.3e}  max={Y_pre.max():.3e}  sum={Y_pre.sum():.4f}")
    print(f"  sigma_pre : min={sigma_pre.min():.3e}  max={sigma_pre.max():.3e}")
    print(f"  Cov_inv_chol: {Cov_inv_chol.shape}")
    print(f"\n  A0={args.A0}   En={args.En:.3e} MeV")

    # ── 2. Build prior & likelihood ──────────────────────────────────────
    section("Prior and likelihood construction")

    prior   = Prior(A0=args.A0, En=args.En)
    lik     = Likelihood(A_pre, Y_pre, sigma_pre, Cov_inv_chol,
                         args.En, args.A0, peak_weight=args.peak_weight)
    ln_post = make_log_posterior(lik, prior)

    print(f"\n  Prior (centred on CGMF nominal):")
    print(f"  {'Param':10s}  {'Nominal':>9s}  {'σ_prior':>9s}  {'Bounds':>22s}")
    print(f"  {'-'*56}")
    for k in range(N_PARAMS):
        lo, hi = prior.bounds[k]
        print(f"  {PARAM_LABELS[k]:10s}  {prior.nom[k]:>9.4f}  "
              f"{prior._prior_sigma[k]:>9.4f}  [{lo:.2g}, {hi:.2g}]")

    # Verify CGMF nominal is in prior support
    lp_nom = prior.ln_prior(prior.nom)
    ll_nom = lik.ln_likelihood(prior.nom)
    Y_nom  = model_yield(prior.nom, A_pre, args.En, args.A0)
    print(f"\n  CGMF nominal validation:")
    print(f"    ln_prior      = {lp_nom:.4f}  (= 0 means exactly at prior mode)")
    print(f"    ln_likelihood = {ll_nom:.4g}")
    print(f"    ln_posterior  = {lp_nom+ll_nom:.4g}")
    print(f"    Y.sum()       = {Y_nom.sum():.6f}")
    dq_nom = derived_quantities(prior.nom, args.En, args.A0)
    print(f"    w1={dq_nom['w1']:.5f}  w2={dq_nom['w2']:.5f}  w3={dq_nom['w3']:.5f}")
    print(f"    mu1={dq_nom['mu1']:.3f}  mu2={dq_nom['mu2']:.3f}")

    if not np.isfinite(lp_nom):
        print("\n  *** FATAL: CGMF nominal violates prior constraints. ***")
        print("  Check that mu_a1 < mu_a2 and both are in [A0/2+2, A0-50]. ***")
        sys.exit(1)

    # ── 3. MAP ───────────────────────────────────────────────────────────
    theta_map, _ = find_map_estimate(ln_post, prior.nom.copy(), prior)

    Y_map     = model_yield(theta_map, A_pre, args.En, args.A0)
    chi2_map  = float(np.sum(((Y_map - Y_pre)/sigma_pre)**2))
    ndof      = len(A_pre) - N_PARAMS
    print(f"\n  MAP yield sum  = {Y_map.sum():.6f}")
    print(f"  χ²/dof (MAP)   = {chi2_map:.1f}/{ndof} = {chi2_map/ndof:.2f}")
    print(f"  MAP peak        = {Y_map.max():.5f} at A={A_pre[np.argmax(Y_map)]}")

    # ── 4. Walker init ───────────────────────────────────────────────────
    section("Initialising walkers")
    nwalkers = max(args.nwalkers, 2*N_PARAMS+4)
    if nwalkers != args.nwalkers:
        print(f"  nwalkers bumped {args.nwalkers}→{nwalkers}")

    p0 = initialise_walkers(theta_map, nwalkers, prior, ln_post,
                             scale=0.005,
                             rng=np.random.default_rng(args.seed+1))
    n_fin = sum(1 for w in p0 if np.isfinite(ln_post(w)))
    print(f"  {n_fin}/{nwalkers} walkers with finite ln_post")
    if n_fin < nwalkers // 2:
        print("  *** WARNING: most walkers rejected — "
              "MAP search may have failed. ***")

    # ── 5. Run emcee ─────────────────────────────────────────────────────
    section(f"emcee: {nwalkers} walkers × {args.nsteps} steps")
    n_prod = args.nsteps - args.burnin
    print(f"  Burn-in:    {args.burnin}  |  Production: {n_prod}  |  Thin: {args.thin}")
    print(f"  Flat chain: ~{nwalkers * n_prod // args.thin} samples")
    if args.peak_weight > 0:
        print(f"  Peak-focus weight: {args.peak_weight}")

    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, N_PARAMS, ln_post)

    print(f"\n  Phase 1: Burn-in ({args.burnin} steps)...")
    sampler.run_mcmc(p0, args.burnin, progress=args.progress)
    p_restart = sampler.get_last_sample()
    sampler.reset()

    print(f"  Phase 2: Production ({n_prod} steps)...")
    sampler.run_mcmc(p_restart, n_prod, progress=args.progress)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f} s ({elapsed/60:.1f} min)")

    # ── 6. Diagnostics ───────────────────────────────────────────────────
    section("Chain diagnostics")

    acc = sampler.acceptance_fraction
    print(f"\n  Acceptance:  mean={acc.mean():.3f}  "
          f"min={acc.min():.3f}  max={acc.max():.3f}")
    if acc.mean() < 0.15:
        print("  *** LOW acceptance (<0.15). Try reducing walker init scale "
              "or increasing burn-in. ***")
    elif acc.mean() > 0.70:
        print("  NOTE: high acceptance (>0.70) — chain may be under-exploring.")

    try:
        tau_arr = sampler.get_autocorr_time(quiet=True)
        print(f"\n  Autocorrelation times (N_prod={n_prod}):")
        print(f"  {'Param':10s}  {'τ':>8s}  {'N/τ':>8s}  Status")
        print(f"  {'-'*44}")
        for k, (nm, tau) in enumerate(zip(PARAM_LABELS, tau_arr)):
            ratio = n_prod / tau if np.isfinite(tau) else np.inf
            flag  = "  *** UNDERSAMPLED (need more steps) ***" if ratio < 50 else "  OK"
            print(f"  {nm:10s}  {tau:>8.1f}  {ratio:>8.1f}{flag}")
    except Exception as e:
        print(f"  Autocorr estimate failed: {e}")

    flat_chain = sampler.get_chain(flat=True, thin=args.thin)
    lp_flat    = sampler.get_log_prob(flat=True, thin=args.thin)
    print(f"\n  Flat chain: {flat_chain.shape}")
    print(f"  Log-posterior: min={lp_flat.min():.4g}  max={lp_flat.max():.4g}  "
          f"median={np.median(lp_flat):.4g}")
    if lp_flat.max() - lp_flat.min() > 500:
        print("  *** Large spread in log-post — walkers may be multi-modal. ***")

    # Gelman-Rubin
    chain_w = np.transpose(sampler.get_chain(), (1, 0, 2))
    Rhat    = gelman_rubin(chain_w)
    print(f"\n  Gelman-Rubin R̂ (< 1.1 = converged):")
    any_bad = False
    for k, (nm, r) in enumerate(zip(PARAM_LABELS, Rhat)):
        flag = "  *** NOT CONVERGED ***" if (np.isfinite(r) and r > 1.1) else ""
        if flag: any_bad = True
        print(f"  {nm:10s}  R̂={r:.4f}{flag}")
    if not any_bad:
        print("  All R̂ < 1.1 ✓")

    ess = effective_sample_size(flat_chain)
    print(f"\n  ESS per parameter:")
    for k, (nm, e) in enumerate(zip(PARAM_LABELS, ess)):
        flag = "  *** LOW ***" if e < 200 else ""
        print(f"  {nm:10s}  ESS={e:.0f}{flag}")

    # ── 7. Posterior summary ─────────────────────────────────────────────
    section("Posterior summary statistics")

    median = np.median(flat_chain, axis=0)
    mean   = flat_chain.mean(axis=0)
    std    = flat_chain.std(axis=0)
    p16    = np.percentile(flat_chain, 16, axis=0)
    p84    = np.percentile(flat_chain, 84, axis=0)

    print(f"\n  {'Param':10s}  {'Nominal':>9s}  {'Median':>9s}  "
          f"{'Std':>9s}  {'16%':>9s}  {'84%':>9s}  {'Δ/σ':>7s}")
    print(f"  {'-'*76}")
    for k in range(N_PARAMS):
        pull = (median[k]-prior.nom[k]) / prior._prior_sigma[k]
        print(f"  {PARAM_LABELS[k]:10s}  {prior.nom[k]:>9.4f}  {median[k]:>9.4f}  "
              f"{std[k]:>9.4f}  {p16[k]:>9.4f}  {p84[k]:>9.4f}  {pull:>7.2f}")

    section("Derived model quantities at posterior median")
    dq_med = derived_quantities(median, args.En, args.A0)
    Y_med  = model_yield(median, A_pre, args.En, args.A0)
    chi2_m = float(np.sum(((Y_med-Y_pre)/sigma_pre)**2))
    print(f"\n  At E = {args.En:.3e} MeV:")
    print(f"    Peak 1:  w1 = {dq_med['w1']:.5f}  "
          f"mu = {dq_med['mu1']:.3f}  sig = {dq_med['sig1']:.3f}  "
          f"(heavy fragment)")
    print(f"    Mirror1: mu = {args.A0 - dq_med['mu1']:.3f}  (light fragment)")
    print(f"    Peak 2:  w2 = {dq_med['w2']:.5f}  "
          f"mu = {dq_med['mu2']:.3f}  sig = {dq_med['sig2']:.3f}  "
          f"(second heavy peak)")
    print(f"    Mirror2: mu = {args.A0 - dq_med['mu2']:.3f}")
    print(f"    Sym:     w3 = {dq_med['w3']:.5f}  "
          f"mu = {dq_med['mu3']:.1f}    sig = {dq_med['sig3']:.3f}")
    print(f"\n  Median yield sum  = {Y_med.sum():.6f}")
    print(f"  χ²(median)/dof    = {chi2_m:.2f}/{ndof} = {chi2_m/ndof:.3f}")
    if chi2_m/ndof > 5:
        print("  NOTE: Large χ²/dof is expected — the 3-Gaussian model is a")
        print("  smooth parameterisation; it cannot exactly match the non-parametric")
        print("  step-2 output.  The important metric is whether the peaks are")
        print("  well-captured (check yield_posterior.png and residuals_posterior.png).")

    # ── 8. Gaussian fit ──────────────────────────────────────────────────
    section("Fitting multivariate Gaussian for online HPC sampling")

    mu_g    = flat_chain.mean(axis=0)
    Sigma_g = np.cov(flat_chain.T)

    try:
        chol_Sigma = linalg.cholesky(Sigma_g, lower=True)
        print(f"  Cholesky(Σ) OK.  cond(L) = {np.linalg.cond(chol_Sigma):.4e}")
    except linalg.LinAlgError:
        eps = 1e-10 * np.diag(Sigma_g).max()
        print(f"  WARNING: Σ not PD; adding jitter {eps:.2e}")
        Sigma_g += eps * np.eye(N_PARAMS)
        chol_Sigma = linalg.cholesky(Sigma_g, lower=True)

    corr_g   = np.corrcoef(flat_chain.T)
    off_diag = corr_g[np.triu_indices(N_PARAMS, k=1)]
    print(f"  Max off-diagonal |ρ|: {np.abs(off_diag).max():.4f}")

    # Non-Gaussianity warnings
    for k in range(N_PARAMS):
        v   = flat_chain[:,k]
        sk  = float(np.mean(((v-v.mean())/v.std())**3))
        ku  = float(np.mean(((v-v.mean())/v.std())**4)) - 3.0
        if abs(sk) > 1.2 or abs(ku) > 2.5:
            print(f"  NOTE: {PARAM_LABELS[k]} is non-Gaussian "
                  f"(skew={sk:.2f}, ex_kurt={ku:.2f}) — "
                  f"Gaussian approx has limited accuracy for this param.")

    print(f"\n  Online HPC sampling:")
    print(f"    z      = np.random.standard_normal((N, 14))")
    print(f"    thetas = gauss_mu + z @ gauss_chol.T   # shape (N, 14)")
    print(f"    for th in thetas:  Y = model_yield(th, A_pre, En, A0)")

    # ── 9. Plots ─────────────────────────────────────────────────────────
    section("Generating diagnostic plots")

    chain_t = np.transpose(sampler.get_chain(), (1, 0, 2))
    plot_chains(chain_t, os.path.join(args.outdir, "chains.png"))
    plot_acceptance(sampler, os.path.join(args.outdir, "acceptance.png"))
    plot_autocorr(sampler, os.path.join(args.outdir, "autocorr.png"))
    plot_yield_posterior(A_pre, Y_pre, sigma_pre, flat_chain,
                         args.En, args.A0, prior.nom,
                         outpath=os.path.join(args.outdir, "yield_posterior.png"))
    plot_marginals(flat_chain, prior,
                   os.path.join(args.outdir, "marginals.png"))
    plot_parameter_summary(flat_chain, prior,
                           os.path.join(args.outdir, "parameter_summary.png"))
    plot_posterior_correlation(flat_chain,
                               os.path.join(args.outdir, "posterior_correlation.png"))
    plot_residuals_posterior(A_pre, Y_pre, sigma_pre, flat_chain,
                             args.En, args.A0,
                             os.path.join(args.outdir, "residuals_posterior.png"))
    plot_gaussian_fit_check(flat_chain, mu_g, Sigma_g,
                            os.path.join(args.outdir, "gaussian_approx_check.png"))
    plot_summary_panel(A_pre, Y_pre, sigma_pre, flat_chain,
                       args.En, args.A0, mu_g, Sigma_g, prior.nom,
                       os.path.join(args.outdir, "summary_panel.png"))

    if HAS_CORNER:
        section("Corner plot")
        try:
            fig_c = corner.corner(
                flat_chain, labels=PARAM_LABELS,
                truths=prior.nom.tolist(), truth_color="red",
                quantiles=[0.16, 0.50, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 8},
                label_kwargs={"fontsize": 9},
            )
            fp = os.path.join(args.outdir, "corner.png")
            fig_c.savefig(fp, dpi=120, bbox_inches="tight")
            plt.close(fig_c)
            print(f"  [plot] {fp}")
        except Exception as e:
            print(f"  Corner failed: {e}")

    # ── 10. Save ─────────────────────────────────────────────────────────
    section("Saving outputs")

    summary = {
        "A0": args.A0, "En": args.En,
        "nwalkers": nwalkers, "nsteps": args.nsteps,
        "burnin": args.burnin, "thin": args.thin,
        "peak_weight": args.peak_weight,
        "acceptance_mean": float(acc.mean()),
        "acceptance_min":  float(acc.min()),
        "chi2_median_model": float(chi2_m),
        "chi2_per_dof": float(chi2_m/ndof),
        "ndof": int(ndof), "elapsed_s": float(elapsed),
        "flat_chain_shape": list(flat_chain.shape),
        "Rhat": {PARAM_LABELS[k]: float(Rhat[k]) for k in range(N_PARAMS)},
        "ESS":  {PARAM_LABELS[k]: float(ess[k])  for k in range(N_PARAMS)},
        "cgmf_nominal":     {PARAM_LABELS[k]: float(prior.nom[k])    for k in range(N_PARAMS)},
        "posterior_median": {PARAM_LABELS[k]: float(median[k])       for k in range(N_PARAMS)},
        "posterior_std":    {PARAM_LABELS[k]: float(std[k])          for k in range(N_PARAMS)},
        "posterior_p16":    {PARAM_LABELS[k]: float(p16[k])          for k in range(N_PARAMS)},
        "posterior_p84":    {PARAM_LABELS[k]: float(p84[k])          for k in range(N_PARAMS)},
        "derived_at_En": {k: float(v) for k, v in dq_med.items()},
        "param_labels": PARAM_LABELS,
    }
    with open(os.path.join(args.outdir, "step3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON: {os.path.join(args.outdir, 'step3_summary.json')}")

    npz_path = os.path.join(args.outdir, "step3_mcmc.npz")
    np.savez(npz_path,
        flat_chain          = flat_chain,
        full_chain          = sampler.get_chain(),
        log_prob_flat       = lp_flat,
        # ── Online HPC: draw thetas = gauss_mu + randn(N,14) @ gauss_chol.T
        gauss_mu            = mu_g,
        gauss_cov           = Sigma_g,
        gauss_chol          = chol_Sigma,
        gauss_corr          = corr_g,
        # ── Point estimates
        theta_nominal       = prior.nom,
        theta_map           = theta_map,
        theta_median        = median,
        theta_mean          = mean,
        theta_std           = std,
        theta_p16           = p16,
        theta_p84           = p84,
        # ── Posterior predictive
        Y_pre_median_model  = Y_med,
        Y_pre_nominal_model = Y_nom,
        A_pre               = A_pre,
        Y_pre_data          = Y_pre,
        sigma_pre_data      = sigma_pre,
        # ── Convergence
        Rhat                = Rhat,
        ESS                 = ess,
        acceptance_fraction = acc,
        # ── Meta
        A0                  = np.array([args.A0]),
        En                  = np.array([args.En]),
        param_labels        = np.array(PARAM_LABELS),
    )
    print(f"  NPZ: {npz_path}")

    section("STEP 3 COMPLETE")
    print(f"  Output directory: {args.outdir}")
    print(f"\n  Plots — check in this order:")
    print(f"    summary_panel.png         — 4-panel overview (START HERE)")
    print(f"    yield_posterior.png       — posterior predictive vs data (log+linear)")
    print(f"    residuals_posterior.png   — pull dist. coloured by yield magnitude")
    print(f"    chains.png                — trace plots (check mixing & stationarity)")
    print(f"    marginals.png             — posterior vs prior + CGMF nominal")
    print(f"    parameter_summary.png     — median ± 68%CI + pull from nominal")
    print(f"    posterior_correlation.png — parameter correlations")
    print(f"    gaussian_approx_check.png — validates N(μ,Σ) for HPC use")
    print(f"    corner.png                — bivariate posteriors (if corner installed)")
    print(f"\n  Online HPC recipe:")
    print(f"    npz    = np.load('step3_mcmc.npz')")
    print(f"    z      = np.random.standard_normal((N, 14))")
    print(f"    thetas = npz['gauss_mu'] + z @ npz['gauss_chol'].T")
    print(f"    for th in thetas:  Y = model_yield(th, A_pre, En, A0)")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
