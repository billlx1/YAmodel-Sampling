#!/usr/bin/env python3
"""
STEP 4 — Online HPC sampling from the MCMC Gaussian approximation.

This script emulates exactly what the HPC online sampler will do at runtime:
draw parameter sets from N(μ, Σ) and evaluate Y(A) for each.
It then produces a comprehensive set of diagnostics so you can verify
that the sampling breadth is appropriate — neither too tight (all samples
nearly identical) nor too wide (unphysical samples dominating).

Usage
-----
  python step4_online_sampling.py \\
      --npz mcmc_out/step3_mcmc.npz \\
      --A0 236 --En 2.53e-8 \\
      --N 2000 \\
      --outdir sampling_diag

Diagnostics produced
---------------------
  01_yield_envelope.png    — full envelope of Y(A) over N samples (log + linear)
  02_peak_coverage.png     — zoomed into peak regions; width of predictive band
  03_weight_distribution.png — marginal distributions of derived w1, w2, w3
  04_peak_positions.png    — scatter of mu1 vs mu2 at sampled energy
  05_peak_widths.png       — scatter of sig1 vs sig2 vs sig3
  06_yield_percentiles.png — 5/16/50/84/95 percentile envelope vs data
  07_parameter_scatter.png — pairwise scatter of key derived quantities
  08_physicality_check.png — fraction physical, w3 distribution, yield sums
  09_uncertainty_profile.png — pointwise CV = std(Y)/mean(Y) across A
  10_summary_dashboard.png — single-page overview of all key metrics
"""

import argparse
import os
import sys
import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Style — clean academic light theme ───────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8f9fb",
    "axes.edgecolor":    "#cdd1d9",
    "axes.labelcolor":   "#2c3e50",
    "axes.titlecolor":   "#1a252f",
    "axes.grid":         True,
    "grid.color":        "#e2e6ea",
    "grid.linewidth":    0.7,
    "grid.linestyle":    "--",
    "xtick.color":       "#5d6d7e",
    "ytick.color":       "#5d6d7e",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "text.color":        "#2c3e50",
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#cdd1d9",
    "legend.labelcolor": "#2c3e50",
    "legend.framealpha": 0.92,
    "figure.dpi":        140,
    "font.family":       "serif",
    "font.serif":        ["Palatino"],
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.constrained_layout.use": False,
})

# ── Academic colour palette ───────────────────────────────────────────────────
C_ENVELOPE = "#2874a6"   # steel blue  — sample band
C_MEDIAN   = "#c0392b"   # deep red    — posterior median model
C_NOMINAL  = "#27ae60"   # forest green — CGMF nominal
C_DATA     = "#d35400"   # burnt orange — step-2 data
C_W1       = "#2980b9"   # mid blue
C_W2       = "#1a5276"   # dark navy
C_W3       = "#1e8449"   # dark green
C_SIG1     = "#b7770d"   # warm amber
C_SIG2     = "#922b21"   # crimson
C_SIG3     = "#6c3483"   # plum purple

PARAM_LABELS = [
    "w_a1", "w_b1", "mu_a1", "mu_b1", "sig_a1", "sig_b1",
    "w_a2", "w_b2", "mu_a2", "mu_b2", "sig_a2", "sig_b2",
    "sig_a3", "sig_b3",
]
N_PARAMS = 14

SEPARATOR = "=" * 72


def section(t):
    print(); print(SEPARATOR); print(f"  {t}"); print(SEPARATOR)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  [plot] {path}")


def _panel_bg(ax):
    """Subtle alternating panel shade and clean spines."""
    ax.set_facecolor("#f8f9fb")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cdd1d9")
        spine.set_linewidth(0.8)


# ── Model (identical to step3) ───────────────────────────────────────────────

def _gauss(A, w, mu, sig):
    return w / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-0.5 * ((A - mu) / sig) ** 2)


def model_yield(theta, A, En, A0):
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta
    a1 = np.clip((En - w_a1) / w_b1, -500, 500)
    a2 = np.clip((En - w_a2) / w_b2, -500, 500)
    w1 = 1.0 / (1.0 + np.exp(a1))
    w2 = 1.0 / (1.0 + np.exp(a2))
    w3 = 2.0 - 2.0 * w1 - 2.0 * w2
    mu1 = mu_a1 + mu_b1 * En;  mu1m = A0 - mu1
    mu2 = mu_a2 + mu_b2 * En;  mu2m = A0 - mu2
    mu3 = A0 / 2.0
    sig1 = sig_a1 + sig_b1 * En
    sig2 = sig_a2 + sig_b2 * En
    sig3 = sig_a3 + sig_b3 * En
    return (_gauss(A, w1, mu1, sig1) + _gauss(A, w1, mu1m, sig1) +
            _gauss(A, w2, mu2, sig2) + _gauss(A, w2, mu2m, sig2) +
            _gauss(A, w3, mu3,  sig3))


def is_physical(theta, En):
    """Return True if sample is physically valid."""
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta
    if abs(w_b1) < 0.05 or abs(w_b2) < 0.05:
        return False
    s1 = sig_a1 + sig_b1 * En
    s2 = sig_a2 + sig_b2 * En
    s3 = sig_a3 + sig_b3 * En
    if s1 <= 0 or s2 <= 0 or s3 <= 0:
        return False
    w1 = 1.0 / (1.0 + np.exp(np.clip((En - w_a1) / w_b1, -500, 500)))
    w2 = 1.0 / (1.0 + np.exp(np.clip((En - w_a2) / w_b2, -500, 500)))
    return (2.0 - 2.0 * w1 - 2.0 * w2) > 0.0


def derived(theta, En, A0):
    """Compute derived physical quantities from a parameter vector."""
    (w_a1, w_b1, mu_a1, mu_b1, sig_a1, sig_b1,
     w_a2, w_b2, mu_a2, mu_b2, sig_a2, sig_b2,
     sig_a3, sig_b3) = theta
    w1   = 1.0 / (1.0 + np.exp(np.clip((En - w_a1) / w_b1, -500, 500)))
    w2   = 1.0 / (1.0 + np.exp(np.clip((En - w_a2) / w_b2, -500, 500)))
    w3   = 2.0 - 2.0 * w1 - 2.0 * w2
    mu1  = mu_a1 + mu_b1 * En
    mu2  = mu_a2 + mu_b2 * En
    sig1 = sig_a1 + sig_b1 * En
    sig2 = sig_a2 + sig_b2 * En
    sig3 = sig_a3 + sig_b3 * En
    return w1, w2, w3, mu1, mu2, sig1, sig2, sig3


# ── Sampling ─────────────────────────────────────────────────────────────────

def draw_samples(gauss_mu, gauss_chol, N, rng):
    """
    Draw N parameter sets from N(μ, Σ) using the Cholesky factor.
    This is exactly what the HPC online sampler does:
        z      = randn(N, 14)
        thetas = mu + z @ L.T
    """
    z = rng.standard_normal((N, N_PARAMS))
    return gauss_mu[None, :] + z @ gauss_chol.T


# ── Shared annotation box style ───────────────────────────────────────────────
_ABOX = dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
             edgecolor="#cdd1d9", alpha=0.90)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Full yield envelope (log + linear)
# ══════════════════════════════════════════════════════════════════════════════

def plot_yield_envelope(A, Y_all, Y_nom, Y_med, Y_data, sigma_data, outpath):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6),
                                      facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.30)

    for ax, do_log in [(ax_l, True), (ax_r, False)]:
        _panel_bg(ax)
        ps = np.percentile(Y_all, [5, 16, 50, 84, 95], axis=0)

        ax.fill_between(A, ps[0], ps[4], alpha=0.12, color=C_ENVELOPE,
                        label="5–95% CI")
        ax.fill_between(A, ps[1], ps[3], alpha=0.28, color=C_ENVELOPE,
                        label="16–84% CI")
        ax.plot(A, ps[2], color=C_ENVELOPE, lw=2.0, label="Sampled median")

        ax.plot(A, Y_nom, color=C_NOMINAL, lw=1.6, ls="--", alpha=0.90,
                label="CGMF nominal", zorder=7)
        ax.plot(A, Y_med, color=C_MEDIAN, lw=1.8, alpha=0.90,
                label="MCMC median", zorder=8)

        if Y_data is not None:
            ax.errorbar(A, Y_data, yerr=sigma_data, fmt="o", ms=3.0,
                        capsize=2.5, lw=0.8, color=C_DATA, alpha=0.85,
                        zorder=9, label="Step-2 data")

        ax.set_xlabel("$A_{\\rm pre}$", fontsize=10)
        ax.set_ylabel("$Y_{\\rm pre}(A)$", fontsize=10)

        if do_log:
            ax.set_yscale("log")
            ax.set_ylim(max(ps[0][ps[0] > 0].min() * 0.3
                           if (ps[0] > 0).any() else 1e-6, 1e-6),
                        ps[4].max() * 3)
            ax.set_title("Yield envelope — log scale", fontweight="bold",
                         fontsize=11)
        else:
            ax.set_ylim(-0.002, ps[4].max() * 1.15)
            ax.set_title("Yield envelope — linear scale", fontweight="bold",
                         fontsize=11)

        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)

    fig.suptitle("Plot 01 — HPC Online Sampling: Full Yield Envelope",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Zoomed peak coverage (heavy + light fragments separately)
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_coverage(A, Y_all, Y_nom, Y_med, Y_data, sigma_data,
                       A0, outpath):
    A_half = A0 // 2
    mask_heavy  = A >= A_half
    mask_light  = A < A_half
    mask_valley = (A >= 105) & (A <= 131)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.30)

    for ax, mask, title in [
        (axes[0], mask_light,  "Light fragments ($A < 118$)"),
        (axes[1], mask_valley, "Valley region (105–131)"),
        (axes[2], mask_heavy,  "Heavy fragments ($A \\geq 118$)"),
    ]:
        _panel_bg(ax)
        A_m = A[mask]
        Y_m = Y_all[:, mask]
        ps  = np.percentile(Y_m, [5, 16, 50, 84, 95], axis=0)

        ax.fill_between(A_m, ps[0], ps[4], alpha=0.12, color=C_ENVELOPE)
        ax.fill_between(A_m, ps[1], ps[3], alpha=0.28, color=C_ENVELOPE,
                        label="16–84% CI")
        ax.plot(A_m, ps[2],         color=C_ENVELOPE, lw=2.0,
                label="Sampled median")
        ax.plot(A_m, Y_nom[mask],   color=C_NOMINAL,  lw=1.5, ls="--",
                label="CGMF nominal")
        ax.plot(A_m, Y_med[mask],   color=C_MEDIAN,   lw=1.8,
                label="MCMC median")

        if Y_data is not None:
            ax.errorbar(A_m, Y_data[mask], yerr=sigma_data[mask],
                        fmt="o", ms=3.5, capsize=2.5, lw=0.8,
                        color=C_DATA, alpha=0.85, label="Data", zorder=9)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("$A_{\\rm pre}$", fontsize=9)
        ax.set_ylabel("$Y_{\\rm pre}(A)$", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.92)

        if "Valley" in title:
            ax.set_yscale("log")
            bottom = max(ps[0][ps[0] > 0].min() * 0.3
                        if (ps[0] > 0).any() else 1e-6, 1e-6)
            ax.set_ylim(bottom, ps[4].max() * 5)

    fig.suptitle("Plot 02 — Zoomed Peak and Valley Coverage",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Derived weight distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_distributions(deriv_vals, nom_deriv, outpath):
    w1s, w2s, w3s = deriv_vals["w1"], deriv_vals["w2"], deriv_vals["w3"]
    w1n, w2n, w3n = nom_deriv["w1"],  nom_deriv["w2"],  nom_deriv["w3"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.32)

    for ax, vals, nom, color, label in [
        (axes[0], w1s, w1n, C_W1, "$w_1$  (minor heavy peak)"),
        (axes[1], w2s, w2n, C_W2, "$w_2$  (dominant heavy peak)"),
        (axes[2], w3s, w3n, C_W3, "$w_3$  (symmetric valley)"),
    ]:
        _panel_bg(ax)
        ax.hist(vals, bins=60, density=True, color=color, alpha=0.55,
                edgecolor=color, linewidth=0.3)
        ax.axvline(nom,              color="#333333", lw=1.8, ls="--",
                   label=f"CGMF nominal = {nom:.4f}")
        ax.axvline(np.median(vals),  color=C_MEDIAN,  lw=1.6,
                   label=f"Sample median = {np.median(vals):.4f}")
        ax.axvline(np.percentile(vals, 16), color=color, lw=1.0, ls=":",
                   alpha=0.7)
        ax.axvline(np.percentile(vals, 84), color=color, lw=1.0, ls=":",
                   alpha=0.7, label="16/84% CI")

        cv = vals.std() / vals.mean() if vals.mean() > 0 else np.nan
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Weight value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.92)
        ax.text(0.97, 0.97, f"CV = {cv:.1%}\nstd = {vals.std():.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#5d6d7e", bbox=_ABOX)

    fig.suptitle(r"Plot 03 — Derived Weight Distributions  $w_1,\,w_2,\,w_3$",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Peak position scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_positions(deriv_vals, nom_deriv, A0, outpath):
    mu1s, mu2s = deriv_vals["mu1"], deriv_vals["mu2"]
    mu1n, mu2n = nom_deriv["mu1"], nom_deriv["mu2"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.32)

    # 1. mu1 vs mu2 scatter
    ax = axes[0]
    _panel_bg(ax)
    ax.scatter(mu1s, mu2s, s=2.5, alpha=0.20, color=C_ENVELOPE,
               rasterized=True)
    ax.scatter([mu1n], [mu2n], s=90, color=C_NOMINAL, zorder=10,
               marker="*", label="CGMF nominal")
    ax.scatter([np.median(mu1s)], [np.median(mu2s)], s=65, color=C_MEDIAN,
               zorder=11, marker="D", label="Sample median")
    ax.set_xlabel("$\\mu_1$  (heavy fragment peak 1)", fontsize=9)
    ax.set_ylabel("$\\mu_2$  (heavy fragment peak 2)", fontsize=9)
    ax.set_title("Peak position joint scatter", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.92)
    ax.text(0.97, 0.03,
            f"$\\mu_1$ mirror: {A0-np.median(mu1s):.1f}\n"
            f"$\\mu_2$ mirror: {A0-np.median(mu2s):.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#5d6d7e", bbox=_ABOX)

    # 2. mu1 marginal
    ax = axes[1]
    _panel_bg(ax)
    bins = np.linspace(mu1s.min(), mu1s.max(), 50)
    ax.hist(mu1s, bins=bins, density=True, color=C_W1, alpha=0.55,
            edgecolor=C_W1, linewidth=0.3)
    ax.axvline(mu1n, color="#333333", lw=1.8, ls="--",
               label=f"Nominal {mu1n:.2f}")
    ax.axvline(np.median(mu1s), color=C_MEDIAN, lw=1.5,
               label=f"Median {np.median(mu1s):.2f}")
    ax.set_xlabel("$\\mu_1$  [u]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("$\\mu_1$ marginal", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.92)

    # 3. mu2 marginal
    ax = axes[2]
    _panel_bg(ax)
    bins = np.linspace(mu2s.min(), mu2s.max(), 50)
    ax.hist(mu2s, bins=bins, density=True, color=C_W2, alpha=0.55,
            edgecolor=C_W2, linewidth=0.3)
    ax.axvline(mu2n, color="#333333", lw=1.8, ls="--",
               label=f"Nominal {mu2n:.2f}")
    ax.axvline(np.median(mu2s), color=C_MEDIAN, lw=1.5,
               label=f"Median {np.median(mu2s):.2f}")
    ax.set_xlabel("$\\mu_2$  [u]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("$\\mu_2$ marginal", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.92)

    fig.suptitle("Plot 04 — Heavy Fragment Peak Positions",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Peak width distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_widths(deriv_vals, nom_deriv, outpath):
    sig1s, sig2s, sig3s = (deriv_vals["sig1"], deriv_vals["sig2"],
                           deriv_vals["sig3"])
    sig1n, sig2n, sig3n = nom_deriv["sig1"], nom_deriv["sig2"], nom_deriv["sig3"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.32)

    for ax, vals, nom, color, label in [
        (axes[0], sig1s, sig1n, C_SIG1, "$\\sigma_1$  (peak 1 width) [u]"),
        (axes[1], sig2s, sig2n, C_SIG2, "$\\sigma_2$  (peak 2 width) [u]"),
        (axes[2], sig3s, sig3n, C_SIG3, "$\\sigma_3$  (valley width) [u]"),
    ]:
        _panel_bg(ax)
        ax.hist(vals, bins=55, density=True, color=color, alpha=0.55,
                edgecolor=color, linewidth=0.3)
        ax.axvline(nom,              color="#333333", lw=2.0, ls="--",
                   label=f"Nominal = {nom:.3f}")
        ax.axvline(np.median(vals),  color=C_MEDIAN,  lw=1.5,
                   label=f"Median = {np.median(vals):.3f}")
        ax.axvline(np.percentile(vals, 16), color=color, lw=1.0, ls=":",
                   alpha=0.7)
        ax.axvline(np.percentile(vals, 84), color=color, lw=1.0, ls=":",
                   alpha=0.7, label="16/84% CI")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("$\\sigma$  [u]", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.92)
        cv = vals.std() / vals.mean() if vals.mean() > 0 else np.nan
        ax.text(0.97, 0.97, f"CV = {cv:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#5d6d7e", bbox=_ABOX)

    fig.suptitle(r"Plot 05 — Peak Width Distributions  $\sigma_1,\,\sigma_2,\,\sigma_3$",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Percentile yield profile vs step-2 data
# ══════════════════════════════════════════════════════════════════════════════

def plot_yield_percentiles(A, Y_all, Y_nom, Y_data, sigma_data, outpath):
    ps = np.percentile(Y_all, [5, 16, 50, 84, 95], axis=0)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5),
                                      facecolor="#ffffff")
    fig.subplots_adjust(wspace=0.30)

    # Percentile line colours — light to dark (centre is boldest)
    pct_colors = ["#adc6e0", "#5b9ec9", C_ENVELOPE, "#5b9ec9", "#adc6e0"]
    lws        = [0.8, 1.2, 2.2, 1.2, 0.8]

    for ax, do_log in [(ax_l, True), (ax_r, False)]:
        _panel_bg(ax)
        ax.fill_between(A, ps[0], ps[4], alpha=0.10, color=C_ENVELOPE)
        ax.fill_between(A, ps[1], ps[3], alpha=0.25, color=C_ENVELOPE)

        for p, c, lw, ps_row in zip([5, 16, 50, 84, 95],
                                     pct_colors, lws, ps):
            ax.plot(A, ps_row, color=c, lw=lw, alpha=0.9)

        ax.plot(A, ps[2], color=C_ENVELOPE, lw=2.2, label="50th pctile")
        ax.plot(A, Y_nom, color=C_NOMINAL,  lw=1.5, ls="--",
                label="CGMF nominal", zorder=8)

        if Y_data is not None:
            ax.errorbar(A, Y_data, yerr=sigma_data, fmt="o", ms=3.0,
                        capsize=2.5, lw=0.7, color=C_DATA, alpha=0.85,
                        zorder=9, label="Step-2 data")
            frac_in = np.mean((Y_data >= ps[1]) & (Y_data <= ps[3]))
            ax.text(0.03, 0.97,
                    f"Data in 16–84% CI: {frac_in:.1%}\n(ideal $\\approx$ 68%)",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color="#5d6d7e", bbox=_ABOX)

        ax.set_xlabel("$A_{\\rm pre}$", fontsize=9)
        ax.set_ylabel("$Y_{\\rm pre}(A)$", fontsize=9)

        if do_log:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, ps[4].max() * 3)
            ax.set_title("Percentile bands vs data — log scale",
                         fontweight="bold", fontsize=11)
        else:
            ax.set_ylim(-0.002, ps[4].max() * 1.15)
            ax.set_title("Percentile bands vs data — linear scale",
                         fontweight="bold", fontsize=11)

        ax.legend(fontsize=8, framealpha=0.92)

    fig.suptitle("Plot 06 — Percentile Yield Profiles (5/16/50/84/95) vs Step-2",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Pairwise scatter of derived physical quantities
# ══════════════════════════════════════════════════════════════════════════════

def plot_parameter_scatter(deriv_vals, nom_deriv, outpath, n_show=1000):
    keys   = ["w1", "w2", "w3", "mu1", "mu2", "sig1", "sig2", "sig3"]
    labels = ["$w_1$", "$w_2$", "$w_3$",
              "$\\mu_1$", "$\\mu_2$",
              "$\\sigma_1$", "$\\sigma_2$", "$\\sigma_3$"]
    nk     = len(keys)
    n_show = min(n_show, len(deriv_vals["w1"]))
    idx    = np.random.choice(len(deriv_vals["w1"]), n_show, replace=False)

    data = np.column_stack([deriv_vals[k][idx] for k in keys])
    noms = np.array([nom_deriv[k] for k in keys])

    fig, axes = plt.subplots(nk, nk, figsize=(20, 20), facecolor="#ffffff")
    plt.subplots_adjust(hspace=0.06, wspace=0.06)

    for i in range(nk):
        for j in range(nk):
            ax = axes[i, j]
            ax.set_facecolor("#f8f9fb")
            for sp in ax.spines.values():
                sp.set_edgecolor("#cdd1d9"); sp.set_linewidth(0.5)

            if i == j:
                ax.hist(data[:, i], bins=35, color=C_ENVELOPE, alpha=0.55,
                        edgecolor=C_ENVELOPE, linewidth=0.2, density=True)
                ax.axvline(noms[i],              color=C_NOMINAL, lw=1.6, ls="--")
                ax.axvline(np.median(data[:, i]), color=C_MEDIAN,  lw=1.2)

            elif j < i:
                ax.scatter(data[:, j], data[:, i], s=1.8, alpha=0.18,
                           color=C_ENVELOPE, rasterized=True)
                ax.scatter([noms[j]], [noms[i]], s=55, color=C_NOMINAL,
                           zorder=10, marker="*")
                ax.scatter([np.median(data[:, j])],
                           [np.median(data[:, i])],
                           s=40, color=C_MEDIAN, zorder=11, marker="D")
            else:
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                # Colour-code: positive = warm, negative = cool, weak = grey
                if abs(r) < 0.30:
                    tc = "#888888"
                elif r > 0:
                    tc = C_MEDIAN    # red for positive correlation
                else:
                    tc = C_ENVELOPE  # blue for negative correlation
                ax.text(0.5, 0.5, f"$\\rho = {r:.2f}$",
                        ha="center", va="center", fontsize=10,
                        color=tc, transform=ax.transAxes, fontweight="bold")
                ax.set_facecolor("#f0f4f8")

            if i == nk - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Plot 07 — Pairwise Scatter of Derived Physical Quantities",
                 fontsize=13, fontweight="bold", y=1.002)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Physicality and normalisation checks
# ══════════════════════════════════════════════════════════════════════════════

def plot_physicality(thetas, yield_sums, phys_mask, w3s, outpath):
    n_total   = len(thetas)
    n_phys    = phys_mask.sum()
    frac_phys = n_phys / n_total

    fig = plt.figure(figsize=(16, 5.5), facecolor="#ffffff")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]

    # 1. Pie
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    wedge_colors = [C_NOMINAL, "#e74c3c"]
    ax.pie([n_phys, n_total - n_phys],
           labels=[f"Physical\n{frac_phys:.1%}",
                   f"Unphysical\n{1-frac_phys:.1%}"],
           colors=wedge_colors, autopct="%1.1f%%",
           textprops={"color": "#2c3e50", "fontsize": 9},
           startangle=90,
           wedgeprops={"edgecolor": "#ffffff", "linewidth": 2.0})
    ax.set_title("Physicality\ncheck", fontweight="bold", fontsize=10)

    # 2. Yield sum
    ax = axes[1]
    _panel_bg(ax)
    ys_phys = yield_sums[phys_mask]
    ax.hist(ys_phys, bins=50, color=C_ENVELOPE, alpha=0.55,
            edgecolor=C_ENVELOPE, linewidth=0.3, density=True)
    ax.axvline(2.0,              color="#333333", lw=2.0, ls="--",
               label="Target = 2.0")
    ax.axvline(np.median(ys_phys), color=C_MEDIAN,  lw=1.5,
               label=f"Median = {np.median(ys_phys):.4f}")
    ax.set_xlabel("$\\sum Y(A)$", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Yield sum\n($\\approx 2.0$ expected)", fontweight="bold",
                 fontsize=10)
    ax.legend(fontsize=7, framealpha=0.92)

    # 3. w3 distribution
    ax = axes[2]
    _panel_bg(ax)
    w3_phys = w3s[phys_mask]
    ax.hist(w3_phys, bins=50, color=C_W3, alpha=0.55,
            edgecolor=C_W3, linewidth=0.3, density=True)
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(w3_phys)
        xg  = np.linspace(w3_phys.min(), w3_phys.max(), 200)
        ax.plot(xg, kde(xg), color="#2c3e50", lw=1.2, alpha=0.7)
    except Exception:
        pass
    ax.axvline(0.00371, color="#333333", lw=1.8, ls="--",
               label="Nominal $w_3 = 0.0037$")
    ax.axvline(np.median(w3_phys), color=C_MEDIAN, lw=1.5,
               label=f"Median = {np.median(w3_phys):.4f}")
    ax.set_xlabel("$w_3$ (valley weight)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Valley weight $w_3$\n(nominal $\\approx 0.004$)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.92)

    # 4. w1+w2
    ax = axes[3]
    _panel_bg(ax)
    w12      = (w3s - 2.0) / (-2.0)
    w12_phys = w12[phys_mask]
    ax.hist(w12_phys, bins=50, color=C_W2, alpha=0.55,
            edgecolor=C_W2, linewidth=0.3, density=True)
    ax.axvline(0.998,              color="#333333", lw=1.8, ls="--",
               label="Nominal $w_1+w_2 = 0.998$")
    ax.axvline(np.median(w12_phys), color=C_MEDIAN, lw=1.5,
               label=f"Median = {np.median(w12_phys):.4f}")
    ax.set_xlabel("$w_1 + w_2$", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Total peak weight\n(nominal $\\approx 0.998$)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.92)

    fig.suptitle("Plot 08 — Physicality and Normalisation Checks",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — Pointwise uncertainty profile (coefficient of variation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_profile(A, Y_all, Y_data, sigma_data, outpath):
    Y_mean = Y_all.mean(axis=0)
    Y_std  = Y_all.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(Y_mean > 0, Y_std / Y_mean, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    sharex=True, facecolor="#ffffff")
    fig.subplots_adjust(hspace=0.28)

    # ── Top: mean ± std ──────────────────────────────────────────────────
    _panel_bg(ax1)
    ax1.fill_between(A, Y_mean - 2*Y_std, Y_mean + 2*Y_std,
                     alpha=0.12, color=C_ENVELOPE, label="Mean $\\pm 2\\sigma$")
    ax1.fill_between(A, Y_mean - Y_std,   Y_mean + Y_std,
                     alpha=0.28, color=C_ENVELOPE, label="Mean $\\pm 1\\sigma$")
    ax1.plot(A, Y_mean, color=C_ENVELOPE, lw=1.8, label="Mean $Y(A)$")

    if Y_data is not None:
        ax1.errorbar(A, Y_data, yerr=sigma_data, fmt="o", ms=3.0,
                     capsize=2.5, lw=0.7, color=C_DATA, alpha=0.85,
                     zorder=9, label="Step-2 data")
    ax1.set_yscale("log")
    pos = Y_mean[Y_mean > 0]
    ax1.set_ylim(max(pos.min() * 0.3, 1e-6), pos.max() * 3)
    ax1.set_ylabel("$Y_{\\rm pre}(A)$", fontsize=9)
    ax1.set_title("Mean $\\pm 1,2\\,\\sigma$ of sampled $Y(A)$",
                  fontweight="bold", fontsize=11)
    ax1.legend(fontsize=8, framealpha=0.92)

    # ── Bottom: CV ───────────────────────────────────────────────────────
    _panel_bg(ax2)
    ax2.fill_between(A, 0, cv * 100, alpha=0.22, color=C_SIG1)
    ax2.plot(A, cv * 100, color=C_SIG1, lw=2.0,
             label="CV = std/mean (%)")
    ax2.axhline(10, color="#888888", lw=0.9, ls="--", alpha=0.7,
                label="10% reference")
    ax2.axhline(50, color="#888888", lw=0.9, ls=":",  alpha=0.5,
                label="50% reference")
    ax2.set_ylabel("CV  [\\%]", fontsize=9)
    ax2.set_xlabel("$A_{\\rm pre}$", fontsize=9)
    ax2.set_title("Pointwise Coefficient of Variation (\\%)\n"
                  "Higher CV = greater model uncertainty at that mass number",
                  fontweight="bold", fontsize=11)
    ax2.legend(fontsize=8, framealpha=0.92)

    peak_mask   = Y_mean > np.percentile(Y_mean, 80)
    valley_mask = Y_mean < np.percentile(Y_mean, 20)
    if peak_mask.any() and valley_mask.any():
        ax2.text(0.97, 0.95,
                 f"Peak CV:   {cv[peak_mask].mean()*100:.1f}\\%\n"
                 f"Valley CV: {np.nanmean(cv[valley_mask])*100:.1f}\\%",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, color="#5d6d7e", bbox=_ABOX)

    fig.suptitle("Plot 09 — Pointwise Uncertainty Profile",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 — Summary dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(A, Y_all, Y_nom, Y_med, Y_data, sigma_data,
                           deriv_vals, nom_deriv, phys_frac, outpath):
    fig = plt.figure(figsize=(22, 14), facecolor="#ffffff")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.46, wspace=0.34)

    # ── Row 0: yield overview ─────────────────────────────────────────────
    ps = np.percentile(Y_all, [5, 16, 50, 84, 95], axis=0)

    ax_yld = fig.add_subplot(gs[0, :2])
    _panel_bg(ax_yld)
    ax_yld.fill_between(A, ps[0], ps[4], alpha=0.10, color=C_ENVELOPE)
    ax_yld.fill_between(A, ps[1], ps[3], alpha=0.25, color=C_ENVELOPE,
                        label="16–84% CI")
    ax_yld.plot(A, ps[2], color=C_ENVELOPE, lw=2.0, label="Sampled median")
    ax_yld.plot(A, Y_nom, color=C_NOMINAL,  lw=1.5, ls="--",
                label="CGMF nominal")
    ax_yld.plot(A, Y_med, color=C_MEDIAN,   lw=1.8, label="MCMC median")
    if Y_data is not None:
        ax_yld.errorbar(A, Y_data, yerr=sigma_data, fmt="o", ms=2.5,
                        capsize=1.5, lw=0.6, color=C_DATA, alpha=0.8,
                        zorder=9, label="Step-2 data")
    ax_yld.set_yscale("log")
    ax_yld.set_ylim(1e-6, ps[4].max() * 3)
    ax_yld.set_xlabel("$A$"); ax_yld.set_ylabel("$Y(A)$")
    ax_yld.set_title("Yield envelope (log scale)", fontweight="bold", fontsize=11)
    ax_yld.legend(fontsize=7, framealpha=0.92, ncol=2)

    ax_yl2 = fig.add_subplot(gs[0, 2:])
    _panel_bg(ax_yl2)
    ax_yl2.fill_between(A, ps[1], ps[3], alpha=0.28, color=C_ENVELOPE)
    ax_yl2.plot(A, ps[2], color=C_ENVELOPE, lw=2.0)
    ax_yl2.plot(A, Y_nom, color=C_NOMINAL,  lw=1.5, ls="--")
    ax_yl2.plot(A, Y_med, color=C_MEDIAN,   lw=1.8)
    if Y_data is not None:
        ax_yl2.errorbar(A, Y_data, yerr=sigma_data, fmt="o", ms=2.5,
                        capsize=1.5, lw=0.6, color=C_DATA, alpha=0.8, zorder=9)
    ax_yl2.set_ylim(-0.002, ps[4].max() * 1.15)
    ax_yl2.set_xlabel("$A$"); ax_yl2.set_ylabel("$Y(A)$")
    ax_yl2.set_title("Yield envelope (linear scale)", fontweight="bold",
                     fontsize=11)

    # ── Row 1: weights + CV ───────────────────────────────────────────────
    for ki, (key, color, lbl) in enumerate([
        ("w1", C_W1, "$w_1$"),
        ("w2", C_W2, "$w_2$"),
        ("w3", C_W3, "$w_3$"),
    ]):
        ax = fig.add_subplot(gs[1, ki])
        _panel_bg(ax)
        vals = deriv_vals[key]
        ax.hist(vals, bins=45, density=True, color=color, alpha=0.55,
                edgecolor=color, linewidth=0.3)
        ax.axvline(nom_deriv[key],  color="#333333", lw=1.8, ls="--",
                   label=f"Nom={nom_deriv[key]:.4f}")
        ax.axvline(np.median(vals), color=C_MEDIAN,  lw=1.3,
                   label=f"Med={np.median(vals):.4f}")
        ax.set_title(lbl, fontweight="bold", fontsize=11)
        ax.set_xlabel("Weight"); ax.set_ylabel("Density")
        ax.legend(fontsize=7, framealpha=0.92)

    Y_mean = Y_all.mean(axis=0); Y_std = Y_all.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(Y_mean > 0, Y_std / Y_mean * 100, np.nan)
    ax_cv = fig.add_subplot(gs[1, 3])
    _panel_bg(ax_cv)
    ax_cv.fill_between(A, 0, cv, alpha=0.20, color=C_SIG1)
    ax_cv.plot(A, cv, color=C_SIG1, lw=1.8)
    ax_cv.axhline(10, color="#888888", lw=0.8, ls="--", alpha=0.6)
    ax_cv.set_xlabel("$A$"); ax_cv.set_ylabel("CV  [\\%]")
    ax_cv.set_title("Pointwise CV (std/mean)", fontweight="bold", fontsize=11)

    # ── Row 2: positions + widths ─────────────────────────────────────────
    ax_mu = fig.add_subplot(gs[2, 0])
    _panel_bg(ax_mu)
    ax_mu.scatter(deriv_vals["mu1"], deriv_vals["mu2"],
                  s=1.8, alpha=0.15, color=C_ENVELOPE, rasterized=True)
    ax_mu.scatter([nom_deriv["mu1"]], [nom_deriv["mu2"]],
                  s=80, color=C_NOMINAL, marker="*", zorder=10,
                  label="Nominal")
    ax_mu.scatter([np.median(deriv_vals["mu1"])],
                  [np.median(deriv_vals["mu2"])],
                  s=60, color=C_MEDIAN, marker="D", zorder=11,
                  label="Median")
    ax_mu.set_xlabel("$\\mu_1$  [u]"); ax_mu.set_ylabel("$\\mu_2$  [u]")
    ax_mu.set_title("Peak positions $\\mu_1$ vs $\\mu_2$",
                    fontweight="bold", fontsize=11)
    ax_mu.legend(fontsize=7, framealpha=0.92)

    for ki, (key, color, lbl) in enumerate([
        ("sig1", C_SIG1, "$\\sigma_1$"),
        ("sig2", C_SIG2, "$\\sigma_2$"),
        ("sig3", C_SIG3, "$\\sigma_3$"),
    ]):
        ax = fig.add_subplot(gs[2, ki + 1])
        _panel_bg(ax)
        vals = deriv_vals[key]
        ax.hist(vals, bins=45, density=True, color=color, alpha=0.55,
                edgecolor=color, linewidth=0.3)
        ax.axvline(nom_deriv[key],  color="#333333", lw=1.8, ls="--",
                   label=f"Nom={nom_deriv[key]:.3f}")
        ax.axvline(np.median(vals), color=C_MEDIAN,  lw=1.3,
                   label=f"Med={np.median(vals):.3f}")
        ax.set_title(f"{lbl}  [u]", fontweight="bold", fontsize=11)
        ax.set_xlabel("$\\sigma$  [u]"); ax.set_ylabel("Density")
        ax.legend(fontsize=7, framealpha=0.92)

    # Stats text box
    n = len(deriv_vals["w1"])
    stats_txt = (
        f"N samples:  {n}\n"
        f"Physical:   {phys_frac:.1%}\n"
        f"w₁ median:  {np.median(deriv_vals['w1']):.4f}\n"
        f"w₂ median:  {np.median(deriv_vals['w2']):.4f}\n"
        f"w₃ median:  {np.median(deriv_vals['w3']):.5f}\n"
        f"μ₁ median:  {np.median(deriv_vals['mu1']):.2f} u\n"
        f"μ₂ median:  {np.median(deriv_vals['mu2']):.2f} u\n"
        f"σ₁ median:  {np.median(deriv_vals['sig1']):.3f} u\n"
        f"σ₂ median:  {np.median(deriv_vals['sig2']):.3f} u\n"
        f"σ₃ median:  {np.median(deriv_vals['sig3']):.3f} u"
    )
    fig.text(0.985, 0.02, stats_txt, ha="right", va="bottom",
             fontsize=8, color="#5d6d7e", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                       edgecolor="#cdd1d9", alpha=0.95))

    fig.suptitle("Plot 10 — Online HPC Sampling: Summary Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    _save(fig, outpath)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Step-4: online HPC sampling diagnostics from step-3 NPZ.")
    ap.add_argument("--npz",    required=True,
                    help="Path to step3_mcmc.npz")
    ap.add_argument("--A0",     type=int,   required=True)
    ap.add_argument("--En",     type=float, default=2.53e-8)
    ap.add_argument("--N",      type=int,   default=2000,
                    help="Number of samples to draw (default 2000)")
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--outdir", default="sampling_diag")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)

    # ── Load NPZ ─────────────────────────────────────────────────────────
    section("Loading step-3 MCMC output")
    if not os.path.exists(args.npz):
        sys.exit(f"ERROR: {args.npz} not found")

    z = np.load(args.npz, allow_pickle=True)
    print(f"  Keys: {list(z.keys())}")

    gauss_mu      = z["gauss_mu"].astype(float)
    gauss_chol    = z["gauss_chol"].astype(float)
    A_pre         = z["A_pre"].astype(int)
    Y_data        = z.get("Y_pre_data",           None)
    sigma_data    = z.get("sigma_pre_data",        None)
    Y_nom_stored  = z.get("Y_pre_nominal_model",   None)
    Y_med_stored  = z.get("Y_pre_median_model",    None)
    theta_nom     = z.get("theta_nominal",         None)
    theta_med     = z.get("theta_median",          None)

    if Y_data     is not None: Y_data     = Y_data.astype(float)
    if sigma_data is not None: sigma_data = sigma_data.astype(float)

    print(f"  gauss_mu shape:   {gauss_mu.shape}")
    print(f"  gauss_chol shape: {gauss_chol.shape}")
    print(f"  A_pre: {A_pre.min()}–{A_pre.max()}  ({len(A_pre)} bins)")
    if Y_data is not None:
        print(f"  Y_data: min={Y_data.min():.3e}  max={Y_data.max():.3e}"
              f"  sum={Y_data.sum():.4f}")

    # ── Draw samples (exactly as HPC will do) ────────────────────────────
    section(f"Drawing {args.N} samples from N(μ, Σ)")
    print(f"  Recipe:  z = randn({args.N}, 14)")
    print(f"           thetas = gauss_mu + z @ gauss_chol.T")

    thetas = draw_samples(gauss_mu, gauss_chol, args.N, rng)
    print(f"  thetas shape: {thetas.shape}")

    # ── Physicality check ─────────────────────────────────────────────────
    section("Physicality check")
    phys_mask  = np.array([is_physical(t, args.En) for t in thetas])
    n_phys     = phys_mask.sum()
    phys_frac  = n_phys / args.N
    print(f"  Physical samples: {n_phys}/{args.N}  ({phys_frac:.1%})")
    if phys_frac < 0.80:
        print("  *** WARNING: less than 80% of samples are physical.")
        print("  Consider using a tighter prior_scale or re-running step-3.")

    thetas_phys = thetas[phys_mask]
    print(f"  Using {len(thetas_phys)} physical samples for yield evaluation.")

    # ── Evaluate Y(A) ─────────────────────────────────────────────────────
    section("Evaluating Y(A) for all physical samples")
    Y_all      = np.zeros((len(thetas_phys), len(A_pre)))
    yield_sums = np.zeros(len(thetas_phys))
    for i, th in enumerate(thetas_phys):
        try:
            Y_all[i]      = model_yield(th, A_pre, args.En, args.A0)
            yield_sums[i] = Y_all[i].sum()
        except Exception:
            Y_all[i]      = np.nan
            yield_sums[i] = np.nan
    valid      = np.all(np.isfinite(Y_all), axis=1)
    Y_all      = Y_all[valid]
    yield_sums = yield_sums[valid]
    thetas_v   = thetas_phys[valid]
    print(f"  Valid yield evaluations: {valid.sum()}/{len(thetas_phys)}")
    print(f"  Yield sum: mean={yield_sums.mean():.5f}  "
          f"std={yield_sums.std():.5f}")
    print(f"  Y max:     {Y_all.max(axis=1).mean():.5f} "
          f"± {Y_all.max(axis=1).std():.5f}")

    # ── Nominal / median curves ───────────────────────────────────────────
    if Y_nom_stored is not None:
        Y_nom = Y_nom_stored.astype(float)
    elif theta_nom is not None:
        Y_nom = model_yield(theta_nom.astype(float), A_pre, args.En, args.A0)
    else:
        Y_nom = model_yield(gauss_mu, A_pre, args.En, args.A0)

    if Y_med_stored is not None:
        Y_med = Y_med_stored.astype(float)
    elif theta_med is not None:
        Y_med = model_yield(theta_med.astype(float), A_pre, args.En, args.A0)
    else:
        Y_med = np.median(Y_all, axis=0)

    # ── Derived physical quantities ───────────────────────────────────────
    section("Computing derived physical quantities")
    d_all = np.array([derived(t, args.En, args.A0) for t in thetas_v])
    deriv_vals = {
        "w1": d_all[:, 0], "w2": d_all[:, 1], "w3": d_all[:, 2],
        "mu1": d_all[:, 3], "mu2": d_all[:, 4],
        "sig1": d_all[:, 5], "sig2": d_all[:, 6], "sig3": d_all[:, 7],
    }

    if theta_nom is not None:
        d_nom = derived(theta_nom.astype(float), args.En, args.A0)
    else:
        d_nom = derived(gauss_mu, args.En, args.A0)
    nom_deriv = dict(zip(
        ["w1", "w2", "w3", "mu1", "mu2", "sig1", "sig2", "sig3"], d_nom))

    print(f"\n  Sampled derived quantities "
          f"(physical samples, N={len(thetas_v)}):")
    print(f"  {'Qty':8s}  {'Nominal':>9s}  {'Median':>9s}  {'Std':>9s}  "
          f"{'16%':>9s}  {'84%':>9s}  {'CV':>7s}")
    print(f"  {'-'*68}")
    for key in ["w1","w2","w3","mu1","mu2","sig1","sig2","sig3"]:
        v   = deriv_vals[key]
        nom = nom_deriv[key]
        cv  = v.std() / v.mean() if v.mean() > 0 else np.nan
        print(f"  {key:8s}  {nom:>9.4f}  {np.median(v):>9.4f}  "
              f"{v.std():>9.4f}  {np.percentile(v,16):>9.4f}  "
              f"{np.percentile(v,84):>9.4f}  {cv:>7.1%}")

    # ── Coverage check ────────────────────────────────────────────────────
    if Y_data is not None and sigma_data is not None:
        section("Coverage check vs step-2 data")
        ps = np.percentile(Y_all, [16, 84], axis=0)
        frac_in   = np.mean((Y_data >= ps[0]) & (Y_data <= ps[1]))
        peak_mask = Y_data > np.percentile(Y_data, 75)
        frac_peak = np.mean((Y_data[peak_mask] >= ps[0][peak_mask]) &
                            (Y_data[peak_mask] <= ps[1][peak_mask]))
        print(f"  Fraction of data bins inside 16–84% CI:")
        print(f"    All bins:   {frac_in:.1%}  (ideal ≈ 68%)")
        print(f"    Peak bins:  {frac_peak:.1%}  (should be ≥ 50%)")
        if frac_in < 0.40:
            print("  NOTE: low coverage — posterior may be too narrow or model")
            print("  may not be expressive enough for these data.")
        elif frac_in > 0.90:
            print("  NOTE: very high coverage — posterior may be too wide.")

    # ── Generate plots ─────────────────────────────────────────────────────
    section("Generating diagnostic plots")

    plot_yield_envelope(
        A_pre, Y_all, Y_nom, Y_med, Y_data, sigma_data,
        os.path.join(args.outdir, "01_yield_envelope.png"))

    plot_peak_coverage(
        A_pre, Y_all, Y_nom, Y_med, Y_data, sigma_data, args.A0,
        os.path.join(args.outdir, "02_peak_coverage.png"))

    plot_weight_distributions(
        deriv_vals, nom_deriv,
        os.path.join(args.outdir, "03_weight_distribution.png"))

    plot_peak_positions(
        deriv_vals, nom_deriv, args.A0,
        os.path.join(args.outdir, "04_peak_positions.png"))

    plot_peak_widths(
        deriv_vals, nom_deriv,
        os.path.join(args.outdir, "05_peak_widths.png"))

    plot_yield_percentiles(
        A_pre, Y_all, Y_nom, Y_data, sigma_data,
        os.path.join(args.outdir, "06_yield_percentiles.png"))

    plot_parameter_scatter(
        deriv_vals, nom_deriv,
        os.path.join(args.outdir, "07_parameter_scatter.png"))

    plot_physicality(
        thetas, yield_sums,
        np.array([is_physical(t, args.En) for t in thetas])[phys_mask][valid],
        deriv_vals["w3"],
        os.path.join(args.outdir, "08_physicality_check.png"))

    plot_uncertainty_profile(
        A_pre, Y_all, Y_data, sigma_data,
        os.path.join(args.outdir, "09_uncertainty_profile.png"))

    plot_summary_dashboard(
        A_pre, Y_all, Y_nom, Y_med, Y_data, sigma_data,
        deriv_vals, nom_deriv, phys_frac,
        os.path.join(args.outdir, "10_summary_dashboard.png"))

    section("STEP 4 COMPLETE")
    print(f"  Output dir: {args.outdir}")
    print(f"\n  10 diagnostic plots — check in this order:")
    print(f"    10_summary_dashboard.png    — START HERE: single-page overview")
    print(f"    01_yield_envelope.png       — full Y(A) band vs data (log+linear)")
    print(f"    02_peak_coverage.png        — light / valley / heavy regions")
    print(f"    06_yield_percentiles.png    — 5/16/50/84/95 pctile vs data")
    print(f"    09_uncertainty_profile.png  — pointwise CV")
    print(f"    03_weight_distribution.png  — w1, w2, w3 marginals")
    print(f"    08_physicality_check.png    — physical fraction + yield sums")
    print(f"    04_peak_positions.png       — μ1 vs μ2 joint scatter")
    print(f"    05_peak_widths.png          — σ1, σ2, σ3 marginals")
    print(f"    07_parameter_scatter.png    — full pairwise scatter")
    print(f"\n  Key health metrics:")
    print(f"    Physical fraction:  {phys_frac:.1%}   (target > 90%)")
    frac_cv = np.nanmean(
        np.where(Y_all.mean(0) > 0,
                 Y_all.std(0) / Y_all.mean(0), np.nan)
    )
    print(f"    Mean pointwise CV:  {frac_cv:.1%}   (5–20% is healthy)")
    print(f"    Yield sum std:      {yield_sums.std():.4f}  (target < 0.01)")
    print(SEPARATOR)


if __name__ == "__main__":
    main()