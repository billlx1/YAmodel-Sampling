# Tikhonov Regularisation

## The Problem: Ill-Posed Inversion

You have measured post-neutron emission fragment mass yields Y_post(A), and you want to recover the pre-neutron yields Y_pre(A) — the physically meaningful quantity. These are related by:

**Y_post = R · Y_pre**

where R is the response matrix from CGMF, with R[i,j] = P(A_post = j | A_pre = i). Inverting this directly is catastrophic: R is nearly singular (many pre-fragment masses map to similar post-fragment masses after emitting 1–3 neutrons), so tiny measurement uncertainties in Y_post get amplified into enormous, oscillating noise in the naive solution. This is an **ill-posed inverse problem**.

## The Fix: Add a Smoothness Penalty

Tikhonov regularisation replaces the naive least-squares minimisation:

**min ||W(R·y − d)||²**

with a penalised version:

**min ||W(R·y − d)||² + λ² ||Ly||²**

The first term demands the solution fits the data; the second penalises solutions that are physically unreasonable. Here **W = diag(1/σ)** weights by measurement uncertainty, and **L** is the second-difference operator (a discrete second derivative), so **||Ly||²** penalises rapid oscillation — enforcing smoothness. The scalar **λ** controls the trade-off: large λ = very smooth but potentially biased; small λ = fits data closely but oscillates wildly.

## Choosing λ: Three Methods

The script scans ~120 values of λ logarithmically and uses three criteria:

- **L-curve corner**: Plot log(residual norm) vs log(seminorm ||Ly||). The curve bends at an "elbow" — the corner balances fit quality against smoothness. The script finds this via maximum curvature in log-log space.
- **GCV (Generalised Cross-Validation)**: Minimises a leave-one-out prediction error estimate — purely data-driven, no geometric judgement needed.
- **Discrepancy principle**: Choose λ so the residual norm equals √m (where m = number of data points) — i.e. the fit is statistically consistent with the measurement uncertainties.

If L-curve and GCV agree within 1.5 decades, the corner is used; otherwise GCV is preferred.

## Uncertainty Propagation

The script propagates **two independent error sources** into a full covariance matrix on Y_pre:

1. **C_data**: From measurement uncertainties σ on Y_post — via the standard sandwich formula M⁻¹(AᵀWA)M⁻¹, where M = AᵀWA + λ²LᵀL.
2. **C_R**: From finite CGMF Monte Carlo statistics in R — each row of R is multinomially distributed, so its uncertainty propagates via first-order sensitivity analysis into additional covariance on Y_pre.

The total covariance **C_total = C_data + C_R** is saved alongside its Cholesky factorisation, ready for use as a correlated Gaussian likelihood in downstream MCMC sampling.


## Implementation
This script uses functions from the scipy Python package.
Specifically:
- scipy.linalg
- scipy.optimize.lsq_linear

## References
- Tikhonov Regularization and Total Least Squares, G. H. GOLUB et al.
  Accessed: https://www.cs.umd.edu/users/oleary/reprints/j51.pdf
  
- Wikipiedia entry: https://en.wikipedia.org/wiki/Ridge_regression
