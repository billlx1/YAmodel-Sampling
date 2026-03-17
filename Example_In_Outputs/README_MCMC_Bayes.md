# MCMC and Bayesian Inference
---

## Bayesian Inference: The General Framework

In classical (frequentist) statistics, a parameter is treated as a fixed unknown quantity and uncertainty is described through the properties of repeated experiments. **Bayesian inference** takes a different view: parameters are treated as random variables with probability distributions that encode our state of knowledge. You begin with a **prior distribution** P(θ) — a quantitative statement of what parameter values seem plausible before seeing any data — and update it in light of measurements to obtain a **posterior distribution** P(θ | data).

The update is performed via **Bayes' theorem**:

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta)\, P(\theta)}{P(\text{data})}$$

Each term has a clear role:

- **P(θ | data)** — the *posterior*: what we believe about θ after seeing the data.
- **P(data | θ)** — the *likelihood*: how probable the observed data are for a given θ. This is conceptually the same quantity minimised in a standard χ² fit.
- **P(θ)** — the *prior*: encodes constraints known before the experiment (e.g. peak widths must be positive, weights must sum to 2).
- **P(data)** — the *evidence*: a normalisation constant, independent of θ, which we rarely need to compute explicitly.

The conceptual shift from frequentist fitting is that rather than finding *the* best θ, we seek the full posterior — a probability *distribution* over all plausible parameter values. This is especially powerful when parameters are correlated or poorly constrained, because the posterior naturally captures the full joint uncertainty structure and can be propagated directly into downstream simulations.

---

## Why Not Just Use Least-Squares or a Grid Search?

For a physicist more familiar with χ² minimisation or least-squares fitting, it is worth being explicit about why those approaches fall short here and why MCMC is preferred.

**Least-squares / maximum-likelihood** finds the single point θ\* that maximises the likelihood (equivalently, minimises χ²). Parameter uncertainties can be estimated from the curvature of χ² at the minimum (the Hessian matrix), giving a Gaussian approximation to the posterior. This is fast and familiar, but it only works well when the posterior is approximately Gaussian and unimodal near the minimum. It gives no information about non-Gaussian tails, multimodality, or strong parameter correlations that deviate from simple ellipses. Critically, it produces a point estimate with error bars — not a full distribution — which is insufficient when the goal is to propagate correlated uncertainty through a downstream simulation.

**Grid search** evaluates the posterior on a regular lattice in parameter space. For *D* parameters with *n* grid points per axis, this requires *n^D* evaluations. Here D = 14. Even at a coarse n = 10, that is 10¹⁴ evaluations — computationally impossible.

**MCMC** sidesteps both problems. It makes no assumption that the posterior is Gaussian, handles arbitrarily complex correlation structures, and scales as O(D) per step rather than O(n^D). It concentrates sampling effort automatically in the high-probability regions of parameter space, making it the standard tool for Bayesian inference in high-dimensional problems across particle physics, cosmology, and nuclear physics alike.

---

## MCMC: The Core Idea

**Markov Chain Monte Carlo** constructs a random walk through parameter space whose **stationary distribution** — the distribution the walk converges to and samples from after a sufficient number of steps — is exactly the posterior P(θ | data). The "Markov" property means each proposed step depends only on the current position, not the full history of the walk. The "Monte Carlo" part refers to the use of random sampling to explore the space.

The key insight is that you never need to compute the normalisation constant P(data), which would require integrating the likelihood over all of parameter space. At every step you only compute the *ratio* of posterior densities at two points:

$$r = \frac{P(\theta' \mid \text{data})}{P(\theta \mid \text{data})} = \frac{P(\text{data} \mid \theta')\, P(\theta')}{P(\text{data} \mid \theta)\, P(\theta)}$$

Since P(data) appears in both numerator and denominator, it cancels exactly.

### Walking Through the Chain: Metropolis–Hastings

The foundational algorithm is **Metropolis–Hastings**. Starting from an initial parameter vector θ⁰ (typically the maximum-likelihood estimate or a physically reasonable guess), at each iteration *t*:

1. **Propose** a new point θ' by drawing from a *proposal distribution* Q(θ' | θᵗ), typically a multivariate Gaussian centred on the current position: θ' = θᵗ + ε, where ε ~ N(0, Σ_prop).

2. **Evaluate the log-posterior** at both the current point and the proposal. For this model, the log-posterior is:

$$\ln P(\theta \mid \text{data}) = \ln P(\text{data} \mid \theta) + \ln P(\theta)$$

3. **Compute the acceptance ratio**:

$$\alpha = \min\!\left(1,\, \frac{P(\text{data} \mid \theta')\, P(\theta')}{P(\text{data} \mid \theta^t)\, P(\theta^t)}\right)$$

4. **Accept or reject**: draw u ~ Uniform(0, 1). If u < α, set θᵗ⁺¹ = θ'; otherwise θᵗ⁺¹ = θᵗ (the chain stays put).

This rule guarantees that moves to higher-probability regions are always accepted, while moves to lower-probability regions are accepted stochastically with a probability proportional to how much lower they are. The chain therefore spends time in each region proportional to the posterior probability there — which is precisely what it means to sample from the posterior.

In practice, the chain must first run through a **burn-in** phase (typically hundreds to thousands of steps) during which it migrates from the arbitrary starting point toward the high-probability region. Burn-in samples are discarded; only the subsequent **production** samples are retained.

### How the Likelihood and Prior Are Evaluated

**Likelihood.** The step-2 output does not provide simple diagonal uncertainties; it supplies a full inverse-covariance Cholesky factor `Cov_inv_chol` (computed from the step-2 uncertainty model) which encodes correlations between mass bins. The log-likelihood is therefore a full-covariance Gaussian:

$$\ln P(\text{data} \mid \theta) = -\frac{1}{2} \left\| \mathbf{L}^{-1}\left(Y_{\text{model}}(\theta) - Y_{\text{data}}\right) \right\|^2$$

where **L** is the lower Cholesky factor of the data covariance matrix, passed in directly from step 2. In practice this is computed as `scipy.linalg.solve_triangular(Cov_inv_chol, residual, lower=True)`, avoiding any explicit matrix inversion. This is more general than the diagonal form −½Σ[(Y_model − Y_data)²/σ²], which is recovered only if the covariance matrix is diagonal.

**Prior.** The prior is a multivariate Gaussian centred on the CGMF nominal parameter values, with per-parameter widths `_prior_sigma` set to reflect physical uncertainty in each parameter. This is combined with hard box bounds (e.g. widths must be positive, peak positions must lie in the heavy-fragment half of the mass distribution, μ₁ < μ₂ to prevent peak-swap degeneracy). Any θ outside the box bounds or failing the physicality check `check_physicality()` is assigned ln P(θ) = −∞ and always rejected.

### Ensemble Sampling with emcee

Single-walker Metropolis–Hastings mixes slowly when parameters are strongly correlated, because an isotropic Gaussian proposal will frequently step across the grain of a tilted, elongated posterior. The `emcee` package implements an **ensemble sampler** (the affine-invariant method of Goodman & Weare, 2010), which runs *N_walkers* parallel chains simultaneously. Each walker proposes moves by stretching along the vector connecting it to a randomly chosen partner walker:

$$\theta'_k = \theta_j + z\,(\theta_k - \theta_j), \quad z \sim p(z) \propto z^{-1/2}$$

Because proposals are constructed from the ensemble itself, they automatically align with the correlation structure of the posterior without any manual tuning of proposal widths. This makes `emcee` far more efficient than standard Metropolis–Hastings for correlated, multi-parameter posteriors like this one.

---

## Gaussian Approximation and Online HPC Sampling

Running a full MCMC chain online during each HPC simulation event would be prohibitively expensive. Step 3 therefore runs the MCMC **once offline**, then summarises the resulting posterior samples with a 14-dimensional **multivariate Gaussian** N(μ, Σ). This approximation is valid when the posterior is roughly unimodal and symmetric — a reasonable assumption once the data are sufficiently informative. Its adequacy is checked explicitly by `plot_gaussian_fit_check()`, which overlays MCMC marginals against draws from N(μ, Σ).

The Gaussian is stored not as the full covariance matrix Σ but as its **Cholesky factor** L, where Σ = LLᵀ — a lower-triangular matrix encoding all pairwise parameter correlations compactly. Drawing a new sample θ from N(μ, Σ) then reduces to:

$$\theta = \mu + L\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is a single matrix–vector multiply — microseconds of compute — compared to the thousands of likelihood evaluations a full MCMC chain would require. `step3_mcmc.npz` stores `gauss_mu` (μ) and `gauss_chol` (L), and the online sampler reproduces this via:

```python
z      = rng.standard_normal((N, 14))
thetas = gauss_mu + z @ gauss_chol.T
```

---

## Postscript: Step 3 Implementation Details

### Packages

| Package | Role |
|---|---|
| `emcee` | Ensemble MCMC sampler |
| `scipy.linalg` | Cholesky decomposition and triangular solves |
| `scipy.optimize.minimize` | Nelder-Mead MAP search used to initialise walkers |
| `numpy` | All numerical operations, chain statistics |
| `corner` (optional) | Bivariate posterior corner plot |
| `matplotlib` | All diagnostic figures |

### Sequence of Operations

**1. Load step-2 data** (`numpy.load`). Reads `A_pre`, `Y_pre`, `sigma_pre`, and critically `Cov_inv_chol` — the Cholesky factor of the inverse data covariance matrix computed in step 2 and passed forward here. This is what makes the likelihood a full-covariance Gaussian rather than a simple χ² sum.

**2. Construct prior and likelihood objects.** The `Prior` class stores the CGMF nominal values as prior centres, the per-parameter widths `_prior_sigma`, and hard box bounds. Its `ln_prior()` method returns −∞ for any parameter vector outside the bounds or failing `check_physicality()`, and otherwise returns the Gaussian log-prior. The `Likelihood` class wraps `scipy.linalg.solve_triangular(Cov_inv_chol, residual, lower=True)` to evaluate the log-likelihood without inverting the covariance matrix explicitly. `make_log_posterior()` combines them into a single callable `ln_posterior(θ) = ln_prior(θ) + ln_likelihood(θ)`.

**3. MAP search** (`scipy.optimize.minimize`, method `Nelder-Mead`). Two-pass Nelder-Mead minimisation of `−ln_posterior`, starting from the CGMF nominal. This locates the posterior mode efficiently and provides a physically validated starting point for the walkers, avoiding the burn-in overhead of initialising from the prior.

**4. Walker initialisation** (`initialise_walkers`). All 96 walkers are placed in a tight ball of radius 0.5% around the MAP estimate (`scale=0.005`). Each candidate position is accepted only if `ln_posterior` is finite. This ensures every walker starts in the physical, double-hump region of parameter space rather than an unphysical local minimum.

**5. Run emcee** (`emcee.EnsembleSampler`, `sampler.run_mcmc`). The sampler is run in two phases: burn-in (default 3000 steps, then `sampler.reset()` to discard) followed by production (default 5000 steps). The separation via `sampler.get_last_sample()` and `sampler.reset()` is important — it ensures the stored chain contains only post-convergence samples.

**6. Extract samples** (`sampler.get_chain(flat=True, thin=10)`, `sampler.get_log_prob(flat=True, thin=10)`). Thinning by a factor of 10 reduces autocorrelation between stored samples. The flat chain has shape (N_walkers × N_production / thin, 14).

**7. Convergence diagnostics.** Three metrics are computed:
- `sampler.acceptance_fraction` — per-walker acceptance rate; target 0.2–0.5.
- `sampler.get_autocorr_time()` — integrated autocorrelation time τ per parameter; the production run should satisfy N_prod/τ > 50 for reliable estimates.
- Gelman-Rubin R̂ (`gelman_rubin()`) — compares within-chain and between-chain variance; R̂ < 1.1 indicates convergence. Computed by splitting each walker's chain in half and treating the halves as independent chains.

**8. Fit multivariate Gaussian.** `mu_g = flat_chain.mean(axis=0)` and `Sigma_g = numpy.cov(flat_chain.T)` compute the sample mean and covariance of the posterior. `scipy.linalg.cholesky(Sigma_g, lower=True)` produces the Cholesky factor. If Σ is not positive-definite (numerical issues with near-degenerate parameters), a small diagonal jitter is added before re-attempting the decomposition. Skewness and excess kurtosis are checked per parameter to flag any for which the Gaussian approximation may be poor.

**9. Save to NPZ.** `numpy.savez` writes the flat chain, point estimates, posterior predictive yields, convergence statistics, and the Gaussian approximation (mean vector and Cholesky factor) to `step3_mcmc.npz` for use by step 4.
