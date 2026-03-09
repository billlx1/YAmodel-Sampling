## `step1_output.npz`

This file contains three NumPy arrays representing the **marginalised mass yield distribution**.

### Contents

| Array Name | Type    | Shape | Description |
|-------------|--------|-------|-------------|
| `A` | `float64` | `(N,)` | Mass numbers of fission fragments |
| `Y` | `float64` | `(N,)` | Marginal mass yields \(Y(A)\) |
| `sigma` | `float64` | `(N,)` | One-sigma uncertainties \(σ(A)\) |

### Notes

- **N** = number of unique mass numbers present in the dataset.
- All arrays are **aligned by index**, meaning each position refers to the same mass number.


## `{PREFIX}_matrices.json`

This file stores the **fragment mass transition probability matrices and metadata** in a portable JSON format.

It contains the probability matrix

\[
P(A_{post} \mid A_{pre})
\]

which represents the probability that a fragment with **pre-neutron emission mass** \(A_{pre}\) is observed with **post-neutron emission mass** \(A_{post}\).

The file also contains the **raw count matrix**, **relative statistical errors**, and **metadata describing the simulation inputs**.

---

### JSON Structure

| Key | Type | Description |
|----|----|----|
| `metadata` | object | Information about the dataset, source files, and generation time |
| `axes` | object | Defines the mass number axes used by the matrices |
| `count_matrix` | object | Raw fragment counts \(N(A_{pre} \rightarrow A_{post})\) |
| `probability_matrix` | object | Conditional probability matrix \(P(A_{post} \mid A_{pre})\) |
| `relative_error_matrix` | object | Relative statistical errors derived from Poisson counting |

---

### 1. Metadata

| Field | Type | Description |
|------|------|-------------|
| `timestamp` | string | ISO timestamp when the file was generated |
| `source_files` | array[string] | CGMF history files used to construct the matrix |
| `total_fission_events` | integer | Total number of fission events processed |
| `total_fragments_analysed` | integer | Total number of fragments analysed |
| `description` | string | Text description of the transition probability definition |

---

### 2. Axes

| Field | Type | Description |
|------|------|-------------|
| `A_pre` | array[int] | Mass number axis for **pre-neutron emission fragments** |
| `A_post` | array[int] | Mass number axis for **post-neutron emission fragments** |

These arrays define the row and column indices of the matrices.

Example indexing:


count_matrix["data"][i][j]


corresponds to

A_pre = axes["A_pre"][i]
A_post = axes["A_post"][j]


---

### 3. Count Matrix

| Field | Type | Description |
|------|------|-------------|
| `description` | string | Description of the matrix |
| `units` | string | `"counts"` |
| `data` | 2D array[int] | Raw fragment counts \(N(A_{pre} \rightarrow A_{post})\) |

Each element represents the number of fragments with mass \(A_{pre}\) that ended up with mass \(A_{post}\).

---

### 4. Probability Matrix

| Field | Type | Description |
|------|------|-------------|
| `description` | string | Description of the probability matrix |
| `units` | string | `"dimensionless"` |
| `data` | 2D array[float] | Conditional probability matrix |

Each row is normalized:

\[
P(A_{post} \mid A_{pre}) =
\frac{N(A_{pre} \rightarrow A_{post})}{N(A_{pre})}
\]

where

\[
N(A_{pre}) = \sum_{A_{post}} N(A_{pre} \rightarrow A_{post})
\]

---

### 5. Relative Error Matrix

| Field | Type | Description |
|------|------|-------------|
| `description` | string | Description of the statistical error definition |
| `units` | string | `"dimensionless"` |
| `data` | 2D array[float] | Relative statistical errors |

Errors follow **Poisson counting statistics**:

\[
\delta_{rel}(A_{post} \mid A_{pre}) =
\frac{1}{\sqrt{N(A_{pre} \rightarrow A_{post})}}
\]

Cells with zero counts contain **0**, since the relative error is undefined.

---

### Matrix Orientation

| Dimension | Represents |
|-----------|------------|
| Rows | \(A_{pre}\) (pre-neutron fragment mass) |
| Columns | \(A_{post}\) (post-neutron fragment mass) |

Example interpretation:

probability_matrix["data"][i][j]

represents

P(A_post = axes["A_post"][j] | A_pre = axes["A_pre"][i])

## `step2_inversion_output.npz`

This file contains NumPy arrays describing the **inferred pre-neutron mass yield solution**, its uncertainties, and diagnostic information from the regularized inversion.

---

### 1. Primary Solution (Pre-neutron yields)

| Array Name | Type | Shape | Description |
|------------|------|-------|-------------|
| `A_pre` | `float64` | `(n_pre,)` | Mass number axis for pre-neutron fragments |
| `Y_pre` | `float64` | `(n_pre,)` | Inferred pre-neutron mass yields |
| `sigma_pre` | `float64` | `(n_pre,)` | Total 1σ uncertainty on `Y_pre` (includes data and response matrix uncertainty) |
| `sigma_pre_data_only` | `float64` | `(n_pre,)` | 1σ uncertainty considering only the input data errors |

---

### 2. Covariance, Precision, & Correlation Matrices

| Array Name | Type | Shape | Description |
|------------|------|-------|-------------|
| `Cov` | `float64` | `(n_pre, n_pre)` | Total posterior covariance matrix \(C_{total} = C_{data} + C_R\) |
| `Cov_chol` | `float64` | `(n_pre, n_pre)` | Lower Cholesky decomposition of `Cov` |
| `Cov_inv` | `float64` | `(n_pre, n_pre)` | Precision matrix (inverse of `Cov`) |
| `Cov_inv_chol` | `float64` | `(n_pre, n_pre)` | Lower Cholesky decomposition of the precision matrix (precomputed for faster MCMC likelihood evaluation) |
| `corr` | `float64` | `(n_pre, n_pre)` | Normalized correlation matrix derived from `Cov` |
| `Cov_data` | `float64` | `(n_pre, n_pre)` | Covariance matrix from measured post-neutron yield data uncertainties only |
| `Cov_R` | `float64` | `(n_pre, n_pre)` | Covariance matrix from Monte Carlo response matrix statistical uncertainty |
| `frac_variance_from_R` | `float64` | `(n_pre,)` | Fraction of total variance in each bin originating from response matrix uncertainty |

---

### 3. Regularization Diagnostics (Tikhonov Lambda Scan)

| Array Name | Type | Shape | Description |
|------------|------|-------|-------------|
| `lambda_grid` | `float64` | `(n_lambda,)` | Grid of Tikhonov regularization parameters tested |
| `residual_norms` | `float64` | `(n_lambda,)` | Weighted residual norm \(||W(Gy - d)||_2\) |
| `seminorms` | `float64` | `(n_lambda,)` | Roughness penalty norm \(||Ly||_2\) |
| `curvature` | `float64` | `(n_lambda,)` | Mathematical curvature of the L-curve |
| `gcv` | `float64` | `(n_lambda,)` | Generalized Cross Validation score |
| `chi2_used` | `float64` | `(n_lambda,)` | χ² statistic using all usable data points |
| `chi2_wc` | `float64` | `(n_lambda,)` | χ² statistic for well-constrained data points |
| `neg_frac` | `float64` | `(n_lambda,)` | Fraction of pre-neutron bins that become negative |
| `lambda_final` | `float64` | `scalar` | Final chosen regularization parameter λ |
| `idx_final` | `int` | `scalar` | Index in `lambda_grid` corresponding to `lambda_final` |
| `idx_corner` | `int` | `scalar` | Index corresponding to the L-curve corner |
| `idx_gcv` | `int` | `scalar` | Index corresponding to the minimum GCV score |
| `idx_disc` | `int` | `scalar` | Index corresponding to the discrepancy principle target |

---

### 4. Input Data & Post-neutron Predictions

| Array Name | Type | Shape | Description |
|------------|------|-------|-------------|
| `A_post` | `float64` | `(n_post,)` | Mass number axis for post-neutron fragments |
| `Y_post_aligned` | `float64` | `(n_post,)` | Evaluated post-neutron yield data aligned to `A_post` |
| `sigma_post_aligned` | `float64` | `(n_post,)` | Uncertainties corresponding to `Y_post_aligned` |
| `Y_post_pred` | `float64` | `(n_post,)` | Forward-folded prediction \(Y_{pre} × R\), expected to reproduce `Y_post_aligned` |

---

### 5. Metadata / Flags

| Field | Type | Description |
|------|------|-------------|
| `nonnegative_used` | `bool` | `True` if the `--nonnegative` option was used to enforce \(Y_{pre} ≥ 0\) |
| `nonnegative_info` | `string` | JSON-formatted optimizer output from the SciPy least-squares solver when non-negative refinement is used |



## `step3_mcmc.npz`

This file stores the **posterior samples and statistical summaries from the MCMC inference stage** used to infer the model parameters controlling the pre-neutron fragment mass yield distribution.

The file contains the raw chains, a Gaussian approximation for fast sampling in downstream transport codes, statistical summaries, posterior predictive model evaluations, convergence diagnostics, and simulation metadata.

---

### 1. MCMC Posterior Chains

These arrays contain the **raw and processed samples from the Markov Chain Monte Carlo exploration of parameter space**.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `flat_chain` | `float64` | `(N_samples, 14)` | Flattened, burn-in removed, and thinned posterior samples. Each row is a sampled parameter vector. |
| `full_chain` | `float64` | `(nwalkers, nsteps, 14)` | Raw MCMC chains for every walker before thinning. Used for diagnostics and trace plots. |
| `log_prob_flat` | `float64` | `(N_samples,)` | Log posterior probability evaluated for each parameter vector in `flat_chain`. |


- Each parameter vector has **14 dimensions**, corresponding to the fitted model parameters.
- `flat_chain` represents the **posterior distribution** used for statistical analysis.
- `full_chain` preserves the full trajectory of each walker.

---

### 2. Online HPC Sampling Data (Gaussian Approximation)

To enable **fast sampling inside transport codes**, the posterior is approximated as a multivariate Gaussian

\[
\mathcal{N}(\mu, \Sigma)
\]

This avoids passing millions of MCMC samples downstream.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `gauss_mu` | `float64` | `(14,)` | Mean vector \( \mu \) of the posterior chain. |
| `gauss_cov` | `float64` | `(14, 14)` | Covariance matrix \( \Sigma \) of the posterior. |
| `gauss_chol` | `float64` | `(14, 14)` | Lower Cholesky decomposition \(L\) of the covariance matrix. |
| `gauss_corr` | `float64` | `(14, 14)` | Normalized correlation matrix showing parameter correlations (values between −1 and 1). |

### Fast Sampling

Downstream codes generate samples using

\[
\theta = \mu + Z L^{T}
\]

where

\[
Z \sim \mathcal{N}(0,1)
\]

This allows **instant generation of posterior-consistent parameter vectors** without running the full MCMC chain.

---

### 3. Point Estimates

These arrays contain **summary statistics of the posterior distribution** for quick reference and plotting.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `theta_nominal` | `float64` | `(14,)` | Baseline parameters used as the prior center (CGMF nominal values for \(^{235}\text{U}+n\)). |
| `theta_map` | `float64` | `(14,)` | Maximum A Posteriori (MAP) estimate found via Nelder–Mead optimization. |
| `theta_median` | `float64` | `(14,)` | Median of the posterior samples. |
| `theta_mean` | `float64` | `(14,)` | Mean of the posterior samples. |
| `theta_std` | `float64` | `(14,)` | Standard deviation of the posterior samples. |
| `theta_p16` | `float64` | `(14,)` | 16th percentile of the posterior distribution. |
| `theta_p84` | `float64` | `(14,)` | 84th percentile of the posterior distribution. |

- The interval between `theta_p16` and `theta_p84` defines the **68% (1σ) credible interval** for each parameter.
- `theta_map` corresponds to the **highest posterior density point** found during optimization.

---

### 4. Model vs. Data (Posterior Predictive)

These arrays allow **direct plotting of the fitted model against the experimental data** without recomputing the model.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `A_pre` | `int32` or `int64` | `(N_bins,)` | Pre-neutron fragment mass numbers \(A\). |
| `Y_pre_data` | `float64` | `(N_bins,)` | Target fission yields \(Y(A)\) imported from Step 2. |
| `sigma_pre_data` | `float64` | `(N_bins,)` | Uncertainties associated with the target yields. |
| `Y_pre_median_model` | `float64` | `(N_bins,)` | Model prediction evaluated using `theta_median`. |
| `Y_pre_nominal_model` | `float64` | `(N_bins,)` | Model prediction evaluated using the baseline `theta_nominal`. |

These arrays allow quick generation of **model vs data comparison plots**.

---

### 5. Convergence Metrics

These values provide **diagnostics for assessing whether the MCMC chains have converged**.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `Rhat` | `float64` | `(14,)` | Gelman–Rubin convergence statistic for each parameter. |
| `ESS` | `float64` | `(14,)` | Effective Sample Size for each parameter. |
| `acceptance_fraction` | `float64` | `(nwalkers,)` | Fraction of proposed MCMC steps accepted by each walker. |


- Convergence is typically considered acceptable when

\[
R_{\hat{}} < 1.1
\]

- `ESS` indicates how many **independent samples** the correlated chain is equivalent to.

---

### 6. Metadata Context

These arrays store **simulation context and parameter labels** used by downstream scripts.

| Array Name | Type | Shape | Description |
|-------------|------|-------|-------------|
| `A0` | `int32` or `int64` | `(1,)` | Compound nucleus mass number (e.g., `[236]`). |
| `En` | `float64` | `(1,)` | Incident neutron energy in MeV (e.g., `[2.53e-8]`). |
| `param_labels` | `string` | `(14,)` | Labels for each parameter (e.g., `w_a1`, `mu_b2`). |


- `param_labels` allows automated labeling in **corner plots, trace plots, and posterior visualizations**.
- `A0` and `En` define the **physical context of the simulation**.
