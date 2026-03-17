# Plan: Core Module Unit Tests

## Verification

```bash
python -m pytest tests/unit/ -v --tb=short
```

## Deliberately Excluded

- R-dependent methods (GAMLSS.fit, load_model, evaluate_worm_plot, predict_patient_oos) — need integration tests with rpy2
- Plotting methods (plot_percentiles, plot_oos_patient, generate_grids) — visual output better validated by snapshots
- brain_structures.py Pydantic models — no custom validators, just field declarations
- REnvironment singleton — requires mocking entire rpy2 import chain; tested implicitly by integration tests
- disk_cache decorator — the decorator itself requires filesystem + pickle mocking. However, generate_cache_key in `app/core/utils/data_cache.py` is a pure function (MD5 of function signature + sorted kwargs) that could be unit tested separately. Consider adding to a future pass.

## Summary

Files 5-7 (test_data_fingerprinting, test_data_cache, test_persistent_cache) removed — the underlying modules were deleted as dead code.

---

# Plan: R Integration Tests for Core Engine

## Context

All existing tests mock away R completely — zero tests exercise the actual rpy2/R runtime. This means the critical path (model fitting → percentile calculation → patient prediction → save/load) has no real integration coverage. A silent rpy2 API change, R package update, or formula regression would go undetected until production.

These tests require a working R installation with gamlss/gamlss.dist packages and will be auto-skipped in environments without R.

## Verification

```bash
# Run only R integration tests (skipped automatically if R unavailable)
python -m pytest tests/integration/r_engine/ -v --tb=short

# Run all tests
python -m pytest tests/ -v --tb=short

# Run excluding R tests (for CI without R)
python -m pytest tests/ -v --tb=short -m "not r_required"
```

Expected: ~42 new tests, all passing when R+gamlss are available, all skipped when not. Total runtime for R tests: ~15s.

## File Structure

```
tests/integration/r_engine/
  __init__.py
  conftest.py                # Skip logic, synthetic data, session-scoped fitted models
  test_r_environment.py      # REnvironment singleton + package availability
  test_gamlss_fit.py         # GAMLSS.fit() convergence + metrics
  test_percentiles.py        # calculate_percentiles() with real R quantile functions
  test_predict_oos.py        # predict_patient_oos() z-score/percentile
  test_model_persistence.py  # save/load RDS roundtrip
  test_worm_plot.py          # evaluate_worm_plot() with real R residuals
  test_model_selector.py     # GAMLSSModelSelector.fit_models() pipeline
```

## Files to Modify

### 1. `pyproject.toml` — Register pytest marker

Add `r_required` marker so users can `pytest -m r_required` or `pytest -m "not r_required"`:

```toml
[tool.pytest.ini_options]
markers = [
    "r_required: tests requiring a working R environment with gamlss packages",
]
```

### 2. `tests/integration/r_engine/conftest.py` — Fixtures and skip logic

Skip mechanism: Module-level try/except around engine imports (importing model.py triggers `r_env = get_r_environment()` at line 19). If R is unavailable, `R_AVAILABLE = False` and all tests in the subpackage are skipped via `pytestmark`.

Synthetic data (seeded `rng = np.random.default_rng(42)`):

- **synthetic_normal_data** — 100 rows, age ∈ [5, 80], volume = 1000 + 2*age + N(0, 30). Simple linear trend with constant variance — Normal family converges reliably.
- **synthetic_lognormal_data** — 100 rows, volume = exp(6.5 + 0.005*age + N(0, 0.1)). All positive values — LogNormal family converges reliably.
- **synthetic_data_with_negatives** — copy of normal data with values shifted to include negative volumes. Used for forced non-convergence test with LOGNO family.

Session-scoped fitted models (avoid refitting per test):

- **fitted_normal_model** — `GAMLSS.fit(family="NO", formula_mu="pb(age, df=3)", formula_sigma="1")` on normal data
- **fitted_lognormal_model** — same with `family="LOGNO"` on lognormal data
- Both use `control_params={"n_cyc": 500, "trace": False}`.

Other fixtures:

- **r_env** — session-scoped `get_r_environment()` singleton
- `STANDARD_PERCENTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`
- Reuse: `get_r_environment` from `app.core.engine.environment`, `GAMLSS`/`FittedGAMLSSModel` from `app.core.engine.model`.

### 3. `tests/integration/r_engine/test_r_environment.py`

| Test | Rationale |
|---|---|
| `test_singleton_returns_same_instance` — `REnvironment() is REnvironment()` | Double-init would redundantly re-import R packages; confirms singleton works. |
| `test_is_available_true` — `r_env.is_available is True` | Verifies the R init completed without error — prerequisite for every other test. |
| `test_gamlss_package_loaded` — `r_env.gamlss_r is not None` | If gamlss import silently failed, all fitting calls would crash with opaque AttributeError. |
| `test_gamlss_dist_package_loaded` — `r_env.gamlss_dist is not None` | Quantile functions (qNO, qLOGNO) live here. Missing package = no percentile curves. |
| `test_check_r_environment_returns_true` — `check_r_environment() is True` | Tests the convenience function used by health endpoints. |

### 4. `tests/integration/r_engine/test_gamlss_fit.py`

Uses session-scoped `fitted_normal_model` and `fitted_lognormal_model`.

| Test | Rationale |
|---|---|
| `test_normal_model_converges` — `.converged is True` | Core contract: NO family on Normal data must converge. Failure means the rpy2→R formula pipeline is broken. |
| `test_lognormal_model_converges` — `.converged is True` | Second family exercises a different R code path (qLOGNO vs qNO). |
| `test_aic_is_finite_positive` — `np.isfinite(model.aic) and model.aic > 0` | AIC extraction calls `r_env.stats.AIC(self.model)` — verifies the rpy2 type conversion (R vector → numpy → float) works. |
| `test_bic_is_finite_positive` — same for BIC | BIC uses the same extraction path but different R function. Both must work for model selection. |
| `test_deviance_is_finite` — `np.isfinite(model.deviance)` | Deviance is extracted via `r_model.rx2("G.deviance")` — different rpy2 access pattern than AIC/BIC. |
| `test_aic_less_than_bic` — `model.aic < model.bic` | For n=100, BIC always penalizes more than AIC. Violation signals metric extraction is swapped or corrupted. |
| `test_fit_with_smooth_sigma` — fit NO with `sigma_formula="pb(age, df=2)"`, assert converges | Tests the sigma formula R path, which constructs a separate `robjects.Formula`. Important because smooth sigma is used in production model candidates. |
| `test_forced_nonconvergence` — fit LOGNO on data with negative values, assert `converged is False` | Using a misspecified family (LogNormal requires positive values) reliably prevents convergence. More robust than n_cyc=1 which could converge on simple Normal data with good initial values. Validates that non-convergence is reported via `r_model.rx2("converged")`, not raised as exception. Critical for selector's convergence filtering. |
| `test_fit_with_nan_in_volume` — inject 5 NaN values into volume column, fit NO | Real clinical data has missing values. Must either silently drop NaN rows and converge, or raise a clear error. Catches silent wrong results. |

### 5. `tests/integration/r_engine/test_percentiles.py`

Uses session-scoped `fitted_normal_model`.

| Test | Rationale |
|---|---|
| `test_percentile_keys_match_requested` — `set(curves.keys()) == set(STANDARD_PERCENTILES)` | Confirms `predictAll()` + quantile function loop returns exactly the requested percentiles, no missing/extra. |
| `test_percentile_curves_have_200_points` — each curve length == 200 | Matches the `np.linspace(..., 200)` in `calculate_percentiles` (line 178-182). Wrong length means the R prediction grid diverged from Python's. |
| `test_percentile_monotonicity` — for adjacent pairs, `np.all(lower <= upper)` | Crossing percentile curves mean the model or quantile function is broken. Hard-assert for a known-good fit. |
| `test_median_curve_near_true_mean` — 50th percentile at midpoint age ≈ 1085, tolerance ±20 | Generative model: volume = 1000 + 2*42.5 = 1085. With n=100 and noise std=30, SE at midpoint ≈ 3, so ±20 is generous enough for CI stability while still catching real regressions. |
| `test_lognormal_percentiles_all_positive` — all LOGNO curve values > 0 | LogNormal support is (0, ∞). Negative values would mean qLOGNO returned invalid results or the rpy2 conversion corrupted the output. |
| `test_percentiles_on_nonconverged_model_raises_or_warns` — call `calculate_percentiles` on a non-converged model | Non-converged model used downstream could produce nonsensical curves. Must either raise, warn, or return clearly flagged results. |

### 6. `tests/integration/r_engine/test_predict_oos.py`

Uses session-scoped `fitted_normal_model` and `synthetic_normal_data`.

| Test | Rationale |
|---|---|
| `test_zscore_at_median_near_zero` — patient at (age=40, volume=mean_at_40) → z ≈ 0, abs=0.3 | A patient at the conditional mean should have z≈0. With n=100 and low noise, z should be within ±0.2; tolerance of 0.3 adds margin for smoothing spline variation. |
| `test_percentile_at_median_near_half` — same patient → percentile ≈ 0.5, abs=0.10 | Validates the `norm.cdf(zscore)` conversion. Tolerance of 0.10 (40th-60th percentile) is tight enough to catch sign errors while allowing for sampling variation. |
| `test_extreme_low_value_negative_zscore` — volume far below mean → z < -1.5 | Confirms directional correctness: low values = negative z-scores. A sign error in the R↔Python bridge would flip this. |
| `test_extreme_high_value_positive_zscore` — volume far above mean → z > 1.5 | Same check on the positive side. |
| `test_returns_float_tuple` — `isinstance(z, float) and isinstance(p, float)` | The conversion chain (R vector → numpy array → [0] → float) has multiple points where the type could be wrong (np.float64 vs float). |
| `test_predict_at_min_age_boundary` — patient at exact min age of training data | Boundary where extrapolation logic kicks in. Prediction must still work and produce valid z-score. |
| `test_predict_at_max_age_boundary` — patient at exact max age of training data | Symmetric check on upper bound. |

### 7. `tests/integration/r_engine/test_model_persistence.py`

Uses session-scoped `fitted_normal_model` and `synthetic_normal_data`. Uses `tmp_path` (function-scoped) for file I/O.

| Test | Rationale |
|---|---|
| `test_save_creates_rds_file` — `model.save(path)`, assert .rds file exists | Validates `r_env.base.saveRDS()` actually writes to disk. Filesystem permission or rpy2 serialization failure would be caught here. |
| `test_save_creates_run_info_json` — assert `_run_info.json` exists after save | Auto-generated JSON sidecar path logic (lines 147-149 of model.py). |
| `test_run_info_contains_expected_keys` — JSON has dataset_length, timestamp, model_family, aic, bic | Consumers (dashboard, model registry) depend on these keys. A missing key would crash downstream tools. |
| `test_load_roundtrip_aic_matches` — `loaded.aic == pytest.approx(original.aic, rel=1e-6)` | The save→load roundtrip passes through R serialization (saveRDS/readRDS). If the R object loses precision or structure, metrics will diverge. |
| `test_load_roundtrip_percentiles_match` — percentile curves from loaded model ≈ original curves | Full fidelity check: the loaded R model must produce identical predictions. |
| `test_load_nonexistent_raises_file_not_found` — `GAMLSS.load_model("nonexistent.rds", ...)` | Validates the guard at line 595-596. Without it, readRDS would raise an opaque R error. |

### 8. `tests/integration/r_engine/test_worm_plot.py`

Uses session-scoped `fitted_normal_model`.

| Test | Rationale |
|---|---|
| `test_worm_plot_returns_diagnostics_object` — `isinstance(result, WormPlotDiagnostics)` | Validates the full pipeline: R `stats.residuals(model, what="z-scores")` → numpy conversion → binning → polynomial fit → dataclass construction. |
| `test_worm_plot_overall_coefficients_finite` — all of b0, b1, b2, b3 are finite floats | The R residual extraction could return NaN/Inf if the model object is malformed. Catches rpy2 conversion issues. |
| `test_worm_plot_returns_per_bin_results` — `len(diagnostics.per_bin) == 4` (default n_bins) | Validates the age-based binning logic works with real R residuals. |
| `test_well_fitting_model_has_small_coefficients` — `abs(overall.b0) < 0.3` | Normal model on Normal data is well-specified. Large coefficients would mean residual extraction from R is corrupted. |
| `test_worm_plot_on_good_fit_passes_or_borderline` — `diagnostics.passed is True` OR failure_reasons only mention borderline issues | Integration check that the full diagnostic pipeline (R residuals → Python polynomial → significance test) works end-to-end. |

### 9. `tests/integration/r_engine/test_model_selector.py`

Creates its own function-scoped `GAMLSSModelSelector` with 2 lightweight candidates (NO simple + LOGNO simple) to keep fitting fast.

| Test | Rationale |
|---|---|
| `test_fit_models_returns_converged_model` — `best is not None and best.converged` | End-to-end pipeline: candidate filtering → multiple fits → metric ranking → worm plot evaluation → best selection. |
| `test_fit_models_populates_results` — `len(selector.results) == 2` (matches injected 2-candidate list) | Confirms both candidates were attempted. A silent exception in the loop would leave results incomplete. Count must match the explicitly injected candidate list, not the global MODEL_CANDIDATES. |
| `test_fit_models_best_has_finite_metrics` — AIC, BIC, deviance all finite | The selected model must have valid metrics for downstream use. |
| `test_invalid_criterion_raises` — `criterion="invalid"` raises ValueError | Validates the guard at line 130-131 of selector.py. |

## Deliberately Excluded

- Plotting methods (plot_percentiles, plot_oos_patient, generate_grids) — visual output better validated by snapshots or manual review
- disk_cache decorator — the decorator requires filesystem + pickle mocking; generate_cache_key is pure-Python testable (see unit test exclusions note)
- Concurrent R access — thread-safety of REnvironment singleton under multiple requests is out of scope; would require a dedicated concurrency test harness
- Service layer (`app/fastapi/services/`) — conditional logic (cache check → fit → save) sits between routers and engine; consider as next priority after these two plans

## Summary

| File | New tests | Coverage |
|---|---|---|
| test_r_environment.py | 5 | Singleton, package availability, convenience functions |
| test_gamlss_fit.py | 9 | Model fitting convergence, metrics, formula paths, NaN handling |
| test_percentiles.py | 6 | Percentile curve calculation, monotonicity, non-converged model |
| test_predict_oos.py | 7 | Out-of-sample z-score/percentile prediction, boundary ages |
| test_model_persistence.py | 6 | RDS save/load roundtrip, run info JSON |
| test_worm_plot.py | 5 | Worm plot diagnostics with real R residuals |
| test_model_selector.py | 4 | Full model selection pipeline |
| **Total** | **~42** | |
