import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from app.core.engine.environment import get_r_environment

logger = logging.getLogger(__name__)

r_env = get_r_environment()


@dataclass
class WormPlotCoefficients:
    """Cubic polynomial coefficients from a worm plot fit.

    Attributes
    ----------
    b0 : float
        Intercept (mean bias).
    b1 : float
        Linear coefficient (variance mis-specification).
    b2 : float
        Quadratic coefficient (skewness mis-specification).
    b3 : float
        Cubic coefficient (kurtosis mis-specification).
    p_values : list[float]
        P-values for b0-b3 from OLS t-tests.
    n : int
        Number of residuals used in the fit.
    """

    b0: float
    b1: float
    b2: float
    b3: float
    p_values: list[float]
    n: int


@dataclass
class WormPlotDiagnostics:
    """Results from an automated worm plot evaluation.

    Attributes
    ----------
    passed : bool
        True if all bins and overall test pass.
    overall : WormPlotCoefficients
        Polynomial fit on the full residual vector.
    per_bin : list[WormPlotCoefficients]
        Polynomial fits per age-quantile bin.
    failure_reasons : list[str]
        Human-readable reasons for failure (empty if passed).
    """

    passed: bool
    overall: WormPlotCoefficients
    per_bin: list[WormPlotCoefficients]
    failure_reasons: list[str] = field(default_factory=list)


def _fit_worm_polynomial(residuals: np.ndarray) -> WormPlotCoefficients:
    """Fit a cubic polynomial to de-trended Q-Q values.

    Follows the van Buuren & Fredriks (2001) worm plot methodology:
    theoretical quantiles on x-axis, deviations (observed - theoretical)
    on y-axis, then fit y ~ b0 + b1*x + b2*x^2 + b3*x^3.

    Parameters
    ----------
    residuals : np.ndarray
        Quantile residuals (should be approximately standard normal
        if the model is adequate).

    Returns
    -------
    WormPlotCoefficients
        Fitted polynomial coefficients with p-values.
    """
    n = len(residuals)
    sorted_residuals = np.sort(residuals)

    # Theoretical quantiles (standard normal)
    theoretical = scipy_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    # Deviations: observed minus theoretical
    deviations = sorted_residuals - theoretical

    # OLS cubic polynomial: deviations ~ b0 + b1*t + b2*t^2 + b3*t^3
    x_mat = np.column_stack([
        np.ones(n),
        theoretical,
        theoretical**2,
        theoretical**3,
    ])

    # Least-squares fit
    coeffs, residual_ss, _, _ = np.linalg.lstsq(x_mat, deviations, rcond=None)

    # Standard errors and p-values via t-distribution
    if len(residual_ss) > 0:
        mse = residual_ss[0] / (n - 4)
    else:
        mse = np.sum((deviations - x_mat @ coeffs) ** 2) / max(n - 4, 1)

    var_covar = mse * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(var_covar))

    t_stats = coeffs / se
    p_values = [
        float(2 * scipy_stats.t.sf(abs(t), df=max(n - 4, 1))) for t in t_stats
    ]

    return WormPlotCoefficients(
        b0=float(coeffs[0]),
        b1=float(coeffs[1]),
        b2=float(coeffs[2]),
        b3=float(coeffs[3]),
        p_values=p_values,
        n=n,
    )


def evaluate_worm_plot(
    model,
    data_table,
    x_column: str,
    n_bins: int = 4,
    alpha: float = 0.05,
) -> WormPlotDiagnostics:
    """Run automated worm plot diagnostics on a fitted GAMLSS model.

    Extracts quantile residuals from the R model, bins them by age
    quantiles, and tests each bin plus the overall residuals for
    distributional adequacy using a cubic polynomial fit.

    Parameters
    ----------
    model : rpy2.robjects.vectors.ListVector
        Fitted R GAMLSS model object.
    data_table : pandas.DataFrame
        Source data used for fitting.
    x_column : str
        Name of the age/independent variable column.
    n_bins : int, optional
        Number of age-quantile bins (default 4).
    alpha : float, optional
        Significance level before Bonferroni correction (default 0.05).

    Returns
    -------
    WormPlotDiagnostics
        Diagnostic results including pass/fail and per-bin coefficients.
    """
    # Extract quantile residuals from R model
    residuals_r = r_env.stats.residuals(model, what="z-scores")
    residuals = np.array(residuals_r)

    # Remove NaN/Inf residuals
    valid_mask = np.isfinite(residuals)
    residuals = residuals[valid_mask]
    x_values = data_table[x_column].values[valid_mask]

    if len(residuals) < 20:
        return WormPlotDiagnostics(
            passed=False,
            overall=_fit_worm_polynomial(residuals),
            per_bin=[],
            failure_reasons=["Too few valid residuals for worm plot analysis"],
        )

    # Overall fit
    overall = _fit_worm_polynomial(residuals)

    # Bin by age quantiles
    bin_edges = np.quantile(x_values, np.linspace(0, 1, n_bins + 1))
    bin_indices = np.digitize(x_values, bin_edges[1:-1])

    per_bin = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() >= 10:
            per_bin.append(_fit_worm_polynomial(residuals[mask]))

    # Bonferroni-corrected alpha: total tests = (n_bins + 1) fits * 4 coefficients
    n_tests = (len(per_bin) + 1) * 4
    corrected_alpha = alpha / n_tests

    failure_reasons = []
    coeff_names = ["b0 (mean)", "b1 (variance)", "b2 (skewness)", "b3 (kurtosis)"]

    # Check overall
    for p_val, name in zip(overall.p_values, coeff_names, strict=True):
        if p_val < corrected_alpha:
            failure_reasons.append(
                f"Overall {name} significant (p={p_val:.4f} < {corrected_alpha:.4f})"
            )

    # Check per-bin
    for b_idx, bin_result in enumerate(per_bin):
        for p_val, name in zip(bin_result.p_values, coeff_names, strict=True):
            if p_val < corrected_alpha:
                failure_reasons.append(
                    f"Bin {b_idx + 1} {name} significant "
                    f"(p={p_val:.4f} < {corrected_alpha:.4f})"
                )

    passed = len(failure_reasons) == 0

    return WormPlotDiagnostics(
        passed=passed,
        overall=overall,
        per_bin=per_bin,
        failure_reasons=failure_reasons,
    )
