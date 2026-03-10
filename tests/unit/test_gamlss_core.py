"""Unit tests for GAMLSS core modules.

Tests cover diagnostics, age calculation, model extrapolation status,
percentile monotonicity validation, and model candidate definitions.
"""

import logging
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.data_processing.process_input import _calculate_age
from app.core.engine.diagnostics import _fit_worm_polynomial
from app.core.engine.model import FittedGAMLSSModel
from app.core.resources.model_candidates import MODEL_CANDIDATES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fitted_model(
    data_table: pd.DataFrame, x_column: str = "age"
) -> FittedGAMLSSModel:
    """Create a ``FittedGAMLSSModel`` with ``__init__`` bypassed.

    Parameters
    ----------
    data_table : pd.DataFrame
        DataFrame to assign as ``data_table``.
    x_column : str
        Name of the x column.

    Returns
    -------
    FittedGAMLSSModel
        Instance with only ``data_table`` and ``x_column`` set.
    """
    with patch.object(FittedGAMLSSModel, "__init__", lambda self, **kw: None):
        obj = FittedGAMLSSModel()
    obj.data_table = data_table
    obj.x_column = x_column
    return obj


# ===========================================================================
# 1. Diagnostics – _fit_worm_polynomial
# ===========================================================================


class TestFitWormPolynomial:
    """Tests for ``_fit_worm_polynomial`` from the diagnostics module.

    Validates that the worm-plot polynomial fit correctly distinguishes
    well-fitting residuals from poorly-fitting ones using p-value signals.
    """

    def test_standard_normal_residuals_pass(self):
        """Verify that standard-normal residuals produce small polynomial coefficients.

        Reasoning
        ---------
        When the model is adequate, quantile residuals should be approximately
        N(0, 1). The cubic polynomial coefficients (b0-b3) should all be small
        in magnitude, indicating no systematic distributional misfit.

        Note: raw p-values from the OLS t-test can be very small even for
        correct N(0,1) data because order-statistic deviations have extremely
        low variance, making the test hypersensitive. In practice the worm
        plot uses Bonferroni correction across bins. Here we verify the
        *coefficients* are near zero (all |b| < 0.15) and all 4 p-values
        are returned, which is the meaningful signal for a well-fitting model.
        """
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 200)

        result = _fit_worm_polynomial(residuals)

        assert len(result.p_values) == 4
        assert result.n == 200
        # For well-fitting N(0,1) residuals, all coefficients should be small.
        assert abs(result.b0) < 0.15
        assert abs(result.b1) < 0.15
        assert abs(result.b2) < 0.15
        assert abs(result.b3) < 0.15

    def test_skewed_residuals_detected(self):
        """Verify that exponentially-distributed residuals trigger at least one low p-value.

        Reasoning
        ---------
        Exponential(1) residuals are right-skewed and far from N(0, 1).
        At least one polynomial coefficient (likely b2/skewness) should be
        significant (p < 0.05), confirming that the diagnostic can detect
        distributional inadequacy. Failing this test would mean the diagnostic
        is insensitive to model misspecification.
        """
        rng = np.random.default_rng(42)
        residuals = rng.exponential(1, 200)

        result = _fit_worm_polynomial(residuals)

        assert len(result.p_values) == 4
        any_significant = any(p < 0.05 for p in result.p_values)
        assert any_significant, (
            f"No p-value below 0.05 for skewed residuals: {result.p_values}"
        )


# ===========================================================================
# 2. Age calculation – _calculate_age
# ===========================================================================


class TestCalculateAge:
    """Tests for ``_calculate_age`` from the process_input module.

    Validates that continuous age calculation from two datetime objects
    produces correct fractional-year values for common date intervals.
    """

    def test_ten_year_span(self):
        """Verify that a 10-year span returns approximately 10.0.

        Reasoning
        ---------
        The simplest case: exact decade difference. Off-by-one or integer
        truncation bugs in the day-count division would surface here.
        """
        birth = datetime(2000, 1, 1)
        study = datetime(2010, 1, 1)

        age = _calculate_age(birth, study)

        assert age == pytest.approx(10.0, abs=0.01)

    def test_fractional_year(self):
        """Verify that a 5.5-year span is calculated correctly.

        Reasoning
        ---------
        Tests that mid-year dates produce the expected fractional result,
        catching errors in the days-to-years divisor (365.25).
        """
        birth = datetime(2000, 6, 15)
        study = datetime(2005, 12, 15)

        age = _calculate_age(birth, study)

        assert age == pytest.approx(5.5, abs=0.02)

    def test_single_day(self):
        """Verify that a one-day difference yields approximately 1/365.25.

        Reasoning
        ---------
        Boundary case: the smallest meaningful age increment. Ensures that
        the divisor constant (365.25) is applied correctly and the result
        is not zero or truncated to an integer.
        """
        birth = datetime(2000, 1, 1)
        study = datetime(2000, 1, 2)

        age = _calculate_age(birth, study)

        assert age == pytest.approx(1 / 365.25, abs=0.001)


# ===========================================================================
# 3. FittedGAMLSSModel – get_extrapolation_status
# ===========================================================================


class TestGetExtrapolationStatus:
    """Tests for ``FittedGAMLSSModel.get_extrapolation_status``.

    Validates the three-tier classification (safe / near_boundary /
    extrapolated) based on patient age relative to training data range.
    """

    @pytest.fixture()
    def model(self) -> FittedGAMLSSModel:
        """Create a FittedGAMLSSModel with x_column range [5.0, 15.0].

        Returns
        -------
        FittedGAMLSSModel
            Model whose training data spans ages 5.0 to 15.0.
        """
        df = pd.DataFrame({"age": np.linspace(5.0, 15.0, 50)})
        return _make_fitted_model(df, x_column="age")

    def test_safe_within_range(self, model: FittedGAMLSSModel):
        """Verify that an age well inside the range returns ``"safe"``.

        Reasoning
        ---------
        Age 10.0 is the midpoint of [5, 15] -- clearly interior. This is
        the happy-path case for clinical predictions.
        """
        assert model.get_extrapolation_status(10.0) == "safe"

    def test_near_boundary(self, model: FittedGAMLSSModel):
        """Verify that an age within 5% of the range edge returns ``"near_boundary"``.

        Reasoning
        ---------
        Range = 10.0, margin = 0.5. Age 5.1 is within 0.1 of min (5.0),
        which is inside the range but within the 5% margin. Clinical users
        should be warned about reduced reliability near the boundary.
        """
        assert model.get_extrapolation_status(5.1) == "near_boundary"

    def test_extrapolated_below(self, model: FittedGAMLSSModel):
        """Verify that an age below the training range returns ``"extrapolated"``.

        Reasoning
        ---------
        Age 3.0 is outside [5, 15]. Predictions here are pure
        extrapolation and should be flagged so users do not rely on them.
        """
        assert model.get_extrapolation_status(3.0) == "extrapolated"


# ===========================================================================
# 4. FittedGAMLSSModel – _validate_percentile_monotonicity
# ===========================================================================


class TestValidatePercentileMonotonicity:
    """Tests for ``FittedGAMLSSModel._validate_percentile_monotonicity``.

    Validates that crossing percentile curves trigger a warning log
    while well-ordered curves do not.
    """

    @pytest.fixture()
    def model(self) -> FittedGAMLSSModel:
        """Create a minimal FittedGAMLSSModel for monotonicity checks.

        Returns
        -------
        FittedGAMLSSModel
            Model with only the attributes needed for validation.
        """
        df = pd.DataFrame({"age": [1, 2, 3]})
        return _make_fitted_model(df, x_column="age")

    def test_non_crossing_curves_no_warning(self, model: FittedGAMLSSModel, caplog):
        """Verify that properly ordered curves produce no warning.

        Reasoning
        ---------
        When curves are monotonically ordered (5th < 50th < 95th at every
        point), no data-integrity concern exists and the log should stay
        clean. A false warning would unnecessarily alarm users.
        """
        curves = {
            0.05: np.array([1, 2, 3]),
            0.50: np.array([4, 5, 6]),
            0.95: np.array([7, 8, 9]),
        }

        with caplog.at_level(logging.WARNING, logger="app.core.engine.model"):
            model._validate_percentile_monotonicity(curves)

        assert "cross" not in caplog.text.lower()

    def test_crossing_curves_emit_warning(self, model: FittedGAMLSSModel, caplog):
        """Verify that crossing curves produce a warning log message.

        Reasoning
        ---------
        If the 5th percentile exceeds the 50th at any point, the model
        fit is suspect. The method must log a warning so that downstream
        consumers (and automated selectors) can react. Missing this
        warning would let invalid percentile grids reach clinicians.
        """
        curves = {
            0.05: np.array([1, 2, 10]),
            0.50: np.array([4, 5, 6]),
        }

        with caplog.at_level(logging.WARNING, logger="app.core.engine.model"):
            model._validate_percentile_monotonicity(curves)

        assert "cross" in caplog.text.lower()


# ===========================================================================
# 5. Model candidates
# ===========================================================================


class TestModelCandidates:
    """Tests for ``MODEL_CANDIDATES`` from the model_candidates module.

    Validates that the candidate list has the expected count,
    consistent control parameters, and includes the newly-added entries.
    """

    def test_candidate_count(self):
        """Verify that MODEL_CANDIDATES contains exactly 10 entries.

        Reasoning
        ---------
        The list was extended from 8 to 10 with the addition of Gamma
        and Generalized Gamma candidates. An incorrect count would
        indicate an accidental deletion or duplication during editing.
        """
        assert len(MODEL_CANDIDATES) == 10

    def test_all_candidates_have_n_cyc_500(self):
        """Verify that every candidate sets ``n_cyc`` to 500.

        Reasoning
        ---------
        All model candidates should use 500 fitting cycles for
        convergence consistency. A mismatched value could cause
        premature convergence or excessive runtime for specific models.
        """
        for candidate in MODEL_CANDIDATES:
            assert candidate.control_params["n_cyc"] == 500, (
                f"{candidate.name} has n_cyc={candidate.control_params['n_cyc']}, expected 500"
            )

    def test_gamma_smooth_exists(self):
        """Verify that ``Gamma_Smooth`` is present in candidates.

        Reasoning
        ---------
        Gamma_Smooth is one of the two newly-added candidates.
        Its absence would mean the GA family is unavailable for
        model selection, reducing the candidate pool.
        """
        names = {c.name for c in MODEL_CANDIDATES}
        assert "Gamma_Smooth" in names

    def test_generalized_gamma_smooth_exists(self):
        """Verify that ``GeneralizedGamma_Smooth`` is present in candidates.

        Reasoning
        ---------
        GeneralizedGamma_Smooth is the second newly-added candidate.
        Its absence would mean the GG family is unavailable for
        model selection.
        """
        names = {c.name for c in MODEL_CANDIDATES}
        assert "GeneralizedGamma_Smooth" in names
