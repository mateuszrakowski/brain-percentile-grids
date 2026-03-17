"""Unit tests for GAMLSS model selector.

Tests cover sample-size-based model candidate filtering, verifying
that complexity thresholds, boundary conditions, and degenerate
inputs are handled correctly.
"""

from unittest.mock import patch

import pytest

from app.core.engine.selector import GAMLSSModelSelector
from app.core.resources.model_candidates import get_all_model_candidates


@pytest.fixture
def model_selector():
    """Create a ``GAMLSSModelSelector`` with ``__init__`` bypassed.

    Returns
    -------
    GAMLSSModelSelector
        Instance with ``model_candidates`` set to the full candidate list.
    """
    with patch.object(GAMLSSModelSelector, "__init__", lambda self, **kw: None):
        obj = GAMLSSModelSelector()
    obj.model_candidates = get_all_model_candidates()
    return obj


class TestGetSampleSizeAppropriateModels:
    """Tests for ``GAMLSSModelSelector._get_sample_size_appropriate_models``.

    Validates that candidate filtering by sample size returns models
    with appropriate complexity: n<30 → max 2, n<100 → max 4,
    n<200 → max 5, n≥200 → all.
    """

    def test_small_sample_excludes_complex(self, model_selector: GAMLSSModelSelector):
        """Verify that n=25 returns only complexity ≤ 2 models.

        Reasoning
        ---------
        With fewer than 30 samples, complex models overfit. Only simple
        (complexity 1) and smooth two-parameter (complexity 2) models
        should be available.
        """
        models = model_selector._get_sample_size_appropriate_models(25)
        assert len(models) == 5
        assert all(m.complexity <= 2 for m in models)

    def test_medium_sample_allows_moderate(self, model_selector: GAMLSSModelSelector):
        """Verify that n=80 returns complexity ≤ 4 models.

        Reasoning
        ---------
        Mid-range threshold covering the most common dataset sizes.
        Three- and four-parameter models become available.
        """
        models = model_selector._get_sample_size_appropriate_models(80)
        assert len(models) == 8
        assert all(m.complexity <= 4 for m in models)

    def test_large_sample_allows_most(self, model_selector: GAMLSSModelSelector):
        """Verify that n=150 returns complexity ≤ 5 models.

        Reasoning
        ---------
        Near-boundary for the 200 threshold. Four-parameter models
        with constant tau become available.
        """
        models = model_selector._get_sample_size_appropriate_models(150)
        assert len(models) == 9
        assert all(m.complexity <= 5 for m in models)

    def test_very_large_sample_returns_all(self, model_selector: GAMLSSModelSelector):
        """Verify that n=300 returns all model candidates.

        Reasoning
        ---------
        No filtering should occur for large datasets. All complexity
        tiers are appropriate.
        """
        models = model_selector._get_sample_size_appropriate_models(300)
        assert len(models) == 10
        assert all(m.complexity <= 6 for m in models)

    def test_boundary_n30_uses_higher_tier(self, model_selector: GAMLSSModelSelector):
        """Verify that n=30 (exactly at threshold) allows complexity ≤ 4.

        Reasoning
        ---------
        Off-by-one check. Code uses ``n < 30``, so n=30 should fall
        into the next tier (max complexity 4), not the lowest tier.
        """
        models = model_selector._get_sample_size_appropriate_models(30)
        assert len(models) == 8
        assert all(m.complexity <= 4 for m in models)

    def test_boundary_n100(self, model_selector: GAMLSSModelSelector):
        """Verify that n=100 (exactly at threshold) allows complexity ≤ 5.

        Reasoning
        ---------
        Off-by-one check. Code uses ``n < 100``, so n=100 should fall
        into the next tier (max complexity 5).
        """
        models = model_selector._get_sample_size_appropriate_models(100)
        assert len(models) == 9
        assert all(m.complexity <= 5 for m in models)

    def test_boundary_n200(self, model_selector: GAMLSSModelSelector):
        """Verify that n=200 (exactly at threshold) returns all candidates.

        Reasoning
        ---------
        Off-by-one check. Code uses ``n < 200``, so n=200 should fall
        into the else branch and return all models.
        """
        models = model_selector._get_sample_size_appropriate_models(200)
        assert len(models) == 10
        assert all(m.complexity <= 6 for m in models)

    def test_returns_nonempty(self, model_selector: GAMLSSModelSelector):
        """Verify that the result is never empty for any positive n.

        Reasoning
        ---------
        An empty candidate list would crash the downstream selector.
        Even for very small datasets, at least the simplest models
        should be returned.
        """
        models = model_selector._get_sample_size_appropriate_models(1)
        assert len(models) > 0

    def test_degenerate_n1_does_not_crash(self, model_selector: GAMLSSModelSelector):
        """Verify that n=1 does not raise an exception.

        Reasoning
        ---------
        Degenerate input. Even if it should never occur in practice,
        the function must not crash. It should return the simplest
        models (complexity ≤ 2).
        """
        models = model_selector._get_sample_size_appropriate_models(1)
        assert all(m.complexity <= 2 for m in models)
