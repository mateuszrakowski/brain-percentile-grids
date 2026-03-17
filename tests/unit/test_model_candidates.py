"""Unit tests for model candidates module.

Tests cover candidate lookup by name and the completeness
of the candidate list and its required fields.
"""

from app.core.resources.model_candidates import (
    MODEL_CANDIDATES,
    get_all_model_candidates,
    get_model_candidate_by_name,
)


class TestGetModelCandidateByName:
    """Tests for ``get_model_candidate_by_name``.

    Validates that the lookup function returns the correct candidate
    object or None for unknown names.
    """

    def test_returns_correct_candidate(self):
        """Verify that a known name returns the matching candidate.

        Reasoning
        ---------
        If the dict key doesn't match the object's name field,
        model loading would use the wrong configuration.
        """
        candidate = get_model_candidate_by_name("Normal_Simple")
        assert candidate is not None
        assert candidate.name == "Normal_Simple"
        assert candidate.family == "NO"

    def test_unknown_name_returns_none(self):
        """Verify that an unknown name returns None.

        Reasoning
        ---------
        Callers rely on None to detect missing models. An exception
        would crash the lookup path.
        """
        assert get_model_candidate_by_name("NonExistent") is None

    def test_all_candidates_retrievable(self):
        """Verify that every candidate in the list can be looked up by name.

        Reasoning
        ---------
        Ensures the dict and list stay in sync. A mismatch would mean
        some candidates are unreachable by name.
        """
        for candidate in MODEL_CANDIDATES:
            result = get_model_candidate_by_name(candidate.name)
            assert result is not None
            assert result.name == candidate.name


class TestGetAllModelCandidates:
    """Tests for ``get_all_model_candidates``.

    Validates that the function returns the full candidate list
    and that all candidates have the required fields.
    """

    def test_returns_list_matching_module_constant(self):
        """Verify that the function returns the same list as the module constant.

        Reasoning
        ---------
        The function wraps the constant. Refactoring could break
        the link between the two.
        """
        assert get_all_model_candidates() is MODEL_CANDIDATES

    def test_all_have_required_fields(self):
        """Verify that every candidate has family, formulas, complexity, and control_params.

        Reasoning
        ---------
        A missing field causes a KeyError deep in the R fitting code
        where the error message would be opaque.
        """
        for candidate in get_all_model_candidates():
            assert candidate.family, f"{candidate.name} missing family"
            assert candidate.mu_formula, f"{candidate.name} missing mu_formula"
            assert candidate.sigma_formula, f"{candidate.name} missing sigma_formula"
            assert isinstance(candidate.complexity, int), (
                f"{candidate.name} complexity is not int"
            )
            assert candidate.control_params, f"{candidate.name} missing control_params"
