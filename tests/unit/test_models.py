"""Unit tests for Pydantic request model validation.

Tests custom validators on request models. We only test validators
that contain our own logic (e.g. percentile range checks, y_columns
non-empty). We do NOT test built-in Pydantic constraints like
min_length, as that would be testing Pydantic itself.
"""

import pytest

from app.fastapi.models.requests import FileUploadMetadata, ReferenceCalculationRequest


class TestReferenceCalculation:
    """Tests for ReferenceCalculationRequest custom validators."""
    def test_valid_percentiles(self):
        """Verify percentiles within (0, 1) exclusive range are accepted."""
        req = ReferenceCalculationRequest(
            x_column="test_x",
            y_columns=["test_y"],
            percentiles=[0.25, 0.50, 0.75, 0.90],
        )

        assert req.percentiles == [0.25, 0.50, 0.75, 0.90]

    @pytest.mark.parametrize("percentiles", [None, [0.0, 0.5, 1.0], [-1.0, 0.5]])
    def test_invalid_percentiles(self, percentiles):
        """Verify None, boundary values (0/1), and negatives are rejected."""
        with pytest.raises(ValueError):
            ReferenceCalculationRequest(
                x_column="test_x",
                y_columns=["test_y"],
                percentiles=percentiles,
            )

    @pytest.mark.parametrize("y_cols", [[""], []])
    def test_invalid_y_columns(self, y_cols):
        """Verify empty list and blank strings are rejected."""
        with pytest.raises(ValueError):
            ReferenceCalculationRequest(
                x_column="test_x",
                y_columns=y_cols,
                percentiles=[0.25, 0.50, 0.75, 0.90],
            )


class TestFileUploadMetadata:
    """Tests for FileUploadMetadata content type validation."""
    @pytest.mark.parametrize(
        "content_type",
        [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ],
    )
    def test_valid_content_type(self, content_type):
        """Verify CSV and Excel MIME types are accepted."""
        req = FileUploadMetadata(
            filename="test-filename", content_type=content_type, size=3000
        )

        assert req.filename == "test-filename"

    @pytest.mark.parametrize(
        "content_type",
        ["text/html", "text/javascript", "image/jpeg"],
    )
    def test_invalid_content_type(self, content_type):
        """Verify non-spreadsheet MIME types are rejected."""
        with pytest.raises(ValueError):
            FileUploadMetadata(
                filename="wrong-file", content_type=content_type, size=3000
            )
