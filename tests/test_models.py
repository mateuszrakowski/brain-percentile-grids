import pytest

from app.fastapi.models.requests import FileUploadMetadata, ReferenceCalculationRequest


class TestReferenceCalculation:
    def test_valid_percentiles(self):
        req = ReferenceCalculationRequest(
            x_column="test_x",
            y_columns=["test_y"],
            percentiles=[0.25, 0.50, 0.75, 0.90],
        )

        assert req.percentiles == [0.25, 0.50, 0.75, 0.90]

    @pytest.mark.parametrize("percentiles", [None, [0.0, 0.5, 1.0], [-1.0, 0.5]])
    def test_invalid_percentiles(self, percentiles):
        with pytest.raises(ValueError):
            ReferenceCalculationRequest(
                x_column="test_x",
                y_columns=["test_y"],
                percentiles=percentiles,
            )

    @pytest.mark.parametrize("y_cols", [[""], []])
    def test_invalid_y_columns(self, y_cols):
        with pytest.raises(ValueError):
            ReferenceCalculationRequest(
                x_column="test_x",
                y_columns=y_cols,
                percentiles=[0.25, 0.50, 0.75, 0.90],
            )


class TestFileUploadMetadata:
    @pytest.mark.parametrize(
        "content_type",
        [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ],
    )
    def test_valid_content_type(self, content_type):
        req = FileUploadMetadata(
            filename="test-filename", content_type=content_type, size=3000
        )

        assert req.filename == "test-filename"

    @pytest.mark.parametrize(
        "content_type",
        ["text/html", "text/javascript", "image/jpeg"],
    )
    def test_invalid_content_type(self, content_type):
        with pytest.raises(ValueError):
            FileUploadMetadata(
                filename="wrong-file", content_type=content_type, size=3000
            )
