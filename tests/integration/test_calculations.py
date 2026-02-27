"""Integration tests for calculation endpoints.

Tests fit, stream, and OOS percentile endpoints through the HTTP
client. Services (CalculationService, ReferenceDataService,
PatientDataProcessor) are monkeypatched because they are created
inside the endpoints and can't be passed in via dependency injection.
"""

import json

import pandas as pd
import pytest

from app.fastapi.services.calculation import (
    CalculationProgress,
    ModelFitResult,
    PatientPercentileResult,
    ReferenceCalculationResult,
)


@pytest.fixture
def mock_fit_results():
    return [
        CalculationProgress(
            current=1,
            total=2,
            structure="hippo",
            status="fitting",
            message="...",
        ),
        CalculationProgress(
            current=2,
            total=2,
            structure="hippo",
            status="fitting",
            message="...",
        ),
        ReferenceCalculationResult(
            results={
                "hippo": ModelFitResult(
                    structure="hippo",
                    converged=True,
                    aic=1.0,
                    bic=1.0,
                    family="NO",
                    formula="x1",
                    percentile_curves={"t": [0.1, 0.2]},
                    x_values=[1.0, 2.0],
                ),
            },
            successful_count=1,
            failed_count=0,
        ),
    ]


async def async_iter(items):
    for item in items:
        yield item


class TestFitDatasetModels:
    """Tests for POST /api/datasets/{id}/fit and /fit/stream."""

    def test_fit_success(
        self,
        client,
        test_dataset,
        test_user_token,
        monkeypatch,
        mock_fit_results,
    ):
        """Verify non-streaming fit returns successful and failed counts."""
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.CalculationService.fit_reference_models",
            lambda self, **kwargs: async_iter(mock_fit_results),
        )
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.ReferenceDataService.get_reference_dataframe",
            lambda self, dataset_id: pd.DataFrame(
                {"AgeYears": [1, 2], "hippo": [0.5, 0.6]}
            ),
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/fit",
            json={
                "y_columns": ["hippo"],
                "percentiles": [0.2, 0.5, 0.7],
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.json()["successful_count"] == 1
        assert response.json()["failed_count"] == 0

    def test_fit_empty_dataset(
        self,
        client,
        test_dataset,
        test_user_token,
    ):
        """Verify fitting with no reference data returns 404."""
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/fit",
            json={
                "y_columns": ["hippo"],
                "percentiles": [0.2, 0.5, 0.7],
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 404

    def test_fit_stream(
        self,
        client,
        test_dataset,
        test_user_token,
        monkeypatch,
        mock_fit_results,
    ):
        """Verify SSE stream emits progress events followed by a complete event."""
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.ReferenceDataService.get_reference_dataframe",
            lambda self, dataset_id: pd.DataFrame(
                {"AgeYears": [1, 2], "hippo": [0.5, 0.6]}
            ),
        )
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.CalculationService.fit_reference_models",
            lambda self, **kwargs: async_iter(mock_fit_results),
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/fit/stream",
            json={
                "y_columns": ["hippo"],
                "percentiles": [0.2, 0.5, 0.7],
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.strip().split("\n\n")
        ]

        assert events[0]["type"] == "progress"
        assert events[-1]["type"] == "complete"

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestCalculateOOSPercentiles:
    """Tests for POST /api/datasets/{id}/calculate.

    Covers happy path with mocked processing, file validation
    (wrong filename, wrong extension), empty dataframe handling,
    and missing fitted models.
    """

    def test_calculate_success(
        self, client, test_dataset, test_user_token, monkeypatch
    ):
        """Verify OOS percentile calculation returns correct processed counts."""
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.CalculationService.calculate_patient_percentiles",
            lambda self, **kwargs: [
                PatientPercentileResult(
                    patient_id="1",
                    structure="hippo",
                    age=25,
                    value=0.5,
                    z_score=0.3,
                    percentile=0.7,
                )
            ],
        )
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.PatientDataProcessor.process_files",
            lambda self, files: [
                pd.DataFrame(
                    {
                        "PatientID": ["p1"],
                        "AgeYears": [25],
                        "StudyDate": ["2024-01-01"],
                        "hippo": [0.5],
                    }
                )
            ],
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/calculate",
            files=[
                (
                    "files",
                    ("patient1.csv", b"content", "text/csv"),
                ),
            ],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        assert response.json()["patients_processed"] == 1
        assert response.json()["structures_processed"] == 1

    def test_calculate_wrong_filename(
        self, client, test_dataset, test_user_token
    ):
        """Verify unsafe filename characters are rejected with 400."""
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/calculate",
            files=[
                (
                    "files",
                    ("???.csv", b"patient_id,age,hippo\np1,25,0.5", "text/csv"),
                ),
            ],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 400

    @pytest.mark.parametrize(
        "extension,content_type",
        [("txt", "text/plain"), ("xml", "text/xml"), ("mp4", "video/mp4")],
    )
    def test_calculate_wrong_extension(
        self, client, test_dataset, test_user_token, extension, content_type
    ):
        """Verify non-CSV/Excel file extensions are rejected with 400."""
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/calculate",
            files=[
                (
                    "files",
                    (
                        f"patient1.{extension}",
                        b"patient_id,age,hippo\np1,25,0.5",
                        content_type,
                    ),
                ),
            ],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 400

    def test_calculate_invalid_dataframe(
        self, client, test_dataset, test_user_token, monkeypatch
    ):
        """Verify empty DataFrame from processing is rejected with 400."""
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.PatientDataProcessor.process_files",
            lambda self, files: [pd.DataFrame({})],
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/calculate",
            files=[
                (
                    "files",
                    ("patient1.csv", b"patient_id,age,hippo\np1,25,0.5", "text/csv"),
                ),
            ],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 400

    def test_calculate_missing_models(
        self, client, test_dataset, test_user_token, monkeypatch
    ):
        """Verify calculation without fitted models is rejected with 400."""
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.PatientDataProcessor.process_files",
            lambda self, files: [
                pd.DataFrame(
                    {
                        "PatientID": ["p1"],
                        "AgeYears": [25],
                        "StudyDate": ["2024-01-01"],
                        "hippo": [0.5],
                    }
                )
            ],
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/calculate",
            files=[
                (
                    "files",
                    ("patient1.csv", b"content", "text/csv"),
                ),
            ],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 400
