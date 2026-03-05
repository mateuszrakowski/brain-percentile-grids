"""Integration tests for calculation endpoints.

Tests fit, stream, and OOS percentile endpoints through the HTTP
client. Services (CalculationService, ReferenceDataService,
PatientDataProcessor) are patched because they are created
inside the endpoints and can't be passed in via dependency injection.
"""

import json
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from app.fastapi.services.calculation import (
    CalculationProgress,
    CalculationService,
    ModelFitResult,
    PatientPercentileResult,
    ReferenceCalculationResult,
)
from app.fastapi.services.reference_data import ReferenceDataService
from app.fastapi.utils.file_utils import PatientDataProcessor


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


@pytest.fixture
def mock_reference_df():
    return pd.DataFrame({"PatientAge": [1, 2], "hippo": [0.5, 0.6]})


def parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    """Parse SSE response text into a list of JSON event payloads."""
    events = []
    for block in raw_text.strip().split("\n\n"):
        for line in block.strip().splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                events.append(json.loads(payload))
    return events


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
        mock_fit_results,
        mock_reference_df,
    ):
        """Verify non-streaming fit returns successful and failed counts."""
        with (
            patch.object(
                CalculationService,
                "fit_reference_models",
                side_effect=lambda **kw: async_iter(mock_fit_results),
            ),
            patch.object(
                ReferenceDataService,
                "get_reference_dataframe",
                return_value=mock_reference_df,
            ),
        ):
            response = client.post(
                f"/api/datasets/{test_dataset['id']}/fit",
                json={
                    "y_columns": ["hippo"],
                    "percentiles": [0.2, 0.5, 0.7],
                },
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["successful_count"] == 1
        assert data["failed_count"] == 0
        assert "results" in data
        assert "total_time" in data
        assert "message" in data

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
        mock_fit_results,
        mock_reference_df,
    ):
        """Verify SSE stream emits progress events followed by a complete event."""
        with (
            patch.object(
                ReferenceDataService,
                "get_reference_dataframe",
                return_value=mock_reference_df,
            ),
            patch.object(
                CalculationService,
                "fit_reference_models",
                side_effect=lambda **kw: async_iter(mock_fit_results),
            ),
        ):
            response = client.post(
                f"/api/datasets/{test_dataset['id']}/fit/stream",
                json={
                    "y_columns": ["hippo"],
                    "percentiles": [0.2, 0.5, 0.7],
                },
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        events = parse_sse_events(response.text)

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

    def test_calculate_success(self, client, test_dataset, test_user_token):
        """Verify OOS percentile calculation returns correct processed counts."""
        mock_results = [
            PatientPercentileResult(
                patient_id="1",
                structure="hippo",
                age=25,
                value=0.5,
                z_score=0.3,
                percentile=0.7,
            )
        ]
        mock_dfs = [
            pd.DataFrame(
                {
                    "PatientID": ["p1"],
                    "PatientAge": [25],
                    "StudyDate": ["2024-01-01"],
                    "hippo": [0.5],
                }
            )
        ]

        with (
            patch.object(
                CalculationService,
                "calculate_patient_percentiles",
                return_value=mock_results,
            ),
            patch.object(
                PatientDataProcessor,
                "process_files",
                return_value=mock_dfs,
            ),
        ):
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
        data = response.json()
        assert data["patients_processed"] == 1
        assert data["structures_processed"] == 1
        assert "results" in data

    def test_calculate_wrong_filename(self, client, test_dataset, test_user_token):
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

    def test_calculate_invalid_dataframe(self, client, test_dataset, test_user_token):
        """Verify empty DataFrame from processing is rejected with 400."""
        with patch.object(
            PatientDataProcessor,
            "process_files",
            return_value=[pd.DataFrame({})],
        ):
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

    def test_calculate_missing_models(self, client, test_dataset, test_user_token):
        """Verify calculation without fitted models is rejected with 400."""
        mock_dfs = [
            pd.DataFrame(
                {
                    "PatientID": ["p1"],
                    "PatientAge": [25],
                    "StudyDate": ["2024-01-01"],
                    "hippo": [0.5],
                }
            )
        ]

        with patch.object(
            PatientDataProcessor,
            "process_files",
            return_value=mock_dfs,
        ):
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
