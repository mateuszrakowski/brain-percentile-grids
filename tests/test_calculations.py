import json
from unittest.mock import Mock

import pandas as pd
import pytest

from app.fastapi.models.requests import ReferenceCalculationRequest
from app.fastapi.routers.calculations import generate_sse_events
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


class TestGenerateSseEvents:
    @pytest.mark.asyncio
    async def test_generate_sse_events(self, mock_fit_results):
        request = ReferenceCalculationRequest()
        mock_service = Mock()
        mock_service.fit_reference_models = Mock(
            return_value=async_iter(mock_fit_results)
        )

        events = [e async for e in generate_sse_events(mock_service, 1, 1, request)]

        first_progress = json.loads(events[0].removeprefix("data: ").strip())
        second_progress = json.loads(events[1].removeprefix("data: ").strip())
        final_response = json.loads(events[2].removeprefix("data: ").strip())

        assert first_progress["progress"] == 50
        assert second_progress["progress"] == 100
        assert len(events) == 3
        assert events[0].startswith("data: ")
        assert list(first_progress.keys()) == [
            "type",
            "current",
            "total",
            "structure",
            "status",
            "message",
            "progress",
        ]
        assert final_response["successful_count"] == 1
        assert final_response["failed_count"] == 0


class TestFitDatasetModels:
    def test_fit_dataset_models(
        self,
        client,
        test_dataset,
        test_user_token,
        monkeypatch,
        mock_fit_results,
    ):
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

    def test_fit_dataset_models_empty_df(
        self,
        client,
        test_dataset,
        test_user_token,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "app.fastapi.routers.calculations.ReferenceDataService.get_reference_dataframe",
            lambda self, dataset_id: pd.DataFrame(),
        )

        response = client.post(
            f"/api/datasets/{test_dataset['id']}/fit",
            json={
                "y_columns": ["hippo"],
                "percentiles": [0.2, 0.5, 0.7],
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 404

    def test_fit_dataset_models_stream(
        self,
        client,
        test_dataset,
        test_user_token,
        monkeypatch,
        mock_fit_results,
    ):
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
    def test_calculate_oos_percentiles(
        self, client, test_dataset, test_user_token, monkeypatch
    ):
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
                pd.DataFrame({"patient_id": ["p1"], "age": [25], "hippo": [0.5]})
            ],
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

        assert response.status_code == 200
        assert response.json()["patients_processed"] == 1
        assert response.json()["structures_processed"] == 1
