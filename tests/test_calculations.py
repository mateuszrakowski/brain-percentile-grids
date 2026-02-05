import json
from unittest.mock import Mock

import pytest

from app.fastapi.models.requests import ReferenceCalculationRequest
from app.fastapi.routers.calculations import generate_sse_events
from app.fastapi.services.calculation import (
    CalculationProgress,
    ModelFitResult,
    ReferenceCalculationResult,
)


async def async_iter(items):
    for item in items:
        yield item


class TestGenerateSseEvents:
    @pytest.mark.asyncio
    async def test_generate_sse_events(self):
        request = ReferenceCalculationRequest()
        mock_service = Mock()
        mock_service.fit_reference_models = Mock(
            return_value=async_iter(
                [
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
                            "structure_1": ModelFitResult(
                                structure="structure_1",
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
            )
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
