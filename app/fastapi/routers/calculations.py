"""
Calculation endpoints for GAMLSS modeling and percentile calculations.

Provides endpoints for:
- Fitting reference models with SSE progress updates
- Calculating patient percentiles against fitted models
"""

import json
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User
from app.fastapi.models.requests import (
    PatientCalculationRequest,
    ReferenceCalculationRequest,
)
from app.fastapi.models.responses import (
    ModelResult,
    PatientCalculationResponse,
    PatientResult,
    ReferenceCalculationResponse,
)
from app.fastapi.services.calculation import (
    CalculationProgress,
    CalculationService,
    ReferenceCalculationResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calculate", tags=["calculations"])


async def generate_sse_events(
    service: CalculationService,
    user_id: int,
    request: ReferenceCalculationRequest,
) -> Any:
    """
    Generate Server-Sent Events for model fitting progress.

    Parameters
    ----------
    service : CalculationService
        The calculation service instance.
    user_id : int
        The user ID.
    request : ReferenceCalculationRequest
        The calculation request parameters.

    Yields
    ------
    str
        SSE formatted event data.
    """
    start_time = time.time()

    async for update in service.fit_reference_models(
        user_id=user_id,
        y_columns=request.y_columns,
        percentiles=request.percentiles,
        criterion="bic",
    ):
        if isinstance(update, CalculationProgress):
            # Send progress event
            event_data = {
                "type": "progress",
                "current": update.current,
                "total": update.total,
                "structure": update.structure,
                "status": update.status,
                "message": update.message,
                "progress": int((update.current / update.total) * 100),
            }
            yield f"data: {json.dumps(event_data)}\n\n"

        elif isinstance(update, ReferenceCalculationResult):
            # Send final result
            elapsed_time = time.time() - start_time

            # Convert to response format
            results = {}
            for structure, model_result in update.results.items():
                results[structure] = ModelResult(
                    structure=structure,
                    converged=model_result.converged,
                    aic=model_result.aic,
                    bic=model_result.bic,
                    family=model_result.family,
                    formula=model_result.formula,
                    percentile_curves=model_result.percentile_curves,
                    error=model_result.error,
                    plot_available=model_result.percentile_curves is not None,
                ).model_dump()

            event_data = {
                "type": "complete",
                "message": (
                    f"Completed fitting {update.successful_count} models "
                    f"({update.failed_count} failed)"
                ),
                "results": results,
                "successful_count": update.successful_count,
                "failed_count": update.failed_count,
                "total_time": round(elapsed_time, 2),
            }
            yield f"data: {json.dumps(event_data)}\n\n"


@router.post("/reference")
async def calculate_reference_dataset(
    current_user: Annotated[User, Depends(get_current_user)],
    request: ReferenceCalculationRequest,
    session: Session = Depends(get_session),
) -> ReferenceCalculationResponse:
    """
    Fit GAMLSS models for user's reference dataset.

    This endpoint fits statistical models for each specified brain structure,
    allowing percentile calculations for patient data.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    request : ReferenceCalculationRequest
        The calculation parameters.
    session : Session
        Database session.

    Returns
    -------
    ReferenceCalculationResponse
        Results of the model fitting operation.

    Raises
    ------
    HTTPException
        404 if no reference data found.
    """
    service = CalculationService(session)
    start_time = time.time()

    # Check if user has reference data
    df = service.get_reference_dataframe(current_user.id)
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No reference data found. Please upload data first.",
        )

    # Fit models (non-streaming version)
    results: dict[str, ModelResult] = {}
    successful_count = 0
    failed_count = 0

    async for update in service.fit_reference_models(
        user_id=current_user.id,
        y_columns=request.y_columns,
        percentiles=request.percentiles,
        criterion="bic",
    ):
        if isinstance(update, ReferenceCalculationResult):
            for structure, model_result in update.results.items():
                results[structure] = ModelResult(
                    structure=structure,
                    converged=model_result.converged,
                    aic=model_result.aic,
                    bic=model_result.bic,
                    family=model_result.family,
                    formula=model_result.formula,
                    percentile_curves=model_result.percentile_curves,
                    error=model_result.error,
                    plot_available=model_result.percentile_curves is not None,
                )
            successful_count = update.successful_count
            failed_count = update.failed_count

    elapsed_time = time.time() - start_time

    return ReferenceCalculationResponse(
        message=(
            f"Completed fitting {successful_count} models ({failed_count} failed)"
        ),
        results=results,
        successful_count=successful_count,
        failed_count=failed_count,
        total_time=round(elapsed_time, 2),
    )


@router.post("/reference/stream")
async def calculate_reference_dataset_stream(
    current_user: Annotated[User, Depends(get_current_user)],
    request: ReferenceCalculationRequest,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """
    Fit GAMLSS models with Server-Sent Events for progress updates.

    This endpoint streams progress updates during model fitting,
    providing real-time feedback on the calculation status.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    request : ReferenceCalculationRequest
        The calculation parameters.
    session : Session
        Database session.

    Returns
    -------
    StreamingResponse
        SSE stream with progress updates and final results.

    Raises
    ------
    HTTPException
        404 if no reference data found.
    """
    service = CalculationService(session)

    # Check if user has reference data
    df = service.get_reference_dataframe(current_user.id)
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No reference data found. Please upload data first.",
        )

    return StreamingResponse(
        generate_sse_events(service, current_user.id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/patient")
async def calculate_patient_percentiles(
    current_user: Annotated[User, Depends(get_current_user)],
    request: PatientCalculationRequest,
    session: Session = Depends(get_session),
) -> PatientCalculationResponse:
    """
    Calculate percentiles for patients against fitted reference models.

    This endpoint computes z-scores and percentiles for specified patients
    using previously fitted GAMLSS models.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    request : PatientCalculationRequest
        The calculation parameters.
    session : Session
        Database session.

    Returns
    -------
    PatientCalculationResponse
        Percentile results for each patient-structure combination.

    Raises
    ------
    HTTPException
        404 if no reference data found.
        400 if no fitted models available.
    """
    service = CalculationService(session)

    # Check if user has reference data
    df = service.get_reference_dataframe(current_user.id)
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No reference data found. Please upload data first.",
        )

    # Get patient IDs from indices
    patient_ids = None
    if request.patient_indices:
        try:
            patient_ids = df.iloc[request.patient_indices]["PatientID"].tolist()
        except IndexError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid patient indices provided.",
            ) from e

    # Calculate percentiles
    calc_results = service.calculate_patient_percentiles(
        user_id=current_user.id,
        patient_ids=patient_ids,
        structures=request.y_columns,
    )

    # Convert to response format
    results = [
        PatientResult(
            patient_id=r.patient_id,
            structure=r.structure,
            z_score=r.z_score,
            percentile=r.percentile,
            age=r.age,
            value=r.value,
            reference_mean=None,
            reference_sd=None,
        )
        for r in calc_results
    ]

    # Collect errors
    errors = [r.error for r in calc_results if r.error is not None]

    # Count unique patients and structures
    patients_processed = len({r.patient_id for r in calc_results})
    structures_processed = len({r.structure for r in calc_results})

    return PatientCalculationResponse(
        message=f"Calculated percentiles for {patients_processed} patients",
        results=results,
        patients_processed=patients_processed,
        structures_processed=structures_processed,
        errors=errors,
    )


@router.get("/structures")
async def get_available_structures(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Get list of available brain structures for calculation.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        List of available structure names.
    """
    service = CalculationService(session)

    structures = service.get_available_structures(current_user.id)

    return {
        "structures": structures,
        "count": len(structures),
    }
