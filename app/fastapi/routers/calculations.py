"""
Calculation endpoints for GAMLSS modeling and percentile calculations.

Provides endpoints for:
- Fitting reference models with SSE progress updates
- Calculating patient percentiles against fitted models (via file upload)
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.fastapi.db.database import get_session
from app.fastapi.db.models import ReferenceDataset
from app.fastapi.dependencies import get_user_dataset, get_validated_files
from app.fastapi.models.requests import ReferenceCalculationRequest
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
from app.fastapi.services.reference_data import ReferenceDataService
from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["calculations"])


async def generate_sse_events(
    service: CalculationService,
    user_id: int,
    dataset_id: int,
    request: ReferenceCalculationRequest,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for model fitting progress.

    Parameters
    ----------
    service : CalculationService
        The calculation service instance.
    user_id : int
        The user ID.
    dataset_id : int
        The dataset ID.
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
        dataset_id=dataset_id,
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


@router.post("/{dataset_id}/fit")
async def fit_dataset_models(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    request: ReferenceCalculationRequest,
    session: Session = Depends(get_session),
) -> ReferenceCalculationResponse:
    """
    Fit GAMLSS models for a dataset's reference data.

    This endpoint fits statistical models for each specified brain structure,
    allowing percentile calculations for patient data. Models are persisted
    to disk for later use.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
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
        404 if dataset not found or has no reference data.
    """
    service = CalculationService(session)
    reference_service = ReferenceDataService(session)
    start_time = time.time()

    # Check if dataset has reference data
    df = reference_service.get_reference_dataframe(dataset.id)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reference data found in dataset '{dataset.name}'. Please upload data first.",
        )

    if len(df) < CalculationService.MIN_SAMPLES_FOR_MODEL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Dataset has {len(df)} samples, but at least "
                f"{CalculationService.MIN_SAMPLES_FOR_MODEL} are required "
                f"for model fitting."
            ),
        )

    # Fit models (non-streaming version)
    results: dict[str, ModelResult] = {}
    successful_count = 0
    failed_count = 0

    async for update in service.fit_reference_models(
        user_id=dataset.user_id,
        dataset_id=dataset.id,
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


@router.post("/{dataset_id}/fit/stream")
async def fit_dataset_models_stream(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    request: ReferenceCalculationRequest,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """
    Fit GAMLSS models with Server-Sent Events for progress updates.

    This endpoint streams progress updates during model fitting,
    providing real-time feedback on the calculation status.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
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
        404 if dataset not found or has no reference data.
    """
    reference_service = ReferenceDataService(session)

    # Check if dataset has reference data
    df = reference_service.get_reference_dataframe(dataset.id)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reference data found in dataset '{dataset.name}'. Please upload data first.",
        )

    if len(df) < CalculationService.MIN_SAMPLES_FOR_MODEL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Dataset has {len(df)} samples, but at least "
                f"{CalculationService.MIN_SAMPLES_FOR_MODEL} are required "
                f"for model fitting."
            ),
        )

    service = CalculationService(session)

    return StreamingResponse(
        generate_sse_events(service, dataset.user_id, dataset.id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{dataset_id}/calculate")
async def calculate_oos_percentiles(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    files: list[ValidatedFile] = Depends(get_validated_files),
    structures: Annotated[
        list[str] | None,
        Query(description="Structures to calculate (None = all available)"),
    ] = None,
    session: Session = Depends(get_session),
) -> PatientCalculationResponse:
    """
    Calculate percentiles for out-of-sample patients against fitted models.

    Upload patient data files (CSV/XLSX) in the same format as reference data.
    The endpoint computes z-scores and percentiles for patients using
    previously fitted GAMLSS models. Patient data is NOT stored - it's
    processed transiently and results are returned immediately.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    files : list[ValidatedFile]
        Uploaded patient data files (CSV/XLSX).
    structures : list[str] | None
        Structures to calculate. If None, uses all available models.
    session : Session
        Database session.

    Returns
    -------
    PatientCalculationResponse
        Percentile results for each patient-structure combination.

    Raises
    ------
    HTTPException
        400 if no files provided or files cannot be processed.
        404 if dataset not found or has no fitted models.
    """

    # Process uploaded files to DataFrames (NOT stored in database)
    processor = PatientDataProcessor()
    try:
        dataframes = processor.process_files(files)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing files: {e}",
        ) from e

    if not dataframes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid patient data found in uploaded files",
        )

    # Combine all DataFrames into one
    patient_df = pd.concat(dataframes, ignore_index=True)

    logger.info(
        f"Processing {len(patient_df)} OOS patients for dataset {dataset.id} "
        f"(from {len(files)} files)"
    )

    # Calculate percentiles
    service = CalculationService(session)
    calc_results = service.calculate_patient_percentiles(
        dataset_id=dataset.id,
        patient_data=patient_df,
        structures=structures,
    )

    if not calc_results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No fitted models found for dataset '{dataset.name}'. Please fit models first.",
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
            is_extrapolated=r.is_extrapolated,
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
