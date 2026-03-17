"""
Calculation endpoints for GAMLSS modeling and percentile calculations.

Provides endpoints for:
- Fitting reference models with SSE progress updates
- Calculating patient percentiles against fitted models (via file upload)
- Retrieving persisted OOS calculation results and plots
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session

from app.fastapi.db.database import get_session
from app.fastapi.db.models import ReferenceDataset
from app.fastapi.dependencies import get_user_dataset, get_validated_files
from app.fastapi.models.requests import ReferenceCalculationRequest
from app.fastapi.models.responses import (
    ModelResult,
    OOSCalculationDetailResponse,
    OOSCalculationListResponse,
    OOSCalculationSummary,
    PatientCalculationResponse,
    PatientResult,
    ReferenceCalculationResponse,
    SavedPatientResult,
)
from app.fastapi.services.calculation import (
    CalculationProgress,
    CalculationService,
    ReferenceCalculationResult,
)
from app.fastapi.services.model_persistence import ModelPersistenceService
from app.fastapi.services.oos_persistence import OOSPersistenceService
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

    source_filenames = ", ".join(f.name for f in files if f.name)

    logger.info(
        f"Processing {len(patient_df)} OOS patients for dataset {dataset.id} "
        f"(from {len(files)} files)"
    )

    # Calculate and persist percentiles
    service = CalculationService(session)
    calculation_id, calc_results = service.calculate_and_persist_patient_percentiles(
        user_id=dataset.user_id,
        dataset_id=dataset.id,
        patient_data=patient_df,
        source_filenames=source_filenames,
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
        calculation_id=calculation_id,
        results=results,
        patients_processed=patients_processed,
        structures_processed=structures_processed,
        errors=errors,
    )


@router.get("/{dataset_id}/calculations")
async def list_oos_calculations(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    include_stale: bool = Query(False, description="Include stale calculations"),
    session: Session = Depends(get_session),
) -> OOSCalculationListResponse:
    """
    List saved OOS calculations for a dataset.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    include_stale : bool
        Whether to include calculations marked stale after model re-fitting.
    session : Session
        Database session.

    Returns
    -------
    OOSCalculationListResponse
        List of calculation summaries.
    """
    oos_service = OOSPersistenceService(session)
    calcs = oos_service.get_dataset_calculations(
        dataset.id, include_stale=include_stale
    )

    summaries = []
    for calc in calcs:
        results = oos_service.get_calculation_results(calc.id)
        summaries.append(
            OOSCalculationSummary(
                id=calc.id,
                dataset_id=calc.dataset_id,
                source_filenames=calc.source_filenames,
                patients_count=len({r.patient_id for r in results}),
                structures_count=len({r.structure for r in results}),
                is_stale=calc.is_stale,
                created_at=calc.created_at,
            )
        )

    return OOSCalculationListResponse(
        calculations=summaries,
        total=len(summaries),
    )


@router.get("/{dataset_id}/calculations/{calculation_id}")
async def get_oos_calculation(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    calculation_id: int,
    session: Session = Depends(get_session),
) -> OOSCalculationDetailResponse:
    """
    Get results for a specific OOS calculation.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    calculation_id : int
        The calculation ID.
    session : Session
        Database session.

    Returns
    -------
    OOSCalculationDetailResponse
        Full calculation details with per-patient results.

    Raises
    ------
    HTTPException
        404 if calculation not found or doesn't belong to dataset.
    """
    from app.fastapi.db.models import OOSCalculation

    oos_service = OOSPersistenceService(session)
    calc = session.get(OOSCalculation, calculation_id)
    if calc is None or calc.dataset_id != dataset.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calculation not found",
        )

    db_results = oos_service.get_calculation_results(calculation_id)

    results = [
        SavedPatientResult(
            id=r.id,
            calculation_id=r.calculation_id,
            patient_id=r.patient_id,
            structure=r.structure,
            age=r.age,
            value=r.value,
            z_score=r.z_score,
            percentile=r.percentile,
            is_extrapolated=r.is_extrapolated,
            error=r.error,
            has_plot=r.plot_path is not None,
        )
        for r in db_results
    ]

    return OOSCalculationDetailResponse(
        id=calc.id,
        dataset_id=calc.dataset_id,
        source_filenames=calc.source_filenames,
        is_stale=calc.is_stale,
        created_at=calc.created_at,
        results=results,
    )


@router.get("/{dataset_id}/calculations/{calculation_id}/results/{result_id}/plot")
async def get_oos_patient_plot(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    calculation_id: int,
    result_id: int,
    session: Session = Depends(get_session),
) -> FileResponse:
    """
    Serve the OOS patient plot PNG for a specific result.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    calculation_id : int
        The calculation ID.
    result_id : int
        The result ID.
    session : Session
        Database session.

    Returns
    -------
    FileResponse
        The plot PNG file.

    Raises
    ------
    HTTPException
        404 if result or plot not found.
    """
    oos_service = OOSPersistenceService(session)
    plot_path = oos_service.get_plot_path(result_id)
    if plot_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plot not found",
        )
    return FileResponse(plot_path, media_type="image/png")


@router.get("/{dataset_id}/models/{structure}/reference-plot")
async def get_reference_plot(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    structure: str,
    session: Session = Depends(get_session),
) -> FileResponse:
    """
    Serve the reference percentile grid plot for a fitted model.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    structure : str
        The brain structure name.
    session : Session
        Database session.

    Returns
    -------
    FileResponse
        The reference plot PNG file.

    Raises
    ------
    HTTPException
        404 if model or plot not found.
    """
    persistence_service = ModelPersistenceService(session)
    plot_path = persistence_service.get_reference_plot_path(dataset.id, structure)
    if plot_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference plot not found for structure '{structure}'",
        )
    return FileResponse(plot_path, media_type="image/png")


@router.delete("/{dataset_id}/calculations/{calculation_id}")
async def delete_oos_calculation(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    calculation_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """
    Delete an OOS calculation and its plot files.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    calculation_id : int
        The calculation ID to delete.
    session : Session
        Database session.

    Returns
    -------
    dict
        Confirmation message.

    Raises
    ------
    HTTPException
        404 if calculation not found or doesn't belong to dataset.
    """
    from app.fastapi.db.models import OOSCalculation

    calc = session.get(OOSCalculation, calculation_id)
    if calc is None or calc.dataset_id != dataset.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calculation not found",
        )

    oos_service = OOSPersistenceService(session)
    oos_service.delete_calculation(dataset.user_id, calculation_id)

    return {"message": f"Calculation {calculation_id} deleted"}
