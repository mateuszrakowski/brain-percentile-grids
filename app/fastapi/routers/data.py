"""
Data management endpoints for file uploads and retrieval.

Endpoints for uploading patient data to datasets and retrieving
dataset information.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from app.fastapi.db.database import get_session
from app.fastapi.db.models import ReferenceDataset
from app.fastapi.dependencies import get_user_dataset, get_validated_files
from app.fastapi.services.reference_data import ReferenceDataService
from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile

router = APIRouter(prefix="/api/datasets", tags=["data"])


# Response models
class ProcessingInfo(BaseModel):
    """Information about data processing results."""

    records_added: int
    duplicates_found: int
    files_processed: int
    total_records: int
    structures: list[str]


class UploadDataResponse(BaseModel):
    """Response model for data upload endpoint."""

    message: str
    dataset_id: int
    dataset_name: str
    processing_info: ProcessingInfo


class DataSample(BaseModel):
    """Sample patient data for preview."""

    patient_id: str
    study_date: str
    created_at: str


class GetDataResponse(BaseModel):
    """Response model for getting dataset data summary."""

    dataset_id: int
    dataset_name: str
    total_records: int
    structures: list[str]
    sample: list[DataSample]


class ClearDataResponse(BaseModel):
    """Response model for clearing dataset data."""

    message: str
    dataset_id: int
    records_deleted: int


class GetStructuresResponse(BaseModel):
    """Response model for getting dataset structures."""

    dataset_id: int
    structures: list[str]
    count: int


@router.post("/{dataset_id}/upload", response_model=UploadDataResponse)
async def upload_dataset_data(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    files: list[ValidatedFile] = Depends(get_validated_files),
    session: Session = Depends(get_session),
) -> UploadDataResponse:
    """
    Upload patient data files to a dataset.

    Processes uploaded files (CSV/Excel), detects duplicates within
    the dataset, and stores patient records.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    files : list[ValidatedFile]
        Validated uploaded files.
    session : Session
        Database session.

    Returns
    -------
    UploadDataResponse
        Processing result with statistics.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    # Process files to DataFrames
    processor = PatientDataProcessor()
    dataframes = processor.process_files(files)

    # Save to database with duplicate detection (scoped to dataset)
    service = ReferenceDataService(session)
    result = service.save_reference_data(dataset.id, dataframes)

    if result.records_added == 0 and result.duplicates_found == 0:
        message = "No new records to add"
    else:
        message = (
            f"Successfully added {result.records_added} records to '{dataset.name}'"
        )

    return UploadDataResponse(
        message=message,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        processing_info=ProcessingInfo(
            records_added=result.records_added,
            duplicates_found=result.duplicates_found,
            files_processed=result.files_processed,
            total_records=result.total_records,
            structures=result.structures,
        ),
    )


@router.get("/{dataset_id}/data", response_model=GetDataResponse)
async def get_dataset_data(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    session: Session = Depends(get_session),
) -> GetDataResponse:
    """
    Get summary of a dataset's patient data.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    session : Session
        Database session.

    Returns
    -------
    GetDataResponse
        Summary including total records, structures, and sample data.

    Raises
    ------
    HTTPException
        404 if dataset not found or has no data.
    """
    service = ReferenceDataService(session)
    summary = service.get_reference_summary(dataset.id)

    if summary is None:
        return GetDataResponse(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            total_records=0,
            structures=[],
            sample=[],
        )

    return GetDataResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        total_records=summary["total_records"],
        structures=summary["structures"],
        sample=[
            DataSample(
                patient_id=s["patient_id"],
                study_date=s["study_date"],
                created_at=s["created_at"],
            )
            for s in summary["sample"]
        ],
    )


@router.delete("/{dataset_id}/data", response_model=ClearDataResponse)
async def clear_dataset_data(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    session: Session = Depends(get_session),
) -> ClearDataResponse:
    """
    Clear all patient data from a dataset (keeps the dataset and models).

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    session : Session
        Database session.

    Returns
    -------
    ClearDataResponse
        Confirmation message with count of deleted records.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    service = ReferenceDataService(session)
    deleted_count = service.clear_reference_data(dataset.id)

    return ClearDataResponse(
        message=f"Deleted {deleted_count} records from '{dataset.name}'",
        dataset_id=dataset.id,
        records_deleted=deleted_count,
    )


@router.get("/{dataset_id}/structures", response_model=GetStructuresResponse)
async def get_dataset_structures(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    session: Session = Depends(get_session),
) -> GetStructuresResponse:
    """
    Get list of available brain structures in a dataset.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    session : Session
        Database session.

    Returns
    -------
    GetStructuresResponse
        List of available structure names.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    service = ReferenceDataService(session)
    structures = service.get_available_structures(dataset.id)

    return GetStructuresResponse(
        dataset_id=dataset.id,
        structures=structures,
        count=len(structures),
    )
