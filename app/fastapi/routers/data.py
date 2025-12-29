"""
Data management endpoints for file uploads and retrieval.

Endpoints for uploading patient data to datasets and retrieving
dataset information.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.db.database import get_session
from app.fastapi.db.models import ReferenceDataset, User
from app.fastapi.dependencies import get_validated_files
from app.fastapi.services.reference_data import ReferenceDataService
from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile

router = APIRouter(prefix="/api/datasets", tags=["data"])


async def get_user_dataset(
    dataset_id: int,
    current_user: User,
    session: Session,
) -> ReferenceDataset:
    """
    Get a dataset belonging to the current user.

    Parameters
    ----------
    dataset_id : int
        The dataset ID.
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    ReferenceDataset
        The dataset.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    dataset = session.exec(
        select(ReferenceDataset).where(
            ReferenceDataset.id == dataset_id,
            ReferenceDataset.user_id == current_user.id,
        )
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    return dataset


@router.post("/{dataset_id}/upload")
async def upload_dataset_data(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    files: list[ValidatedFile] = Depends(get_validated_files),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Upload patient data files to a dataset.

    Processes uploaded files (CSV/Excel), detects duplicates within
    the dataset, and stores patient records.

    Parameters
    ----------
    dataset_id : int
        The dataset ID to upload to.
    current_user : User
        The authenticated user.
    files : list[ValidatedFile]
        Validated uploaded files.
    session : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        Processing result with statistics.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    # Verify dataset belongs to user
    dataset = await get_user_dataset(dataset_id, current_user, session)

    # Process files to DataFrames
    processor = PatientDataProcessor()
    dataframes = processor.process_files(files)

    # Save to database with duplicate detection (scoped to dataset)
    service = ReferenceDataService(session)
    result = service.save_reference_data(dataset_id, dataframes)

    if result.records_added == 0 and result.duplicates_found == 0:
        return {
            "message": "No new records to add",
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "processing_info": result.to_dict(),
        }

    return {
        "message": f"Successfully added {result.records_added} records to '{dataset.name}'",
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "processing_info": result.to_dict(),
    }


@router.get("/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Get summary of a dataset's patient data.

    Parameters
    ----------
    dataset_id : int
        The dataset ID.
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        Summary including total records, structures, and sample data.

    Raises
    ------
    HTTPException
        404 if dataset not found or has no data.
    """
    # Verify dataset belongs to user
    dataset = await get_user_dataset(dataset_id, current_user, session)

    service = ReferenceDataService(session)
    summary = service.get_reference_summary(dataset_id)

    if summary is None:
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "total_records": 0,
            "structures": [],
            "sample": [],
        }

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        **summary,
    }


@router.delete("/{dataset_id}/data")
async def clear_dataset_data(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Clear all patient data from a dataset (keeps the dataset and models).

    Parameters
    ----------
    dataset_id : int
        The dataset ID.
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        Confirmation message with count of deleted records.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    # Verify dataset belongs to user
    dataset = await get_user_dataset(dataset_id, current_user, session)

    service = ReferenceDataService(session)
    deleted_count = service.clear_reference_data(dataset_id)

    return {
        "message": f"Deleted {deleted_count} records from '{dataset.name}'",
        "dataset_id": dataset_id,
        "records_deleted": deleted_count,
    }


@router.get("/{dataset_id}/structures")
async def get_dataset_structures(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Get list of available brain structures in a dataset.

    Parameters
    ----------
    dataset_id : int
        The dataset ID.
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        List of available structure names.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    # Verify dataset belongs to user
    await get_user_dataset(dataset_id, current_user, session)

    service = ReferenceDataService(session)
    structures = service.get_available_structures(dataset_id)

    return {
        "dataset_id": dataset_id,
        "structures": structures,
        "count": len(structures),
    }
