"""
Data management endpoints for file uploads and retrieval.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.db.database import get_session
from app.fastapi.db.models import User
from app.fastapi.dependencies import get_validated_files
from app.fastapi.services.reference_data import ReferenceDataService
from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/upload/reference")
async def upload_reference_data(
    current_user: Annotated[User, Depends(get_current_user)],
    files: list[ValidatedFile] = Depends(get_validated_files),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Upload reference dataset files (CSV/Excel).

    Processes uploaded files, detects duplicates, and stores
    patient records in the database.

    Parameters
    ----------
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
    """
    user_id = current_user.id

    # Process files to DataFrames
    processor = PatientDataProcessor()
    dataframes = processor.process_files(files)

    # Save to database with duplicate detection
    service = ReferenceDataService(session)
    result = service.save_reference_data(user_id, dataframes)

    if result.records_added == 0 and result.duplicates_found == 0:
        return {
            "message": "No new records to add",
            "processing_info": result.to_dict(),
        }

    return {
        "message": f"Successfully added {result.records_added} records",
        "processing_info": result.to_dict(),
    }


@router.get("/reference")
async def get_reference_data(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Get summary of user's reference dataset.

    Parameters
    ----------
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
        404 if no reference data found.
    """
    user_id = current_user.id

    service = ReferenceDataService(session)
    summary = service.get_reference_summary(user_id)

    if summary is None:
        raise HTTPException(status_code=404, detail="No reference data found")

    return summary


@router.delete("/reference")
async def clear_reference_data(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """
    Clear all reference data for the user.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    dict[str, str]
        Confirmation message with count of deleted records.
    """
    user_id = current_user.id

    service = ReferenceDataService(session)
    deleted_count = service.clear_reference_data(user_id)

    return {"message": f"Deleted {deleted_count} records"}
