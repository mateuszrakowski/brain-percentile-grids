"""
Data management endpoints for file uploads and retrieval.
"""

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.fastapi.db.database import get_session
from app.fastapi.db.models import PatientRecord, PatientStructureValue
from app.fastapi.dependencies import get_validated_files
from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile

router = APIRouter(prefix="/api/data", tags=["data"])

# TODO: Add authentication dependency
# from app.fastapi.auth import get_current_user
# current_user: User = Depends(get_current_user)

# Temporary: hardcoded user_id until auth is implemented
TEMP_USER_ID = 1


@router.post("/upload/reference")
async def upload_reference_data(
    files: list[ValidatedFile] = Depends(get_validated_files),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Upload reference dataset files (CSV/Excel).

    Processes uploaded files, detects duplicates, and stores
    patient records in the database.
    """
    # TODO: Get user_id from authenticated user
    user_id = TEMP_USER_ID

    # Get current reference data for this user (for duplicate detection)
    existing_records = session.exec(
        select(PatientRecord).where(PatientRecord.user_id == user_id)
    ).all()

    # Convert existing records to DataFrame for duplicate detection
    current_df = None
    if existing_records:
        current_df = pd.DataFrame([
            {
                "PatientID": r.patient_id,
                "StudyDate": r.study_date,
                "StudyDescription": r.study_description,
            }
            for r in existing_records
        ])

    # Process uploaded files
    processor = PatientDataProcessor()
    final_df, processing_info = processor.update_reference_dataset(
        current_df_state=current_df,
        uploaded_files=files,
    )

    if final_df.empty:
        return {
            "message": "No new records to add",
            "processing_info": processing_info,
        }

    # Store new records in database
    records_added = 0
    for _, row in final_df.iterrows():
        # Create patient record
        patient_record = PatientRecord(
            user_id=user_id,
            patient_id=str(row.get("PatientID", "")),
            birth_date=str(row.get("BirthDate", "")),
            study_date=str(row.get("StudyDate", "")),
            study_description=row.get("StudyDescription"),
        )
        session.add(patient_record)
        session.flush()  # Get the ID

        # Add structure values
        metadata_cols = ["Filename", "PatientID", "AgeYears", "AgeMonths",
                         "BirthDate", "StudyDate", "StudyDescription"]
        for col in final_df.columns:
            if col not in metadata_cols:
                value = row.get(col)
                if value is not None and not pd.isna(value):
                    structure_value = PatientStructureValue(
                        patient_record_id=patient_record.id,  # type: ignore
                        structure_name=col,
                        value=float(value),
                    )
                    session.add(structure_value)

        records_added += 1

    session.commit()

    return {
        "message": f"Successfully added {records_added} records",
        "processing_info": processing_info,
        "total_records": len(existing_records) + records_added,
    }


@router.get("/reference")
async def get_reference_data(
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Get summary of user's reference dataset.
    """
    # TODO: Get user_id from authenticated user
    user_id = TEMP_USER_ID

    records = session.exec(
        select(PatientRecord).where(PatientRecord.user_id == user_id)
    ).all()

    if not records:
        raise HTTPException(status_code=404, detail="No reference data found")

    # Get unique structures
    structure_names = session.exec(
        select(PatientStructureValue.structure_name)
        .join(PatientRecord)
        .where(PatientRecord.user_id == user_id)
        .distinct()
    ).all()

    return {
        "total_records": len(records),
        "structures": list(structure_names),
        "sample": [
            {
                "patient_id": r.patient_id,
                "study_date": r.study_date,
                "created_at": r.created_at.isoformat(),
            }
            for r in records[:5]
        ],
    }


@router.delete("/reference")
async def clear_reference_data(
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """
    Clear all reference data for the user.
    """
    # TODO: Get user_id from authenticated user
    user_id = TEMP_USER_ID

    records = session.exec(
        select(PatientRecord).where(PatientRecord.user_id == user_id)
    ).all()

    for record in records:
        session.delete(record)

    session.commit()

    return {"message": f"Deleted {len(records)} records"}
