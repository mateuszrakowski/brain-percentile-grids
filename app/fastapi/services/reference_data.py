"""
Service for managing reference dataset operations.

This module handles database operations for patient reference data,
including duplicate detection and batch inserts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from app.fastapi.db.models import PatientRecord, PatientStructureValue
from app.fastapi.utils.file_utils import PatientDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing reference data upload."""

    records_added: int = 0
    duplicates_found: int = 0
    files_processed: int = 0
    total_records: int = 0
    structures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "records_added": self.records_added,
            "duplicates_found": self.duplicates_found,
            "files_processed": self.files_processed,
            "total_records": self.total_records,
            "structures": self.structures,
        }


class ReferenceDataService:
    """
    Service for managing reference dataset operations.

    Handles database operations including duplicate detection,
    batch inserts, and data retrieval.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_reference_data(
        self,
        user_id: int,
        dataframes: list[pd.DataFrame],
    ) -> ProcessingResult:
        """
        Save processed DataFrames to the database with duplicate detection.

        Parameters
        ----------
        user_id : int
            The user ID to associate records with.
        dataframes : list[pd.DataFrame]
            List of processed DataFrames to save.

        Returns
        -------
        ProcessingResult
            Statistics about the processing operation.
        """
        result = ProcessingResult(files_processed=len(dataframes))

        if not dataframes:
            result.total_records = self._count_user_records(user_id)
            return result

        # Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Get existing records for duplicate detection
        existing_keys = self._get_existing_keys(user_id)

        # Filter out duplicates
        if existing_keys:
            combined_df, duplicates = self._remove_duplicates(
                combined_df, existing_keys
            )
            result.duplicates_found = duplicates

        # Save new records
        if not combined_df.empty:
            result.records_added = self._insert_records(user_id, combined_df)
            result.structures = PatientDataProcessor.get_structure_columns(combined_df)

        self.session.commit()

        result.total_records = self._count_user_records(user_id)
        return result

    def get_reference_summary(self, user_id: int) -> dict[str, Any] | None:
        """
        Get summary of user's reference dataset.

        Parameters
        ----------
        user_id : int
            The user ID to get data for.

        Returns
        -------
        dict[str, Any] | None
            Summary dictionary or None if no data exists.
        """
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.user_id == user_id)
        ).all()

        if not records:
            return None

        structure_names = self.session.exec(
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

    def clear_reference_data(self, user_id: int) -> int:
        """
        Clear all reference data for a user.

        Parameters
        ----------
        user_id : int
            The user ID to clear data for.

        Returns
        -------
        int
            Number of records deleted.
        """
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.user_id == user_id)
        ).all()

        for record in records:
            self.session.delete(record)

        self.session.commit()
        return len(records)

    def _get_existing_keys(self, user_id: int) -> set[str]:
        """
        Get composite keys for existing records.

        Parameters
        ----------
        user_id : int
            The user ID to get keys for.

        Returns
        -------
        set[str]
            Set of composite keys (PatientID_StudyDate_StudyDescription).
        """
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.user_id == user_id)
        ).all()

        return {
            f"{r.patient_id}_{r.study_date}_{r.study_description or ''}"
            for r in records
        }

    def _remove_duplicates(
        self,
        df: pd.DataFrame,
        existing_keys: set[str],
    ) -> tuple[pd.DataFrame, int]:
        """
        Remove rows that already exist in the database.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to filter.
        existing_keys : set[str]
            Set of existing composite keys.

        Returns
        -------
        tuple[pd.DataFrame, int]
            Filtered DataFrame and count of duplicates removed.
        """
        # Create composite keys for new data
        new_keys = (
            df[PatientDataProcessor.UNIQUE_COLUMNS]
            .fillna("")
            .astype(str)
            .agg("_".join, axis=1)
        )

        # Find duplicates
        is_duplicate = new_keys.isin(existing_keys)
        duplicates_count = is_duplicate.sum()

        if duplicates_count > 0:
            logger.info(f"Found {duplicates_count} duplicate records, skipping")
            df = df[~is_duplicate].reset_index(drop=True)

        return df, int(duplicates_count)

    def _insert_records(self, user_id: int, df: pd.DataFrame) -> int:
        """
        Insert patient records and structure values into the database.

        Parameters
        ----------
        user_id : int
            The user ID to associate records with.
        df : pd.DataFrame
            DataFrame containing patient data.

        Returns
        -------
        int
            Number of records inserted.
        """
        structure_columns = PatientDataProcessor.get_structure_columns(df)
        records_added = 0

        for _, row in df.iterrows():
            # Create patient record
            patient_record = PatientRecord(
                user_id=user_id,
                patient_id=str(row.get("PatientID", "")),
                birth_date=str(row.get("BirthDate", "")),
                study_date=str(row.get("StudyDate", "")),
                study_description=row.get("StudyDescription"),
            )
            self.session.add(patient_record)
            self.session.flush()  # Get the ID

            # Add structure values
            for col in structure_columns:
                value = row.get(col)
                if value is not None and not pd.isna(value):
                    structure_value = PatientStructureValue(
                        patient_record_id=patient_record.id,  # type: ignore
                        structure_name=col,
                        value=float(value),
                    )
                    self.session.add(structure_value)

            records_added += 1

        logger.info(f"Inserted {records_added} patient records")
        return records_added

    def _count_user_records(self, user_id: int) -> int:
        """Count total records for a user."""
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.user_id == user_id)
        ).all()
        return len(records)
