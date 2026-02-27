"""
Service for managing reference dataset operations.

This module handles database operations for patient reference data,
including duplicate detection and batch inserts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import delete as sa_delete
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.fastapi.db.models import (
    PatientRecord,
    PatientStructureValue,
    ReferenceDataset,
)
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


@dataclass
class SampleRecord:
    """A sample patient record for preview."""

    patient_id: str
    study_date: str
    created_at: str


@dataclass
class ReferenceSummary:
    """Summary of a dataset's reference data."""

    total_records: int = 0
    structures: list[str] = field(default_factory=list)
    sample: list[SampleRecord] = field(default_factory=list)


class ReferenceDataService:
    """
    Service for managing reference dataset operations.

    Handles database operations including duplicate detection,
    batch inserts, and data retrieval. All operations are scoped
    to a specific dataset.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_reference_data(
        self,
        dataset_id: int,
        dataframes: list[pd.DataFrame],
    ) -> ProcessingResult:
        """
        Save processed DataFrames to the database with duplicate detection.

        Duplicates are detected within the dataset scope only.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to add records to.
        dataframes : list[pd.DataFrame]
            List of processed DataFrames to save.

        Returns
        -------
        ProcessingResult
            Statistics about the processing operation.
        """
        result = ProcessingResult(files_processed=len(dataframes))

        if not dataframes:
            result.total_records = self._count_dataset_records(dataset_id)
            return result

        # Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Get existing records for duplicate detection (scoped to dataset)
        existing_keys = self._get_existing_keys(dataset_id)

        # Filter out duplicates
        if existing_keys:
            combined_df, duplicates = self._remove_duplicates(
                combined_df, existing_keys
            )
            result.duplicates_found = duplicates

        # Save new records
        if not combined_df.empty:
            result.records_added = self._insert_records(dataset_id, combined_df)
            result.structures = PatientDataProcessor.get_structure_columns(combined_df)

            # Update dataset sample count
            self._update_dataset_sample_count(dataset_id)

        self.session.commit()

        result.total_records = self._count_dataset_records(dataset_id)
        return result

    def get_reference_summary(self, dataset_id: int) -> ReferenceSummary:
        """
        Get summary of a dataset's reference data.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to get data for.

        Returns
        -------
        ReferenceSummary
            Summary with total records, structures, and sample data.
        """
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.dataset_id == dataset_id)
        ).all()

        if not records:
            return ReferenceSummary()

        structure_names = self.session.exec(
            select(PatientStructureValue.structure_name)
            .join(PatientRecord)
            .where(PatientRecord.dataset_id == dataset_id)
            .distinct()
        ).all()

        return ReferenceSummary(
            total_records=len(records),
            structures=list(structure_names),
            sample=[
                SampleRecord(
                    patient_id=r.patient_id,
                    study_date=r.study_date,
                    created_at=r.created_at.isoformat(),
                )
                for r in records[:5]
            ],
        )

    def get_reference_dataframe(self, dataset_id: int) -> pd.DataFrame:
        """
        Retrieve dataset's reference data as a DataFrame.

        Uses eager loading to avoid N+1 query problem.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to retrieve data for.

        Returns
        -------
        pd.DataFrame
            DataFrame with patient records and structure values.
            Empty DataFrame if no data exists.
        """
        # Use eager loading to fetch all structure values in a single query
        records = self.session.exec(
            select(PatientRecord)
            .where(PatientRecord.dataset_id == dataset_id)
            .options(selectinload(PatientRecord.structure_values))
        ).all()

        if not records:
            return pd.DataFrame()

        # Build DataFrame from records (structure_values already loaded)
        data_rows = []
        for record in records:
            row = {
                "PatientID": record.patient_id,
                "BirthDate": record.birth_date,
                "StudyDate": record.study_date,
                "StudyDescription": record.study_description,
                "AgeYears": record.age_years,
                "AgeMonths": record.age_months,
            }

            for sv in record.structure_values:
                row[sv.structure_name] = sv.value

            data_rows.append(row)

        return pd.DataFrame(data_rows)

    def clear_reference_data(self, dataset_id: int) -> int:
        """
        Clear all reference data for a dataset.

        Uses bulk delete for efficient database operations.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to clear data for.

        Returns
        -------
        int
            Number of records deleted.
        """
        # Get record IDs for counting
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.dataset_id == dataset_id)
        ).all()
        record_ids = [r.id for r in records]
        record_count = len(record_ids)

        if record_ids:
            # Bulk delete structure values first (foreign key constraint)
            self.session.exec(
                sa_delete(PatientStructureValue).where(
                    PatientStructureValue.patient_record_id.in_(record_ids)
                )
            )

            # Bulk delete patient records
            self.session.exec(
                sa_delete(PatientRecord).where(PatientRecord.dataset_id == dataset_id)
            )

        # Update dataset sample count
        self._update_dataset_sample_count(dataset_id, count=0)

        self.session.commit()
        return record_count

    def get_available_structures(self, dataset_id: int) -> list[str]:
        """
        Get list of available structure columns for a dataset.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to get structures for.

        Returns
        -------
        list[str]
            List of structure column names.
        """
        structure_names = self.session.exec(
            select(PatientStructureValue.structure_name)
            .join(PatientRecord)
            .where(PatientRecord.dataset_id == dataset_id)
            .distinct()
        ).all()

        return list(structure_names)

    def _get_existing_keys(self, dataset_id: int) -> set[str]:
        """
        Get composite keys for existing records in a dataset.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to get keys for.

        Returns
        -------
        set[str]
            Set of composite keys (PatientID_StudyDate_StudyDescription).
        """
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.dataset_id == dataset_id)
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

    def _insert_records(self, dataset_id: int, df: pd.DataFrame) -> int:
        """
        Insert patient records and structure values into the database.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to add records to.
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
            # Get age values (already calculated in process_csv_input)
            age_years = row.get("AgeYears")
            age_months = row.get("AgeMonths")

            # Create patient record with pre-calculated age
            patient_record = PatientRecord(
                dataset_id=dataset_id,
                patient_id=str(row.get("PatientID", "")),
                birth_date=str(row.get("BirthDate", "")),
                study_date=str(row.get("StudyDate", "")),
                study_description=row.get("StudyDescription"),
                age_years=int(age_years) if pd.notna(age_years) else None,
                age_months=int(age_months) if pd.notna(age_months) else None,
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

    def _count_dataset_records(self, dataset_id: int) -> int:
        """Count total records for a dataset."""
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.dataset_id == dataset_id)
        ).all()
        return len(records)

    def _update_dataset_sample_count(
        self, dataset_id: int, count: int | None = None
    ) -> None:
        """Update the sample count on the dataset record."""
        dataset = self.session.exec(
            select(ReferenceDataset).where(ReferenceDataset.id == dataset_id)
        ).first()

        if dataset:
            if count is not None:
                dataset.sample_count = count
            else:
                dataset.sample_count = self._count_dataset_records(dataset_id)
            self.session.add(dataset)
