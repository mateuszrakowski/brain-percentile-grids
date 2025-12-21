"""
File processing utilities for FastAPI app.

This module provides file validation and DataFrame conversion utilities.
"""

import io
import logging
import os

import pandas as pd
from fastapi import UploadFile

from app.core.data_processing.process_input import (
    process_csv_input,
    sum_structure_volumes,
)

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Custom exception for file validation errors."""

    pass


class ValidatedFile:
    """
    Wrapper for FastAPI UploadFile with validated content.

    Attributes
    ----------
    name : str
        Sanitized filename.
    extension : str
        File extension (lowercase, including dot).
    content : bytes
        File bytes content.
    content_type : str | None
        MIME type.
    """

    def __init__(self, upload_file: UploadFile, content: bytes):
        self.name = self._secure_filename(upload_file.filename or "unknown")
        _, self.extension = os.path.splitext(self.name.lower())
        self.content = content
        self.content_type = upload_file.content_type

    def _secure_filename(self, filename: str) -> str:
        """Remove dangerous characters from filename."""
        return "".join(c for c in filename if c.isalnum() or c in "._-")

    def read(self) -> bytes:
        """Return file content as bytes."""
        return self.content

    def to_buffer(self) -> io.BytesIO:
        """Return file content as BytesIO buffer for pandas."""
        return io.BytesIO(self.content)


class PatientDataProcessor:
    """
    Converts validated files to processed DataFrames.

    This class handles the conversion of uploaded patient data files (CSV/Excel)
    into processed pandas DataFrames with calculated age and aggregated
    brain structure volumes.
    """

    # Metadata columns that are not brain structure measurements
    METADATA_COLUMNS = [
        "Filename",
        "PatientID",
        "AgeYears",
        "AgeMonths",
        "BirthDate",
        "StudyDate",
        "StudyDescription",
    ]

    # Columns used for duplicate detection
    UNIQUE_COLUMNS = ["PatientID", "StudyDate", "StudyDescription"]

    def process_files(
        self, files: list[ValidatedFile]
    ) -> list[pd.DataFrame]:
        """
        Process validated files into DataFrames.

        Parameters
        ----------
        files : list[ValidatedFile]
            List of validated files to process.

        Returns
        -------
        list[pd.DataFrame]
            List of processed DataFrames, one per file.

        Raises
        ------
        ValueError
            If a file cannot be read or processed.
        """
        processed = []

        for file in files:
            df = self._read_and_process_file(file)
            if not df.empty:
                processed.append(df)

        logger.info(f"Processed {len(processed)} of {len(files)} files")
        return processed

    def _read_and_process_file(self, file: ValidatedFile) -> pd.DataFrame:
        """
        Read and process a single file.

        Parameters
        ----------
        file : ValidatedFile
            The validated file to process.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with metadata and structure volumes.
        """
        try:
            raw_df = self._read_file(file)

            if raw_df.empty:
                logger.warning(f"File {file.name} is empty")
                return pd.DataFrame()

            # Apply core processing pipeline
            processed_df = process_csv_input(raw_df)
            processed_df = sum_structure_volumes(processed_df)
            processed_df["Filename"] = file.name

            # Clean and reorder columns
            processed_df = processed_df.dropna(how="all")
            processed_df = self._reorder_columns(processed_df)

            logger.info(
                f"Processed {file.name}: {len(processed_df)} rows, "
                f"{len(processed_df.columns)} columns"
            )
            return processed_df

        except Exception as e:
            error_msg = f"Error processing {file.name}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def _read_file(self, file: ValidatedFile) -> pd.DataFrame:
        """
        Read file content into a raw DataFrame.

        Parameters
        ----------
        file : ValidatedFile
            The validated file to read.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame from file content.
        """
        if file.extension == ".csv":
            return pd.read_csv(
                file.to_buffer(),
                encoding="utf-8",
                low_memory=False,
            )
        elif file.extension == ".xlsx":
            return pd.read_excel(
                file.to_buffer(),
                engine="openpyxl",
            )
        elif file.extension == ".xls":
            return pd.read_excel(
                file.to_buffer(),
                engine="xlrd",
            )
        else:
            raise ValueError(f"Unsupported file type: {file.extension}")

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns with metadata first, then structures sorted.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to reorder.

        Returns
        -------
        pd.DataFrame
            DataFrame with reordered columns.
        """
        metadata_cols = [c for c in self.METADATA_COLUMNS if c in df.columns]
        structure_cols = sorted(
            c for c in df.columns if c not in self.METADATA_COLUMNS
        )
        return df[metadata_cols + structure_cols]

    @classmethod
    def get_structure_columns(cls, df: pd.DataFrame) -> list[str]:
        """
        Get list of structure column names from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to extract structure columns from.

        Returns
        -------
        list[str]
            List of column names that are brain structure measurements.
        """
        return [c for c in df.columns if c not in cls.METADATA_COLUMNS]
