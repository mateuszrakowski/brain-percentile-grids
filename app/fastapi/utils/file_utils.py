"""
File processing utilities for FastAPI app.

This module provides secure file processing capabilities including
file validation, type checking, and DataFrame conversion.
"""

import io
import logging
import os
from typing import Union

import pandas as pd
from fastapi import UploadFile

from app.core.data_processing.process_input import (
    process_csv_input,
    sum_structure_volumes,
)

logger = logging.getLogger(__name__)

MAX_FILES_COUNT = 300


class FileValidationError(Exception):
    """Custom exception for file validation errors."""

    pass


class ValidatedFile:
    """
    Wrapper for FastAPI UploadFile with validated content.

    Attributes:
        name: Sanitized filename
        extension: File extension (lowercase)
        content: File bytes content
        content_type: MIME type
    """

    def __init__(self, upload_file: UploadFile, content: bytes):
        self.name = self._secure_filename(upload_file.filename or "unknown")
        _, self.extension = os.path.splitext(self.name.lower())
        self.content = content
        self.content_type = upload_file.content_type

    def _secure_filename(self, filename: str) -> str:
        """Remove dangerous characters from filename."""
        # Keep only safe characters
        return "".join(c for c in filename if c.isalnum() or c in "._-")

    def read(self) -> bytes:
        """Return file content as bytes."""
        return self.content

    def to_buffer(self) -> io.BytesIO:
        """Return file content as BytesIO buffer for pandas."""
        return io.BytesIO(self.content)


class PatientDataProcessor:
    """
    Handles loading, processing, and duplicate detection for patient data files.
    """

    def __init__(self):
        self.unique_columns = ["PatientID", "StudyDate", "StudyDescription"]
        self.desired_column_order = [
            "Filename",
            "PatientID",
            "AgeYears",
            "AgeMonths",
            "BirthDate",
            "StudyDate",
            "StudyDescription",
        ]
        self.max_files_count = MAX_FILES_COUNT
        self._supported_readers = {
            ".csv": self._read_csv,
            ".xlsx": self._read_excel,
            ".xls": self._read_excel,
        }

    def update_reference_dataset(
        self,
        current_df_state: Union[pd.DataFrame, None],
        uploaded_files: list[ValidatedFile],
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load and process patient files with duplicate detection.

        Args:
            current_df_state: Existing reference data or None
            uploaded_files: List of validated files

        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        processing_info = self._initialize_processing_info(
            current_df_state, uploaded_files
        )

        result = self._process_uploaded_files(uploaded_files, concat_dataframes=True)
        new_combined_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()

        if self._is_empty_data_scenario(new_combined_df, current_df_state):
            return pd.DataFrame(), processing_info

        if self._should_perform_duplicate_detection(current_df_state, new_combined_df):
            new_combined_df = self._remove_duplicates(
                current_df_state,  # type: ignore[arg-type]
                new_combined_df,
                processing_info,
            )

        final_df = self._combine_dataframes(current_df_state, new_combined_df)
        final_df = self._finalize_dataframe(final_df)

        self._update_final_processing_info(processing_info, new_combined_df, final_df)

        logger.info(f"Processing summary: {processing_info}")
        return final_df, processing_info

    def update_patients_dataset(
        self, uploaded_files: list[ValidatedFile]
    ) -> list[dict]:
        """
        Load and process patient data for percentile calculations.

        Args:
            uploaded_files: List of validated files

        Returns:
            List of patient data dictionaries
        """
        result = self._process_uploaded_files(uploaded_files, concat_dataframes=False)
        processed_dataframes: list[pd.DataFrame] = (
            result if isinstance(result, list) else [result]
        )

        finalized_dataframes = [
            self._finalize_dataframe(df) for df in processed_dataframes
        ]

        return [
            {
                "Filename": df["Filename"].item(),
                "data": df.to_dict("records")[0],
                "columns": list(df.columns),
            }
            for df in finalized_dataframes
        ]

    def _initialize_processing_info(
        self,
        current_df_state: Union[pd.DataFrame, None],
        uploaded_files: list[ValidatedFile],
    ) -> dict:
        """Initialize metadata dictionary for tracking processing information."""
        return {
            "files_processed": len(uploaded_files),
            "new_records_uploaded": 0,
            "duplicates_found": 0,
            "duplicate_detection_method": "none",
            "existing_records": (
                0
                if current_df_state is None or current_df_state.empty
                else len(current_df_state)
            ),
            "total_records_after": 0,
        }

    def _process_uploaded_files(
        self, uploaded_files: list[ValidatedFile], concat_dataframes: bool = True
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Process uploaded files and return combined dataframe."""
        dataframes_data = self._convert_to_dataframes(uploaded_files)

        if not dataframes_data:
            return pd.DataFrame()

        processed_dataframes = []
        for name, dataframe in dataframes_data.items():
            processed_df = sum_structure_volumes(process_csv_input(dataframe))
            processed_df["Filename"] = name
            processed_dataframes.append(processed_df)

        if concat_dataframes:
            return pd.concat(processed_dataframes, ignore_index=True)
        else:
            return processed_dataframes

    def _convert_to_dataframes(
        self, files: list[ValidatedFile]
    ) -> dict[str, pd.DataFrame]:
        """
        Convert ValidatedFile objects to pandas DataFrames.

        Args:
            files: List of validated files

        Returns:
            Dictionary mapping filename to DataFrame
        """
        if len(files) > self.max_files_count:
            raise FileValidationError(
                f"Too many files uploaded. Maximum allowed: {self.max_files_count}"
            )

        dataframes_data = {}

        for validated_file in files:
            try:
                df = self._read_file(validated_file)

                if df.empty:
                    logger.warning(
                        f"File {validated_file.name} resulted in empty dataframe"
                    )
                    continue

                dataframes_data[validated_file.name] = df
                logger.info(
                    f"Successfully processed file: {validated_file.name} "
                    f"({len(df)} rows, {len(df.columns)} columns)"
                )

            except Exception as e:
                error_msg = f"Error reading file {validated_file.name}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(
            f"Successfully processed {len(dataframes_data)} files out of {len(files)}"
        )
        return dataframes_data

    def _read_file(self, validated_file: ValidatedFile) -> pd.DataFrame:
        """Read a single file based on its extension."""
        if validated_file.extension not in self._supported_readers:
            raise ValueError(f"Unsupported file extension: {validated_file.extension}")

        reader_method = self._supported_readers[validated_file.extension]
        return reader_method(validated_file)

    def _read_csv(self, validated_file: ValidatedFile) -> pd.DataFrame:
        """Read CSV file with safety parameters."""
        return pd.read_csv(
            validated_file.to_buffer(),
            encoding="utf-8",
            low_memory=False,
        )

    def _read_excel(self, validated_file: ValidatedFile) -> pd.DataFrame:
        """Read Excel file with appropriate engine."""
        engine = "openpyxl" if validated_file.extension == ".xlsx" else None
        return pd.read_excel(
            validated_file.to_buffer(),
            engine=engine,
        )

    def _is_empty_data_scenario(
        self, new_combined_df: pd.DataFrame, current_df_state: Union[pd.DataFrame, None]
    ) -> bool:
        """Check if we're dealing with an empty data scenario."""
        return new_combined_df.empty and current_df_state is None

    def _should_perform_duplicate_detection(
        self, current_df_state: Union[pd.DataFrame, None], new_combined_df: pd.DataFrame
    ) -> bool:
        """Determine if duplicate detection should be performed."""
        return (
            current_df_state is not None
            and not current_df_state.empty
            and not new_combined_df.empty
        )

    def _remove_duplicates(
        self,
        current_df_state: pd.DataFrame,
        new_combined_df: pd.DataFrame,
        processing_info: dict,
    ) -> pd.DataFrame:
        """Remove duplicates from new data based on existing data."""
        logger.info(
            f"Starting duplicate detection: existing={len(current_df_state)} rows, "
            f"new={len(new_combined_df)} rows"
        )

        try:
            current_keys, new_keys = self._create_composite_keys(
                current_df_state, new_combined_df
            )
            duplicate_mask = new_keys.isin(current_keys)
            duplicates_found = duplicate_mask.sum()

            self._update_duplicate_info(processing_info, duplicates_found)

            if duplicates_found > 0:
                logger.info(
                    f"Found {duplicates_found} duplicate records to be excluded"
                )
                new_combined_df = new_combined_df[~duplicate_mask].reset_index(
                    drop=True
                )

        except Exception as e:
            logger.warning(
                f"Could not perform duplicate detection with {self.unique_columns}: {e}"
            )

        return new_combined_df

    def _create_composite_keys(
        self, current_df_state: pd.DataFrame, new_combined_df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Create composite keys for duplicate detection."""
        current_keys = (
            current_df_state[self.unique_columns].astype(str).agg("_".join, axis=1)
        )
        new_keys = (
            new_combined_df[self.unique_columns].astype(str).agg("_".join, axis=1)
        )
        return current_keys, new_keys

    def _update_duplicate_info(
        self, processing_info: dict, duplicates_found: int
    ) -> None:
        """Update processing info with duplicate detection results."""
        processing_info["duplicates_found"] = duplicates_found
        processing_info["duplicate_detection_method"] = "_".join(self.unique_columns)

    def _combine_dataframes(
        self, current_df_state: Union[pd.DataFrame, None], new_combined_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine current and new dataframes."""
        dfs_to_concat = []

        if current_df_state is not None and not current_df_state.empty:
            logger.info(
                f"Adding existing data to concatenation: {len(current_df_state)} rows"
            )
            dfs_to_concat.append(current_df_state)

        if not new_combined_df.empty:
            logger.info(
                f"Adding new data to concatenation: {len(new_combined_df)} rows"
            )
            dfs_to_concat.append(new_combined_df)

        if not dfs_to_concat:
            return pd.DataFrame()

        logger.info(f"Concatenating {len(dfs_to_concat)} dataframes...")
        final_df = pd.concat(dfs_to_concat, ignore_index=True)
        logger.info(f"After concatenation: {len(final_df)} rows")

        return final_df

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final processing steps to the dataframe."""
        if df.empty:
            return df

        df = df.dropna(how="all")
        logger.info(f"After dropna(how='all'): {len(df)} rows")

        df = self._reorder_columns(df)

        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder dataframe columns according to desired order."""
        structure_columns = [
            col for col in df.columns if col not in self.desired_column_order
        ]

        available_metadata_cols = [
            col for col in self.desired_column_order if col in df.columns
        ]
        final_column_order = available_metadata_cols + sorted(structure_columns)

        return df[final_column_order]

    def _update_final_processing_info(
        self,
        processing_info: dict,
        new_combined_df: pd.DataFrame,
        final_df: pd.DataFrame,
    ) -> None:
        """Update processing info with final statistics."""
        processing_info["new_records_uploaded"] = (
            len(new_combined_df) if not new_combined_df.empty else 0
        )
        processing_info["total_records_after"] = (
            len(final_df) if not final_df.empty else 0
        )
