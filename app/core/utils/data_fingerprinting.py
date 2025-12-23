"""
Data fingerprinting utilities for creating stable hashes from datasets.

This module provides functions to create consistent fingerprints of datasets
for cache validation purposes.
"""

import hashlib
import json
from typing import Any

import pandas as pd


def create_dataframe_fingerprint(df: pd.DataFrame, columns: list[str] = None) -> str:
    """
    Create a stable fingerprint/hash for a pandas DataFrame.

    This function creates a deterministic hash that remains consistent
    across different sessions as long as the underlying data is the same.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to fingerprint.
    columns : list[str], optional
        List of columns to include in fingerprint. If None, all columns are used.

    Returns
    -------
    str
        A hexadecimal string representing the dataset fingerprint.
    """
    if df.empty:
        return hashlib.sha256(b"empty_dataframe").hexdigest()

    # Use specified columns or all columns
    if columns is None:
        columns = list(df.columns)

    # Sort columns to ensure consistent ordering
    columns = sorted([col for col in columns if col in df.columns])

    if not columns:
        return hashlib.sha256(b"no_valid_columns").hexdigest()

    # Create a subset DataFrame with only the relevant columns
    subset_df = df[columns].copy()

    # Sort by all columns to ensure deterministic row ordering
    subset_df = subset_df.sort_values(by=columns, na_position="first")

    # Reset index to ensure consistent indexing
    subset_df = subset_df.reset_index(drop=True)

    # Create fingerprint components
    fingerprint_data = {
        "shape": subset_df.shape,
        "columns": columns,
        "dtypes": {col: str(subset_df[col].dtype) for col in columns},
        "data_hash": _create_data_hash(subset_df),
        "column_stats": _create_column_stats(subset_df),
    }

    # Convert to JSON string with sorted keys for consistency
    fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=True)

    # Create SHA256 hash
    return hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()


def _create_data_hash(df: pd.DataFrame) -> str:
    """
    Create a hash of the actual data values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash.

    Returns
    -------
    str
        MD5 hash of the data values.
    """
    # Convert DataFrame to string representation that's deterministic
    data_strings = []

    for col in df.columns:
        col_data = df[col]

        # Handle different data types appropriately
        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric data, round to avoid floating point precision issues
            if pd.api.types.is_float_dtype(col_data):
                col_values = col_data.round(10).astype(str)
            else:
                col_values = col_data.astype(str)
        else:
            col_values = col_data.astype(str)

        # Handle NaN values consistently
        col_values = col_values.fillna("__NaN__")

        # Create column hash
        col_string = f"{col}:{','.join(col_values.values)}"
        data_strings.append(col_string)

    # Combine all column data
    combined_data = "|".join(data_strings)

    return hashlib.md5(combined_data.encode("utf-8")).hexdigest()


def _create_column_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Create statistical summary for each column to aid in validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.

    Returns
    -------
    Dict[str, Any]
        Dictionary with column statistics including count, null count, unique count,
        and numeric stats (min, max, mean, std) for numeric columns.
    """
    stats = {}

    for col in df.columns:
        col_data = df[col]
        col_stats: dict[str, Any] = {
            "count": int(len(col_data)),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
        }

        # Add numeric statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            try:
                col_min = col_data.min()
                col_max = col_data.max()
                col_mean = col_data.mean()
                col_std = col_data.std()

                col_stats.update(
                    {
                        "min": float(col_min) if not pd.isna(col_min) else None,
                        "max": float(col_max) if not pd.isna(col_max) else None,
                        "mean": float(col_mean) if not pd.isna(col_mean) else None,
                        "std": float(col_std) if not pd.isna(col_std) else None,
                    }
                )
            except Exception:
                # If conversion fails, skip numeric stats
                pass

        stats[col] = col_stats

    return stats


def validate_dataframe_fingerprint(
    df: pd.DataFrame, expected_fingerprint: str, columns: list[str] = None
) -> bool:
    """
    Validate that a DataFrame matches the expected fingerprint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    expected_fingerprint : str
        The expected fingerprint hash.
    columns : list[str], optional
        List of columns to include in validation.

    Returns
    -------
    bool
        True if the fingerprint matches, False otherwise.
    """
    current_fingerprint = create_dataframe_fingerprint(df, columns)
    return current_fingerprint == expected_fingerprint


def create_model_cache_key(
    y_column: str,
    x_column: str,
    data_fingerprint: str,
    percentiles: list[float] | None = None,
) -> str:
    """
    Create a cache key for a specific model configuration.

    Parameters
    ----------
    y_column : str
        Target column name.
    x_column : str
        Feature column name.
    data_fingerprint : str
        Fingerprint of the training data.
    percentiles : list[float] | None, optional
        List of percentiles used in the model.

    Returns
    -------
    str
        A cache key string for the model.
    """
    if percentiles is None:
        percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Create model configuration hash
    config_data = {
        "y_column": y_column,
        "x_column": x_column,
        "data_fingerprint": data_fingerprint,
        "percentiles": sorted(percentiles),
    }

    config_json = json.dumps(config_data, sort_keys=True)
    config_hash = hashlib.md5(config_json.encode("utf-8")).hexdigest()

    # Return a readable cache key
    return f"model_{y_column}_{config_hash[:8]}"


def create_simple_model_cache_key(y_column: str) -> str:
    """
    Create a simple cache key based only on the structure name.

    This allows using the latest model regardless of training dataset.

    Parameters
    ----------
    y_column : str
        Target column name (brain structure).

    Returns
    -------
    str
        A simple cache key string for the model.
    """
    return f"model_{y_column}"
