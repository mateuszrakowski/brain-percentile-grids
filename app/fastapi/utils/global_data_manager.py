"""
Global data manager for sharing reference data across all users.

This module provides a singleton-based approach to store and retrieve
reference datasets that should be accessible to all users of the application.
"""

import json
import logging
import os
import pickle
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GlobalDataManager:
    """
    Singleton class to manage global reference data shared across all users.

    This class stores reference datasets on disk and provides thread-safe
    access to the data for all application users.
    """

    _instance: Optional["GlobalDataManager"] = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls, cache_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if self._initialized:
            return

        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache")
        self.data_dir = os.path.join(self.cache_dir, "global_data")
        self.reference_data_file = os.path.join(self.data_dir, "reference_data.pkl")
        self.reference_metadata_file = os.path.join(
            self.data_dir, "reference_metadata.json"
        )

        # Thread lock for data operations
        self.data_lock = threading.Lock()

        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing reference data if available
        self._reference_data: Optional[pd.DataFrame] = None
        self._reference_metadata: Dict[str, Any] = {}
        self._load_reference_data()

        self._initialized = True
        logger.info(f"GlobalDataManager initialized with cache_dir: {self.cache_dir}")

    def _load_reference_data(self) -> None:
        """Load reference data from disk if it exists."""
        try:
            if os.path.exists(self.reference_data_file):
                with open(self.reference_data_file, "rb") as f:
                    self._reference_data = pickle.load(f)
                logger.info(
                    f"Loaded reference data with {len(self._reference_data)} records"
                )

            if os.path.exists(self.reference_metadata_file):
                with open(self.reference_metadata_file, "r") as f:
                    self._reference_metadata = json.load(f)
                logger.info("Loaded reference data metadata")
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self._reference_data = None
            self._reference_metadata = {}

    def _save_reference_data(self) -> bool:
        """Save reference data to disk."""
        try:
            # Save DataFrame
            if self._reference_data is not None:
                with open(self.reference_data_file, "wb") as f:
                    pickle.dump(self._reference_data, f)

            # Save metadata
            with open(self.reference_metadata_file, "w") as f:
                json.dump(self._reference_metadata, f, indent=2, default=str)

            logger.info("Reference data saved to disk")
            return True
        except Exception as e:
            logger.error(f"Error saving reference data: {e}")
            return False

    def set_reference_data(
        self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set global reference data that will be shared across all users.

        Args:
            data: Reference DataFrame
            metadata: Optional metadata about the dataset

        Returns:
            bool: True if data was saved successfully
        """
        with self.data_lock:
            try:
                self._reference_data = data.copy()

                # Update metadata
                self._reference_metadata = {
                    "uploaded_at": datetime.now().isoformat(),
                    "total_records": len(data),
                    "columns": list(data.columns),
                    "data_types": {
                        col: str(dtype) for col, dtype in data.dtypes.items()
                    },
                    **(metadata or {}),
                }

                success = self._save_reference_data()
                if success:
                    logger.info(f"Set global reference data with {len(data)} records")
                return success
            except Exception as e:
                logger.error(f"Error setting reference data: {e}")
                return False

    def get_reference_data(self) -> Optional[pd.DataFrame]:
        """
        Get global reference data.

        Returns:
            DataFrame or None if no reference data is available
        """
        if self._reference_data is not None:
            return self._reference_data.copy()
        return None

    def get_reference_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the reference data.

        Returns:
            Dictionary containing metadata
        """
        return self._reference_metadata.copy()

    def has_reference_data(self) -> bool:
        """
        Check if reference data is available.

        Returns:
            bool: True if reference data exists
        """
        return self._reference_data is not None and not self._reference_data.empty

    def clear_reference_data(self) -> bool:
        """
        Clear global reference data.

        Returns:
            bool: True if data was cleared successfully
        """
        with self.data_lock:
            try:
                self._reference_data = None
                self._reference_metadata = {}

                # Remove files
                for filepath in [
                    self.reference_data_file,
                    self.reference_metadata_file,
                ]:
                    if os.path.exists(filepath):
                        os.remove(filepath)

                logger.info("Global reference data cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing reference data: {e}")
                return False

    def get_available_structures(self) -> List[str]:
        """
        Get list of available brain structures from reference data.

        Returns:
            List of structure column names
        """
        if self._reference_data is not None:
            # Filter out metadata columns to get structure columns
            metadata_columns = [
                "PatientID",
                "AgeYears",
                "AgeMonths",
                "BirthDate",
                "StudyDate",
            ]
            structure_columns = [
                col
                for col in self._reference_data.columns
                if col not in metadata_columns
            ]
            return structure_columns
        return []

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the global data state.

        Returns:
            Dictionary containing data information
        """
        with self.data_lock:
            info = {
                "has_reference_data": self.has_reference_data(),
                "reference_metadata": self._reference_metadata.copy(),
                "available_structures": self.get_available_structures(),
                "cache_dir": self.cache_dir,
                "data_files_exist": {
                    "reference_data": os.path.exists(self.reference_data_file),
                    "reference_metadata": os.path.exists(self.reference_metadata_file),
                },
            }

            if self._reference_data is not None:
                info["data_shape"] = self._reference_data.shape
                info["memory_usage_mb"] = (
                    self._reference_data.memory_usage(deep=True).sum() / 1024 / 1024
                )

            return info


# Global instance
_global_data_manager = None


def get_global_data_manager(cache_dir: Optional[str] = None) -> GlobalDataManager:
    """
    Get the global data manager singleton instance.

    Args:
        cache_dir: Cache directory path (only used on first call)

    Returns:
        GlobalDataManager instance
    """
    global _global_data_manager
    if _global_data_manager is None:
        _global_data_manager = GlobalDataManager(cache_dir)
    return _global_data_manager
