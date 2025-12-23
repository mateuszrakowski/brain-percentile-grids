"""
Persistent model caching system for GAMLSS models.

This module provides a filesystem-based caching system that persists
fitted GAMLSS models and their associated plots across server restarts.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from core.engine.model import GAMLSS, FittedGAMLSSModel
from core.utils.data_fingerprinting import (
    create_dataframe_fingerprint,
    create_model_cache_key,
    create_simple_model_cache_key,
)

logger = logging.getLogger(__name__)


class PersistentModelCache:
    """
    Persistent filesystem-based cache for GAMLSS models and plots.

    This cache stores models as .rds files using R's native serialization,
    plots as binary PNG data, and metadata as JSON files for validation.

    Attributes
    ----------
    cache_dir : str
        Root directory for cache files.
    validation_enabled : bool
        Whether to validate data fingerprints.
    models_dir : str
        Directory for model files.
    plots_dir : str
        Directory for plot files.
    metadata_dir : str
        Directory for metadata files.
    """

    def __init__(self, cache_dir: str, validation_enabled: bool = True):
        """
        Initialize the persistent cache.

        Parameters
        ----------
        cache_dir : str
            Directory to store cache files.
        validation_enabled : bool, optional
            Whether to validate data fingerprints.
        """
        self.cache_dir = cache_dir
        self.validation_enabled = validation_enabled
        self.models_dir = os.path.join(cache_dir, "models")
        self.plots_dir = os.path.join(cache_dir, "plots")
        self.metadata_dir = os.path.join(cache_dir, "metadata")

        # Create subdirectories
        for directory in [self.models_dir, self.plots_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"PersistentModelCache initialized with cache_dir: {cache_dir}")

        # In-memory cache for fast access during session
        self._memory_cache: dict[str, Any] = {}

    def _get_model_path(self, cache_key: str) -> str:
        """
        Get the filesystem path for a model file.

        Parameters
        ----------
        cache_key : str
            Cache key for the model.

        Returns
        -------
        str
            Full path to the model file.
        """
        return os.path.join(self.models_dir, f"{cache_key}.rds")

    def _get_plot_path(self, cache_key: str) -> str:
        """
        Get the filesystem path for a plot file.

        Parameters
        ----------
        cache_key : str
            Cache key for the plot.

        Returns
        -------
        str
            Full path to the plot file.
        """
        return os.path.join(self.plots_dir, f"{cache_key}.png")

    def _get_metadata_path(self, cache_key: str) -> str:
        """
        Get the filesystem path for a metadata file.

        Parameters
        ----------
        cache_key : str
            Cache key for the metadata.

        Returns
        -------
        str
            Full path to the metadata file.
        """
        return os.path.join(self.metadata_dir, f"{cache_key}.json")

    def _save_metadata(self, cache_key: str, metadata: dict[str, Any]) -> None:
        """
        Save metadata for a cached model.

        Parameters
        ----------
        cache_key : str
            Cache key for the model.
        metadata : Dict[str, Any]
            Metadata dictionary to save.
        """
        metadata_path = self._get_metadata_path(cache_key)

        # Add cache timestamp
        metadata["cached_at"] = datetime.now().isoformat()
        metadata["cache_key"] = cache_key

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to save metadata for {cache_key}: {e}")

    def _load_metadata(self, cache_key: str) -> dict[str, Any] | None:
        """
        Load metadata for a cached model.

        Parameters
        ----------
        cache_key : str
            Cache key for the model.

        Returns
        -------
        Optional[Dict[str, Any]]
            Metadata dictionary or None if not found.
        """
        metadata_path = self._get_metadata_path(cache_key)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {cache_key}: {e}")
            return None

    def _validate_cache_entry(
        self, cache_key: str, data_fingerprint: str, x_column: str, y_column: str
    ) -> bool:
        """
        Validate that a cached model is still valid for the given data.

        Parameters
        ----------
        cache_key : str
            Cache key to validate.
        data_fingerprint : str
            Expected data fingerprint.
        x_column : str
            Expected x column name.
        y_column : str
            Expected y column name.

        Returns
        -------
        bool
            True if the cache entry is valid, False otherwise.
        """
        if not self.validation_enabled:
            return True

        metadata = self._load_metadata(cache_key)
        if not metadata:
            return False

        # Check if all required files exist
        model_path = self._get_model_path(cache_key)
        if not os.path.exists(model_path):
            logger.warning(f"Model file missing for cache key {cache_key}")
            return False

        # Validate data fingerprint
        if metadata.get("data_fingerprint") != data_fingerprint:
            logger.info(f"Data fingerprint mismatch for cache key {cache_key}")
            return False

        # Validate column configuration
        if metadata.get("x_column") != x_column or metadata.get("y_column") != y_column:
            logger.info(f"Column configuration mismatch for cache key {cache_key}")
            return False

        return True

    def save_model(
        self,
        cache_key: str,
        model: FittedGAMLSSModel,
        data_fingerprint: str,
        plot_data: bytes | None = None,
    ) -> bool:
        """
        Save a fitted GAMLSS model to persistent cache.

        Parameters
        ----------
        cache_key : str
            Unique cache key for the model.
        model : FittedGAMLSSModel
            Fitted GAMLSS model to cache.
        data_fingerprint : str
            Fingerprint of the training data.
        plot_data : bytes | None, optional
            Plot data as bytes.

        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        try:
            model_path = self._get_model_path(cache_key)

            # Save the R model using existing save method
            model.save(model_path)

            # Save plot if provided
            if plot_data:
                plot_path = self._get_plot_path(cache_key)
                with open(plot_path, "wb") as f:
                    f.write(plot_data)

            # Save metadata
            metadata = {
                "data_fingerprint": data_fingerprint,
                "x_column": model.x_column,
                "y_column": model.y_column,
                "percentiles": model.percentiles,
                "converged": model.converged,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "deviance": float(model.deviance),
                "model_family": str(model.model.rx2("family")[0]),
                "data_shape": list(model.data_table.shape),
                "has_plot": plot_data is not None,
            }
            self._save_metadata(cache_key, metadata)

            # Also cache in memory for current session
            self._memory_cache[f"model_{cache_key}"] = model
            if plot_data:
                self._memory_cache[f"plot_{cache_key}"] = plot_data

            logger.info(f"Successfully cached model {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model {cache_key}: {e}")
            return False

    def save_latest_model(
        self,
        y_column: str,
        model: FittedGAMLSSModel,
        data_fingerprint: str,
        plot_data: bytes | None = None,
    ) -> bool:
        """
        Save a model using simple cache key and replace any existing model.

        Parameters
        ----------
        y_column : str
            Structure name.
        model : FittedGAMLSSModel
            Fitted GAMLSS model to cache.
        data_fingerprint : str
            Fingerprint of the training data.
        plot_data : bytes | None, optional
            Plot data as bytes.

        Returns
        -------
        bool
            True if save was successful, False otherwise.
        """
        try:
            # Use simple cache key based only on structure name
            cache_key = create_simple_model_cache_key(y_column)
            logger.info(
                f"Saving latest model for {y_column} with simple key: {cache_key}"
            )

            # Remove any existing model for this structure first
            self.remove_models_for_structure(y_column)

            # Save the new model using the simple key
            return self.save_model(cache_key, model, data_fingerprint, plot_data)

        except Exception as e:
            logger.error(f"Failed to save latest model for {y_column}: {e}")
            return False

    def load_latest_model(
        self,
        y_column: str,
        reference_data: pd.DataFrame,
        x_column: str,
        percentiles: list[float] | None = None,
    ) -> FittedGAMLSSModel | None:
        """
        Load the latest model for a structure, regardless of training dataset.

        Parameters
        ----------
        y_column : str
            Structure name.
        reference_data : pd.DataFrame
            Reference dataset (for model creation).
        x_column : str
            X column name.
        percentiles : list[float] | None, optional
            Percentiles used in the model.

        Returns
        -------
        Optional[FittedGAMLSSModel]
            Loaded FittedGAMLSSModel or None if not found.
        """
        try:
            # Try simple cache key first (latest approach)
            simple_key = create_simple_model_cache_key(y_column)
            model = self.load_model(
                simple_key, reference_data, x_column, y_column, percentiles
            )
            if model:
                logger.info(f"Loaded latest model for {y_column} using simple key")
                return model

            # Fallback: try to find any model for this structure with complex keys
            logger.info(
                f"No simple key model found for {y_column}, searching for any cached "
                f"model..."
            )

            if os.path.exists(self.metadata_dir):
                for filename in os.listdir(self.metadata_dir):
                    if filename.endswith(".json"):
                        cache_key = filename[:-5]  # Remove .json extension
                        metadata = self._load_metadata(cache_key)
                        if metadata and metadata.get("y_column") == y_column:
                            logger.info(
                                f"Found existing model for {y_column} with key: "
                                f"{cache_key}"
                            )
                            model = self.load_model(
                                cache_key,
                                reference_data,
                                x_column,
                                y_column,
                                percentiles,
                            )
                            if model:
                                logger.info(
                                    f"Successfully loaded existing model for {y_column}"
                                )
                                return model

            logger.info(f"No cached model found for {y_column}")
            return None

        except Exception as e:
            logger.error(f"Failed to load latest model for {y_column}: {e}")
            return None

    def remove_models_for_structure(self, y_column: str) -> bool:
        """
        Remove all cached models for a specific structure.

        Parameters
        ----------
        y_column : str
            Structure name.

        Returns
        -------
        bool
            True if removal was successful.
        """
        try:
            removed_count = 0

            # Find all models for this structure
            if os.path.exists(self.metadata_dir):
                for filename in os.listdir(self.metadata_dir):
                    if filename.endswith(".json"):
                        cache_key = filename[:-5]  # Remove .json extension
                        metadata = self._load_metadata(cache_key)
                        if metadata and metadata.get("y_column") == y_column:
                            if self.remove_cached_model(cache_key):
                                removed_count += 1

            if removed_count > 0:
                logger.info(f"Removed {removed_count} existing models for {y_column}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove models for {y_column}: {e}")
            return False

    def load_model(
        self,
        cache_key: str,
        reference_data: pd.DataFrame,
        x_column: str,
        y_column: str,
        percentiles: list[float] | None = None,
    ) -> FittedGAMLSSModel | None:
        """
        Load a fitted GAMLSS model from persistent cache.

        Parameters
        ----------
        cache_key : str
            Unique cache key for the model.
        reference_data : pd.DataFrame
            Reference dataset (used for validation and model creation).
        x_column : str
            X column name.
        y_column : str
            Y column name.
        percentiles : list[float] | None, optional
            Percentiles used in the model.

        Returns
        -------
        Optional[FittedGAMLSSModel]
            Loaded FittedGAMLSSModel or None if not found/invalid.
        """
        # Check memory cache first
        memory_key = f"model_{cache_key}"
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]

        model_path = self._get_model_path(cache_key)

        if not os.path.exists(model_path):
            return None

        try:
            # Validate cache entry
            data_fingerprint = create_dataframe_fingerprint(reference_data)
            if not self._validate_cache_entry(
                cache_key, data_fingerprint, x_column, y_column
            ):
                logger.info(f"Cache validation failed for {cache_key}")
                return None

            # Load the model using existing load method
            if percentiles is None:
                percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

            model = GAMLSS.load_model(
                model_path=model_path,
                source_data=reference_data,
                x_column=x_column,
                y_column=y_column,
                percentiles=percentiles,
            )

            # Cache in memory for current session
            self._memory_cache[memory_key] = model

            logger.info(f"Successfully loaded cached model {cache_key}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {e}")
            return None

    def get_plot(self, cache_key: str) -> bytes | None:
        """
        Get plot data from cache.

        Parameters
        ----------
        cache_key : str
            Cache key for the plot.

        Returns
        -------
        Optional[bytes]
            Plot data as bytes or None if not found.
        """
        # Check memory cache first
        memory_key = f"plot_{cache_key}"
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]

        plot_path = self._get_plot_path(cache_key)

        if not os.path.exists(plot_path):
            return None

        try:
            with open(plot_path, "rb") as f:
                plot_data = f.read()

            # Cache in memory for current session
            self._memory_cache[memory_key] = plot_data

            return plot_data

        except Exception as e:
            logger.error(f"Failed to load plot {cache_key}: {e}")
            return None

    def save_plot(self, cache_key: str, plot_data: bytes) -> bool:
        """
        Save plot data to cache.

        Parameters
        ----------
        cache_key : str
            Cache key for the plot.
        plot_data : bytes
            Plot data as bytes.

        Returns
        -------
        bool
            True if save was successful.
        """
        try:
            plot_path = self._get_plot_path(cache_key)

            with open(plot_path, "wb") as f:
                f.write(plot_data)

            # Cache in memory for current session
            self._memory_cache[f"plot_{cache_key}"] = plot_data

            # Update metadata to indicate plot is available
            metadata = self._load_metadata(cache_key)
            if metadata:
                metadata["has_plot"] = True
                self._save_metadata(cache_key, metadata)

            return True

        except Exception as e:
            logger.error(f"Failed to save plot {cache_key}: {e}")
            return False

    def has_cached_model(
        self,
        y_column: str,
        x_column: str,
        data_fingerprint: str,
        percentiles: list[float] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if a model is already cached for the given parameters.

        Parameters
        ----------
        y_column : str
            Target column name.
        x_column : str
            Feature column name.
        data_fingerprint : str
            Fingerprint of the data.
        percentiles : list[float] | None, optional
            List of percentiles.

        Returns
        -------
        Tuple[bool, str]
            Tuple of (is_cached, cache_key).
        """
        cache_key = create_model_cache_key(
            y_column, x_column, data_fingerprint, percentiles
        )

        # Check memory cache first
        if f"model_{cache_key}" in self._memory_cache:
            return True, cache_key

        # Check if valid cache entry exists on disk
        if self._validate_cache_entry(cache_key, data_fingerprint, x_column, y_column):
            return True, cache_key

        return False, cache_key

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about cached items.

        Returns
        -------
        Dict[str, Any]
            Dictionary with cache statistics.
        """
        info = {
            "cache_dir": self.cache_dir,
            "memory_items": len(self._memory_cache),
            "disk_models": 0,
            "disk_plots": 0,
            "disk_metadata": 0,
        }

        try:
            if os.path.exists(self.models_dir):
                info["disk_models"] = len(
                    [f for f in os.listdir(self.models_dir) if f.endswith(".rds")]
                )
            if os.path.exists(self.plots_dir):
                info["disk_plots"] = len(
                    [f for f in os.listdir(self.plots_dir) if f.endswith(".png")]
                )
            if os.path.exists(self.metadata_dir):
                info["disk_metadata"] = len(
                    [f for f in os.listdir(self.metadata_dir) if f.endswith(".json")]
                )
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")

        return info

    def clear_cache(self) -> bool:
        """
        Clear all cached items from both memory and disk.

        Returns
        -------
        bool
            True if successful.
        """
        try:
            # Clear memory cache
            self._memory_cache.clear()

            # Clear disk cache
            for directory in [self.models_dir, self.plots_dir, self.metadata_dir]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        os.remove(file_path)

            logger.info("Successfully cleared persistent cache")
            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def remove_cached_model(self, cache_key: str) -> bool:
        """
        Remove a specific cached model and its associated files.

        Parameters
        ----------
        cache_key : str
            Cache key to remove.

        Returns
        -------
        bool
            True if removal was successful.
        """
        try:
            # Remove from memory cache
            keys_to_remove = [k for k in self._memory_cache.keys() if cache_key in k]
            for key in keys_to_remove:
                del self._memory_cache[key]

            # Remove from disk
            files_to_remove = [
                self._get_model_path(cache_key),
                self._get_plot_path(cache_key),
                self._get_metadata_path(cache_key),
            ]

            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)

            logger.info(f"Successfully removed cached model {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove cached model {cache_key}: {e}")
            return False
