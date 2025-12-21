"""
Async cache management for FastAPI.
Replaces synchronous cache operations with async versions.
"""

import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles


class CacheManager:
    """
    Async cache manager for models and data.
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.metadata_dir = os.path.join(cache_dir, "metadata")
        self.models_dir = os.path.join(cache_dir, "models")
        self.plots_dir = os.path.join(cache_dir, "plots")

        # Create directories
        for dir_path in [
            self.cache_dir,
            self.metadata_dir,
            self.models_dir,
            self.plots_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def save_model(
        self, structure: str, model_data: Any, metadata: Dict[str, Any]
    ) -> bool:
        """
        Save model to cache asynchronously.

        Args:
            structure: Structure name
            model_data: Model object to cache
            metadata: Model metadata

        Returns:
            Success status
        """
        try:
            cache_key = self._get_cache_key(structure)

            # Save metadata as JSON
            metadata_path = os.path.join(self.metadata_dir, f"{cache_key}.json")
            metadata["cached_at"] = datetime.now().isoformat()
            metadata["structure"] = structure

            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(json.dumps(metadata, indent=2))

            # Save model as pickle
            model_path = os.path.join(self.models_dir, f"{cache_key}.pkl")

            # Pickle in thread pool (CPU-intensive)
            import asyncio

            loop = asyncio.get_event_loop()
            model_bytes = await loop.run_in_executor(None, pickle.dumps, model_data)

            async with aiofiles.open(model_path, "wb") as f:
                await f.write(model_bytes)

            return True

        except Exception as e:
            logger.error(f"Failed to cache model for {structure}: {e}")
            return False

    async def load_model(self, structure: str) -> Optional[Any]:
        """
        Load model from cache asynchronously.

        Args:
            structure: Structure name

        Returns:
            Cached model or None
        """
        try:
            cache_key = self._get_cache_key(structure)

            # Check if cache exists
            model_path = os.path.join(self.models_dir, f"{cache_key}.pkl")
            if not os.path.exists(model_path):
                return None

            # Load model
            async with aiofiles.open(model_path, "rb") as f:
                model_bytes = await f.read()

            # Unpickle in thread pool
            import asyncio

            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, pickle.loads, model_bytes)

            return model

        except Exception as e:
            logger.error(f"Failed to load cached model for {structure}: {e}")
            return None

    async def save_plot(self, structure: str, plot_data: bytes) -> bool:
        """Save plot data to cache."""
        try:
            cache_key = self._get_cache_key(structure)
            plot_path = os.path.join(self.plots_dir, f"{cache_key}.png")

            async with aiofiles.open(plot_path, "wb") as f:
                await f.write(plot_data)

            return True

        except Exception as e:
            logger.error(f"Failed to cache plot for {structure}: {e}")
            return False

    async def load_plot(self, structure: str) -> Optional[bytes]:
        """Load plot data from cache."""
        try:
            cache_key = self._get_cache_key(structure)
            plot_path = os.path.join(self.plots_dir, f"{cache_key}.png")

            if not os.path.exists(plot_path):
                return None

            async with aiofiles.open(plot_path, "rb") as f:
                return await f.read()

        except Exception as e:
            logger.error(f"Failed to load cached plot for {structure}: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Count cached items
            model_files = os.listdir(self.models_dir)
            plot_files = os.listdir(self.plots_dir)

            # Calculate cache size
            total_size = 0
            for dir_path in [self.models_dir, self.plots_dir, self.metadata_dir]:
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    total_size += os.path.getsize(file_path)

            return {
                "cached_models": len(model_files),
                "cached_plots": len(plot_files),
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_directory": self.cache_dir,
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    async def clear_cache(self) -> bool:
        """Clear all cached data."""
        try:
            import shutil

            for dir_path in [self.models_dir, self.plots_dir, self.metadata_dir]:
                shutil.rmtree(dir_path, ignore_errors=True)
                os.makedirs(dir_path, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
