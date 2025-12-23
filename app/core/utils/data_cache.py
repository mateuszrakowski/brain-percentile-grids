import hashlib
import inspect
import os
import pickle
from typing import Any, Callable


def disk_cache(cache_dir: str = ".cache"):
    """
    Decorator for caching function results to disk.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to store cache files.

    Returns
    -------
    Callable
        Decorator function that wraps the target function with caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    memory_cache: dict[str, Any] = {}

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = generate_cache_key(func, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{key}.pickle")

            if key in memory_cache:
                return memory_cache[key]

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    memory_cache[key] = result
                    return result
                except (pickle.PickleError, EOFError):
                    pass

            result = func(*args, **kwargs)
            memory_cache[key] = result

            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except (pickle.PickleError, EOFError):
                pass

            return result

        return wrapper

    return decorator


def generate_cache_key(func, args, kwargs):
    """
    Generate a unique cache key for a function call.

    Parameters
    ----------
    func : Callable
        The function being cached.
    args : tuple
        Positional arguments to the function.
    kwargs : dict
        Keyword arguments to the function.

    Returns
    -------
    str
        MD5 hash string serving as the cache key.
    """
    func_id = f"{func.__module__}.{func.__name__}"

    try:
        source_code = inspect.getsource(func)
    except (IOError, TypeError):
        source_code = ""

    try:
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
    except Exception:
        args_str = str([str(arg) for arg in args])
        kwargs_str = str([(k, str(v)) for k, v in sorted(kwargs.items())])

    key_string = f"{func_id}:{source_code}:{args_str}:{kwargs_str}"

    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


def clear_cache(cache_dir: str = ".cache"):
    """
    Clear all files from the cache directory.

    Parameters
    ----------
    cache_dir : str, optional
        Directory containing cache files.
    """
    if not os.path.exists(cache_dir):
        return
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))


def clear_model_cache(model_path: str = "/app/data/models/"):
    """
    Clear all files from the model cache directory.

    Parameters
    ----------
    model_path : str, optional
        Directory containing cached model files.
    """
    if not os.path.exists(model_path):
        return
    for file in os.listdir(model_path):
        os.remove(os.path.join(model_path, file))


class DataCache:
    """
    Simple in-memory cache for storing models and data.

    Attributes
    ----------
    _cache : dict[str, Any]
        Internal dictionary storing cached values.
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to store.
        """
        self._cache[key] = value

    def get(self, key: str) -> Any:
        """
        Retrieve a value from the cache.

        Parameters
        ----------
        key : str
            Cache key to retrieve.

        Returns
        -------
        Any
            Cached value or None if not found.
        """
        return self._cache.get(key)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Parameters
        ----------
        key : str
            Cache key to check.

        Returns
        -------
        bool
            True if key exists, False otherwise.
        """
        return key in self._cache

    def remove(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Parameters
        ----------
        key : str
            Cache key to remove.

        Returns
        -------
        bool
            True if key existed and was removed, False otherwise.
        """
        return self._cache.pop(key, None) is not None

    def keys(self):
        """
        Get all cache keys.

        Returns
        -------
        KeysView
            View of all keys in the cache.
        """
        return self._cache.keys()

    def size(self) -> int:
        """
        Get the number of items in cache.

        Returns
        -------
        int
            Number of cached items.
        """
        return len(self._cache)
