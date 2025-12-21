import hashlib
import inspect
import os
import pickle
from typing import Any, Callable


def disk_cache(cache_dir: str = ".cache"):
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
    if not os.path.exists(cache_dir):
        return
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))


def clear_model_cache(model_path: str = "/app/data/models/"):
    if not os.path.exists(model_path):
        return
    for file in os.listdir(model_path):
        os.remove(os.path.join(model_path, file))


class DataCache:
    """Simple in-memory cache for storing models and data."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = value

    def get(self, key: str) -> Any:
        """Retrieve a value from the cache."""
        return self._cache.get(key)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    def remove(self, key: str) -> bool:
        """Remove a key from the cache. Returns True if key existed."""
        return self._cache.pop(key, None) is not None

    def keys(self):
        """Get all cache keys."""
        return self._cache.keys()

    def size(self) -> int:
        """Get the number of items in cache."""
        return len(self._cache)
