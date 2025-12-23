"""
Shared dependencies for FastAPI endpoints.
"""

import time
from typing import Any, Dict

from fastapi import Depends, File, HTTPException, Request, UploadFile

from .config import Settings, get_settings
from .utils.file_utils import ValidatedFile


async def get_request_id(request: Request) -> str:
    """
    Get request ID for tracking.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    str
        Request ID or 'unknown' if not set.
    """
    return getattr(request.state, "request_id", "unknown")


async def get_system_metrics() -> Dict[str, Any]:
    """
    Get system metrics for monitoring.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing CPU, memory, disk usage percentages and uptime.
    """
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "uptime": time.time() - psutil.boot_time(),
    }


async def get_validated_files(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> list[ValidatedFile]:
    """
    Validate uploaded files and return validated file objects.

    Checks file count, filename presence, extension, size, and content.

    Parameters
    ----------
    files : list[UploadFile]
        List of uploaded files.
    settings : Settings
        Application settings from dependency injection.

    Returns
    -------
    list[ValidatedFile]
        List of ValidatedFile objects with content loaded.

    Raises
    ------
    HTTPException
        400 if no files provided, too many files, invalid filename,
        unsupported extension, file too large, or empty file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Check file count
    if len(files) > settings.max_files_count:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed: {settings.max_files_count}",
        )

    validated = []
    for file in files:
        # Check filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="File missing filename")

        # Check extension
        if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed: {file.filename}. "
                f"Allowed: {settings.allowed_extensions}",
            )

        # Read content
        content = await file.read()

        # Check size
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file.filename}. "
                f"Max size: {settings.max_upload_size} bytes",
            )

        # Check not empty
        if len(content) == 0:
            raise HTTPException(
                status_code=400, detail=f"File is empty: {file.filename}"
            )

        validated.append(ValidatedFile(file, content))

    return validated


# Dependency for pagination
class PaginationParams:
    """
    Common pagination parameters.

    Attributes
    ----------
    skip : int
        Number of items to skip.
    limit : int
        Maximum number of items to return (capped at 1000).
    """

    def __init__(self, skip: int = 0, limit: int = 100):
        """
        Initialize pagination parameters.

        Parameters
        ----------
        skip : int, optional
            Number of items to skip.
        limit : int, optional
            Maximum number of items to return.
        """
        self.skip = skip
        self.limit = min(limit, 1000)  # Cap at 1000
