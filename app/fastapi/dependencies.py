"""
Shared dependencies for FastAPI endpoints.
"""

import time
from typing import Annotated, Any

from fastapi import Depends, File, HTTPException, Request, UploadFile, status
from sqlmodel import Session, select

from .auth.dependencies import get_current_user
from .config import Settings, get_settings
from .db.database import get_session
from .db.models import ReferenceDataset, User
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


async def get_system_metrics() -> dict[str, Any]:
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
        if not any(
            file.filename.lower().endswith(ext) for ext in settings.allowed_extensions
        ):
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


async def get_user_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> ReferenceDataset:
    """
    Get a dataset belonging to the current user.

    This is a shared dependency that validates dataset ownership
    and returns the dataset if it belongs to the authenticated user.

    Parameters
    ----------
    dataset_id : int
        The dataset ID from the path parameter.
    current_user : User
        The authenticated user from JWT token.
    session : Session
        Database session.

    Returns
    -------
    ReferenceDataset
        The dataset if it exists and belongs to the user.

    Raises
    ------
    HTTPException
        404 if dataset not found or doesn't belong to user.
    """
    dataset = session.exec(
        select(ReferenceDataset).where(
            ReferenceDataset.id == dataset_id,
            ReferenceDataset.user_id == current_user.id,
        )
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    return dataset
