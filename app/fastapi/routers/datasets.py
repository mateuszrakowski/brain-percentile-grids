"""
Dataset management endpoints for reference datasets.

Provides endpoints for creating, listing, and managing reference datasets
that contain patient records and fitted models.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.fastapi.auth.dependencies import get_current_user
from app.fastapi.db.database import get_session
from app.fastapi.db.models import (
    FittedModel,
    PatientRecord,
    PatientStructureValue,
    ReferenceDataset,
    User,
)
from app.fastapi.dependencies import get_user_dataset
from app.fastapi.services.model_persistence import ModelPersistenceService

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# Request/Response models
class DatasetCreate(BaseModel):
    """Request model for creating a new dataset."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = None


class DatasetUpdate(BaseModel):
    """Request model for updating a dataset."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None


class DatasetResponse(BaseModel):
    """Response model for a dataset."""

    id: int
    name: str
    description: str | None
    sample_count: int
    created_at: str
    has_models: bool
    structures: list[str]


class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""

    datasets: list[DatasetResponse]
    total: int


class ModelInfo(BaseModel):
    """Information about a fitted model."""

    structure: str
    family: str
    aic: float
    bic: float
    created_at: str


class DatasetDetailResponse(BaseModel):
    """Detailed response for a single dataset."""

    id: int
    name: str
    description: str | None
    sample_count: int
    created_at: str
    structures: list[str]
    models: list[ModelInfo]


class DeleteDatasetResponse(BaseModel):
    """Response model for dataset deletion."""

    message: str
    patients_deleted: int
    models_deleted: int
    values_deleted: int


@router.post("", response_model=DatasetResponse)
async def create_dataset(
    current_user: Annotated[User, Depends(get_current_user)],
    request: DatasetCreate,
    session: Session = Depends(get_session),
) -> DatasetResponse:
    """
    Create a new reference dataset.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    request : DatasetCreate
        The dataset creation request.
    session : Session
        Database session.

    Returns
    -------
    DatasetResponse
        The created dataset.

    Raises
    ------
    HTTPException
        409 if dataset name already exists for this user.
    """
    # Check if name already exists
    existing = session.exec(
        select(ReferenceDataset).where(
            ReferenceDataset.user_id == current_user.id,
            ReferenceDataset.name == request.name,
        )
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset '{request.name}' already exists",
        )

    # Create dataset
    dataset = ReferenceDataset(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        sample_count=0,
        created_at=dataset.created_at.isoformat(),
        has_models=False,
        structures=[],
    )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
) -> DatasetListResponse:
    """
    List all datasets for the current user.

    Parameters
    ----------
    current_user : User
        The authenticated user.
    session : Session
        Database session.

    Returns
    -------
    DatasetListResponse
        List of user's datasets.
    """
    datasets = session.exec(
        select(ReferenceDataset)
        .where(ReferenceDataset.user_id == current_user.id)
        .order_by(ReferenceDataset.created_at.desc())
    ).all()

    result = []
    for dataset in datasets:
        # Get structures for this dataset
        structures = session.exec(
            select(PatientStructureValue.structure_name)
            .join(PatientRecord)
            .where(PatientRecord.dataset_id == dataset.id)
            .distinct()
        ).all()

        # Check if has models
        has_models = (
            session.exec(
                select(FittedModel).where(FittedModel.dataset_id == dataset.id)
            ).first()
            is not None
        )

        result.append(
            DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                sample_count=dataset.sample_count,
                created_at=dataset.created_at.isoformat(),
                has_models=has_models,
                structures=list(structures),
            )
        )

    return DatasetListResponse(datasets=result, total=len(result))


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    session: Session = Depends(get_session),
) -> DatasetDetailResponse:
    """
    Get detailed information about a dataset.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    session : Session
        Database session.

    Returns
    -------
    DatasetDetailResponse
        Detailed dataset information including models.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    # Get structures
    structures = session.exec(
        select(PatientStructureValue.structure_name)
        .join(PatientRecord)
        .where(PatientRecord.dataset_id == dataset.id)
        .distinct()
    ).all()

    # Get models
    models = session.exec(
        select(FittedModel).where(FittedModel.dataset_id == dataset.id)
    ).all()

    model_infos = [
        ModelInfo(
            structure=m.structure,
            family=m.family,
            aic=m.aic,
            bic=m.bic,
            created_at=m.created_at.isoformat(),
        )
        for m in models
    ]

    return DatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        sample_count=dataset.sample_count,
        created_at=dataset.created_at.isoformat(),
        structures=list(structures),
        models=model_infos,
    )


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    request: DatasetUpdate,
    session: Session = Depends(get_session),
) -> DatasetResponse:
    """
    Update a dataset's name or description.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    request : DatasetUpdate
        The update request.
    session : Session
        Database session.

    Returns
    -------
    DatasetResponse
        The updated dataset.

    Raises
    ------
    HTTPException
        404 if dataset not found.
        409 if new name conflicts with existing dataset.
    """
    # Check name uniqueness if changing
    if request.name and request.name != dataset.name:
        existing = session.exec(
            select(ReferenceDataset).where(
                ReferenceDataset.user_id == dataset.user_id,
                ReferenceDataset.name == request.name,
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Dataset '{request.name}' already exists",
            )
        dataset.name = request.name

    if request.description is not None:
        dataset.description = request.description

    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    # Get structures
    structures = session.exec(
        select(PatientStructureValue.structure_name)
        .join(PatientRecord)
        .where(PatientRecord.dataset_id == dataset.id)
        .distinct()
    ).all()

    # Check if has models
    has_models = (
        session.exec(
            select(FittedModel).where(FittedModel.dataset_id == dataset.id)
        ).first()
        is not None
    )

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        sample_count=dataset.sample_count,
        created_at=dataset.created_at.isoformat(),
        has_models=has_models,
        structures=list(structures),
    )


@router.delete("/{dataset_id}", response_model=DeleteDatasetResponse)
async def delete_dataset(
    dataset: Annotated[ReferenceDataset, Depends(get_user_dataset)],
    session: Session = Depends(get_session),
) -> DeleteDatasetResponse:
    """
    Delete a dataset and all its data (patients, models).

    Uses SQLModel cascade relationships to automatically delete
    related patient records, structure values, and model records.
    Model files (.rds) are deleted separately via ModelPersistenceService.

    Parameters
    ----------
    dataset : ReferenceDataset
        The validated dataset (injected via dependency).
    session : Session
        Database session.

    Returns
    -------
    DeleteDatasetResponse
        Confirmation message with deletion counts.

    Raises
    ------
    HTTPException
        404 if dataset not found.
    """
    dataset_name = dataset.name

    # Count records before deletion (for response)
    patients_deleted = len(dataset.patients)
    values_deleted = sum(len(p.structure_values) for p in dataset.patients)
    models_deleted = len(dataset.fitted_models)

    # Delete model files from disk (cascade only handles DB records)
    model_service = ModelPersistenceService(session)
    model_service.delete_model_files(dataset.user_id, dataset.id)

    # Delete dataset - cascade deletes patients, structure_values, fitted_models
    session.delete(dataset)
    session.commit()

    return DeleteDatasetResponse(
        message=f"Dataset '{dataset_name}' deleted",
        patients_deleted=patients_deleted,
        models_deleted=models_deleted,
        values_deleted=values_deleted,
    )
