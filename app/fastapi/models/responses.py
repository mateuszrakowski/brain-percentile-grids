"""
Response models for API endpoints.
These models ensure consistent response formats.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ProgressUpdate(BaseModel):
    """SSE progress update message."""

    type: str = "progress"
    message: str
    progress: int = Field(ge=0, le=100)
    status: str  # running, completed, error
    operation: str | None = None
    metadata: dict[str, Any] | None = None


class ReferenceUploadResponse(BaseModel):
    """Response for reference dataset upload."""

    message: str
    rows: int
    columns: int
    processing_info: dict[str, Any]
    structures_detected: list[str]
    upload_id: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Reference dataset uploaded successfully",
                "rows": 1500,
                "columns": 45,
                "processing_info": {"files_processed": 3, "total_time": 2.5},
                "structures_detected": ["TotalGreyVol", "TotalWhiteVol"],
                "upload_id": "ref_20240101_120000",
            }
        }
    )


class ModelResult(BaseModel):
    """Result from GAMLSS model fitting."""

    structure: str
    converged: bool
    aic: float | None = None
    bic: float | None = None
    family: str | None = None
    formula: str | None = None
    percentile_curves: dict[str, list[float]] | None = None
    plot_available: bool = False
    error: str | None = None


class ReferenceCalculationResponse(BaseModel):
    """Response for reference calculations."""

    message: str
    results: dict[str, ModelResult]
    successful_count: int
    failed_count: int
    total_time: float


class PatientResult(BaseModel):
    """Individual patient calculation result."""

    patient_id: str
    structure: str
    z_score: float | None
    percentile: float | None
    age: float | None
    value: float
    reference_mean: float | None
    reference_sd: float | None


class PatientCalculationResponse(BaseModel):
    """Response for patient calculations."""

    message: str
    results: list[PatientResult]
    patients_processed: int
    structures_processed: int
    errors: list[str] = []
