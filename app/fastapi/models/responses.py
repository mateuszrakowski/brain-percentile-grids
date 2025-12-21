"""
Response models for API endpoints.
These models ensure consistent response formats.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ProgressUpdate(BaseModel):
    """SSE progress update message."""

    type: str = "progress"
    message: str
    progress: int = Field(ge=0, le=100)
    status: str  # running, completed, error
    operation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ReferenceUploadResponse(BaseModel):
    """Response for reference dataset upload."""

    message: str
    rows: int
    columns: int
    processing_info: Dict[str, Any]
    structures_detected: List[str]
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
    aic: Optional[float] = None
    bic: Optional[float] = None
    family: Optional[str] = None
    formula: Optional[str] = None
    percentile_curves: Optional[Dict[str, List[float]]] = None
    plot_available: bool = False
    error: Optional[str] = None


class ReferenceCalculationResponse(BaseModel):
    """Response for reference calculations."""

    message: str
    results: Dict[str, ModelResult]
    successful_count: int
    failed_count: int
    total_time: float
    cache_hits: int = 0


class PatientResult(BaseModel):
    """Individual patient calculation result."""

    patient_id: str
    structure: str
    z_score: Optional[float]
    percentile: Optional[float]
    age: Optional[float]
    value: float
    reference_mean: Optional[float]
    reference_sd: Optional[float]


class PatientCalculationResponse(BaseModel):
    """Response for patient calculations."""

    message: str
    results: List[PatientResult]
    patients_processed: int
    structures_processed: int
    errors: List[str] = []
