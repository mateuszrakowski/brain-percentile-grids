"""
Request models for API endpoints.
These models validate incoming data automatically.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReferenceCalculationRequest(BaseModel):
    """Request model for reference percentile calculations."""

    x_column: str = Field(
        default="AgeYears", description="Independent variable column name"
    )

    y_columns: List[str] = Field(
        min_length=1,
        description="Dependent variable columns to model",
    )

    percentiles: List[float] = Field(
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        description="Percentiles to calculate",
    )

    model_families: Optional[List[str]] = Field(
        default=None, description="GAMLSS distribution families to try"
    )

    @field_validator("percentiles", mode="after")
    @classmethod
    def validate_percentiles(cls, v: List[float]) -> List[float]:
        """Ensure percentiles are between 0 and 1."""
        for p in v:
            if not 0 < p < 1:
                raise ValueError(f"Percentile {p} must be between 0 and 1")
        return v

    @field_validator("y_columns", mode="after")
    @classmethod
    def validate_y_columns(cls, v: List[str]) -> List[str]:
        """Ensure column names are valid."""
        for col in v:
            if not col or not col.strip():
                raise ValueError("Column names cannot be empty")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x_column": "AgeYears",
                "y_columns": ["TotalGreyVol", "TotalWhiteVol", "CSF"],
                "percentiles": [0.05, 0.25, 0.50, 0.75, 0.95],
            }
        }
    )


class PatientCalculationRequest(BaseModel):
    """Request model for patient percentile calculations."""

    patient_indices: List[int] = Field(
        min_length=1, description="Indices of patients to process"
    )

    y_columns: Optional[List[str]] = Field(
        description="Structures to calculate (None = all available)"
    )

    use_cached_models: bool = Field(
        default=True, description="Whether to use cached reference models"
    )


class FileUploadMetadata(BaseModel):
    """Metadata for uploaded files."""

    filename: str
    content_type: str
    size: int

    @field_validator("content_type", mode="after")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Ensure file type is supported."""
        allowed = [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]
        if v not in allowed:
            raise ValueError(f"File type {v} not supported")
        return v
