from datetime import UTC, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    """User account for authentication."""

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relationships
    datasets: list["ReferenceDataset"] = Relationship(back_populates="user")


class ReferenceDataset(SQLModel, table=True):
    """
    A named collection of reference patients for model fitting.

    Users can have multiple datasets (e.g., "Pediatric Cohort", "Adult Study")
    each with its own fitted models.
    """

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    name: str = Field(index=True)
    description: str | None = None
    sample_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="unique_dataset_name_per_user"),
    )

    # Relationships
    user: User = Relationship(back_populates="datasets")
    patients: list["PatientRecord"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    fitted_models: list["FittedModel"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    oos_calculations: list["OOSCalculation"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class PatientRecord(SQLModel, table=True):
    """A patient record in a reference dataset."""

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="referencedataset.id", index=True)
    patient_id: str = Field(index=True)
    birth_date: str
    study_date: str
    study_description: str | None = None
    patient_age: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "patient_id",
            "study_date",
            name="unique_patient_per_dataset",
        ),
    )

    # Relationships
    dataset: ReferenceDataset = Relationship(back_populates="patients")
    structure_values: list["PatientStructureValue"] = Relationship(
        back_populates="patient_record",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class PatientStructureValue(SQLModel, table=True):
    """Brain structure volume value for a patient record."""

    id: int | None = Field(default=None, primary_key=True)
    patient_record_id: int = Field(foreign_key="patientrecord.id", index=True)
    structure_name: str = Field(index=True)
    value: float

    # Relationships
    patient_record: PatientRecord = Relationship(back_populates="structure_values")


class FittedModel(SQLModel, table=True):
    """
    A fitted GAMLSS model for a specific structure in a dataset.

    Stores metadata about the model; actual .rds file is on disk.
    """

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="referencedataset.id", index=True)
    structure: str = Field(index=True)
    family: str
    aic: float
    bic: float
    file_path: str
    reference_plot_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "structure",
            name="unique_model_per_dataset_structure",
        ),
    )

    # Relationships
    dataset: ReferenceDataset = Relationship(back_populates="fitted_models")


class OOSCalculation(SQLModel, table=True):
    """
    Groups results from one out-of-sample calculation run.

    Each calculation represents a single upload+compute session
    where patient files were processed against fitted models.
    """

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="referencedataset.id", index=True)
    source_filenames: str | None = None
    is_stale: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relationships
    dataset: ReferenceDataset = Relationship(back_populates="oos_calculations")
    results: list["OOSPatientResult"] = Relationship(
        back_populates="calculation",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class OOSPatientResult(SQLModel, table=True):
    """
    One row per patient-structure pair in an OOS calculation.

    Stores the computed z-score, percentile, and optional plot path
    for a single patient's brain structure measurement.
    """

    id: int | None = Field(default=None, primary_key=True)
    calculation_id: int = Field(foreign_key="ooscalculation.id", index=True)
    patient_id: str
    structure: str
    age: float
    value: float
    z_score: float | None = None
    percentile: float | None = None
    is_extrapolated: bool = False
    error: str | None = None
    plot_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "calculation_id",
            "patient_id",
            "structure",
            name="unique_result_per_calc_patient_structure",
        ),
    )

    # Relationships
    calculation: OOSCalculation = Relationship(back_populates="results")
