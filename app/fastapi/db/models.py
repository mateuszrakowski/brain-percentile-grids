from datetime import datetime

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.now)


class PatientRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    patient_id: str = Field(index=True)
    birth_date: str
    study_date: str
    study_description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class PatientStructureValue(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_record_id: int = Field(foreign_key="patientrecord.id", index=True)
    structure_name: str = Field(index=True)
    value: float
