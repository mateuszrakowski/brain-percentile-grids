"""
Service for GAMLSS model fitting and percentile calculations.

This module handles the core calculation logic for fitting reference models
and calculating patient percentiles.
"""

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.engine.model import GAMLSS, FittedGAMLSSModel
from app.core.engine.selector import GAMLSSModelSelector
from app.core.resources.model_candidates import get_all_model_candidates
from app.fastapi.db.models import PatientRecord, PatientStructureValue

logger = logging.getLogger(__name__)


@dataclass
class ModelFitResult:
    """Result of fitting a single GAMLSS model."""

    structure: str
    converged: bool
    aic: float | None = None
    bic: float | None = None
    family: str | None = None
    formula: str | None = None
    percentile_curves: dict[str, list[float]] | None = None
    x_values: list[float] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "structure": self.structure,
            "converged": self.converged,
            "aic": self.aic,
            "bic": self.bic,
            "family": self.family,
            "formula": self.formula,
            "percentile_curves": self.percentile_curves,
            "x_values": self.x_values,
            "error": self.error,
        }


@dataclass
class PatientPercentileResult:
    """Result of calculating percentile for a single patient structure."""

    patient_id: str
    structure: str
    age: float
    value: float
    z_score: float | None = None
    percentile: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "patient_id": self.patient_id,
            "structure": self.structure,
            "age": self.age,
            "value": self.value,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "error": self.error,
        }


@dataclass
class CalculationProgress:
    """Progress update for SSE streaming."""

    current: int
    total: int
    structure: str
    status: str  # fitting, completed, error
    message: str


@dataclass
class ReferenceCalculationResult:
    """Result of fitting all reference models."""

    results: dict[str, ModelFitResult] = field(default_factory=dict)
    successful_count: int = 0
    failed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
        }


class CalculationService:
    """
    Service for GAMLSS model fitting and percentile calculations.

    Handles the core business logic for:
    - Retrieving reference data from the database
    - Fitting GAMLSS models for each brain structure
    - Calculating percentile curves
    - Computing patient z-scores and percentiles
    """

    DEFAULT_PERCENTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    X_COLUMN = "AgeYears"

    def __init__(self, session: Session):
        self.session = session
        self._fitted_models: dict[str, FittedGAMLSSModel] = {}

    def get_reference_dataframe(self, user_id: int) -> pd.DataFrame | None:
        """
        Retrieve user's reference data from database as a DataFrame.

        Parameters
        ----------
        user_id : int
            The user ID to retrieve data for.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with patient records and structure values,
            or None if no data exists.
        """
        # Get all patient records for user
        records = self.session.exec(
            select(PatientRecord).where(PatientRecord.user_id == user_id)
        ).all()

        if not records:
            return None

        # Build DataFrame from records
        data_rows = []
        for record in records:
            # Get structure values for this record
            values = self.session.exec(
                select(PatientStructureValue).where(
                    PatientStructureValue.patient_record_id == record.id
                )
            ).all()

            row = {
                "PatientID": record.patient_id,
                "BirthDate": record.birth_date,
                "StudyDate": record.study_date,
                "StudyDescription": record.study_description,
                "AgeYears": record.age_years,
                "AgeMonths": record.age_months,
            }

            # Add structure values
            for sv in values:
                row[sv.structure_name] = sv.value

            data_rows.append(row)

        if not data_rows:
            return None

        return pd.DataFrame(data_rows)

    def get_available_structures(self, user_id: int) -> list[str]:
        """
        Get list of available structure columns for a user.

        Parameters
        ----------
        user_id : int
            The user ID to get structures for.

        Returns
        -------
        list[str]
            List of structure column names.
        """
        structure_names = self.session.exec(
            select(PatientStructureValue.structure_name)
            .join(PatientRecord)
            .where(PatientRecord.user_id == user_id)
            .distinct()
        ).all()

        return list(structure_names)

    async def fit_reference_models(
        self,
        user_id: int,
        y_columns: list[str] | None = None,
        percentiles: list[float] | None = None,
        criterion: str = "bic",
    ) -> AsyncGenerator[CalculationProgress | ReferenceCalculationResult, None]:
        """
        Fit GAMLSS models for reference data with progress updates.

        Parameters
        ----------
        user_id : int
            The user ID to fit models for.
        y_columns : list[str] | None
            Structures to fit. If None, fits all available.
        percentiles : list[float] | None
            Percentiles to calculate. If None, uses defaults.
        criterion : str
            Model selection criterion ('aic', 'bic', 'deviance').

        Yields
        ------
        CalculationProgress | ReferenceCalculationResult
            Progress updates during fitting, then final result.
        """
        percentiles = percentiles or self.DEFAULT_PERCENTILES

        # Get reference data
        df = self.get_reference_dataframe(user_id)
        if df is None:
            yield ReferenceCalculationResult(failed_count=0, successful_count=0)
            return

        # Determine structures to fit
        if y_columns is None:
            y_columns = self.get_available_structures(user_id)

        # Filter to columns that exist in the data
        y_columns = [col for col in y_columns if col in df.columns]

        if not y_columns:
            yield ReferenceCalculationResult(failed_count=0, successful_count=0)
            return

        result = ReferenceCalculationResult()
        total = len(y_columns)

        for i, structure in enumerate(y_columns):
            yield CalculationProgress(
                current=i + 1,
                total=total,
                structure=structure,
                status="fitting",
                message=f"Fitting model for {structure}",
            )

            try:
                model_result = self._fit_single_model(
                    df=df,
                    structure=structure,
                    percentiles=percentiles,
                    criterion=criterion,
                )
                result.results[structure] = model_result

                if model_result.converged:
                    result.successful_count += 1
                    # Store fitted model for later patient calculations
                    if structure in self._fitted_models:
                        pass  # Already stored during fitting
                else:
                    result.failed_count += 1

            except Exception as e:
                logger.error(f"Error fitting model for {structure}: {e}")
                result.results[structure] = ModelFitResult(
                    structure=structure,
                    converged=False,
                    error=str(e),
                )
                result.failed_count += 1

        yield result

    def _fit_single_model(
        self,
        df: pd.DataFrame,
        structure: str,
        percentiles: list[float],
        criterion: str,
    ) -> ModelFitResult:
        """
        Fit a single GAMLSS model for a structure.

        Parameters
        ----------
        df : pd.DataFrame
            Reference data.
        structure : str
            Structure column name.
        percentiles : list[float]
            Percentiles to calculate.
        criterion : str
            Model selection criterion.

        Returns
        -------
        ModelFitResult
            Result of the model fitting.
        """
        # Filter data for this structure (remove NaN values)
        model_df = df[[self.X_COLUMN, structure]].dropna()

        if len(model_df) < 10:
            return ModelFitResult(
                structure=structure,
                converged=False,
                error=f"Insufficient data: {len(model_df)} samples (minimum 10 required)",
            )

        # Create GAMLSS fitter
        fitter = GAMLSS(
            data_table=model_df,
            x_column=self.X_COLUMN,
            y_column=structure,
            percentiles=percentiles,
        )

        # Get model candidates and selector
        candidates = get_all_model_candidates()
        selector = GAMLSSModelSelector(fitter, candidates)

        # Fit models (without saving to disk for API use)
        best_model = selector.fit_models(criterion=criterion)

        if best_model is None or not best_model.converged:
            return ModelFitResult(
                structure=structure,
                converged=False,
                error="No model converged successfully",
            )

        # Store fitted model for patient calculations
        self._fitted_models[structure] = best_model

        # Calculate percentile curves
        try:
            curves = best_model.calculate_percentiles()

            # Convert curves to JSON-serializable format
            percentile_curves = {str(p): curve.tolist() for p, curve in curves.items()}

            # Get x values for plotting
            x_values = np.linspace(
                model_df[self.X_COLUMN].min(),
                model_df[self.X_COLUMN].max(),
                200,
            ).tolist()

        except Exception as e:
            logger.warning(f"Could not calculate percentiles for {structure}: {e}")
            percentile_curves = None
            x_values = None

        try:
            family = str(best_model.model.rx2("family")[0])
        except Exception:
            family = None

        return ModelFitResult(
            structure=structure,
            converged=True,
            aic=best_model.aic,
            bic=best_model.bic,
            family=family,
            percentile_curves=percentile_curves,
            x_values=x_values,
        )

    def calculate_patient_percentiles(
        self,
        user_id: int,
        patient_ids: list[str] | None = None,
        structures: list[str] | None = None,
    ) -> list[PatientPercentileResult]:
        """
        Calculate percentiles for patients against reference models.

        Parameters
        ----------
        user_id : int
            The user ID.
        patient_ids : list[str] | None
            Patient IDs to calculate. If None, calculates for all.
        structures : list[str] | None
            Structures to calculate. If None, uses all fitted models.

        Returns
        -------
        list[PatientPercentileResult]
            List of percentile results for each patient-structure combination.
        """
        results = []

        # Get patient data
        df = self.get_reference_dataframe(user_id)
        if df is None:
            return results

        # Filter patients if specified
        if patient_ids is not None:
            df = df[df["PatientID"].isin(patient_ids)]

        if df.empty:
            return results

        # Determine structures to calculate
        if structures is None:
            structures = list(self._fitted_models.keys())
        else:
            # Filter to fitted models only
            structures = [s for s in structures if s in self._fitted_models]

        # Calculate for each patient-structure combination
        for _, row in df.iterrows():
            patient_id = str(row["PatientID"])
            age = float(row["AgeYears"])

            for structure in structures:
                if structure not in df.columns or pd.isna(row[structure]):
                    continue

                value = float(row[structure])
                model = self._fitted_models.get(structure)

                if model is None:
                    results.append(
                        PatientPercentileResult(
                            patient_id=patient_id,
                            structure=structure,
                            age=age,
                            value=value,
                            error="No fitted model available",
                        )
                    )
                    continue

                try:
                    # Create patient DataFrame for prediction
                    patient_df = pd.DataFrame(
                        {self.X_COLUMN: [age], structure: [value]}
                    )

                    z_score, percentile = model.predict_patient_oos(patient_df)

                    results.append(
                        PatientPercentileResult(
                            patient_id=patient_id,
                            structure=structure,
                            age=age,
                            value=value,
                            z_score=float(z_score) if not np.isnan(z_score) else None,
                            percentile=(
                                float(percentile) if not np.isnan(percentile) else None
                            ),
                        )
                    )

                except Exception as e:
                    logger.error(
                        f"Error calculating percentile for {patient_id}/{structure}: {e}"
                    )
                    results.append(
                        PatientPercentileResult(
                            patient_id=patient_id,
                            structure=structure,
                            age=age,
                            value=value,
                            error=str(e),
                        )
                    )

        return results

    def get_fitted_model(self, structure: str) -> FittedGAMLSSModel | None:
        """
        Get a fitted model for a structure.

        Parameters
        ----------
        structure : str
            The structure name.

        Returns
        -------
        FittedGAMLSSModel | None
            The fitted model, or None if not available.
        """
        return self._fitted_models.get(structure)
