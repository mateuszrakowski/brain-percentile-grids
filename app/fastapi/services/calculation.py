"""
Service for GAMLSS model fitting and percentile calculations.

This module handles the core calculation logic for fitting reference models
and calculating patient percentiles. Models are persisted to disk.
"""

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.engine.model import GAMLSS, FittedGAMLSSModel
from app.core.engine.selector import GAMLSSModelSelector
from app.core.resources.model_candidates import get_all_model_candidates
from app.fastapi.services.model_persistence import ModelPersistenceService
from app.fastapi.services.reference_data import ReferenceDataService

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
    - Persisting models to disk
    - Calculating percentile curves
    - Computing patient z-scores and percentiles
    """

    DEFAULT_PERCENTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    X_COLUMN = "AgeYears"

    def __init__(self, session: Session):
        self.session = session
        self._reference_service = ReferenceDataService(session)
        self._persistence_service = ModelPersistenceService(session)

    async def fit_reference_models(
        self,
        user_id: int,
        dataset_id: int,
        y_columns: list[str] | None = None,
        percentiles: list[float] | None = None,
        criterion: str = "bic",
    ) -> AsyncGenerator[CalculationProgress | ReferenceCalculationResult, None]:
        """
        Fit GAMLSS models for reference data with progress updates.

        Models are automatically persisted to disk after fitting.

        Parameters
        ----------
        user_id : int
            The user ID (for file path organization).
        dataset_id : int
            The dataset ID to fit models for.
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

        # Get reference data (caller must ensure data exists)
        df = self._reference_service.get_reference_dataframe(dataset_id)
        assert df is not None, "Reference data must exist (caller should validate)"

        # Determine structures to fit
        if y_columns is None:
            y_columns = self._reference_service.get_available_structures(dataset_id)

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
                model_result, fitted_model = self._fit_single_model(
                    df=df,
                    structure=structure,
                    percentiles=percentiles,
                    criterion=criterion,
                )
                result.results[structure] = model_result

                if model_result.converged and fitted_model is not None:
                    result.successful_count += 1
                    # Persist the model
                    self._persistence_service.save_model(
                        fitted_model=fitted_model,
                        user_id=user_id,
                        dataset_id=dataset_id,
                        structure=structure,
                    )
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
    ) -> tuple[ModelFitResult, FittedGAMLSSModel | None]:
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
        tuple[ModelFitResult, FittedGAMLSSModel | None]
            Result of the model fitting and the fitted model (if successful).
        """
        # Filter data for this structure (remove NaN values)
        model_df = df[[self.X_COLUMN, structure]].dropna()

        if len(model_df) < 10:
            return (
                ModelFitResult(
                    structure=structure,
                    converged=False,
                    error=f"Insufficient data: {len(model_df)} samples (minimum 10 required)",
                ),
                None,
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

        # Fit models
        best_model = selector.fit_models(criterion=criterion)

        if best_model is None or not best_model.converged:
            return (
                ModelFitResult(
                    structure=structure,
                    converged=False,
                    error="No model converged successfully",
                ),
                None,
            )

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

        return (
            ModelFitResult(
                structure=structure,
                converged=True,
                aic=best_model.aic,
                bic=best_model.bic,
                family=family,
                percentile_curves=percentile_curves,
                x_values=x_values,
            ),
            best_model,
        )

    def calculate_patient_percentiles(
        self,
        dataset_id: int,
        patient_data: pd.DataFrame,
        structures: list[str] | None = None,
    ) -> list[PatientPercentileResult]:
        """
        Calculate percentiles for out-of-sample patients.

        Loads fitted models from disk and calculates z-scores and percentiles
        for the provided patient data.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to use models from.
        patient_data : pd.DataFrame
            DataFrame with patient data (must have AgeYears and structure columns).
        structures : list[str] | None
            Structures to calculate. If None, uses all available models.

        Returns
        -------
        list[PatientPercentileResult]
            List of percentile results for each patient-structure combination.
        """
        results = []

        if patient_data.empty:
            return results

        # Get reference data for model loading
        ref_df = self._reference_service.get_reference_dataframe(dataset_id)
        if ref_df is None:
            logger.error(f"No reference data found for dataset {dataset_id}")
            return results

        # Get available models
        available_models = self._persistence_service.get_dataset_models(dataset_id)
        model_structures = {m.structure for m in available_models}

        # Determine structures to calculate
        if structures is None:
            structures = list(model_structures)
        else:
            structures = [s for s in structures if s in model_structures]

        if not structures:
            logger.warning(f"No fitted models found for dataset {dataset_id}")
            return results

        # Load models and calculate percentiles
        loaded_models: dict[str, FittedGAMLSSModel] = {}

        for structure in structures:
            model = self._persistence_service.load_model(
                dataset_id=dataset_id,
                structure=structure,
                source_data=ref_df,
                x_column=self.X_COLUMN,
                percentiles=self.DEFAULT_PERCENTILES,
            )
            if model is not None:
                loaded_models[structure] = model

        # Calculate for each patient-structure combination
        for _, row in patient_data.iterrows():
            patient_id = str(row.get("PatientID", "unknown"))
            age = row.get(self.X_COLUMN)

            if age is None or pd.isna(age):
                continue

            age = float(age)

            for structure in structures:
                if structure not in patient_data.columns or pd.isna(row[structure]):
                    continue

                value = float(row[structure])
                model = loaded_models.get(structure)

                if model is None:
                    results.append(
                        PatientPercentileResult(
                            patient_id=patient_id,
                            structure=structure,
                            age=age,
                            value=value,
                            error="Model not loaded",
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
