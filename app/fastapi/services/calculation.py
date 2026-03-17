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
from app.fastapi.services.oos_persistence import OOSPersistenceService
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
    is_extrapolated: bool = False
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
            "is_extrapolated": self.is_extrapolated,
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

    # Configuration constants
    DEFAULT_PERCENTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    X_COLUMN = "PatientAge"
    MIN_SAMPLES_FOR_MODEL = 20
    PERCENTILE_CURVE_POINTS = 200

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

        # Get reference data
        df = self._reference_service.get_reference_dataframe(dataset_id)
        if df.empty:
            raise ValueError(
                f"No reference data found for dataset {dataset_id}. "
                "Caller should validate data exists before fitting."
            )

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

        if len(model_df) < self.MIN_SAMPLES_FOR_MODEL:
            return (
                ModelFitResult(
                    structure=structure,
                    converged=False,
                    error=(
                        f"Insufficient data: {len(model_df)} samples "
                        f"(minimum {self.MIN_SAMPLES_FOR_MODEL} required)"
                    ),
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
                self.PERCENTILE_CURVE_POINTS,
            ).tolist()

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Could not calculate percentiles for {structure}: {e}")
            percentile_curves = None
            x_values = None

        try:
            family = str(best_model.model.rx2("family")[0])
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            logger.debug(f"Could not extract model family: {e}")
            family = "unknown"

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
    ) -> tuple[list[PatientPercentileResult], dict[str, FittedGAMLSSModel]]:
        """
        Calculate percentiles for out-of-sample patients.

        Loads fitted models from disk and calculates z-scores and percentiles
        for the provided patient data.

        Parameters
        ----------
        dataset_id : int
            The dataset ID to use models from.
        patient_data : pd.DataFrame
            DataFrame with patient data (must have PatientAge and structure columns).
        structures : list[str] | None
            Structures to calculate. If None, uses all available models.

        Returns
        -------
        tuple[list[PatientPercentileResult], dict[str, FittedGAMLSSModel]]
            Percentile results and the loaded models dict (for plot generation).
        """
        results: list[PatientPercentileResult] = []
        loaded_models: dict[str, FittedGAMLSSModel] = {}

        if patient_data.empty:
            return results, loaded_models

        # Get reference data for model loading
        ref_df = self._reference_service.get_reference_dataframe(dataset_id)
        if ref_df.empty:
            logger.error(f"No reference data found for dataset {dataset_id}")
            return results, loaded_models

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
            return results, loaded_models

        # Load models
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
                    extrapolated = model.is_extrapolated(age)

                    warning = None
                    if extrapolated:
                        warning = (
                            f"Patient age {age} is outside training data range; "
                            "prediction is extrapolated and may be unreliable"
                        )

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
                            is_extrapolated=extrapolated,
                            error=warning,
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

        return results, loaded_models

    def calculate_and_persist_patient_percentiles(
        self,
        user_id: int,
        dataset_id: int,
        patient_data: pd.DataFrame,
        source_filenames: str | None = None,
        structures: list[str] | None = None,
    ) -> tuple[int, list[PatientPercentileResult]]:
        """
        Calculate OOS percentiles and persist results with plots.

        Creates an OOSCalculation record, computes z-scores/percentiles,
        generates per-patient plots, and saves everything to DB and disk.

        Parameters
        ----------
        user_id : int
            The user ID (for file path organization).
        dataset_id : int
            The dataset ID to use models from.
        patient_data : pd.DataFrame
            DataFrame with patient data.
        source_filenames : str | None
            Comma-separated list of uploaded filenames.
        structures : list[str] | None
            Structures to calculate. If None, uses all available models.

        Returns
        -------
        tuple[int, list[PatientPercentileResult]]
            The calculation ID and list of results.
        """
        oos_service = OOSPersistenceService(self.session)

        # Create calculation record
        calc = oos_service.create_calculation(
            dataset_id=dataset_id,
            source_filenames=source_filenames,
        )

        # Compute percentiles + get loaded models
        calc_results, loaded_models = self.calculate_patient_percentiles(
            dataset_id=dataset_id,
            patient_data=patient_data,
            structures=structures,
        )

        # Pre-compute percentile curves per structure (reuse across patients)
        percentile_curves_cache: dict[str, dict[float, np.ndarray]] = {}
        for structure, model in loaded_models.items():
            try:
                percentile_curves_cache[structure] = model.calculate_percentiles()
            except Exception as e:
                logger.warning(
                    f"Could not calculate percentile curves for {structure}: {e}"
                )

        # Save each result with optional plot
        for r in calc_results:
            plot_path: str | None = None

            # Generate OOS plot if model and curves are available
            model = loaded_models.get(r.structure)
            curves = percentile_curves_cache.get(r.structure)
            if (
                model is not None
                and curves is not None
                and r.z_score is not None
                and r.percentile is not None
            ):
                try:
                    patient_df = pd.DataFrame(
                        {self.X_COLUMN: [r.age], r.structure: [r.value]}
                    )
                    fig = model.plot_oos_patient(
                        patient_df, curves, r.z_score, r.percentile
                    )
                    plot_path = oos_service.save_oos_plot(
                        fig=fig,
                        user_id=user_id,
                        dataset_id=dataset_id,
                        calculation_id=calc.id,
                        patient_id=r.patient_id,
                        structure=r.structure,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not generate OOS plot for "
                        f"{r.patient_id}/{r.structure}: {e}"
                    )

            oos_service.save_result(
                calculation_id=calc.id,
                patient_id=r.patient_id,
                structure=r.structure,
                age=r.age,
                value=r.value,
                z_score=r.z_score,
                percentile=r.percentile,
                is_extrapolated=r.is_extrapolated,
                error=r.error,
                plot_path=plot_path,
            )

        return calc.id, calc_results
