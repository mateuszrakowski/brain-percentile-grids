"""
Service for persisting OOS (out-of-sample) calculation results and plots.

Handles creating calculation records, saving per-patient results,
and managing OOS plot files on disk.
"""

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from sqlalchemy import desc
from sqlmodel import Session, select

from app.fastapi.config import get_settings
from app.fastapi.db.models import OOSCalculation, OOSPatientResult
from app.fastapi.services.model_persistence import ModelPersistenceService

logger = logging.getLogger(__name__)


class OOSPersistenceService:
    """
    Service for persisting OOS calculation results and plots.

    Parameters
    ----------
    session : Session
        Database session.
    """

    def __init__(self, session: Session):
        self.session = session
        self.models_dir = Path(get_settings().models_dir)

    def _get_oos_directory(
        self, user_id: int, dataset_id: int, calculation_id: int
    ) -> Path:
        """
        Get the directory path for an OOS calculation's plots.

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.
        calculation_id : int
            The calculation ID.

        Returns
        -------
        Path
            Directory path for the OOS calculation's plots.
        """
        return (
            self.models_dir
            / f"user_{user_id}"
            / f"dataset_{dataset_id}"
            / "oos"
            / f"calc_{calculation_id}"
        )

    def create_calculation(
        self, dataset_id: int, source_filenames: str | None = None
    ) -> OOSCalculation:
        """
        Create a new OOS calculation record.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.
        source_filenames : str | None
            Comma-separated list of uploaded filenames.

        Returns
        -------
        OOSCalculation
            The created calculation record.
        """
        calc = OOSCalculation(
            dataset_id=dataset_id,
            source_filenames=source_filenames,
        )
        self.session.add(calc)
        self.session.commit()
        self.session.refresh(calc)
        logger.info(f"Created OOS calculation {calc.id} for dataset {dataset_id}")
        return calc

    def save_result(
        self,
        calculation_id: int,
        patient_id: str,
        structure: str,
        age: float,
        value: float,
        z_score: float | None = None,
        percentile: float | None = None,
        is_extrapolated: bool = False,
        error: str | None = None,
        plot_path: str | None = None,
    ) -> OOSPatientResult:
        """
        Save a single patient-structure OOS result.

        Parameters
        ----------
        calculation_id : int
            The parent calculation ID.
        patient_id : str
            Patient identifier.
        structure : str
            Brain structure name.
        age : float
            Patient age.
        value : float
            Structure volume value.
        z_score : float | None
            Computed z-score.
        percentile : float | None
            Computed percentile.
        is_extrapolated : bool
            Whether the prediction is extrapolated.
        error : str | None
            Error message if computation failed.
        plot_path : str | None
            Path to the patient plot PNG.

        Returns
        -------
        OOSPatientResult
            The created result record.
        """
        result = OOSPatientResult(
            calculation_id=calculation_id,
            patient_id=patient_id,
            structure=structure,
            age=age,
            value=value,
            z_score=z_score,
            percentile=percentile,
            is_extrapolated=is_extrapolated,
            error=error,
            plot_path=plot_path,
        )
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        return result

    def save_oos_plot(
        self,
        fig: plt.Figure,
        user_id: int,
        dataset_id: int,
        calculation_id: int,
        patient_id: str,
        structure: str,
    ) -> str:
        """
        Save an OOS patient plot to disk.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to save.
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.
        calculation_id : int
            The calculation ID.
        patient_id : str
            Patient identifier.
        structure : str
            Brain structure name.

        Returns
        -------
        str
            File path of the saved plot.
        """
        plot_dir = self._get_oos_directory(user_id, dataset_id, calculation_id)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{patient_id}_{structure}.png"
        ModelPersistenceService._save_figure_to_png(fig, plot_path)
        return str(plot_path)

    def get_dataset_calculations(
        self, dataset_id: int, include_stale: bool = False
    ) -> list[OOSCalculation]:
        """
        Get OOS calculations for a dataset.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.
        include_stale : bool
            Whether to include stale calculations.

        Returns
        -------
        list[OOSCalculation]
            List of calculation records.
        """
        query = select(OOSCalculation).where(
            OOSCalculation.dataset_id == dataset_id
        )
        if not include_stale:
            query = query.where(OOSCalculation.is_stale == False)  # noqa: E712
        query = query.order_by(desc(OOSCalculation.created_at))
        return list(self.session.exec(query).all())

    def get_calculation_results(
        self, calculation_id: int
    ) -> list[OOSPatientResult]:
        """
        Get all results for a calculation.

        Parameters
        ----------
        calculation_id : int
            The calculation ID.

        Returns
        -------
        list[OOSPatientResult]
            List of patient result records.
        """
        return list(
            self.session.exec(
                select(OOSPatientResult).where(
                    OOSPatientResult.calculation_id == calculation_id
                )
            ).all()
        )

    def get_plot_path(self, result_id: int) -> str | None:
        """
        Get the plot path for a specific result.

        Parameters
        ----------
        result_id : int
            The result ID.

        Returns
        -------
        str | None
            Path to the plot PNG, or None if not available.
        """
        result = self.session.get(OOSPatientResult, result_id)
        if result is None or result.plot_path is None:
            return None
        if not Path(result.plot_path).exists():
            return None
        return result.plot_path

    def delete_calculation(self, user_id: int, calculation_id: int) -> None:
        """
        Delete a calculation and its plot files.

        Parameters
        ----------
        user_id : int
            The user ID (for file path).
        calculation_id : int
            The calculation ID to delete.
        """
        calc = self.session.get(OOSCalculation, calculation_id)
        if calc is None:
            return

        # Delete plot files
        plot_dir = self._get_oos_directory(
            user_id, calc.dataset_id, calculation_id
        )
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
            logger.info(f"Deleted OOS plot directory: {plot_dir}")

        # Cascade delete handles OOSPatientResult records
        self.session.delete(calc)
        self.session.commit()
        logger.info(f"Deleted OOS calculation {calculation_id}")

    def delete_dataset_oos_data(self, user_id: int, dataset_id: int) -> int:
        """
        Delete all OOS data for a dataset.

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.

        Returns
        -------
        int
            Number of calculations deleted.
        """
        calcs = self.get_dataset_calculations(
            dataset_id, include_stale=True
        )
        count = len(calcs)
        for calc in calcs:
            if calc.id is not None:
                self.delete_calculation(user_id, calc.id)
        return count
