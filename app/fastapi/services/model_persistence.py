"""
Service for persisting and loading fitted GAMLSS models.

Handles saving models to .rds files and tracking them in the database.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from sqlmodel import Session, select

if TYPE_CHECKING:
    import pandas as pd

from app.core.engine.model import GAMLSS, FittedGAMLSSModel
from app.fastapi.config import get_settings
from app.fastapi.db.models import FittedModel, OOSCalculation

logger = logging.getLogger(__name__)


class ModelPersistenceService:
    """
    Service for saving and loading fitted GAMLSS models.

    Models are stored as .rds files on disk with metadata in the database.
    Directory structure: {models_dir}/user_{user_id}/dataset_{dataset_id}/

    Parameters
    ----------
    session : Session
        Database session for model metadata operations.
    """

    def __init__(self, session: Session):
        self.session = session
        self.models_dir = Path(get_settings().models_dir)

    def _get_model_directory(self, user_id: int, dataset_id: int) -> Path:
        """
        Get the directory path for a dataset's models.

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.

        Returns
        -------
        Path
            Directory path for the dataset's models.
        """
        return self.models_dir / f"user_{user_id}" / f"dataset_{dataset_id}"

    @staticmethod
    def _save_figure_to_png(fig: plt.Figure, path: Path) -> None:
        """
        Save a matplotlib figure to PNG and close it.

        Parameters
        ----------
        fig : plt.Figure
            The figure to save.
        path : Path
            Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _get_model_path(self, user_id: int, dataset_id: int, structure: str) -> Path:
        """
        Get the file path for a specific model.

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.
        structure : str
            The brain structure name.

        Returns
        -------
        Path
            File path for the model's .rds file.
        """
        return self._get_model_directory(user_id, dataset_id) / f"{structure}.rds"

    def save_model(
        self,
        fitted_model: FittedGAMLSSModel,
        user_id: int,
        dataset_id: int,
        structure: str,
    ) -> FittedModel:
        """
        Save a fitted model to disk and create database record.

        Parameters
        ----------
        fitted_model : FittedGAMLSSModel
            The fitted GAMLSS model to save.
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.
        structure : str
            The brain structure name.

        Returns
        -------
        FittedModel
            The database record for the saved model.
        """
        # Create directory if needed
        model_dir = self._get_model_directory(user_id, dataset_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save .rds file
        model_path = self._get_model_path(user_id, dataset_id, structure)
        fitted_model.save(str(model_path))

        # Get model family
        try:
            family = str(fitted_model.model.rx2("family")[0])
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            logger.debug(f"Could not extract model family: {e}")
            family = "unknown"

        # Check for existing record and update or create
        existing = self.session.exec(
            select(FittedModel).where(
                FittedModel.dataset_id == dataset_id,
                FittedModel.structure == structure,
            )
        ).first()

        # Generate reference plot
        reference_plot_path: str | None = None
        try:
            fig = fitted_model.generate_grids()
            plot_file = model_dir / f"{structure}_reference_plot.png"
            self._save_figure_to_png(fig, plot_file)
            reference_plot_path = str(plot_file)
            logger.info(f"Saved reference plot for {structure}")
        except Exception as e:
            logger.warning(f"Could not generate reference plot for {structure}: {e}")

        if existing:
            # Update existing record
            existing.family = family
            existing.aic = fitted_model.aic
            existing.bic = fitted_model.bic
            existing.file_path = str(model_path)
            existing.reference_plot_path = reference_plot_path
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            logger.info(f"Updated model for {structure} in dataset {dataset_id}")
            db_record = existing
        else:
            # Create new record
            db_model = FittedModel(
                dataset_id=dataset_id,
                structure=structure,
                family=family,
                aic=fitted_model.aic,
                bic=fitted_model.bic,
                file_path=str(model_path),
                reference_plot_path=reference_plot_path,
            )
            self.session.add(db_model)
            self.session.commit()
            self.session.refresh(db_model)
            logger.info(f"Saved new model for {structure} in dataset {dataset_id}")
            db_record = db_model

        # Mark all OOS calculations for this dataset as stale
        stale_calcs = self.session.exec(
            select(OOSCalculation).where(
                OOSCalculation.dataset_id == dataset_id,
                OOSCalculation.is_stale == False,  # noqa: E712
            )
        ).all()
        for calc in stale_calcs:
            calc.is_stale = True
            self.session.add(calc)
        if stale_calcs:
            self.session.commit()
            logger.info(
                f"Marked {len(stale_calcs)} OOS calculations as stale "
                f"for dataset {dataset_id}"
            )

        return db_record

    def load_model(
        self,
        dataset_id: int,
        structure: str,
        source_data: "pd.DataFrame",
        x_column: str,
        percentiles: list[float],
    ) -> FittedGAMLSSModel | None:
        """
        Load a fitted model from disk.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.
        structure : str
            The brain structure name.
        source_data : pd.DataFrame
            The source data (needed for predictions).
        x_column : str
            The x column name.
        percentiles : list[float]
            The percentiles to use.

        Returns
        -------
        FittedGAMLSSModel | None
            The loaded model, or None if not found.
        """
        # Get model record from database
        db_model = self.session.exec(
            select(FittedModel).where(
                FittedModel.dataset_id == dataset_id,
                FittedModel.structure == structure,
            )
        ).first()

        if db_model is None:
            logger.warning(f"No model found for {structure} in dataset {dataset_id}")
            return None

        # Check if file exists
        if not os.path.exists(db_model.file_path):
            logger.error(f"Model file not found: {db_model.file_path}")
            return None

        # Load from .rds file
        try:
            return GAMLSS.load_model(
                model_path=db_model.file_path,
                source_data=source_data,
                x_column=x_column,
                y_column=structure,
                percentiles=percentiles,
            )
        except Exception as e:
            logger.error(f"Failed to load model from {db_model.file_path}: {e}")
            return None

    def get_dataset_models(self, dataset_id: int) -> list[FittedModel]:
        """
        Get all fitted models for a dataset.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.

        Returns
        -------
        list[FittedModel]
            List of fitted model records.
        """
        return list(
            self.session.exec(
                select(FittedModel).where(FittedModel.dataset_id == dataset_id)
            ).all()
        )

    def delete_model_files(self, user_id: int, dataset_id: int) -> None:
        """
        Delete model files from disk (not database records).

        Use this when cascade delete handles DB records.

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.
        """
        model_dir = self._get_model_directory(user_id, dataset_id)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model directory: {model_dir}")

    def delete_dataset_models(self, user_id: int, dataset_id: int) -> int:
        """
        Delete all models for a dataset (files and database records).

        Parameters
        ----------
        user_id : int
            The user ID.
        dataset_id : int
            The dataset ID.

        Returns
        -------
        int
            Number of models deleted.
        """
        # Get all models for this dataset
        models = self.get_dataset_models(dataset_id)

        # Delete database records
        for model in models:
            self.session.delete(model)

        self.session.commit()

        # Delete model files
        self.delete_model_files(user_id, dataset_id)

        return len(models)

    def get_reference_plot_path(self, dataset_id: int, structure: str) -> str | None:
        """
        Get the reference plot path for a fitted model.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.
        structure : str
            The brain structure name.

        Returns
        -------
        str | None
            Path to the reference plot PNG, or None if not available.
        """
        db_model = self.session.exec(
            select(FittedModel).where(
                FittedModel.dataset_id == dataset_id,
                FittedModel.structure == structure,
            )
        ).first()
        if db_model is None or db_model.reference_plot_path is None:
            return None
        if not os.path.exists(db_model.reference_plot_path):
            return None
        return db_model.reference_plot_path

    def has_fitted_models(self, dataset_id: int) -> bool:
        """
        Check if a dataset has any fitted models.

        Parameters
        ----------
        dataset_id : int
            The dataset ID.

        Returns
        -------
        bool
            True if the dataset has fitted models.
        """
        model = self.session.exec(
            select(FittedModel).where(FittedModel.dataset_id == dataset_id)
        ).first()
        return model is not None
