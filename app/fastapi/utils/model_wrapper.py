"""
Model wrapper to provide a consistent interface for the Flask app.

This module provides a simplified interface for GAMLSS model selection and fitting,
abstracting the complexity of the underlying statistical engine for Flask application use.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.engine.model import GAMLSS, FittedGAMLSSModel
from core.engine.selector import GAMLSSModelSelector
from core.resources.model_candidates import ModelCandidate

# Configure logging
logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Wrapper class to provide a simplified interface for GAMLSS model selection.

    This class bridges the gap between the existing GAMLSSModelSelector and Flask app needs,
    providing a clean interface for model fitting and selection operations.

    Attributes
    ----------
    data_table : pd.DataFrame
        DataFrame containing the training data.
    x_column : str
        Name of the independent variable column (typically age).
    y_column : str
        Name of the dependent variable column (brain structure volume).
    percentiles : List[float]
        List of percentiles to calculate.
    gamlss_fitter : GAMLSS
        GAMLSS engine instance.
    model_candidates : List[ModelCandidate]
        List of model candidates for selection.
    selector : GAMLSSModelSelector
        GAMLSSModelSelector instance.
    """

    def __init__(
        self,
        data_table: pd.DataFrame,
        x_column: str,
        y_column: str,
        percentiles: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the ModelSelector.

        Parameters
        ----------
        data_table : pd.DataFrame
            DataFrame containing the training data.
        x_column : str
            Name of the independent variable column.
        y_column : str
            Name of the dependent variable column.
        percentiles : Optional[List[float]], optional
            List of percentiles to calculate.

        Raises
        ------
        ValueError
            If data_table is empty or columns are not found.
        """
        # Validate inputs
        if data_table.empty:
            raise ValueError("data_table cannot be empty")
        if x_column not in data_table.columns:
            raise ValueError(f"Column '{x_column}' not found in data_table")
        if y_column not in data_table.columns:
            raise ValueError(f"Column '{y_column}' not found in data_table")

        self.data_table = data_table
        self.x_column = x_column
        self.y_column = y_column
        self.percentiles = percentiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        # Create GAMLSS fitter instance
        self.gamlss_fitter = GAMLSS(
            data_table=data_table,
            x_column=x_column,
            y_column=y_column,
            percentiles=percentiles,
        )

        # Load default model candidates
        self.model_candidates = self._get_default_model_candidates()

        # Create the selector
        self.selector = GAMLSSModelSelector(
            gamlss_fitter=self.gamlss_fitter, model_candidates=self.model_candidates
        )

    def _get_default_model_candidates(self) -> List[ModelCandidate]:
        """
        Get default model candidates for brain structure modeling.

        Returns
        -------
        List[ModelCandidate]
            List of model candidates for GAMLSS fitting.
        """
        # Import here to avoid circular imports
        from core.resources.model_candidates import get_all_model_candidates

        try:
            return get_all_model_candidates()
        except (ImportError, AttributeError):
            # Fallback to basic model candidates if the resource file doesn't exist
            return self._create_fallback_candidates()

    def _create_fallback_candidates(self) -> List[ModelCandidate]:
        """
        Create basic fallback model candidates if the resource file is not available.

        Returns
        -------
        List[ModelCandidate]
            List of basic fallback model candidates.
        """
        return [
            ModelCandidate(
                name="Linear_Normal",
                family="NO",
                mu_formula="{x}",
                sigma_formula="1",
                complexity=1,
                control_params={"n.cyc": 200},
            ),
            ModelCandidate(
                name="Linear_Log_Normal",
                family="LOGNO",
                mu_formula="{x}",
                sigma_formula="1",
                complexity=1,
                control_params={"n.cyc": 200},
            ),
            ModelCandidate(
                name="Quadratic_Normal",
                family="NO",
                mu_formula="poly({x}, 2)",
                sigma_formula="1",
                complexity=2,
                control_params={"n.cyc": 200},
            ),
            ModelCandidate(
                name="Spline_Normal",
                family="NO",
                mu_formula="cs({x}, 3)",
                sigma_formula="1",
                complexity=3,
                control_params={"n.cyc": 200},
            ),
        ]

    def get_best_model(self, criterion: str = "bic") -> Optional[FittedGAMLSSModel]:
        """
        Get the best model using the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            Model selection criterion ('aic', 'bic', 'deviance').

        Returns
        -------
        Optional[FittedGAMLSSModel]
            Best fitted model or None if no model converged.

        Raises
        ------
        ValueError
            If criterion is not supported.
        """
        valid_criteria = {"aic", "bic", "deviance"}
        if criterion.lower() not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Must be one of: {valid_criteria}")

        try:
            logger.info(f"Fitting models with criterion: {criterion}")
            # Use in-memory fitting instead of file-based approach for Flask
            best_model = self.selector.fit_models(criterion=criterion)
            if best_model:
                logger.info(f"Successfully fitted model with {criterion} criterion")
            else:
                logger.warning("No models converged successfully")
            return best_model
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return None

    def get_available_families(self) -> List[str]:
        """
        Get list of available distribution families.

        Returns
        -------
        List[str]
            List of unique distribution family names.
        """
        return list(set(candidate.family for candidate in self.model_candidates))

    def get_available_formulas(self) -> List[str]:
        """
        Get list of available formula types.

        Returns
        -------
        List[str]
            List of unique mu formula templates.
        """
        return list(set(candidate.mu_formula for candidate in self.model_candidates))
