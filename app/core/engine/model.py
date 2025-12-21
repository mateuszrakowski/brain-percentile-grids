import json
import os
import re
from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import PackageNotInstalledError
from scipy.stats import norm

from app.core.engine.environment import get_r_environment

r_env = get_r_environment()


class FittedGAMLSSModel:
    """
    Represents a single, fitted GAMLSS model.

    This object is self-contained and holds the R model object, its metrics,
    and the methods to interact with it (predict, plot, save).
    """

    def __init__(
        self,
        r_model: robjects.vectors.ListVector,
        source_data: pd.DataFrame,
        x_column: str,
        y_column: str,
        percentiles: list[float],
    ):
        self.model = r_model
        self.data_table = source_data
        self.x_column = x_column
        self.y_column = y_column
        self.percentiles = percentiles

        # Key metrics calculated on initialization
        self.converged: bool = bool(self.model.rx2("converged")[0])
        self.aic: float = self._calculate_metric("AIC")
        self.bic: float = self._calculate_metric("BIC")
        self.deviance: float = self.model.rx2("G.deviance")[0]

    def _calculate_metric(self, metric_name: str) -> float:
        """Helper to safely calculate AIC or BIC."""
        try:
            metric_func = getattr(r_env.stats, metric_name)
            metric_r_object = metric_func(self.model)
            return float(np.array(metric_r_object)[0])
        except (AttributeError, PackageNotInstalledError):
            return float("inf")

    def _convert_table_to_r(self, table: pd.DataFrame) -> robjects.DataFrame:
        """Converts a pandas DataFrame to an R DataFrame."""
        with r_env.localconverter(robjects.default_converter + r_env.pandas2ri.converter):
            return robjects.conversion.py2rpy(table)  # type: ignore

    @staticmethod
    def _split_structure_name(structure_name: str) -> str:
        """Adds spaces to a PascalCase string for plotting titles."""
        return re.sub(r"(?<!^)(?=[A-Z])", " ", structure_name)

    def save(self, model_path: str, run_info_path: str | None = None) -> None:
        """Saves the R model object to a .rds file and optionally saves run info."""
        # Save the model
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        r_env.base.saveRDS(self.model, file=model_path)
        print(f"Model saved to {model_path}")

        # Auto-generate run_info_path if not provided
        if run_info_path is None:
            base_name, _ = os.path.splitext(model_path)
            run_info_path = f"{base_name}_run_info.json"

        # Save run info
        run_info_dir = os.path.dirname(run_info_path)
        if not os.path.exists(run_info_dir):
            os.makedirs(run_info_dir)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        data_to_save = {
            "dataset_length": len(self.data_table),
            "timestamp": timestamp_str,
            "model_family": str(self.model.rx2("family")[0]),
            "aic": self.aic,
            "bic": self.bic,
        }
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f)
        print(f"Run info saved to {run_info_path}")

    def calculate_percentiles(self) -> dict[float, np.ndarray]:
        """Calculates percentile curves for the fitted model."""
        x_pred_points = np.linspace(
            self.data_table[self.x_column].min(),
            self.data_table[self.x_column].max(),
            200,
        )
        df_pred_r = self._convert_table_to_r(pd.DataFrame({self.x_column: x_pred_points}))

        pred_params = r_env.gamlss_r.predictAll(
            object=self.model, newdata=df_pred_r, type="response"
        )

        # Dynamically get the quantile function for the model's family
        family_name = self.model.rx2("family")[0]
        q_function_name = f"q{family_name}"
        q_function = getattr(r_env.gamlss_dist, q_function_name)

        # Prepare parameters for the quantile function
        params_for_prediction = {"p": self.percentiles}
        for param_name in self.model.rx2("parameters"):
            if param_name in pred_params.names:
                params_for_prediction[param_name] = pred_params.rx2(param_name)

        percentile_curves = {}
        for p in self.percentiles:
            params_for_prediction["p"] = p  # type: ignore
            p_curve = q_function(**params_for_prediction)
            percentile_curves[p] = np.array(p_curve)

        return percentile_curves

    def plot_percentiles(self, percentile_curves: dict[float, np.ndarray]) -> plt.Figure:
        """Generates and returns a matplotlib figure of the percentile curves."""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.scatter(
            self.data_table[self.x_column],
            self.data_table[self.y_column],
            alpha=0.1,
            label="Data Points",
            s=5,
        )

        x_pred_points = np.linspace(
            self.data_table[self.x_column].min(),
            self.data_table[self.x_column].max(),
            200,
        )

        for p, curve_data in percentile_curves.items():
            linestyle = "-" if p == 0.50 else "--"
            linewidth = 1.5 if p == 0.50 else 1.0
            ax.plot(
                x_pred_points,
                curve_data,
                label=f"{int(p*100)}th Percentile",
                linestyle=linestyle,
                linewidth=linewidth,
            )

        ax.set_xlabel("Age")
        ax.set_ylabel("Volume")
        ax.set_title(self._split_structure_name(self.y_column))
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        plot_min_y = np.percentile(self.data_table[self.y_column], 1)
        plot_max_y = np.percentile(self.data_table[self.y_column], 99)
        plot_range = plot_max_y - plot_min_y
        ax.set_ylim(plot_min_y - 0.1 * plot_range, plot_max_y + 0.1 * plot_range)

        return fig  # type: ignore

    def predict_patient_oos(self, patient_data: pd.DataFrame) -> tuple[float, float]:
        oos_zscore = None
        oos_percentile = None

        try:
            centiles_pred_func = r_env.gamlss_r.centiles_pred
            x_value_r = robjects.FloatVector([patient_data[self.x_column].values[0]])
            y_value_r = robjects.FloatVector([patient_data[self.y_column].values[0]])

            zscore_result_r = centiles_pred_func(
                self.model,
                xvalues=x_value_r,
                yval=y_value_r,
                type="z-scores",
                xname=self.x_column,
            )

            oos_zscore = np.array(zscore_result_r)[0]
            oos_percentile = norm.cdf(oos_zscore)
        except (AttributeError, PackageNotInstalledError):
            print("R environment not available. Skipping OOS prediction.")
            oos_zscore, oos_percentile = np.nan, np.nan

        return oos_zscore, oos_percentile

    def plot_oos_patient(
        self,
        patient_data: pd.DataFrame,
        percentile_curves: dict[float, np.ndarray],
        oos_zscore,
        oos_percentile,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))

        x_pred_points = np.linspace(
            self.data_table[self.x_column].min(),
            self.data_table[self.x_column].max(),
            200,
        )

        for p, curve_data in percentile_curves.items():
            linestyle = "-" if p == 0.50 else "--"
            linewidth = 1.5 if p == 0.50 else 1.0
            alpha_line = 0.5

            label = f"{int(p*100)}th Percentile" if p in [0.05, 0.50, 0.95] else None

            ax.plot(
                x_pred_points,
                curve_data,
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha_line,
                color="gray",
            )

        ax.scatter(
            self.data_table[self.x_column],
            self.data_table[self.y_column],
            alpha=0.15,
            label=(
                f"Training Data ({min(self.data_table[self.x_column])}-"
                f"{max(self.data_table[self.x_column])} yrs)"
            ),
            s=15,
            color="lightblue",
        )

        ax.scatter(
            [patient_data[self.x_column].values[0]],
            [patient_data[self.y_column].values[0]],
            color="red",
            s=80,
            zorder=5,
            label=(
                f"Patient (Age={patient_data[self.x_column].values[0]}, "
                f"Vol={patient_data[self.y_column].values[0]})"
            ),
        )

        if oos_percentile:
            ax.annotate(
                f" P {oos_percentile*100:.1f}\n (Z={oos_zscore:.2f})",
                (
                    patient_data[self.x_column].values[0],
                    patient_data[self.y_column].values[0],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )
        else:
            raise Exception("No percentile found for OOS patient!")

        ax.set_xlabel("Age")
        ax.set_ylabel("Volume")
        ax.set_title(self._split_structure_name(self.y_column))
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        plot_min_y = min(
            np.percentile(self.data_table[self.y_column], 1),
            patient_data[self.y_column].values[0] * 0.9,
        )
        plot_max_y = max(
            np.percentile(self.data_table[self.y_column], 99),
            patient_data[self.y_column].values[0] * 1.1,
        )
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.set_xlim(
            self.data_table[self.x_column].min() - 1,
            self.data_table[self.x_column].max() + 1,
        )

        return fig

    def generate_grids(self) -> plt.Figure:
        percentile_curves = self.calculate_percentiles()
        plot_figure = self.plot_percentiles(percentile_curves)

        return plot_figure

    def generate_grids_oos(self, patient_data: pd.DataFrame) -> plt.Figure:
        percentile_curves = self.calculate_percentiles()
        oos_zscore, oos_percentile = self.predict_patient_oos(patient_data)

        plot_figure = self.plot_oos_patient(
            patient_data, percentile_curves, oos_zscore, oos_percentile
        )

        return plot_figure


class GAMLSS:
    """
    A factory class for fitting GAMLSS models.

    This class holds the data and configuration, but not the state of a
    single fitted model. Its 'fit' method returns a new FittedGAMLSSModel object.
    """

    def __init__(
        self,
        data_table: pd.DataFrame,
        x_column: str,
        y_column: str,
        percentiles: list[float],
    ):
        self.data_table = data_table
        self.x_column = x_column
        self.y_column = y_column
        self.percentiles = percentiles

    def _convert_table_to_r(self, table: pd.DataFrame) -> robjects.DataFrame:
        with r_env.localconverter(robjects.default_converter + r_env.pandas2ri.converter):
            return robjects.conversion.py2rpy(table)

    @staticmethod
    def load_model(
        model_path: str,
        source_data: pd.DataFrame,
        x_column: str,
        y_column: str,
        percentiles: list[float],
    ) -> "FittedGAMLSSModel":
        """
        Loads a pre-trained GAMLSS model from an RDS file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
        try:
            readRDS = r_env.base.readRDS
            r_model = readRDS(file=model_path)
            return FittedGAMLSSModel(
                r_model=r_model,
                source_data=source_data,
                x_column=x_column,
                y_column=y_column,
                percentiles=percentiles,
            )
        except (AttributeError, PackageNotInstalledError) as e:
            print(f"Failed to load model: {e}")
            raise

    @staticmethod
    def load_run_info(filename: str) -> None:
        try:
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            return None
        except Exception as e:
            print(f"Failed to load run info: {e}")

    def fit(
        self,
        family: str,
        formula_mu: str,
        formula_sigma: str,
        formula_nu: str | None = None,
        formula_tau: str | None = None,
        control_params: Dict[str, Any] | None = None,
    ) -> FittedGAMLSSModel:
        """
        Fits a GAMLSS model with the given parameters and returns a
        FittedGAMLSSModel object.
        """
        r_table = self._convert_table_to_r(self.data_table)
        formulas = {
            "formula": robjects.Formula(f"{self.y_column} ~ {formula_mu}"),
            "sigma_formula": robjects.Formula(f"~ {formula_sigma}"),
        }
        if formula_nu:
            formulas["nu_formula"] = robjects.Formula(f"~ {formula_nu}")
        if formula_tau:
            formulas["tau_formula"] = robjects.Formula(f"~ {formula_tau}")

        control = r_env.gamlss_r.gamlss_control(**(control_params or {}))

        r_model_object = r_env.gamlss_r.gamlss(
            family=family,
            data=r_table,
            control=control,
            **formulas,
        )

        return FittedGAMLSSModel(
            r_model=r_model_object,
            source_data=self.data_table,
            x_column=self.x_column,
            y_column=self.y_column,
            percentiles=self.percentiles,
        )
