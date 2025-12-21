import os
from typing import Any, Dict

from app.core.engine.model import GAMLSS, FittedGAMLSSModel
from app.core.resources.model_candidates import ModelCandidate


class GAMLSSModelSelector:
    """
    Selects the best GAMLSS model from a list of candidates.
    """

    def __init__(self, gamlss_fitter: GAMLSS, model_candidates: list[ModelCandidate]):
        self.fitter = gamlss_fitter
        self.model_candidates = model_candidates
        self.results: Dict[str, Any] = {}

    def select_best_model(
        self, model_path: str = "/app/data/models/", criterion: str = "bic"
    ) -> FittedGAMLSSModel | None:
        """
        Checks if a saved model exists at the given path and loads it.
        If not found, it runs the model selection process, saves the best
        model to the path, and returns it.

        Args:
            model_path str: The full path to the .rds model file.
            criterion str: The criterion for model selection ('aic', 'bic', 'deviance').

        Returns:
            FittedGAMLSSModel | None: The loaded or newly fitted model, or None if
            fitting fails.
        """
        model_path = os.path.abspath(model_path + f"gamlss_{getattr(self.fitter, 'y_column')}.rds")

        if os.path.exists(model_path):
            print(f"Found existing model at {model_path}. Loading...")
            try:
                loaded_model = GAMLSS.load_model(
                    model_path=model_path,
                    source_data=self.fitter.data_table,
                    x_column=self.fitter.x_column,
                    y_column=self.fitter.y_column,
                    percentiles=self.fitter.percentiles,
                )
                print("Model loaded successfully.")
                return loaded_model
            except Exception as e:
                print(f"Warning: Failed to load existing model at {model_path}. " f"Error: {e}")
                print("Proceeding to fit a new model.")

        print(f"No existing model found at {model_path}. Starting model selection...")
        best_model = self.fit_models(criterion=criterion)

        if best_model:
            print(f"Saving the best model to {model_path}...")
            try:
                # The save method will now also save the corresponding .json run info
                best_model.save(model_path=model_path)
            except Exception as e:
                print(f"Warning: Failed to save the best model. Error: {e}")

        return best_model

    def _get_sample_size_appropriate_models(self, n: int) -> list[ModelCandidate]:
        if n < 30:
            max_complexity = 2
        elif n < 100:
            max_complexity = 4
        elif n < 200:
            max_complexity = 5
        else:
            max_complexity = 6

        return [m for m in self.model_candidates if m.complexity <= max_complexity]

    def fit_models(self, criterion: str = "bic") -> FittedGAMLSSModel | None:
        if criterion not in ["aic", "bic", "deviance"]:
            raise ValueError("Criterion must be 'aic', 'bic', or 'deviance'.")

        n_samples = len(self.fitter.data_table)
        appropriate_models = self._get_sample_size_appropriate_models(n_samples)

        print(f"Testing {len(appropriate_models)} models for n={n_samples} samples...")

        for model_config in appropriate_models:
            model_name = model_config.name
            print(f"Fitting {model_name}...")
            try:
                # Build fit parameters, including nu and tau if present
                fit_params = {
                    "family": model_config.family,
                    "formula_mu": model_config.mu_formula.format(x=self.fitter.x_column),
                    "formula_sigma": model_config.sigma_formula.format(x=self.fitter.x_column),
                    "control_params": model_config.control_params,
                }

                # Add nu formula if present in model config
                if model_config.nu_formula:
                    fit_params["formula_nu"] = model_config.nu_formula.format(
                        x=self.fitter.x_column
                    )

                # Add tau formula if present in model config
                if model_config.tau_formula:
                    fit_params["formula_tau"] = model_config.tau_formula.format(
                        x=self.fitter.x_column
                    )

                fitted_model = self.fitter.fit(**fit_params)  # type: ignore
                self.results[model_name] = fitted_model

                status = "✓" if fitted_model.converged else "⚠"
                metric_value = getattr(fitted_model, criterion)
                print(
                    f"  {status} Converged: {fitted_model.converged}, "
                    f"{criterion.upper()}: {metric_value:.2f}"
                )

            except Exception as e:
                print(f"  ✗ Failed to fit {model_name}: {e}")
                self.results[model_name] = None

        successful_models = {
            name: model for name, model in self.results.items() if model and model.converged
        }

        if not successful_models:
            print("\nNo models converged successfully.")
            return None

        best_name = min(
            successful_models,
            key=lambda name: getattr(successful_models[name], criterion),
        )

        best_model = successful_models[best_name]

        print(f"\n🏆 Best model found: {best_name}")
        print(f"   BIC: {best_model.bic:.2f}")
        print(f"   AIC: {best_model.aic:.2f}")
        print(f"   Deviance: {best_model.deviance:.2f}")

        return best_model
