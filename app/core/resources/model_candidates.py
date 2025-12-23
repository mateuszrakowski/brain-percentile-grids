from typing import Any, Dict
from pydantic import BaseModel

# Model definitions as data
MODEL_DEFINITIONS = [
    {
        "name": "Normal_Simple",
        "family": "NO",
        "mu_formula": "pb({x}, df=2)",
        "sigma_formula": "1",
        "complexity": 1,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "LogNormal_Simple",
        "family": "LOGNO",
        "mu_formula": "pb({x}, df=2)",
        "sigma_formula": "1",
        "complexity": 1,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "Normal_Smooth_Sigma",
        "family": "NO",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "complexity": 2,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "LogNormal_Smooth",
        "family": "LOGNO",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "complexity": 2,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "BCT_Constant_Nu",
        "family": "BCT",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "nu_formula": "1",
        "complexity": 3,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "BCT_Smooth_Nu",
        "family": "BCT",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "nu_formula": "pb({x}, df=2)",
        "complexity": 4,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
        },
    },
    {
        "name": "BCPE_Constant_Nu_Tau",
        "family": "BCPEo",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "nu_formula": "1",
        "tau_formula": "1",
        "complexity": 5,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
            "tau_step": 0.01,
            "tau_max": 10.0,
        },
    },
    {
        "name": "BCPE_Smooth_Nu_Constant_Tau",
        "family": "BCPEo",
        "mu_formula": "pb({x}, df=3)",
        "sigma_formula": "pb({x}, df=2)",
        "nu_formula": "pb({x}, df=2)",
        "tau_formula": "1",
        "complexity": 6,
        "control_params": {
            "n_cyc": 2000,
            "trace": False,
            "mu_step": 0.5,
            "sigma_step": 0.5,
            "tau_step": 0.01,
            "tau_max": 10.0,
        },
    },
]


class ModelCandidate(BaseModel):
    name: str
    family: str
    mu_formula: str
    sigma_formula: str
    nu_formula: str | None = None
    tau_formula: str | None = None
    complexity: int
    control_params: Dict[str, Any]


MODEL_CANDIDATES = [ModelCandidate(**definition) for definition in MODEL_DEFINITIONS]  # type: ignore
MODEL_CANDIDATES_DICT = {model.name: model for model in MODEL_CANDIDATES}


def get_all_model_candidates():
    """
    Return all available model candidates.

    Returns
    -------
    list[ModelCandidate]
        List of all predefined GAMLSS model configurations.
    """
    return MODEL_CANDIDATES


def get_model_candidate_by_name(name: str):
    """
    Get a specific model candidate by name.

    Parameters
    ----------
    name : str
        Name of the model candidate to retrieve.

    Returns
    -------
    ModelCandidate | None
        The model candidate if found, None otherwise.
    """
    return MODEL_CANDIDATES_DICT.get(name)
