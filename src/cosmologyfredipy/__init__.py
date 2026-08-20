"""A compact FrediPy example for primordial-spectrum reconstruction."""

from .analysis import (
    GP_SENSITIVITY_SETTINGS,
    PLANCK_KEYS,
    SYNTHETIC_KEYS,
    PlanckAnalysis,
    PlanckSensitivityRow,
    PowerLawFit,
    SyntheticAnalysis,
    analyse_planck,
    analyse_planck_sensitivity,
    analyse_synthetic,
    make_summary,
    write_summary,
)
from .inversion import MatrixIntegrator, Posterior, invert

__all__ = [
    "GP_SENSITIVITY_SETTINGS",
    "PLANCK_KEYS",
    "SYNTHETIC_KEYS",
    "MatrixIntegrator",
    "PlanckAnalysis",
    "PlanckSensitivityRow",
    "Posterior",
    "PowerLawFit",
    "SyntheticAnalysis",
    "analyse_planck",
    "analyse_planck_sensitivity",
    "analyse_synthetic",
    "invert",
    "make_summary",
    "write_summary",
]
