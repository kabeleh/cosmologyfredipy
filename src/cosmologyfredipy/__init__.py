"""A compact FrediPy example for primordial-spectrum reconstruction."""

from .analysis import (
    PLANCK_KEYS,
    SYNTHETIC_KEYS,
    PlanckAnalysis,
    PowerLawFit,
    SyntheticAnalysis,
    analyse_planck,
    analyse_synthetic,
    make_summary,
    write_summary,
)
from .inversion import MatrixIntegrator, Posterior, invert

__all__ = [
    "PLANCK_KEYS",
    "SYNTHETIC_KEYS",
    "MatrixIntegrator",
    "PlanckAnalysis",
    "Posterior",
    "PowerLawFit",
    "SyntheticAnalysis",
    "analyse_planck",
    "analyse_synthetic",
    "invert",
    "make_summary",
    "write_summary",
]
