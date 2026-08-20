"""The two analyses used in the paper: CLASS closure and Planck PR3 TT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from .inversion import Posterior, invert


SYNTHETIC_KEYS = (
    "a0",
    "gp_sigma",
    "gp_gamma",
    "k_per_mpc",
    "multipoles",
    "operator",
    "data_d_ell",
    "data_variance",
    "primordial_truth",
    "evaluation_mask",
    "raw_cl_tt",
    "direct_cl_tt",
)

PLANCK_KEYS = (
    "a0",
    "gp_sigma",
    "gp_gamma",
    "pivot_per_mpc",
    "k_per_mpc",
    "band_operator",
    "observed_bandpowers",
    "band_covariance",
    "ell_eff",
    "lensing_template_bandpowers",
    "fiducial_primordial_power",
    "reported_band_sigma",
)

SENSITIVITY_THRESHOLD = 0.01


@dataclass(frozen=True)
class SyntheticAnalysis:
    posterior: Posterior
    operator: np.ndarray
    multipoles: np.ndarray
    data_d_ell: np.ndarray
    data_variance: np.ndarray
    truth: np.ndarray
    evaluation_mask: np.ndarray
    max_kernel_closure_fraction: float
    max_interior_fractional_error: float
    median_interior_fractional_error: float
    rms_interior_fractional_error: float


@dataclass(frozen=True)
class PowerLawFit:
    """Best fit of P(k) = amplitude (k/pivot)^(spectral_index - 1)."""

    amplitude: float
    spectral_index: float
    pivot_per_mpc: float
    spectrum: np.ndarray
    q: float


@dataclass(frozen=True)
class PlanckAnalysis:
    posterior: Posterior
    operator: np.ndarray
    ell_eff: np.ndarray
    conditional_bandpowers: np.ndarray
    observed_bandpowers: np.ndarray
    lensing_template_bandpowers: np.ndarray
    covariance: np.ndarray
    reported_sigma: np.ndarray
    fiducial_spectrum: np.ndarray
    sensitivity: np.ndarray
    sensitivity_mask: np.ndarray
    power_law: PowerLawFit
    q_flexible: float
    max_power_law_fractional_difference: float
    rms_power_law_fractional_difference: float
    max_power_law_difference_sigma: float

    @property
    def delta_q(self) -> float:
        """Power-law Q minus flexible-reconstruction Q."""

        return self.power_law.q - self.q_flexible


def _load_npz(path: str | Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        return {name: np.asarray(archive[name]).copy() for name in required}


def _scalar(arrays: dict[str, np.ndarray], name: str) -> float:
    value = np.asarray(arrays[name], dtype=float)
    if value.shape != () or not np.isfinite(value):
        raise ValueError(f"{name} must be one finite scalar")
    return float(value)


def _whitener(covariance: np.ndarray):
    covariance = np.asarray(covariance, dtype=float)
    factor = np.linalg.cholesky(covariance)

    def whiten(values: np.ndarray) -> np.ndarray:
        return solve_triangular(factor, values, lower=True, check_finite=False)

    return whiten


def _q(residual: np.ndarray, whiten) -> float:
    whitened = whiten(residual)
    return float(whitened @ whitened)


def analyse_synthetic(path: str | Path) -> SyntheticAnalysis:
    """Reconstruct the spectrum that generated the compact CLASS example."""

    arrays = _load_npz(path, SYNTHETIC_KEYS)
    a0 = _scalar(arrays, "a0")
    posterior = invert(
        arrays["operator"],
        arrays["k_per_mpc"],
        arrays["data_d_ell"],
        arrays["data_variance"],
        amplitude_scale=a0,
        gp_sigma=_scalar(arrays, "gp_sigma"),
        gp_gamma=_scalar(arrays, "gp_gamma"),
        prior_mean=a0,
    )
    truth = np.asarray(arrays["primordial_truth"], dtype=float)
    mask = np.asarray(arrays["evaluation_mask"], dtype=bool)
    if (
        truth.shape != posterior.mean.shape
        or mask.shape != truth.shape
        or not np.any(mask)
    ):
        raise ValueError("synthetic truth and evaluation mask must match the k grid")
    fractional_error = np.abs(posterior.mean / truth - 1.0)
    raw_cl = np.asarray(arrays["raw_cl_tt"], dtype=float)
    direct_cl = np.asarray(arrays["direct_cl_tt"], dtype=float)
    if raw_cl.shape != direct_cl.shape or np.any(raw_cl == 0.0):
        raise ValueError("raw_cl_tt and direct_cl_tt must be matching nonzero vectors")

    return SyntheticAnalysis(
        posterior=posterior,
        operator=np.asarray(arrays["operator"], dtype=float),
        multipoles=np.asarray(arrays["multipoles"], dtype=float),
        data_d_ell=np.asarray(arrays["data_d_ell"], dtype=float),
        data_variance=np.asarray(arrays["data_variance"], dtype=float),
        truth=truth,
        evaluation_mask=mask,
        max_kernel_closure_fraction=float(np.max(np.abs(direct_cl / raw_cl - 1.0))),
        max_interior_fractional_error=float(np.max(fractional_error[mask])),
        median_interior_fractional_error=float(np.median(fractional_error[mask])),
        rms_interior_fractional_error=float(
            np.sqrt(np.mean(np.square(fractional_error[mask])))
        ),
    )


def _fit_power_law(
    k: np.ndarray,
    operator: np.ndarray,
    data: np.ndarray,
    covariance: np.ndarray,
    a0: float,
    pivot: float,
) -> PowerLawFit:
    whiten = _whitener(covariance)

    def spectrum(parameters: np.ndarray) -> np.ndarray:
        log_amplitude_ratio, tilt = parameters
        return a0 * np.exp(log_amplitude_ratio) * (k / pivot) ** tilt

    def residual(parameters: np.ndarray) -> np.ndarray:
        prediction = operator @ (spectrum(parameters) / a0)
        return whiten(data - prediction)

    fit = least_squares(
        residual,
        x0=np.array([0.0, -0.035]),
        bounds=([-2.0, -0.5], [2.0, 0.5]),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=500,
    )
    if not fit.success:
        raise RuntimeError(f"power-law fit failed: {fit.message}")
    fitted_spectrum = spectrum(fit.x)
    return PowerLawFit(
        amplitude=float(a0 * np.exp(fit.x[0])),
        spectral_index=float(1.0 + fit.x[1]),
        pivot_per_mpc=float(pivot),
        spectrum=fitted_spectrum,
        q=float(fit.fun @ fit.fun),
    )


def analyse_planck(path: str | Path) -> PlanckAnalysis:
    """Run the fixed-calibration Planck PR3 TT reconstruction.

    The fixed lensing template is subtracted from the observed band powers
    before applying the linear primordial-spectrum operator.  Calibration is
    fixed at ``A_planck = 1``; no calibration marginalisation is performed.
    """

    arrays = _load_npz(path, PLANCK_KEYS)
    a0 = _scalar(arrays, "a0")
    operator = np.asarray(arrays["band_operator"], dtype=float)
    observed = np.asarray(arrays["observed_bandpowers"], dtype=float)
    lensing = np.asarray(arrays["lensing_template_bandpowers"], dtype=float)
    covariance = np.asarray(arrays["band_covariance"], dtype=float)
    conditional = observed - lensing

    posterior = invert(
        operator,
        arrays["k_per_mpc"],
        conditional,
        covariance,
        amplitude_scale=a0,
        gp_sigma=_scalar(arrays, "gp_sigma"),
        gp_gamma=_scalar(arrays, "gp_gamma"),
        prior_mean=a0,
    )
    whiten = _whitener(covariance)
    whitened_operator = whiten(operator)
    sensitivity = np.linalg.norm(whitened_operator, axis=0)
    mask = sensitivity >= SENSITIVITY_THRESHOLD * float(np.max(sensitivity))
    if not np.any(mask):
        raise RuntimeError("the operator-sensitivity mask is empty")

    power_law = _fit_power_law(
        posterior.k_per_mpc,
        operator,
        conditional,
        covariance,
        a0,
        _scalar(arrays, "pivot_per_mpc"),
    )
    flexible_prediction = operator @ (posterior.mean / a0)
    q_flexible = _q(conditional - flexible_prediction, whiten)
    difference_sigma = np.abs(posterior.mean - power_law.spectrum) / posterior.sigma
    fractional_difference = posterior.mean / power_law.spectrum - 1.0

    return PlanckAnalysis(
        posterior=posterior,
        operator=operator,
        ell_eff=np.asarray(arrays["ell_eff"], dtype=float),
        conditional_bandpowers=conditional,
        observed_bandpowers=observed,
        lensing_template_bandpowers=lensing,
        covariance=covariance,
        reported_sigma=np.asarray(arrays["reported_band_sigma"], dtype=float),
        fiducial_spectrum=np.asarray(arrays["fiducial_primordial_power"], dtype=float),
        sensitivity=sensitivity,
        sensitivity_mask=mask,
        power_law=power_law,
        q_flexible=q_flexible,
        max_power_law_fractional_difference=float(
            np.max(np.abs(fractional_difference[mask]))
        ),
        rms_power_law_fractional_difference=float(
            np.sqrt(np.mean(np.square(fractional_difference[mask])))
        ),
        max_power_law_difference_sigma=float(np.max(difference_sigma[mask])),
    )


def make_summary(
    synthetic: SyntheticAnalysis,
    planck: PlanckAnalysis,
) -> dict[str, Any]:
    """Return the small, JSON-ready set of numbers quoted in the paper."""

    direct_k = planck.posterior.k_per_mpc[planck.sensitivity_mask]
    return {
        "planck": {
            "a_planck_fixed": 1.0,
            "delta_q_power_law_minus_flexible": planck.delta_q,
            "direct_sensitivity_k_max_per_mpc": float(direct_k[-1]),
            "direct_sensitivity_k_min_per_mpc": float(direct_k[0]),
            "direct_sensitivity_nodes": int(direct_k.size),
            "sensitivity_threshold_fraction_of_peak": SENSITIVITY_THRESHOLD,
            "max_power_law_fractional_difference": (
                planck.max_power_law_fractional_difference
            ),
            "max_power_law_difference_sigma": planck.max_power_law_difference_sigma,
            "number_of_bands": int(planck.ell_eff.size),
            "power_law_amplitude": planck.power_law.amplitude,
            "power_law_pivot_per_mpc": planck.power_law.pivot_per_mpc,
            "power_law_spectral_index": planck.power_law.spectral_index,
            "q_flexible": planck.q_flexible,
            "q_power_law": planck.power_law.q,
            "rms_power_law_fractional_difference": (
                planck.rms_power_law_fractional_difference
            ),
        },
        "synthetic": {
            "evaluation_nodes": int(np.count_nonzero(synthetic.evaluation_mask)),
            "max_interior_fractional_error": synthetic.max_interior_fractional_error,
            "max_kernel_closure_fraction": synthetic.max_kernel_closure_fraction,
            "median_interior_fractional_error": synthetic.median_interior_fractional_error,
            "number_of_multipoles": int(synthetic.multipoles.size),
            "rms_interior_fractional_error": synthetic.rms_interior_fractional_error,
        },
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> str:
    """Write deterministic JSON and return the exact serialized text."""

    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return text
