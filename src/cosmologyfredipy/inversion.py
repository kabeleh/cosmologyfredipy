"""The small adapter between a precomputed Fredholm matrix and FrediPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fredipy.constraints import LinearEquality
from fredipy.integrators import Integrator
from fredipy.kernels import RadialBasisFunction
from fredipy.models import GaussianProcess
from fredipy.operators import Integral


class MatrixIntegrator(Integrator):
    """Use the rows of a precomputed matrix as FrediPy quadrature weights.

    The matrix already contains the physical kernel, quadrature weights, and
    interpolation onto the reconstruction nodes.  Constraint coordinates are
    therefore just integer row numbers.
    """

    def __init__(self, nodes: np.ndarray, matrix: np.ndarray) -> None:
        nodes = np.asarray(nodes, dtype=float)
        matrix = np.asarray(matrix, dtype=float)
        if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError(
                "nodes must be a strictly increasing one-dimensional array"
            )
        if matrix.ndim != 2 or matrix.shape[1] != nodes.size:
            raise ValueError("matrix columns must match the reconstruction nodes")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(matrix)):
            raise ValueError("nodes and matrix must be finite")
        self.w = nodes[:, None]
        self.matrix = matrix

    def _rows(self, constraint: LinearEquality) -> np.ndarray:
        values = np.asarray(constraint.x, dtype=float).reshape(-1)
        rows = values.astype(int)
        if not np.array_equal(values, rows.astype(float)):
            raise ValueError("matrix row labels must be integers")
        if np.any(rows < 0) or np.any(rows >= self.matrix.shape[0]):
            raise ValueError("matrix row label is out of range")
        return self.matrix[rows]

    def doubleIntegrationSymmetric(self, constraint, kernel) -> np.ndarray:
        rows = self._rows(constraint)
        return rows @ kernel(self.w, self.w) @ rows.T

    def doubleIntegration(self, constraint1, kernel, constraint2) -> np.ndarray:
        rows1 = self._rows(constraint1)
        rows2 = self._rows(constraint2)
        return rows1 @ kernel(self.w, self.w) @ rows2.T

    def singleIntegration(self, constraint, kernel, w_pred) -> np.ndarray:
        return self._rows(constraint) @ kernel(self.w, w_pred)


@dataclass(frozen=True)
class Posterior:
    """FrediPy posterior on the supplied wavenumber grid, in physical units."""

    k_per_mpc: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    prior_mean: np.ndarray
    model: GaussianProcess
    amplitude_scale: float

    @property
    def sigma(self) -> np.ndarray:
        """Pointwise posterior standard deviation."""

        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))


def _covariance_array(covariance: np.ndarray, size: int) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (size,):
        if np.any(covariance <= 0.0):
            raise ValueError("all data variances must be positive")
        matrix = np.diag(covariance)
    elif covariance.shape == (size, size):
        matrix = covariance
    else:
        raise ValueError(
            f"data covariance must have shape ({size},) or ({size}, {size})"
        )
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError("data covariance must be finite and symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data covariance must be positive definite") from exc
    return matrix


def invert(
    operator: np.ndarray,
    k_per_mpc: np.ndarray,
    data: np.ndarray,
    covariance: np.ndarray,
    *,
    amplitude_scale: float,
    gp_sigma: float,
    gp_gamma: float,
    prior_mean: float | np.ndarray,
) -> Posterior:
    """Condition an RBF Gaussian process with FrediPy.

    ``operator`` maps the dimensionless field
    ``f = primordial_power / amplitude_scale`` to the data.  FrediPy uses a
    zero-mean GP, so a nonzero mean is handled exactly by conditioning on
    ``data - operator @ mean`` and adding the mean back afterwards.
    """

    operator = np.asarray(operator, dtype=float)
    k = np.asarray(k_per_mpc, dtype=float)
    data = np.asarray(data, dtype=float)
    if operator.ndim != 2 or k.shape != (operator.shape[1],):
        raise ValueError("operator and wavenumber grid have incompatible shapes")
    if data.shape != (operator.shape[0],):
        raise ValueError("data length must match the operator rows")
    if np.any(k <= 0.0) or np.any(np.diff(k) <= 0.0):
        raise ValueError("wavenumbers must be positive and strictly increasing")
    if not np.all(np.isfinite(operator)) or not np.all(np.isfinite(data)):
        raise ValueError("operator and data must be finite")
    if amplitude_scale <= 0.0 or gp_sigma <= 0.0 or gp_gamma <= 0.0:
        raise ValueError("amplitude scale and GP hyperparameters must be positive")

    physical_mean = np.broadcast_to(np.asarray(prior_mean, dtype=float), k.shape).copy()
    if not np.all(np.isfinite(physical_mean)):
        raise ValueError("prior mean must be finite")
    normalized_mean = physical_mean / float(amplitude_scale)
    data_covariance = _covariance_array(covariance, data.size)

    coordinates = np.log(k)
    row_labels = np.arange(data.size, dtype=float)
    integrator = MatrixIntegrator(coordinates, operator)
    integral = Integral(lambda *args: None, integrator)
    constraint = LinearEquality(
        integral,
        {
            "x": row_labels,
            "y": data - operator @ normalized_mean,
            "cov_y": data_covariance,
        },
    )
    # FrediPy 0.2.1 requires built-in floats for the kernel parameters.
    model = GaussianProcess(
        RadialBasisFunction(float(gp_sigma), float(gp_gamma)),
        [constraint],
    )
    residual_mean, residual_covariance = model.predict(coordinates, full_cov=True)
    residual_mean = np.asarray(residual_mean, dtype=float).reshape(-1)
    residual_covariance = np.asarray(residual_covariance, dtype=float)
    if residual_mean.shape != k.shape or residual_covariance.shape != (k.size, k.size):
        raise RuntimeError("FrediPy returned an unexpected posterior shape")

    scale = float(amplitude_scale)
    return Posterior(
        k_per_mpc=k.copy(),
        mean=physical_mean + scale * residual_mean,
        covariance=scale**2 * residual_covariance,
        prior_mean=physical_mean,
        model=model,
        amplitude_scale=scale,
    )
