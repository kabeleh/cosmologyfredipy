from importlib.metadata import version
from pathlib import Path

import numpy as np
from fredipy.kernels import RadialBasisFunction
from fredipy.models import GaussianProcess

from cosmologyfredipy.analysis import (
    GP_SENSITIVITY_SETTINGS,
    analyse_planck,
    make_summary,
    write_summary,
)
from cosmologyfredipy.plotting import plot_operator, plot_planck, plot_synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_both_reconstructions_use_public_fredipy(synthetic_result, planck_result):
    assert version("fredipy") == "0.2.1"
    assert isinstance(synthetic_result.posterior.model, GaussianProcess)
    assert isinstance(planck_result.posterior.model, GaussianProcess)
    assert np.allclose(
        planck_result.posterior.model.constraints[0].cov_y,
        planck_result.covariance,
    )


def test_synthetic_class_recovery_is_better_than_one_per_mille(synthetic_result):
    assert synthetic_result.max_kernel_closure_fraction < 1.0e-3
    assert synthetic_result.max_interior_fractional_error < 1.0e-3
    assert synthetic_result.rms_interior_fractional_error < 5.0e-4


def test_planck_power_law_is_close_to_flexible_reconstruction(planck_result):
    assert planck_result.q_flexible < planck_result.power_law.q
    assert planck_result.delta_q < 1.0
    assert planck_result.max_power_law_fractional_difference < 0.013
    assert planck_result.rms_power_law_fractional_difference < 0.004
    assert planck_result.max_power_law_difference_sigma <= 0.5


def test_gp_hyperparameter_sensitivity(planck_result, planck_sensitivity):
    settings = tuple((row.case, row.sigma_f, row.gamma) for row in planck_sensitivity)
    assert settings == GP_SENSITIVITY_SETTINGS

    baseline = planck_sensitivity[0]
    assert baseline.q_flexible == planck_result.q_flexible
    assert (
        baseline.max_power_law_fractional_difference
        == planck_result.max_power_law_fractional_difference
    )

    measured = np.array(
        [
            (
                row.q_flexible,
                row.delta_q_power_law_minus_flexible,
                row.max_power_law_fractional_difference,
                row.max_power_law_difference_sigma,
            )
            for row in planck_sensitivity
        ]
    )
    expected = np.array(
        [
            [
                218.51071345521,
                0.39215663792649025,
                0.012396814772795972,
                0.4841369148231625,
            ],
            [
                218.55507273209474,
                0.3477973610417564,
                0.006251784718123821,
                0.4418112927070447,
            ],
            [
                218.49139616731034,
                0.41147392582615794,
                0.01477051514973926,
                0.48562231291546415,
            ],
            [
                217.10410995925776,
                1.7987601338787442,
                0.02876819288116539,
                0.9327436602187189,
            ],
            [
                218.68478137034495,
                0.21808872279154912,
                0.0038736504958100104,
                0.3073854709693377,
            ],
            [
                215.4515123325657,
                3.451357760570801,
                0.05381621084895194,
                1.3401155466857004,
            ],
        ]
    )
    np.testing.assert_allclose(measured, expected, rtol=1.0e-8, atol=1.0e-10)

    rows = {row.case: row for row in planck_sensitivity}
    assert rows["gamma_half"].max_power_law_fractional_difference < 0.03
    assert rows["gamma_half"].max_power_law_difference_sigma < 1.0
    assert rows["gamma_aggressive"].max_power_law_fractional_difference < 0.055
    assert rows["gamma_aggressive"].max_power_law_difference_sigma < 1.35
    assert rows["gamma_aggressive"].delta_q_power_law_minus_flexible > 3.4


def test_gamma_half_matches_dense_planck_conditioning(planck_result):
    gamma = 0.75
    result = analyse_planck(
        ROOT / "data" / "planck_pr3_tt.npz", gp_sigma=1.0, gp_gamma=gamma
    )
    coordinates = np.log(result.posterior.k_per_mpc)
    separation = coordinates[:, None] - coordinates[None, :]
    kernel = np.exp(-0.5 * (separation / gamma) ** 2)
    public_kernel = RadialBasisFunction(2.0, gamma)(
        coordinates[:, None], coordinates[:, None], cache=False
    )
    assert np.allclose(public_kernel, 4.0 * kernel)

    operator = result.operator
    data_kernel = operator @ kernel @ operator.T + result.covariance
    residual = result.conditional_bandpowers - operator @ np.ones(coordinates.size)
    alpha = np.linalg.solve(data_kernel, residual)
    dense_mean = result.posterior.amplitude_scale * (
        np.ones(coordinates.size) + kernel @ operator.T @ alpha
    )
    np.testing.assert_allclose(
        result.posterior.mean,
        dense_mean,
        rtol=1.0e-10,
        atol=1.0e-20,
    )

    data_residual = result.conditional_bandpowers - operator @ (
        dense_mean / result.posterior.amplitude_scale
    )
    dense_q = float(data_residual @ np.linalg.solve(result.covariance, data_residual))
    np.testing.assert_allclose(dense_q, result.q_flexible, rtol=1.0e-10, atol=1.0e-10)
    dense_fraction = dense_mean / planck_result.power_law.spectrum - 1.0
    assert np.max(np.abs(dense_fraction[planck_result.sensitivity_mask])) > 0.028


def test_summary_and_figures_are_created(
    tmp_path, synthetic_result, planck_result, planck_sensitivity
):
    summary = make_summary(synthetic_result, planck_result, planck_sensitivity)
    first = write_summary(tmp_path / "summary.json", summary)
    second = write_summary(tmp_path / "summary.json", summary)
    assert first == second
    assert len(summary["planck"]["gp_hyperparameter_sensitivity"]) == 6

    figures = {
        "fredholm_operator.pdf": lambda path: plot_operator(synthetic_result, path),
        "synthetic_reconstruction.pdf": lambda path: plot_synthetic(
            synthetic_result, path
        ),
        "planck_reconstruction.pdf": lambda path: plot_planck(planck_result, path),
    }
    for name, render in figures.items():
        path = tmp_path / name
        render(path)
        assert path.stat().st_size > 1_000
