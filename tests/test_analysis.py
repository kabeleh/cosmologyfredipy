from importlib.metadata import version

import numpy as np
from fredipy.models import GaussianProcess

from cosmologyfredipy.analysis import make_summary, write_summary
from cosmologyfredipy.plotting import plot_operator, plot_planck, plot_synthetic


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


def test_summary_and_figures_are_created(tmp_path, synthetic_result, planck_result):
    summary = make_summary(synthetic_result, planck_result)
    first = write_summary(tmp_path / "summary.json", summary)
    second = write_summary(tmp_path / "summary.json", summary)
    assert first == second

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
