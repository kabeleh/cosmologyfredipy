import numpy as np

from cosmologyfredipy.inversion import invert


def test_fredipy_matches_dense_gaussian_conditioning_with_full_covariance():
    k = np.array([0.01, 0.02, 0.05, 0.10])
    operator = np.array(
        [
            [0.8, 0.3, 0.1, 0.0],
            [0.1, 0.7, 0.4, 0.1],
            [0.0, 0.2, 0.6, 0.9],
        ]
    )
    covariance = np.array(
        [
            [0.040, 0.008, 0.002],
            [0.008, 0.030, 0.006],
            [0.002, 0.006, 0.050],
        ]
    )
    scale = 2.0
    prior_mean = np.array([1.8, 2.0, 2.2, 2.4])
    data = operator @ np.array([0.95, 1.04, 1.08, 1.16]) + np.array(
        [0.02, -0.01, 0.015]
    )
    gp_sigma = 0.7
    gp_gamma = 0.9

    posterior = invert(
        operator,
        k,
        data,
        covariance,
        amplitude_scale=scale,
        gp_sigma=gp_sigma,
        gp_gamma=gp_gamma,
        prior_mean=prior_mean,
    )

    coordinates = np.log(k)
    separation = coordinates[:, None] - coordinates[None, :]
    kernel = gp_sigma**2 * np.exp(-0.5 * (separation / gp_gamma) ** 2)
    normalized_mean = prior_mean / scale
    data_kernel = operator @ kernel @ operator.T + covariance
    gain = np.linalg.solve(data_kernel, operator @ kernel).T
    expected_mean = prior_mean + scale * gain @ (data - operator @ normalized_mean)
    expected_covariance = scale**2 * (kernel - gain @ operator @ kernel)

    assert np.allclose(posterior.mean, expected_mean, rtol=2.0e-12, atol=2.0e-12)
    assert np.allclose(
        posterior.covariance,
        expected_covariance,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
