"""Three deliberately simple, colour-blind-friendly paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import PlanckAnalysis, SyntheticAnalysis

# Keep text editable and avoid Type 3 bitmap fonts in journal PDFs.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
BLACK = "#202020"


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_operator(result: SyntheticAnalysis, path: str | Path) -> None:
    """Plot the scale and multipole localisation of the Fredholm operator."""

    response = np.abs(result.operator)
    row_peak = np.max(response, axis=1, keepdims=True)
    response = np.divide(
        response, row_peak, out=np.zeros_like(response), where=row_peak > 0.0
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    image = ax.pcolormesh(
        result.posterior.k_per_mpc,
        result.multipoles,
        response,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"wavenumber $k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"multipole $\ell$")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"row-normalised $|W_{\ell j}|$")
    _save(fig, path)


def plot_synthetic(result: SyntheticAnalysis, path: str | Path) -> None:
    """Plot the injected CLASS spectrum and its FrediPy reconstruction."""

    k = result.posterior.k_per_mpc
    mean = result.posterior.mean
    sigma = result.posterior.sigma
    residual = mean / result.truth - 1.0
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.2),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    top.fill_between(
        k, 1.0e9 * (mean - sigma), 1.0e9 * (mean + sigma), color=BLUE, alpha=0.22
    )
    top.plot(k, 1.0e9 * mean, color=BLUE, lw=1.8, label="FrediPy posterior")
    top.plot(k, 1.0e9 * result.truth, color=BLACK, lw=1.5, ls="--", label="CLASS input")
    top.set_xscale("log")
    top.tick_params(labelbottom=False)
    top.set_ylabel(r"$10^9\mathcal{P}_{\mathcal{R}}(k)$")
    top.legend(frameon=False)
    top.grid(alpha=0.2)

    bottom.axhline(0.0, color=BLACK, lw=0.8)
    mask = result.evaluation_mask
    bottom.plot(k[mask], residual[mask], color=BLUE, lw=1.4)
    bottom.set_xlim(k[mask][0], k[mask][-1])
    bottom.set_xscale("log")
    bottom.set_xlabel(r"wavenumber $k$ [Mpc$^{-1}$]")
    bottom.set_ylabel("fractional\nresidual")
    bottom.grid(alpha=0.2)
    _save(fig, path)


def plot_planck(result: PlanckAnalysis, path: str | Path) -> None:
    """Plot the Planck reconstruction and band-power residuals."""

    k = result.posterior.k_per_mpc
    mean = result.posterior.mean
    sigma = result.posterior.sigma
    mask = result.sensitivity_mask
    a0 = result.posterior.amplitude_scale
    flexible_bands = result.operator @ (mean / a0) + result.lensing_template_bandpowers
    power_law_bands = (
        result.operator @ (result.power_law.spectrum / a0)
        + result.lensing_template_bandpowers
    )
    flexible_residual = (
        result.observed_bandpowers - flexible_bands
    ) / result.reported_sigma
    power_law_residual = (
        result.observed_bandpowers - power_law_bands
    ) / result.reported_sigma

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.4),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.28},
    )
    top.fill_between(
        k, 1.0e9 * (mean - sigma), 1.0e9 * (mean + sigma), color=BLUE, alpha=0.22
    )
    top.plot(k, 1.0e9 * mean, color=BLUE, lw=1.8, label="FrediPy posterior")
    top.plot(
        k,
        1.0e9 * result.power_law.spectrum,
        color=ORANGE,
        lw=1.6,
        ls="--",
        label="best-fit power law",
    )
    top.axvspan(k[0], k[mask][0], color="0.75", alpha=0.18, lw=0)
    top.axvspan(k[mask][-1], k[-1], color="0.75", alpha=0.18, lw=0)
    top.set_xscale("log")
    top.set_xlabel(r"wavenumber $k$ [Mpc$^{-1}$]")
    top.set_ylabel(r"$10^9\mathcal{P}_{\mathcal{R}}(k)$")
    top.legend(frameon=False)
    top.grid(alpha=0.2)

    bottom.axhline(0.0, color=BLACK, lw=0.8)
    bottom.scatter(
        result.ell_eff,
        flexible_residual,
        color=BLUE,
        s=11,
        marker="o",
        label="flexible",
    )
    bottom.scatter(
        result.ell_eff,
        power_law_residual,
        facecolors="none",
        edgecolors=ORANGE,
        s=17,
        marker="s",
        label="power law",
    )
    bottom.set_xlabel(r"effective multipole $\ell_{\rm eff}$")
    bottom.set_ylabel("residual /\n" + r"reported $\sigma$")
    bottom.legend(frameon=False, ncol=2)
    bottom.grid(alpha=0.2)
    _save(fig, path)
