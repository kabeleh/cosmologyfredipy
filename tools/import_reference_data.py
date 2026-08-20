#!/usr/bin/env python3
"""Extract the two compact paper inputs from the maintained COREOPSIS artifacts.

The generated files contain the physical operators and data vectors needed to
rerun the FrediPy conditioning.  They deliberately omit cached posteriors,
environment attestations, and the many diagnostics of the larger audit project.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


SYNTHETIC_SHA256 = "eaffd50c40c832a74f5407a30ec87096fd4b18f4197b350afdb70eb75b4c3183"
PLANCK_SHA256 = "9708a48308994f2c7fe5808e1f50256c8dbad9c1f17ee59d2a617e645000e3c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Unexpected SHA-256 for {path}: {actual}")


def extract(coreopsis: Path, output: Path) -> None:
    synthetic_source = coreopsis / "paper/artifacts/fredipy-synthetic-v2/arrays.npz"
    planck_source = coreopsis / "paper/artifacts/planck-pr3-conditional-v3/arrays.npz"
    require_hash(synthetic_source, SYNTHETIC_SHA256)
    require_hash(planck_source, PLANCK_SHA256)

    output.mkdir(parents=True, exist_ok=True)
    with np.load(synthetic_source, allow_pickle=False) as source:
        np.savez_compressed(
            output / "synthetic_class.npz",
            a0=np.array(2.1e-9),
            gp_sigma=np.array(1.0),
            gp_gamma=np.array(1.5),
            k_per_mpc=source["k_coarse_1_per_mpc"],
            multipoles=source["multipoles"],
            operator=source["operator_dk_2e_5"],
            data_d_ell=source["data_D_ell"],
            data_variance=source["data_variance"],
            primordial_truth=source["primordial_truth"],
            evaluation_mask=source["evaluation_mask"],
            raw_cl_tt=source["raw_cl_tt"],
            direct_cl_tt=source["direct_cl_tt_dk_2e_5"],
        )

    with np.load(planck_source, allow_pickle=False) as source:
        np.savez_compressed(
            output / "planck_pr3_tt.npz",
            a0=np.array(2.1e-9),
            gp_sigma=np.array(1.0),
            gp_gamma=np.array(1.5),
            pivot_per_mpc=np.array(0.05),
            k_per_mpc=source["k_per_mpc"],
            band_operator=source["band_operator"],
            observed_bandpowers=source["observed_bandpowers"],
            band_covariance=source["band_covariance"],
            ell_eff=source["ell_eff"],
            lensing_template_bandpowers=source["lensing_template_bandpowers"],
            fiducial_primordial_power=source["fiducial_primordial_power"],
            reported_band_sigma=source["reported_band_sigma"],
        )

    for name in ("synthetic_class.npz", "planck_pr3_tt.npz"):
        path = output / name
        print(f"{sha256(path)}  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coreopsis",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "coreopsis",
        help="path to the maintained COREOPSIS checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="directory for the two compact NPZ files",
    )
    args = parser.parse_args()
    extract(args.coreopsis.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
