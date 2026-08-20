from pathlib import Path

import pytest

from cosmologyfredipy.analysis import analyse_planck, analyse_synthetic


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_result():
    return analyse_synthetic(ROOT / "data" / "synthetic_class.npz")


@pytest.fixture(scope="session")
def planck_result():
    return analyse_planck(ROOT / "data" / "planck_pr3_tt.npz")
