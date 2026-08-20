#!/usr/bin/env python3
"""Verify the two compact, provenance-documented analysis inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    Path("data/synthetic_class.npz"): (
        "c287fcb4eb68531a805b4526bee3041b088533ba5e536a0c4c11229600f0f06f"
    ),
    Path("data/planck_pr3_tt.npz"): (
        "abaf020efa6d1e6a71c2ee4cb1759af3619d7f363fb2ae86264b9437d963ef54"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing input: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"checksum mismatch for {relative}: {actual} != {expected}"
            )
        print(f"{relative}: OK ({actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
