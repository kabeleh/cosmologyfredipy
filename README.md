# The primordial power spectrum as a Fredholm problem

This is a compact proof of concept for reconstructing the primordial curvature
power spectrum with [FrediPy](https://pypi.org/project/fredipy/).  It accompanies
a short, cosmologist-facing paper.  The compiled manuscript is
[paper/main.pdf](paper/main.pdf), with source in [paper/main.tex](paper/main.tex).
The repository contains only the two calculations needed for its argument:

1. a controlled reconstruction of a known power law from synthetic CLASS TT
   data; and
2. a conditional reconstruction from Planck PR3 high-\(\ell\) TT bandpowers.

This repository does not introduce a new inversion method.  It supplies CMB
operators, data vectors and covariances to the unmodified public FrediPy
interface, so the CLASS and Planck examples use the same general Fredholm
solver.

The first calculation checks that the Fredholm inversion recovers the input
spectrum.  Over the fixed evaluation interval
\(3\times10^{-4}<k<0.15\,\mathrm{Mpc}^{-1}\), its largest fractional error is
about \(7.5\times10^{-4}\).  In the Planck example, the flexible posterior mean
improves the data quadratic only from 218.9 to 218.5 relative to an optimised
power law.  Across the reported direct-sensitivity range, the two differ by at
most about 1.24% or 0.48 pointwise posterior standard deviations.

These numbers show that FrediPy can perform the reconstruction and that a
featureless primordial power law remains adequate in this worked observational
example.  They are not a feature-significance or model-selection result: the
background cosmology, lensing correction, calibration and covariance are
fixed.  The baseline Gaussian-process hyperparameters are subjected to a small
sensitivity scan, but they are not marginalised.  Factor-two variations keep
the maximum power-law separation below 2.9% and \(\Delta Q\leq1.80\); the more
permissive \(\gamma=0.5\) case gives 5.38% and \(\Delta Q=3.45\).

## Run the analysis

Python 3.10 or newer is required.  A fresh environment can be prepared with

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
```

Then reproduce the numerical summaries and all paper figures:

```bash
.venv/bin/python scripts/run_analysis.py
.venv/bin/pytest
```

The script writes `results/summary.json` and the three PDFs in
`paper/figures/`.  It conditions both examples through the public FrediPy
0.2.1 API; the tests compare its output with direct Gaussian conditioning.
The last floating-point digits in `summary.json` can depend on the BLAS
implementation: across the tested CPU kernels, the sensitivity diagnostics
agree within roughly \(5\times10^{-8}\) relative, and this variation changes
none of the values quoted in the paper.

Build the manuscript with

```bash
cd paper
latexmk -pdf main.tex
```

or use `make analysis`, `make test`, and `make paper` after installing the
package.

The manuscript uses the official JCAP `jcappub.sty` and `JHEP.bst` files,
which are kept in `paper/` so that the build is self-contained.  The generated
`paper/main.bbl` is retained because JCAP requests it in BibTeX submission
archives.

## What is stored here

The compact files in `data/` contain the sampled Fredholm operators, data
vectors and covariances needed by the paper.  They contain no cached posterior:
`run_analysis.py` reruns the inversion.  Their exact provenance, checksums and
third-party terms are recorded in [data/README.md](data/README.md).

The earlier `FredIP` experiments and the comprehensive `coreopsis` validation
repository remain separate.  In particular, this project does not use the
ambiguous 2023 `prim_k.txt` lineage, compiled CLASS binaries, vendored FrediPy,
or the large audit products.

## Terminology

The expected primordial spectrum is described here as a *power law*, or as a
straight line in \(\ln \mathcal P_{\mathcal R}\) versus \(\ln k\).  “Linear
power spectrum” is avoided because in cosmology it normally denotes a distinct
late-time matter statistic.

## Licence and citation

Original code and documentation are MIT licensed.  The Planck-derived input is
not covered by that licence; see `data/README.md` before redistributing it.
Citation metadata are provided in `CITATION.cff`.
