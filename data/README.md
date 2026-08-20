# Compact paper inputs

These two files are inputs, not cached inversion results.  They store the
operators and measurements required to rerun the two FrediPy conditionings in
the paper.  `tools/import_reference_data.py` records the extraction and rejects
unexpected source artifacts by SHA-256.

## `synthetic_class.npz`

- SHA-256: `c287fcb4eb68531a805b4526bee3041b088533ba5e536a0c4c11229600f0f06f`
- Size: about 244 kB.
- Source artifact: `coreopsis/paper/artifacts/fredipy-synthetic-v2/arrays.npz`.
- Source SHA-256: `eaffd50c40c832a74f5407a30ec87096fd4b18f4197b350afdb70eb75b4c3183`.
- Producer: clean `coreopsis` commit
  `eec88e62ceab7f3d79b44d1925b07773923b6aba`.
- Physical setup: CLASS 3.3.4, unlensed scalar TT, 125 sampled multipoles
  between 2 and 2000, and 240 nodes in physical wavenumber
  \(1.05\times10^{-5}<k<0.455\,\mathrm{Mpc}^{-1}\).
- Input spectrum: \(A_s=2.1\times10^{-9}\), \(n_s=0.9649\), pivot
  \(k_*=0.05\,\mathrm{Mpc}^{-1}\).
- Statistical convention: the target is deterministic CLASS output; independent
  standard deviations \(\sigma_\ell=0.01D_\ell\), hence variances
  \((0.01D_\ell)^2\), regularise an illustrative closure test and are not an
  observing model.

The active fixture was produced with CLASS, whose source code is distributed
under the GNU General Public License.  This compact file contains numerical
outputs, not CLASS source or binaries.  Cite CLASS and the cosmological setup
given in the paper when reusing it.

## `planck_pr3_tt.npz`

- SHA-256: `abaf020efa6d1e6a71c2ee4cb1759af3619d7f363fb2ae86264b9437d963ef54`.
- Size: about 744 kB.
- Source artifact: `coreopsis/paper/artifacts/planck-pr3-conditional-v3/arrays.npz`.
- Source SHA-256: `9708a48308994f2c7fe5808e1f50256c8dbad9c1f17ee59d2a617e645000e3c0`.
- Producer: the same clean `coreopsis` commit as above.
- Observations: the first 215 Planck PR3 `plik_lite` TT bins
  (\(30\leq\ell\leq2508\)) and their full covariance.  The exact published
  binning windows were applied upstream and are encoded in `band_operator`.
- Theory input: an unlensed CLASS response at fixed cosmology plus a fixed
  fiducial lensing template.  The calibration is fixed to unity in this small
  example.  The principal fixed values are \(h=0.6736\),
  \(\omega_b=0.02237\), \(\omega_{\rm cdm}=0.1200\),
  \(\tau_{\rm reio}=0.0544\), \(N_{\rm ncdm}=0\) and \(N_{\rm ur}=3.044\),
  with flat scalar adiabatic initial conditions.

The original likelihood archive is the Planck 2018 legacy likelihood release,
`COM_Likelihood_Data-baseline_R3.00.tar.gz`, SHA-256
`0b73171e3acc671c28184466a45485a2d1c1d93676b832abdfe688c7b04024e6`.
It is available from the ESA Planck Legacy Archive under the terms associated
with the [Planck PR3 dataset](https://doi.org/10.5270/esa-gb3sw1a).  ESA and the
Planck Collaboration must be credited.  This Planck-derived NPZ is not covered
by the repository's MIT licence.  Users intending to redistribute it should
verify that their use complies with the current ESA/Planck terms; the fully
conservative alternative is to obtain the official archive and regenerate the
derived input.

## Re-extract from the audited local source

If the sibling `coreopsis` checkout is available, reproduce both compact files
with

```bash
python tools/import_reference_data.py --coreopsis ../coreopsis --output data
```

The importer deliberately refuses the historical FredIP arrays.  A later
byte-level audit found that their primordial-wavenumber transformation cannot
be reconstructed consistently, so they are not admissible fixtures here.
