# Matched-Mode Testing

Code for "Matched-Mode Testing: Two-Sample Inference for the Locations of Density Modes".

## Layout

```
modal_transport.py                  estimator and tests (mode search, covariance, tolerance and exact-equality tests)
mt_tools.py                         comparison tests, simulation families, Monte Carlo runner, figure style
simulation/run_*.py                 simulation studies (Section 6.1 and Appendix D)
application_mmochi_cd14/            CITE-seq landmark registration (Section 6.2)
application_zircon_provenance/      detrital-zircon provenance (Appendix)
application_afhq_diffusion_final/   pretrained diffusion model on AFHQv2 (Section 6.3); see its README
results/                            tables written by the scripts
Plots/                              figures used in the paper
```

## Requirements

Python 3.10 or later with numpy, scipy, pandas, matplotlib, joblib, and hyppo
(`pip install numpy scipy pandas matplotlib joblib hyppo`). The CITE-seq application also
reads 10x HDF5 files (`pip install h5py`). The diffusion application has its own
requirements, listed in `application_afhq_diffusion_final/README.md`.

## Running

Run every script from the repository root, for example

```
python simulation/run_prevalence.py
python application_mmochi_cd14/run_mmochi_cd14.py
python application_zircon_provenance/run_zircon_provenance.py
```

Each script writes its tables to `results/<script name>/` and its figures to
`Plots/<script name>/`. Monte Carlo replicates use fixed seeds
(`numpy.random.SeedSequence`), so results do not depend on the number of worker
processes; completed settings are cached in `results/<script name>/raw/` and
recomputed only if the code or settings change.

## Data

* CITE-seq: the public 10x PBMC protein matrices are included in `application_mmochi_cd14/`.
* Detrital zircon: the public supplementary ages are included in `application_zircon_provenance/`.
* AFHQv2 and the EDM checkpoints are downloaded automatically by the diffusion pipeline.