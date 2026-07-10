# DESI Y3 Redshift Errors

Analysis code for studying how spectroscopic redshift errors propagate into
DESI Year 3 clustering measurements and full-shape cosmological fits. The
workflow starts from repeat-observation redshift differences, turns those
measurements into redshift-error models, injects representative errors into
mock catalogs, measures two- and three-point statistics, and fits the resulting
data vectors.

This repository is written for a DESI/NERSC working environment rather than as
a portable Python package. Several scripts use hardcoded paths to collaboration
data, CFS products, and scratch outputs.

## Repository Layout

```text
.
├── README.md
├── main/
│   ├── README.md              # detailed guide to the analysis code
│   ├── helper.py              # constants, redshift bins, tracer metadata
│   ├── cat_tools.py           # catalog/statistic path builders and readers
│   ├── dv_tools.py            # repeat-observation metrics and error sampling
│   ├── fit_support.py         # fit defaults, priors, labels, config hashes
│   ├── fit_tools.py           # likelihood, covariance, and statistic loading
│   ├── mock_tools.py          # mock-catalog readers and conversion helpers
│   ├── plotting_tools.py      # plotting utilities
│   ├── jax_support.py         # distributed JAX setup and interpolation helpers
│   ├── utils.py               # logging and lightweight utilities
│   ├── repeat_obs/            # build repeat-observation redshift-error models
│   ├── mocks/                 # convert mocks and add redshift-error columns
│   ├── clustering/            # measure 2-point, power-spectrum, and bispectrum statistics
│   ├── full_shape/            # desilike full-shape likelihoods and chains
│   └── old_scripts/           # previous or reference analyses
└── overleaf/                  # paper/notes material and figures
```

## Main Workflow

1. `main/repeat_obs/` builds repeat-observation inputs, velocity-difference
   summaries, uncertainty tables, and CDF products for error sampling.
2. `main/mocks/` converts external mock products into the local catalog format
   and adds redshift-error realizations such as `repeat`, `verr_empirical`, and
   `verr_nonparam`.
3. `main/clustering/` measures configuration-space statistics with `pycorr`
   and mesh-based power spectrum or bispectrum products with `jaxpower`.
4. `main/full_shape/` constructs `desilike` likelihoods and runs likelihood
   checks, Minuit profiles, or sampler chains for `mesh2` and `mesh3`
   observables.

For the detailed directory guide, script entry points, and fit options, see
[`main/README.md`](main/README.md).

## Key Entry Points

| Stage | Scripts |
| --- | --- |
| Repeat observations | `main/repeat_obs/get_repeat_redshifts.py`, `main/repeat_obs/desi_main_repeats.py`, `main/repeat_obs/repeats_variance.py`, `main/repeat_obs/model_repeats.py` |
| Mock preparation | `main/mocks/convert_mocks.py`, `main/mocks/build_zerr_mocks.py` |
| Clustering measurements | `main/clustering/compute_2pt.py`, `main/clustering/compute_mesh_jax.py` |
| Full-shape fits | `main/full_shape/run_fits.py` |
| Slurm helpers | `main/repeat_obs/srun.sh`, `main/mocks/srun_mocks.sh`, `main/clustering/srun_stat.sh`, `main/full_shape/srun_fit.sh`, `main/full_shape/srun_QSO_test.sh` |

## Common Labels

The redshift-error labels used across catalog building, clustering, and fitting
are:

| Label | Meaning |
| --- | --- |
| `None` | No added redshift-error realization. |
| `repeat` | Draw errors from repeat-observation distributions. |
| `verr_empirical` | Draw from empirical velocity-error CDF products. |
| `verr_nonparam` | Draw from non-parametric velocity-error CDF products. |
| `*_zevol` | Use redshift-dependent bins where supported by the measurement/fitting script. |

## Environment And Dependencies

The code assumes access to DESI collaboration software, NERSC file systems, and
scratch directories referenced directly in source files and Slurm launchers.
Environment activation is handled outside the repository, for example with
`/global/homes/s/shengyu/env.sh` or shared DESI conda environments.

Common Python dependencies include `numpy`, `scipy`, `pandas`, `astropy`,
`fitsio`, `matplotlib`, `mockfactory`, `pycorr`, `pypower`, `jax`, `jaxpower`,
`cosmoprimo`, `desilike`, `lsstypes`, `mpi4py`, and DESI stack modules such as
`LSS` and `desitarget`.

## Configuration Notes

Important data locations and run conventions are defined directly in source:

| File | Configuration |
| --- | --- |
| `main/helper.py` | Tracer definitions, redshift bins, survey regions, and mock redshift sets. |
| `main/cat_tools.py` | Catalog paths, measurement filenames, error-label parsing, and catalog readers. |
| `main/dv_tools.py` | Repeat-observation products and redshift-error sampling inputs. |
| `main/fit_support.py` | Full-shape fit defaults, nuisance priors, and output naming helpers. |
| `main/fit_tools.py` | Statistic loading, covariance handling, and likelihood construction. |

