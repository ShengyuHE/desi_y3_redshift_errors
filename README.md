# DESI Y3 Redshift Errors

Analysis code for studying DESI Year 3 spectroscopic redshift errors, including repeat-observation based error modeling, redshift-error injection into mock catalogs, and clustering measurements in configuration space and Fourier space.

## What This Repository Does

This project brings together three connected pieces of the redshift-error pipeline:

1. Measure repeat-observation statistics from DESI spectroscopy and build empirical or nonparametric models for redshift uncertainties and catastrophic failures.
2. Inject those redshift-error prescriptions into mock catalogs such as Abacus high-fidelity boxes and survey-like samples.
3. Quantify the impact on clustering observables, including 2-point functions, power spectrum multipoles, and JAX-based mesh estimators for 2-point and 3-point statistics.

The code is written for a DESI/NERSC working environment rather than as a portable Python package, so several scripts use hardcoded paths to collaboration data and scratch outputs.

## Repository Layout

```text
.
├── README.md
└── main/
    ├── helper.py              # constants, redshift bins, tracer metadata
    ├── cat_tools.py           # catalog I/O, weighting, path builders
    ├── dv_tools.py            # repeat-observation utilities and sampling
    ├── fitting_tools.py       # loading, fitting, covariance helpers
    ├── plotting_tools.py      # plotting utilities
    ├── clustering/
    │   ├── build_catalogs.py  # inject redshift errors into catalogs
    │   ├── compute_2pt.py     # pycorr/pypower-based xi and P(k)
    │   ├── compute_mesh_jax.py# jaxpower-based mesh 2pt/3pt estimators
    │   ├── srun.sh            # interactive launch helper
    │   ├── submit_mesh.sh     # batch job template for mesh runs
    │   └── notebooks/         # tests, checks, and plotting notebooks
    ├── repeat_obs/
    │   ├── desi_main_repeats.py
    │   ├── get_repeat_redshifts.py
    │   ├── model_repeats.py
    │   ├── repeats_variance.py
    │   ├── notebooks/
    │   └── results/
    └── old_scripts/           # exploratory or superseded analysis scripts
```

## Main Workflow

### 1. Build repeat-observation redshift-error models

The `main/repeat_obs/` scripts extract repeat spectra, compute `delta v` distributions, and save histogram- or kernel-based CDF models that can later be sampled when contaminating mocks.

Key entry points:

- `main/repeat_obs/desi_main_repeats.py`: assemble repeat-observation parent samples from DESI spectroscopy products.
- `main/repeat_obs/get_repeat_redshifts.py`: construct repeat redshift pair tables.
- `main/repeat_obs/model_repeats.py`: save empirical / kernel CDF models for each tracer and redshift bin.

Typical example:

```bash
python main/repeat_obs/model_repeats.py \
  --tracers LRG ELG QSO \
  --ztypes LSS \
  --bin_mode log_signed \
  --cdf_mode both
```

### 2. Inject redshift errors into catalogs

`main/clustering/build_catalogs.py` augments mock catalogs with redshift-space positions and alternative redshift-error prescriptions. Supported labels in the current scripts include:

- `None`
- `repeat`
- `verr_empirical`
- `verr_nonparam`

Example:

```bash
python main/clustering/build_catalogs.py \
  --version AbacusHF-v2 \
  --domains cubic \
  --tracers LRG \
  --zerrs repeat verr_empirical verr_nonparam \
  --mockid 0-24
```

### 3. Measure clustering statistics

There are two main measurement paths.

`main/clustering/compute_2pt.py`
- Uses `pycorr` and `pypower`.
- Computes correlation-function multipoles and FFT power-spectrum multipoles.
- Best suited for standard 2-point analyses.

Example:

```bash
srun -N 1 -n 4 -C gpu -t 04:00:00 --gpus 4 python main/clustering/compute_2pt.py \
  --version AbacusHF-v2 \
  --domains cubic \
  --tracers LRG \
  --zerrs verr_nonparam \
  --mockid 0-24
```

`main/clustering/compute_mesh_jax.py`
- Uses `jaxpower` plus MPI-distributed JAX.
- Computes mesh-based 2-point and 3-point spectra, and optional window terms.
- Supports `cubic`, `cutsky`, and `altmtl` domains.

Important runtime switches:

- `--zerrs`: `None`, `repeat`, `repeat_zevol`, `verr_empirical`, `verr_nonparam`, `verr_nonparam_zevol`
- `--todos`: `mesh2`, `mesh2_window`, `mesh3_scoccimarro`, `mesh3_sugiyama`, `mesh3_scoccimarro_window`, `mesh3_sugiyama_window`
- `--regions`: relevant for cutsky or altmtl runs

Example:

```bash
srun -N 1 -n 4 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 python main/clustering/compute_mesh_jax.py \
  --version holi-v3 \
  --domain altmtl \
  --tracers LRG ELG QSO \
  --mockid 0-99 \
  --zerrs None verr_nonparam \
  --todos mesh2
```

## Convenience Launchers

For interactive runs on Perlmutter, use:

```bash
bash main/clustering/srun.sh cat
bash main/clustering/srun.sh 2pt
bash main/clustering/srun.sh mesh
```

For batch mesh jobs, start from:

```bash
sbatch main/clustering/submit_mesh.sh
```

## Environment And Dependencies

This repository assumes access to DESI collaboration software, NERSC filesystems, and scratch directories referenced directly in the code. It is not configured as a standalone installable package.

Common Python dependencies used across the repo include:

- `numpy`
- `scipy`
- `pandas`
- `astropy`
- `fitsio`
- `matplotlib`
- `mockfactory`
- `pycorr`
- `pypower`
- `jax`
- `jaxpower`
- `cosmoprimo`
- `desilike`
- `lsstypes`
- DESI stack modules such as `LSS`, `desitarget`, and related survey tooling

Environment activation is currently handled outside the repo, for example through site-specific scripts such as `/global/homes/s/shengyu/env.sh` or shared DESI conda environments referenced in `main/clustering/srun.sh`.

## Configuration Notes

Several important paths are defined directly in source files:

- catalog and scratch locations in `main/cat_tools.py`
- repeat-observation products in `main/dv_tools.py`
- tracer and redshift-bin definitions in `main/helper.py`

If you are adapting this analysis to a new machine, survey release, or directory structure, these modules are the first places to update.

## Outputs

Outputs are generally written outside the repository tree to scratch or collaboration storage, while lightweight summary products and notebooks live inside the repo. In particular:

- repeat summary tables are under `main/repeat_obs/results/`
- notebooks for validation and plotting are under `main/repeat_obs/notebooks/` and `main/clustering/notebooks/`
- large clustering measurements are written to paths constructed by `get_measurement_fn()` in `main/cat_tools.py`

## Current Caveats

- The project is tightly coupled to DESI data products and NERSC paths.
- Some scripts support more domains or options than are fully production-ready.
- `main/README.md` is currently just a placeholder.
- The repository contains exploratory notebooks and older scripts alongside the main pipeline.

## Suggested First Steps

If you are returning to this project after some time, a good order is:

1. Read `main/helper.py` and `main/cat_tools.py` to understand the tracer, redshift-bin, and file-layout conventions.
2. Inspect `main/repeat_obs/model_repeats.py` if you are working on the redshift-error model itself.
3. Use `main/clustering/compute_2pt.py` for conventional 2-point tests.
4. Use `main/clustering/compute_mesh_jax.py` for production mesh-based 2-point or 3-point measurements.
