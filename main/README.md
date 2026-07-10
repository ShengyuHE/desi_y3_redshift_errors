# DESI Y3 Redshift-Error Analyses

This directory contains the analysis code for studying how spectroscopic
redshift errors propagate into DESI Y3 clustering measurements and
full-shape fits. The central idea is to measure redshift-difference
distributions from repeated observations, inject representative errors into
mock catalogs, measure clustering statistics, and test their impact on
cosmological inference.

The main tracers are `BGS`, `LRG`, `ELG`, and `QSO`. Redshift bins, tracer
conventions, region selections, and other common settings are defined in
`helper.py`.

## Analysis Flow

1. `repeat_obs/` extracts repeat observations and models their velocity or
   redshift-difference distributions.
2. `mocks/` converts external mock products and injects representative
   redshift-error realizations into local mock catalogs.
3. `clustering/` measures two- and three-point clustering statistics from
   the prepared catalogs.
4. `full_shape/` constructs likelihoods, profiles, and MCMC/nested-sampling
   fits of the measured statistics to quantify parameter shifts or
   constraints.

The redshift-error labels used in the clustering and fitting stages are:

| Label | Meaning |
| --- | --- |
| `None` | No added redshift-error realization. |
| `repeat` | Draw errors from a distribution based directly on repeat observations. |
| `verr_empirical` | Draw an empirical velocity-error model from stored CDF products. |
| `verr_nonparam` | Draw a non-parametric velocity-error realization. |
| `*_zevol` | Apply the corresponding model with redshift-dependent bins where supported. |

## Directory Guide

### `repeat_obs/`: Repeat-Observation Redshift Errors

This folder builds the observational input to the project. It identifies
targets observed more than once, measures velocity differences between
redshift estimates, estimates uncertainty on summary metrics, and produces
CDF-based models that can be sampled in mocks.

| File or folder | Purpose |
| --- | --- |
| `get_repeat_redshifts.py` | Select repeated DESI observations for each tracer and write paired-redshift catalogs. |
| `desi_main_repeats.py` | Alternative repeat-catalog construction pipeline, including parent/pair products and validation-redshift inputs. |
| `repeats_variance.py` | Compute bootstrap or jackknife errors on repeat-derived median scatter, RMS core scatter, and catastrophic fraction. |
| `model_repeats.py` | Build histogram or kernel-smoothed CDF products for the measured velocity-difference distributions. |
| `notebooks/` | Exploration and figures for distributions, line confusions, spectra, and error-model checks. |
| `results/` | Tables of repeat counts and derived uncertainty summaries used by later analyses. |
| `srun.sh` | NERSC/Slurm helper commands for producing repeats and repeat-error models. |

### `mocks/`: Mock Catalog Preparation

This folder prepares the mock catalogs consumed by the clustering stage. It
can convert external QSO cut-sky or lightcone ASCII mocks to the local FITS
catalog format, and it can add redshift-space coordinates and sampled
redshift-error columns to AbacusHF cubic mocks.

| File or folder | Purpose |
| --- | --- |
| `convert_mocks.py` | Convert external QSO mock data or random catalogs into the path and column conventions used by `cat_tools.py`. |
| `build_zerr_mocks.py` | Add redshift-space coordinates and sampled `repeat`, `verr_empirical`, or `verr_nonparam` redshift-error realizations to mock catalogs. |
| `srun_mocks.sh` | NERSC/Slurm helper commands for mock conversion or redshift-error injection. |
| `notebooks/` | Mock-catalog checks and exploratory validation. |

### `clustering/`: Impact on Clustering Statistics

This folder measures clustering statistics from the prepared mock catalogs. It
supports cubic Abacus/HOD catalogs and cut-sky or alternate-MTL style catalogs,
with two-point correlation measurements and JAX-based power spectrum or
bispectrum measurements.

| File or folder | Purpose |
| --- | --- |
| `compute_2pt.py` | Measure configuration-space two-point statistics, including multipoles and projected correlation functions, with `pycorr`. |
| `compute_mesh_jax.py` | Measure mesh-based power spectra (`mesh2`), bispectra (`mesh3`), and survey-window products with `jaxpower`. |
| `srun_stat.sh` | Interactive Slurm launch examples for catalog, 2-point, and mesh-statistic runs. |
| `help_scripts/copy_saved_mesh2.sh` | Utility for moving or collecting stored mesh2 results. |
| `help_scripts/copy_saved_mesh3.sh` | Utility for moving or collecting stored mesh3 results. |
| `notebooks/` | Validation and plotting notebooks for mocks, windows, 2-point measurements, power spectra, and bispectra. |
| `slurms/` | Batch submission scripts and run logs for production clustering calculations. |
| `results/` | Local output location for clustering-analysis products when used. |

The principal clustering products are `mpslog` and `wplog` from
`compute_2pt.py`, and `mesh2_spectrum_poles` or
`mesh3_spectrum_poles_<basis>` plus their window functions from
`compute_mesh_jax.py`.

### `full_shape/`: Full-Shape Cosmological Fits

This folder performs the full-shape part of the redshift-error analysis. It
uses the shared likelihood builder and `desilike` theory/sampler interfaces to
fit power-spectrum multipoles (`mesh2`) and, where supported, bispectrum
multipoles (`mesh3`) measured from the mock catalogs. Fits can compare
different redshift-error realizations, combine multiple mock IDs into one data
vector, vary covariance choices, and switch between cosmology, theory, prior,
and sampler configurations.

| File or folder | Purpose |
| --- | --- |
| `run_fits.py` | Command-line entry point for likelihood checks, Minuit profiling, and chain production for `mesh2` and `mesh3`. |
| `srun_fit.sh` | Example NERSC/Slurm launch for production alt-MTL QSO full-shape sampling. |
| `srun_QSO_test.sh` | Slurm helper for cubic QSO test fits with scaled EZmock covariance and k-range scans. |
| `slurms/submit_fits.sh` | Batch submission template for full-shape production fits. |
| `notebooks/check_covariance.ipynb` | Covariance validation and inspection. |
| `notebooks/test_desilike.ipynb` | Interactive checks of the `desilike` likelihood setup. |
| `notebooks/test_profile.ipynb` | Profiling tests and diagnostic checks. |
| `notebooks/plot_corner.ipynb` | Corner plots for sampled chains. |
| `notebooks/plot_chain_bar.ipynb` | Compact comparison plots for chain-derived parameter shifts or constraints. |

Common `run_fits.py` options include:

| Option | Meaning |
| --- | --- |
| `--todos test profile sample` | Build/check the likelihood, run a Minuit profile, or produce sampler chains. |
| `--stats mesh2 mesh3` | Select the power-spectrum and/or bispectrum observables to fit. |
| `--kmax 0.35-0.25-0.10` | Set the maximum fitted wavenumber for `mesh2` monopole, `mesh2` quadrupole, and `mesh3` monopole. |
| `--theory_model` | Choose among `folpsD`, `folpsEFT`, and `reptvelocileptors`; `reptvelocileptors` is implemented only for `mesh2`. |
| `--cosmo_params` | Use `base`, `base_ns-fixed`, or `fixed` cosmological-parameter freedom. |
| `--prior_basis` | Select the nuisance-prior basis, such as `physical_aap`. |
| `--sampler` | Choose `emcee`, `mcmc`, `nautilus`, or `pocomc`. |
| `--resume` | Resume sampling from existing saved chain files. |

### `old_scripts/`: Previous or Reference Analyses

This folder preserves earlier implementations and one-off studies, including
QSO matching, earlier redshift-systematics modeling, and legacy cut-sky
clustering comparisons. These scripts are useful for provenance and
cross-checks; current workflows should generally use the folders above.

## Shared Modules

The modules directly in `main/` provide the functions shared between analysis
stages:

| Module | Main responsibility |
| --- | --- |
| `helper.py` | Constants, tracer/redshift-bin definitions, mock settings, coordinate conversion, and survey-region selection. |
| `dv_tools.py` | Repeat-observation velocity-difference metrics, CDF sampling, and parametric/non-parametric redshift-error modeling tools. |
| `cat_tools.py` | Catalog paths, error-label parsing, mock or survey catalog reading, weights, positions, regions, and measurement filenames. |
| `mock_tools.py` | Readers and conversion helpers for ASCII mock products. |
| `fit_support.py` | Default fit options, nuisance priors, compact output labels, and stable configuration hashes. |
| `fit_tools.py` | Statistic/window loading, covariance corrections, `LikelihoodBuilder`, and cosmology/theory/likelihood construction. |
| `plotting_tools.py` | Plot styling and helpers for repeat-observation diagnostics and clustering-fit displays. |
| `jax_support.py` | Distributed JAX initialization and interpolation support for GPU mesh calculations. |
| `utils.py` | Lightweight formatting and logging utilities. |

## Inputs And Outputs

Several scripts currently refer to DESI/NERSC file systems and user scratch
areas directly, for example repeat catalogs under `/pscratch` and DESI survey
catalogs under `/global/cfs/cdirs/desi`. The shell launchers also source
site-specific environments. Before running on a different system or data
release, check the paths in `dv_tools.py`, `cat_tools.py`, the analysis
scripts, and the `srun*.sh` helpers.

Generated analysis products are generally kept outside the repository on
scratch storage. Small summary tables, validation products, notebooks, and
batch logs are retained inside the relevant analysis folder for inspection.
