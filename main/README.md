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
| `compute_cross_jax.py` | Measure two-tracer cross-power (`cross2`) and its survey window (`cross2_window`) using each tracer's random catalog. |
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

Use `compute_mesh_jax.py --output-dir PATH` to save validation measurements
under a separate root. Regional outputs and GCcomb inputs use the same root.

`clustering/notebooks/plot_clustering.ipynb` loads its mesh-statistic error bars from saved
`lsstypes.CovarianceMatrix` HDF5 files in
`/global/cfs/cdirs/desi/users/shengyu/galaxies/catalogs/Y3/holi-v3/altmtl/cov`.
Plotting reads the covariance directly; missing files raise without rebuilding mocks.
Filenames follow the measurements, for example
`covariance_mesh2_spectrum_poles_QSO_z0.8-2.1_holi_v3.h5` and
`covariance_mesh2_spectrum_poles_QSO_z0.8-2.1_holi_v3+dv_verr_nonparam.h5`.
The covariance honors `cov_args['use_dv']` (False by default).
For mesh statistics, `get_error_bars(cov_args, corr, data, ells=...)` takes the
native observable from `read_data_from_fn(..., return_observable=True)` and uses
`covariance.at.observable.match(data)` directly. The caches preserve the native
bin edges, weights, and full bispectrum triangle coordinates, allowing scale
cuts, multipole selection and supported coarser rebinning of the full covariance.
The `mpslog` plot instead reads the 100 usable clean holi `.npy` measurements
among mock IDs 0--119 directly, once per tracer/bin. It applies the same
separation cuts as the data and computes `std(axis=0, ddof=1)`, bypassing
lsstypes covariance matching. Attributes of the mesh caches record the mock IDs
actually read and any missing mocks. Errors remain the
single-realization scatter, `sqrt(diag(covariance))`, without division by sqrt(N).
Use the separate builder with `--rebuild` after replacing or adding measurements.
The builder validates metadata (binning and mock selection) when reusing files.
To precompute power and bispectrum plot caches for all standard bins (clean-only
for BGS/ELG; clean and `verr_nonparam` for LRG/QSO), using `region='ALL'`, run
`python main/clustering/tests/cache_plot_covariances.py` from the repository root in
`fit_env` on a compute node. MPI ranks distribute independent covariances.
Use `--tracers` and `--zerrs` to select subsets. Missing measurement
sets are reported and do not produce covariance files. ELG has no `ALL`-region
bispectrum measurements, so only its clean power-spectrum covariance is saved.

The builder also includes `mpslog` for LRG/QSO, reading the `.npy` measurements
and saving `covariance_mpslog_<tracer>_<zrange>_holi_v3[+dv_verr_nonparam].h5`.
It defaults to mock IDs 0--119 for this statistic (1000 for the mesh statistics),
with the same mock exclusion list. There are 102 clean measurements in each
standard LRG/QSO bin and 82 smeared measurements for LRG1; the other smeared
sets are unavailable. After the exclusion list, the caches use 100 clean mocks
per bin and 82 smeared LRG1 mocks. The small-scale plot cuts remove the first 11 separation
bins for LRG and 16 for QSO, excluding undefined small-separation QSO values.
Use `--tracers LRG1 LRG2 LRG3 QSO1 --stats mpslog` to build only these caches.

For cross spectra and windows, run the following in the clustering environment
and a suitable Slurm allocation:

```bash
python main/clustering/compute_cross_jax.py --version AbacusHF-v2 \
    --domain altmtl --tracers LRGxELG --mockid 0-24 --zerrs None verr_nonparam \
    --regions NGC SGC GCcomb --todos cross2 cross2_window
```

Each `--zerrs` value requests a separate run with that setting applied to both
tracers. Thus `--zerrs None verr_nonparam` computes None × None and
verr_nonparam × verr_nonparam; mixed error settings are not supported.

`cross2_window` reads the matching saved cross spectrum's mesh geometry,
output bins, normalization, and both data weight sums. If the spectrum is
missing, it computes and saves it first. For altmtl catalogs, spectra and windows
use `NRAN_Y3` from `helper.py` by default (LRG: 10, ELG: 12 random files).
The reference `clustering_statistics/tools.py` uses 15 for ELG; this project retains 12.
Use `--nran N1 N2` to override these counts in tracer order. Each independent random catalog is
normalized to its own tracer's data weight sum, preserving tracer order for
the first-point line of sight. The window uses the same two box scales and
FFTLog construction as the auto-spectrum window. Window files have the prefix
`window_mesh2_spectrum_poles_` in the cross measurement directory.

Use `--todos cross2_window` to compute only windows, and `--overwrite` to
replace existing requested products. A window-only run keeps the saved
spectrum, including its geometry. GCcomb combines the NGC and SGC windows;
regenerate it with `--overwrite` after updating the regional windows.
`cross2_window` requires a survey domain and is unavailable for periodic boxes.
`--dry-run` prints the requested output paths without computing measurements.

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
| `srun_fit.sh` | NERSC/Slurm launcher for LRG1/2/3 and QSO1 likelihood checks, profiling, and PocoMC sampling. |
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
| `--todos build test profile sample` | Train/cache the emulator, check the likelihood, run a Minuit profile, or produce sampler chains. |
| `--stats mesh2 mesh3` | Select the power-spectrum and/or bispectrum observables to fit. |
| `--kmax 0.20-0.20` | Two values for `--stats mesh2`: monopole and quadrupole. Use four values, e.g. `0.20-0.20-0.10-0.03`, for `--stats mesh2 mesh3` (adding bispectrum 000 and 202). |
| `--theory_model` | Choose among `folpsD`, `folpsEFT`, `reptvelocileptors`, and `COMET` (also accepts `comet`); `reptvelocileptors` is implemented only for `mesh2`. |
| `--comet-model` | COMET internal model: `VDG_infty` (default) or `EFT`. COMET uses `physical` or `physical_aap` priors and its native emulator; FOLPS damping and Chebyshev settings do not apply. |
| `--cosmo_params` | Use `base`, `base_ns-fixed`, or `fixed` cosmological-parameter freedom. |
| `--prior_basis` | Select the nuisance-prior basis, such as `physical_aap`. |
| `--sampler` | Choose `emcee`, `mcmc`, `nautilus`, or `pocomc`. |
| `--resume` | Require existing sampler checkpoints before resuming; current PocoMC uses `samples/samples_<id>/pmc_*.state`. |

The `folpsEFT` power-spectrum configuration enables the isotropic stochastic
parameter `sn1`, retaining desilike's physical-basis Gaussian prior (mean 0,
standard deviation 5). It is analytically marginalized when `theory.marg=True`
(the default), including in joint P+B fits. With `marg=False`, it is explicitly
sampled. The `sn1` setting is included in fit identities, separating these runs
from previous SN1-disabled chains. It adds no bispectrum parameter.

COMET supports both `mesh2` and `mesh3`. Its `srun_fit.sh` preset sets `--kmax` to
`0.35-0.30-0.20-0.10` for P0, P2, B000, and B202; slash-separated values such as
`0.35/0.30/0.20/0.10` also work. A two-value override applies to a single statistic.
Cosmological priors are intersected with COMET's published training ranges using
desilike's `truncate_priors`; its physical nuisance priors are retained, with affine
parameters marginalized and the shared `sn0` and `avir` explicitly sampled.
COMET-only fits default to the `eisenstein_hu` cosmology provider and save
`H0`, `Omega_m`, `sigma8_m`, `sigma8_cb`, and `rs_drag`, matching `QSO_zbins`.
These derived quantities use the selected cosmology provider; the saved sigma8
values are Eisenstein-Hu outputs, not CLASS post-processing results. COMET supplies
the fitted spectra and growth. Minuit keeps `precision=1e-4`.
The profiling kernel rejects non-finite parameter proposals before calling the
likelihood, and treats non-finite model values as infinite chi-square. This
prevents NaN optimizer coordinates from reaching CLASS's native solver.
During profiling, each rank reports progress about once per minute of completed
evaluations: call count, elapsed time, best objective, and time since its last
improvement. Separate MIGRAD and HESSE messages report validity, EDM and its goal,
call-limit status, and covariance accuracy. These are log diagnostics; they are
not stored in `profiles.h5`. A finite returned point alone does not imply convergence.
The cosmology settings are included in the fit identity, so these results have a
different directory hash from earlier COMET runs saving only `H0` and `Omega_m`,
and from CLASS runs.
`plot_chain_bar.ipynb` plots `H0` and `Omega_m` for COMET; other theories also
include their saved `sigma8_m`.

The fit environment must include `comet-emu` and its downloaded model data.
The COMET shell preset defaults `COMET_DATA_DIR` to the shared cosmodesi
`desilike/comet-emulators` installation. Set it yourself to use another directory
containing `tables/` and `models/`. Missing data produces an error before
measurements are loaded. From `main/full_shape`, run:

```bash
# Optional: export COMET_DATA_DIR=/path/to/comet/data
FIT_THEORY=COMET bash srun_fit.sh --todos test
FIT_THEORY=COMET bash srun_fit.sh --todos sample
```

The COMET shell preset writes to `redshift_error_fits/COMET` and uses the four
scale cuts above. Use `--comet-model EFT` to select COMET's EFT variant.
The QSO launcher `srun_QSO_fit.sh` defaults to no redshift-error theory correction
and inherits `--cosmo_params base` (free `n_s`). Set `--corr_dv` to enable a
correction, or `--corr_dv None` to explicitly disable it.

Batch and parallel launch scripts live in `full_shape/slurms/`. The basic
`srun_fit.sh`, `srun_QSO_fit.sh`, and `srun_data_fit.sh` launchers remain in
`full_shape/`. Parallel launchers resolve these from their parent directory;
`FIT_SCRIPT_DIR` can override that directory.

For concurrent LRG COMET reruns, `full_shape/slurms/sbatch_LRG_COMET.sh` defines
12 tasks: LRG1/2/3, each with P-only and P+B, for both `None` and
`verr_nonparam` injected-error mocks. Tasks 0–5 use `None`; tasks 6–11 use
`verr_nonparam`; even tasks are P-only and odd tasks are P+B. Every task calls
`srun_fit.sh` with `--todos test sample`, keeping theory corrections off and
using one four-GPU node. Supply an existing log directory with Slurm's
`--output` option when submitting the regular-queue array.

`full_shape/slurms/run_LRG_COMET_parallel.sh LOG_DIR TASK_ID...` runs those tasks
inside an existing allocation, with one concurrent task per allocated node.
For example, use two four-node interactive allocations, with task lists
`1 3 5 0 2 4` and `7 9 11 6 8 10`, to run eight fits at once and start the
remaining tasks as slots become free. Each allocation must request 16 tasks,
32 CPUs per task, and four GPUs per node. The launcher records each task's
log and exit status in `LOG_DIR`.

The QSO counterpart, `full_shape/slurms/sbatch_QSO_COMET.sh`, defines four COMET
fits for QSO1 (0.8 < z < 2.1): tasks 0/1 use `None` injected errors and tasks
2/3 use `verr_nonparam`; even tasks are P-only and odd tasks are P+B. It
explicitly passes `--corr_dv None`, so the theory correction stays off even
if `CORR_DV` is set in the environment. The runs inherit free `n_s`, all five
COMET derived outputs, and the `redshift_error_fits/COMET` output directory.
Each task checks the likelihood before sampling.

To launch all four QSO fits on separate nodes, from `main/full_shape`:

```bash
salloc -N 4 -n 16 -c 32 -C gpu --gpus-per-node=4 -A desi_g \
    --qos interactive -t 04:00:00 bash slurms/run_QSO_COMET_parallel.sh \
    /global/cfs/cdirs/desi/users/shengyu/Y3/full-shape/redshift_error_fits/COMET/logs/QSO_uncorrected
```

This requires an available interactive-allocation slot. The launcher creates
the log directory and records each fit's log and exit status. Alternatively,
submit `slurms/sbatch_QSO_COMET.sh` as a regular-queue array with an existing log
directory supplied through `sbatch --output`.

For the complete COMET rerun with cuts `0.35/0.30/0.20/0.10`, use
`full_shape/slurms/run_COMET_rerun_parallel.sh LOG_DIR TASK_ID...` inside an allocation.
It runs one fit per node, starting the next task as a node becomes free:

- Tasks 0–11: the LRG mapping above, without theory correction.
- Tasks 12–15: QSO P/P+B for `None` and `verr_nonparam`, without correction.
- Tasks 16/17: QSO P/P+B on `verr_nonparam` mocks, with **PDF-only** correction.
- Tasks 18/19: optional PDF-only QSO control fits on `None` mocks.

PDF tasks use `REDSHIFT_ERROR_PDF` or the default measured QSO velocity PDF
configured in the launcher. They pass `--corr_dv pdf`; no Gaussian or Lorentzian
residual parameter is added. Corrected outputs have distinct `rs-pdf` fit identities.
The launcher writes task logs and exit codes beneath the supplied `LOG_DIR`.

`full_shape/slurms/run_folpsEFT_parallel.sh LOG_DIR TASK_ID...` uses the same task
mapping for folpsEFT at cuts `0.20/0.20/0.20/0.03`, with SN1 enabled. Each task
first builds its Chebyshev emulators on 16 CPU ranks, then checks and samples
the likelihood on four GPU ranks. A failed build prevents sampling for that task.

`full_shape/submit_fits_after.py MANIFEST.json` can defer allocation commands
until the manifest's `afterok` Slurm jobs complete successfully and all listed
`required_chain_files` exist. It records submission status and controller PIDs
in the manifest; allocation stdout/stderr goes to each specified log file.

Redshift-error **theory corrections** are selected separately from `--zerrs`,
which chooses the errors injected into the measured mock catalogs. Corrections
are off by default. COMET (JAX backend), folpsD and folpsEFT support:

| `--corr_dv` | Theory correction |
| --- | --- |
| `gauss` | Fit a Gaussian single-galaxy displacement distribution. |
| `lor` | Fit a Lorentzian/Cauchy single-galaxy displacement distribution. |
| `pdf` | Use a supplied measured single-galaxy velocity PDF, with no fitted width. |
| `pdf+gauss` / `pdf+lor` | Measured PDF plus an additional fitted residual width. |

From `main/full_shape`, for example:

```bash
FIT_THEORY=COMET bash srun_fit.sh test --corr_dv gauss
FIT_THEORY=COMET bash srun_fit.sh sample --corr_dv pdf \
    --redshift-error-pdf '/path/to/CDF_verr_nonparam_{tracer}_z{zmin:.1f}-{zmax:.1f}.npz'
```

The PDF path accepts `{tracer}` (e.g. LRG), `{tracer_bin}` (LRG1),
`{zmin}`, `{zmax}`, and `{zsnap}` when those catalog fields exist. NPZ files
must contain `grid` (km/s) and `pdf`; lsstypes HDF5 files may instead use
`dv` and `pdf`. Supply the inferred **single-galaxy** error PDF, not a
distribution of pairwise repeat differences. Choose a PDF appropriate to the
catalog/error realization; it is not inferred from `--zerrs`.

The implementation in `fit_support.py` constructs a JAX single-field
kernel D(k mu) and passes it directly to desilike. Desilike applies D squared
to the power spectrum and D1 D2 D3 to the bispectrum before angular projection.
The Gaussian kernel is exp[-(k mu vsmear)^2/2]; the Lorentzian error PDF gives
exp[-|k mu| vsmear]. The shared per-tracer parameter `logvsmear` has a uniform
prior [-6, ln(4)], with vsmear in Mpc/h and initial value 0.5 Mpc/h.
Notebook callers can set `theory={'model': 'COMET', 'redshift_smearing': 'gauss'}`
on a `LikelihoodBuilder` observable. PDF modes use a dictionary, e.g.
`'redshift_smearing': {'mode': 'pdf', 'pdf_file': '/path/to/pdf.npz'}`.

Measured PDFs are normalized and their even part is used via a cosine transform,
with dv converted to displacement using the fiducial aH at the effective redshift.
This is one effective-redshift correction per bin, including for `_zevol` catalogs;
it does not integrate over the bin's redshift evolution or model asymmetric shifts.
The transform is tabulated over |k mu| <= 5 h/Mpc; outside it the kernel returns NaN.
Correction settings and the PDF content digest enter the fit/cache identity.
The existing FoG settings and injected-error labels retain their separate meanings.

Run from the repository root on Perlmutter:

```bash
# Print the command without activating the environment or requesting resources.
bash main/full_shape/srun_fit.sh --dry-run

# Train/cache on CPU across 16 MPI ranks, as in example_emulator.py.
bash main/full_shape/srun_fit.sh build

# Evaluate a finite fiducial log posterior for one tracer using the cache.
bash main/full_shape/srun_fit.sh test --tracers LRG1

# Profile all four tracers, then sample them.
bash main/full_shape/srun_fit.sh profile
bash main/full_shape/srun_fit.sh sample
```

The launcher activates `fit_env` and requests one GPU node. `build` uses the CPU
JAX backend with 16 MPI ranks and four OpenMP threads per rank. Other actions use
four MPI ranks, one GPU per rank, and 32 CPU threads per rank. `FIT_TIME` and `FIT_QOS` override
the default four-hour interactive request. Extra Python arguments override its
defaults, for example `--regions NGC --mockid 0`. Tracers run sequentially; MPI
ranks cooperate within each fit. `--mockid 0-24` averages 25 data vectors and
windows into one fit per tracer. The covariance is **not** divided by 25 unless
you request `--cov_scale 0.04`. For QSO1, `run_fits.py` converts `--domain altmtl`
to cutsky for data/windows and uses `hod=base_dv` for QSO by convention.
LRG uses `--hod`, which defaults to `base`.

The launcher's selected region is `GCcomb`. By convention, `run_fits.py` uses
`region='ALL'` for its mock covariance while retaining `GCcomb` for data and
windows. Other regions retain their own covariance region. The covariance
override is included in the fit and covariance-cache identities.

A file check on 2026-09-11 found all 25 GCcomb data/window products for each of
LRG1/2/3 with `hod=base`, and 846 of the 1000 requested ALL-region covariance
mocks. QSO requires the `base_dv` products; existing QSO `base` files do not
satisfy this run's configuration. At that check, QSO `base_dv` had 25/25 data
and window files for `ALL`, but 0/25 for `GCcomb`; prepare the latter before
running the full launcher. Separate GCcomb covariance products are not required.
The commands above retain the launcher's GCcomb selection. The loader skips missing covariance
mocks and uses the number actually loaded for its corrections. These are file
availability checks, not validation of file contents or a completed fit.

The Chebyshev configuration follows
`/global/homes/s/shengyu/Y3/blinded_data_splits/desi-clustering/full_shape/job_scripts/example_emulator.py`
and the active return value of `propose_fiducial_emulator_options()` in its
`full_shape/tools.py` (checked 2026-09-11). It uses `Emulator(theory.pt, space,
engine='chebyshev')`, leaving the emulator class to the theory. For FOLPS this
selects its scaling-aware emulator, including a trained scalar provider.
There is one emulator per observable; no Taylor emulator is used.

The default sparse-grid budget is 3, with `levels={'h': 3, 'm_ncdm': 3}` restricted
to varied parameters. The base cosmology's bounds are:

| Parameter | Training range |
| --- | --- |
| `h` | `[0.5, 1.0]` |
| `omega_b` | `[0.020005, 0.024995]` |
| `omega_cdm` | `[0.08, 0.16]` |
| `logA` | `[2.0, 3.7]` |
| `n_s` | `[0.8, 1.1]` |

These are the example's declared ranges, not ranges inferred from finite-difference
steps. Fixed axes are removed. Override the grid budget with `--emulator-budget 4`
or a range with `--emulator-bound h 0.55 0.9` (repeat for other parameters).
The budget controls sparse-grid cross terms; it is not a polynomial order.
Coverage checks stay enabled: outside supported ranges, eager predictions raise
and traced predictions return non-finite values, which the posterior rejects.
Thus the emulator's effective support is limited even though the configured
cosmological priors can be wider. Check accuracy and boundary effects before
interpreting production chains; the example's exact-posterior `bridge` step
has not been added to this runner.

Training checkpoints use `<emulator>.h5.ckpt.npz`. The root rank writes the
trained HDF5 file, all ranks synchronize, and the emulator is reloaded before
deployment to avoid carrying training state into the posterior. Parameter values
are restored after training. New Chebyshev cache and fit identities differ from
the Taylor ones; existing Taylor caches are neither loaded nor deleted. Fixed
cosmologies skip training when PT has no varied parameters.

Profiles are saved as `<fit directory>/profiles.h5`; current PocoMC samples
are saved as `<fit directory>/samples/samples_0.h5` (and similarly for other
chain IDs), with resumable state under `samples/samples_0/`. The current
desilike backend automatically reuses checkpoints in that directory; use a
different `--fits_dir` for an independent fresh run. `--resume` additionally
checks that checkpoints exist. The `test` command trains/loads the emulator
and evaluates the posterior; it requires all selected data and window files
and enough covariance mocks, and is intended to run on the compute node.

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
