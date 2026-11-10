"""Default fit configuration, redshift-error kernels, and compact naming support."""

import hashlib
import json
import numbers
import copy
from pathlib import Path

import numpy as np

# Single-galaxy kernels: desilike applies D**2 to P and D1*D2*D3 to B.
# PDF inputs describe individual velocity errors, not pair differences.
REDSHIFT_SMEARING_MODES = ('gauss', 'lor', 'pdf', 'pdf+gauss', 'pdf+lor')

def normalize_smearing(options):
    """Return serializable options; include the PDF digest in fit/cache identities."""
    if options is None:
        return None
    options = {'mode': options} if isinstance(options, str) else dict(options)
    mode = options.get('mode')
    if mode not in REDSHIFT_SMEARING_MODES:
        raise ValueError(f'redshift_smearing mode must be one of {REDSHIFT_SMEARING_MODES}; got {mode!r}')
    if mode.startswith('pdf'):
        if not options.get('pdf_file'):
            raise ValueError('PDF smearing requires pdf_file (CLI: --redshift-error-pdf).')
        path = Path(options['pdf_file']).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Redshift-error PDF not found: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if options.get('pdf_sha256', digest) != digest:
            raise ValueError(f'Redshift-error PDF changed since options were created: {path}')
        options.update(pdf_file=str(path), pdf_sha256=digest)
    elif options.get('pdf_file'):
        raise ValueError('pdf_file requires a pdf, pdf+gauss, or pdf+lor mode.')
    return options


def read_velocity_pdf(filename):
    """Read grid/pdf NPZ products or dv/pdf lsstypes HDF5 products."""
    path = Path(filename)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            dv, pdf = np.asarray(data['grid']), np.asarray(data['pdf'])
    else:
        import lsstypes as types
        leaf = types.read(path)
        dv, pdf = np.asarray(leaf.coords('dv')), np.asarray(leaf.values('pdf'))
    dv, pdf = np.asarray(dv, dtype=float), np.asarray(pdf, dtype=float)
    if (dv.ndim != 1 or pdf.shape != dv.shape or dv.size < 2
            or not np.isfinite(dv).all() or not np.isfinite(pdf).all()
            or not np.all(np.diff(dv) > 0) or np.any(pdf < 0)):
        raise ValueError('PDF needs finite, increasing 1D velocity coordinates and nonnegative densities.')
    weights = np.zeros_like(dv)
    weights[:-1] += np.diff(dv) / 2.
    weights[1:] += np.diff(dv) / 2.
    weights *= pdf
    norm = weights.sum()
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Redshift-error PDF must have positive finite integral.')
    return dv, weights / norm


def make_redshift_smearing(options, z, fiducial):
    """Build the JAX callable accepted by desilike's redshift_smearing keyword.

    Measured PDFs use their even part (cosine transform), tabulated on
    0 <= |k mu| <= 5 h/Mpc. Values outside that grid return NaN.
    Residual width exp(logvsmear) is in Mpc/h, with a flat log prior [-6, ln(4)].
    """
    options = normalize_smearing(options)
    if options is None:
        return None
    import jax.numpy as jnp
    from desilike import Parameter

    mode = options['mode']
    residual = mode.partition('+')[2] if mode.startswith('pdf') else mode
    if mode.startswith('pdf'):
        if not np.isfinite(z) or z < 0:
            raise ValueError('PDF smearing requires a finite, nonnegative effective redshift.')
        dv, weights = read_velocity_pdf(options['pdf_file'])
        aH = float(100. * fiducial.efunc(z) / (1. + z))
        if not np.isfinite(aH) or aH <= 0:
            raise ValueError('Fiducial aH must be positive and finite.')
        grid = np.linspace(0., 5., 100001)
        values = np.empty_like(grid)
        chunk = max(1, 2_000_000 // dv.size)
        for start in range(0, grid.size, chunk):
            values[start:start + chunk] = np.cos(grid[start:start + chunk, None] * dv / aH) @ weights
        grid, values = jnp.asarray(grid), jnp.asarray(values)

        def measured(kmu):
            return jnp.interp(jnp.abs(kmu), grid, values, right=jnp.nan)
    else:
        def measured(kmu):
            return jnp.ones_like(kmu)

    if not residual:
        return measured

    def kernel(kmu, logvsmear):
        x = kmu * jnp.exp(logvsmear)
        damping = jnp.exp(-x**2 / 2.) if residual == 'gauss' else jnp.exp(-jnp.abs(x))
        return measured(kmu) * damping

    kernel.params = Parameter('logvsmear', value=np.log(0.5),
                              prior=dict(dist='uniform', limits=[-6., np.log(4.)]),
                              ref=dict(dist='norm', loc=np.log(0.5), scale=1.),
                              latex=r'\ln v_{\mathrm{smear}}')
    return kernel


def propose_fiducial_covariance_options():
    """Return fiducial covariance options."""
    return {'source': 'mock', 'version': 'holi-v3-altmtl',
            'corrections': ['hartlap', 'percival'], 'rescale': False}

def propose_fiducial_cosmology_options():
    """Return fiducial cosmology options."""
    return {'model': 'base_ns-fixed', 'template': 'direct'}

def propose_fiducial_sampler_options(sampler=None):
    """Return fiducial sampler options."""
    if sampler is None:
        sampler = 'emcee'
    init, run = {}, {}
    init['rng'] = 42
    if sampler in ['emcee', 'zeus', 'mhmcmc', 'nuts', 'numpyro_nuts', 'numpyro_barker']:
        init['nparallel'] = 4
        run['min_steps'] = 50
        run['gelman_rubin'] = 1.05
        run['ess'] = 400
    if sampler in ['emcee']:
        init['batch_size'] = 16
        run['thinning'] = 5
    if sampler in ['mcmc', 'mhmcmc']:
        init['oversample_power'] = 0
    if sampler in ['nuts']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        run['adaptation'] = dict(initial_step_size=0.01, target_acceptance_rate=0.8,
                                 steps=1000, is_mass_matrix_diagonal=False)
    if sampler in ['numpyro_nuts', 'numpyro_barker']:
        init['rescale'] = 'diag'
        init['step_size'] = 0.1
        run['adaptation'] = dict(steps=500, dense_mass=True)
    if sampler in ['mhmcmc']:
        run['check_every'] = 1000
    if sampler in ['nautilus']:
        init['rescale'] = 'diag'
        init['n_live'] = 1000
        run['n_eff'] = 200
        run['verbose'] = True
    if sampler in ['pocomc']:
        init['batch_size'] = 8
        init['rescale'] = 'diag'
        init['n_effective'] = 1024
        init['n_active'] = 512
        init['flow'] = 'nsf6'
        run['n_total'] = 20000
    return {'sampler': sampler, 'init': init, 'run': run, 'nchains': 1, 'resume': False}

def propose_fiducial_profiler_options(profiler=None):
    """Return fiducial profiler options."""
    if profiler is None: profiler = 'minuit'
    return {'profiler': profiler, 'init': {}, 'maximize': {}}

def propose_fiducial_emulator_options():
    """Match the active defaults used by desi-clustering's example_emulator.py.

    Source: blinded_data_splits/desi-clustering/full_shape/tools.py,
    propose_fiducial_emulator_options (2026-09-11). Keep only varied axes at use.
    """
    return {'name': 'chebyshev', 'budget': 3, 'levels': {'h': 3, 'm_ncdm': 3},
            'space': {'limits': {'h': [0.5, 1.], 'omega_b': [0.020005, 0.024995],
                                 'omega_cdm': [0.08, 0.16], 'logA': [2., 3.7],
                                 'n_s': [0.8, 1.1], 'w0_fld': [-3., 0.5],
                                 'wa_fld': [-3., 2.], 'm_ncdm': [1e-6, 1.],
                                 'N_eff': [2., 4.], 'tau_reio': [0.01014, 0.14986]}}}


def propose_fiducial_observable_options(stat, tracer=None, zrange=None, theory_model=None):
    """Return fiducial observable options for a statistic."""
    propose_fiducial = {'stat': {'kind': stat},
                        'catalog': {'weight': 'default-FKP'},
                        'theory': {},
                        'emulator': propose_fiducial_emulator_options(),
                        'window': {}}
    propose_stat = {
        'mesh2': {'select': [{'ells': ell, 'k': [0.02, 0.3, 0.01]} for ell in [0, 2]]},
        'mesh3': {'select': [{'ells': (0, 0, 0), 'k': [0.02, 0.20, 0.01]},
                                      {'ells': (2, 0, 2), 'k': [0.02, 0.08, 0.01]}],
                           'basis': 'sugiyama-diagonal'},
    }
    # base_full_shape_theory = {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'lor', 'marg': True}
    base_full_shape_theory = {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'vdg', 'marg': True}
    propose_theory = {
        'mesh2': base_full_shape_theory | {'damping_method': 'tree', 'b3_coev': False, 'A_full': False},
        'mesh3': base_full_shape_theory | {'A_full': False},
    }
    for name in propose_stat:
        if name in stat:
            propose_fiducial['stat'].update(propose_stat[name])
            propose_fiducial['theory'].update(propose_theory[name])
    if str(theory_model).lower() == 'comet':
        propose_fiducial['theory'] = {'model': 'COMET', 'comet_model': 'VDG_infty',
                                    'prior_basis': 'physical_aap', 'marg': True}
        propose_fiducial['emulator'] = {'name': None}
    return propose_fiducial

def fill_fiducial_observable_options(options):
    """Fill missing observable options with fiducial values."""
    options = dict(options)
    if str(options.get('theory', {}).get('model')).lower() == 'comet':
        options['theory'] = options['theory'] | {'model': 'COMET'}
    stat = options['stat']['kind']
    tracer = options.get('catalog', {}).get('tracer', None)
    zrange = options.get('catalog', {}).get('zrange', None)
    fiducial_options = propose_fiducial_observable_options(
        stat, tracer, zrange, theory_model=options.get('theory', {}).get('model'))
    options = fiducial_options | options
    for key, value in fiducial_options.items():
        options[key] = value | options[key]
    if options['theory'].get('redshift_smearing') is not None:
        options['theory'] = dict(options['theory'])
        options['theory']['redshift_smearing'] = normalize_smearing(options['theory']['redshift_smearing'])
    return options

def fill_fiducial_likelihood_options(options):
    """Fill missing likelihood options with fiducial values."""
    if isinstance(options, dict):
        options = dict(options)
        options['observables'] = [fill_fiducial_observable_options(obs) for obs in options['observables']]
        options['covariance'] = propose_fiducial_covariance_options() | (options.get('covariance', {}) or {})
        return options
    return type(options)(fill_fiducial_likelihood_options(opts) for opts in options)

def fill_fiducial_options(options):
    """Fill missing top-level fitting options with fiducial values."""
    options = dict(options)
    options['cosmology'] = propose_fiducial_cosmology_options() | options.get('cosmology', {})
    likelihoods = options.get('likelihoods', None)
    if likelihoods is not None:
        options['likelihoods'] = fill_fiducial_likelihood_options(likelihoods)
        theories = [obs['theory']['model'] for likelihood in options['likelihoods']
                    for obs in likelihood['observables']]
        if theories and all(model.lower() == 'comet' for model in theories):
            # Match QSO_zbins: COMET supplies the spectra, while the cosmology
            # provider supplies the saved derived quantities.
            options['cosmology'].setdefault('engine', 'eisenstein_hu')
            options['cosmology'].setdefault('derived', ['H0', 'Omega_m', 'sigma8_m', 'sigma8_cb', 'rs_drag'])
            profiler = dict(options.get('profiler') or {})
            profiler['init'] = dict(profiler.get('init') or {})
            profiler['init'].setdefault('precision', 1e-4)
            options['profiler'] = profiler
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory']['cosmology'] = options['cosmology']
    for name in ['sampler', 'profiler']:
        options.setdefault(name, {})
        options[name] = globals()[f'propose_fiducial_{name}_options'](options[name].get(name)) | options[name]
    return options

def get_full_tracer_zrange(tracerz=None, zrange=None):
    """Translate compact tracer-bin labels, e.g. LRG1, to tracer and z-range."""
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _translate(one):
        if 'x' in one:
            return list(zip(*[_translate(item) for item in one.split('x')]))
        if one in translate_zrange:
            return one[:-1], translate_zrange[one]
        if zrange is None:
            raise ValueError(f'zrange not found for {one}; choose one from {list(translate_zrange)}')
        return one, zrange

    if isinstance(tracerz, str):
        return _translate(tracerz)
    return type(tracerz)(zip(*map(_translate, tracerz)))


def get_tracerz(tracer, zrange=None):
    """Return the compact tracer-bin label matching ``tracer`` and ``zrange``."""
    translate_tracerz = get_full_tracer_zrange(tracerz=None)

    try:
        from cat_tools import get_simple_tracer
    except ImportError:
        def get_simple_tracer(tracer):
            return str(tracer).rstrip('0123456789')

    if not isinstance(tracer, str) or zrange is None or tracer in translate_tracerz:
        return tracer
    simple_tracer = get_simple_tracer(tracer)
    if tracer != simple_tracer:
        return tracer
    for tracerz, tracerz_zrange in translate_tracerz.items():
        if tracerz.startswith(simple_tracer) and np.allclose(zrange, tracerz_zrange):
            return tracerz
    return tracer


def normalize_catalog_tracerz(catalog):
    """Return a catalog copy using compact tracer-bin labels when possible."""
    catalog = dict(catalog)
    if 'tracer' in catalog:
        catalog['tracer'] = get_tracerz(catalog['tracer'], catalog.get('zrange', None))
    return catalog


def _is_scalar_mock_id(value):
    return isinstance(value, (int, np.integer))


def _as_mock_id_list(value, name):
    if _is_scalar_mock_id(value):
        return [int(value)]
    if isinstance(value, str) or np.isscalar(value):
        raise ValueError(f'catalog["{name}"] must be an integer mock id or a non-string sequence of integer mock ids')
    mock_ids = [int(mock_id) for mock_id in list(value)]
    if not mock_ids:
        raise ValueError(f'catalog["{name}"] must contain at least one mock realization')
    return mock_ids


def normalize_catalog_mock_ids(catalog):
    """Return a catalog copy with one canonical mock-id option."""
    catalog = dict(catalog)
    has_mock_id = 'mock_id' in catalog and catalog['mock_id'] is not None
    has_mock_ids = 'mock_ids' in catalog and catalog['mock_ids'] is not None
    if has_mock_id and has_mock_ids:
        raise ValueError('Use only one of catalog["mock_id"] or catalog["mock_ids"]')
    if has_mock_id:
        mock_ids = _as_mock_id_list(catalog.pop('mock_id'), 'mock_id')
    elif has_mock_ids:
        mock_ids = _as_mock_id_list(catalog.pop('mock_ids'), 'mock_ids')
    else:
        return catalog
    if len(mock_ids) == 1:
        catalog['mock_id'] = mock_ids[0]
    else:
        catalog['mock_ids'] = mock_ids
    return catalog


def normalize_catalog_use_dv(catalog):
    """Return a catalog copy with one no-redshift-error spelling."""
    catalog = dict(catalog)
    if 'use_dv' in catalog and catalog['use_dv'] in [None, False, 'None', 'False']:
        catalog['use_dv'] = False
    return catalog


def normalize_catalog_options(catalog):
    """Return a catalog copy with canonical tracer-bin and mock-id names."""
    return normalize_catalog_use_dv(normalize_catalog_mock_ids(normalize_catalog_tracerz(catalog)))


def _get_default_ref_from_prior(prior, value=None):
    """Build a compact reference distribution from a prior for sampler initialization."""
    if not prior: return None
    dist = prior.get('dist', None)
    limits = prior.get('limits', [-np.inf, np.inf])
    if limits is None: limits = [-np.inf, np.inf]
    limits = list(limits)
    if dist == 'norm':
        scale = prior.get('scale', None)
        if scale is None or scale <= 0:
            return None
        return {'dist': 'norm',
                'loc': prior.get('loc', value if value is not None else 0.),
                'scale': scale / 5.,
                'limits': limits,}
    
    if dist == 'uniform':
        if len(limits) != 2 or not np.all(np.isfinite(limits)):
            return None
        lo, hi = limits
        return {'dist': 'norm',
                'loc': value if value is not None else 0.5 * (lo + hi),
                'scale': (hi - lo) / 20.,
                'limits': [lo, hi],
        }
    return None

def _normal_prior(loc, scale, limits=None, nsigma=6):
    """Return a Gaussian prior with finite limits for prior-volume samplers."""
    if limits is None:
        limits = [loc - nsigma * scale, loc + nsigma * scale]
    return {'dist': 'norm', 'loc': loc, 'scale': scale, 'limits': limits}

def get_default_cosmology_priors(cosmology_options: dict=None):
    """
    Return parameter configs used to set cosmological priors.

    Parameters
    ----------
    cosmology_options : dict, optional
        Cosmology options. Supports ``model`` plus optional ``params`` updates
        and ``priors`` prior-only updates, both keyed by parameter name.
        ``derived`` selects which of H0, Omega_m, sigma8_m, sigma8_cb, and
        rs_drag to report; by default the first three are included.

    Returns
    -------
    params : dict
        Mapping from cosmological parameter names to desilike parameter configs.
    """
    cosmology_options = cosmology_options or {}
    model = cosmology_options.get('model', 'base_ns-fixed')
    is_fixed_model = model == 'fixed'
    params = {
        'tau_reio': {'fixed': True},
        'n_s':      {'fixed': is_fixed_model or 'ns-fixed' in model, 'prior': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}},
        'omega_b':  {'fixed': is_fixed_model, 'prior': {'dist': 'norm', 'loc': 0.02237,  'scale': 0.00055}},
        'h':        {'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [0.2,  1.0]}},
        'omega_cdm':{'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [0.01, 0.99]}},
        'logA':     {'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [1.61,  3.91]}},
    }
    derived = ['H0', 'Omega_m', 'sigma8_m', 'sigma8_cb', 'rs_drag']
    requested = cosmology_options.get('derived', derived[:3])
    unknown = set(requested) - set(derived)
    if unknown:
        raise ValueError(f'Unknown derived cosmology parameters {sorted(unknown)}; choose from {derived}.')
    for name in derived:
        if name in requested:
            params[name] = {'derived': True}
    if 'w0wa' in model:
        params['w0_fld'] = {'fixed': is_fixed_model}
        params['wa_fld'] = {'fixed': is_fixed_model}

    for name, prior in cosmology_options.get('priors', {}).items():
        params.setdefault(name, {})
        params[name]['prior'] = copy.deepcopy(prior)
        params[name].setdefault('fixed', False)

    for name, config in cosmology_options.get('params', {}).items():
        params.setdefault(name, {})
        params[name].update(copy.deepcopy(config))

    for name, config in list(params.items()):
        if config.get('fixed', False):
            params[name] = {key: value for key, value in config.items() if key != 'prior'}
    return params

def get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=False, tracer=None, sigma8_fid=1.):
    """
    Build FOLPS nuisance priors matching desi-clustering/full_shape/tools.py.

    Parameters
    ----------
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2', 'mesh3'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float, optional
        Retained for caller compatibility; physical-basis priors are zero-centered.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        :meth:`Parameter.update` (e.g. ``{'fixed': True}`` or ``{'prior': {...}}``).
    """
    params = {}

    def gaussian(scale, fixed=False):
        return {'value': 0., 'fixed': fixed,
                'prior': {'dist': 'norm', 'loc': 0., 'scale': scale}}

    # Source: blinded_data_splits/desi-clustering/full_shape/tools.py,
    # update_theory_nuisance_priors (2026-09-16). Keep Gaussian priors unbounded.
    if 'physical' in prior_basis:
        params['b1'] = {'value': 1.5, 'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.1, 8.]}}
        params['b2'] = gaussian(20.)
        params['bs'] = gaussian(20.)
        params['sn0'] = gaussian(2.)
        params['X_FoG'] = {'value': 0., 'fixed': True, 'prior': {'dist': 'uniform', 'limits': [0, 20]}}
        if 'mesh2' in stat:
            params['b3'] = gaussian(1., fixed=b3_coev)
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = gaussian(50.)
            params['ct'] = {'value': 0., 'fixed': True, 'prior': {}}
            params['sn2'] = gaussian(5.)
        elif 'mesh3' in stat:
            params['c1'] = gaussian(20.)
            params['c2'] = gaussian(20., fixed=True)
            params['snb0'] = gaussian(1.)
    if 'EFT' in model.upper():
        params['X_FoG*'] = {'fixed': True}
    else:
        params['X_FoG*'] = {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0, 20]}}
    return params


def update_theory_nuisance_priors(params, model, stat, prior_basis, b3_coev=False,
                                  marg=False, user_params=None):
    """Apply nuisance priors and solve only the affine nuisance sector."""
    # COMET supplies physical-basis priors, including its extra sn22 parameter.
    configs = {} if str(model).lower() == 'comet' else get_default_theory_nuisance_priors(
        model, stat, prior_basis, b3_coev=b3_coev)
    if str(model).lower() == 'comet' and b3_coev:
        configs['b3'] = {'value': 0., 'fixed': True}
    if marg:
        # sn0 is shared by P and B and remains explicitly varied in both.
        affine = ['alpha*', 'sn2', 'sn4', 'b3', 'snb0']
        if str(model).lower() == 'comet':
            affine.append('sn22')
        elif str(model).lower() == 'folpseft' and 'mesh2' in stat:
            affine.append('sn1')
        for param in params.select(basename=affine):
            param.update(derived='marg')
    # Apply explicit overrides last, including after wildcard FoG defaults.
    for updates in [configs, user_params or {}]:
        for basename, config in updates.items():
            for param in params.select(basename=basename):
                param.update(**config)
    return params

def _get_level(level: int | dict = None):
    """Normalize verbosity level dictionaries used by string helpers."""
    default_level = {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}
    if level is None:
        level = {}
    if not isinstance(level, dict):
        level = {name: level for name in default_level}
    return default_level | level


def _base_type_options(options):
    """Cast option values to JSON-serializable Python base types."""
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return convert(value.item())
            return [convert(item) for item in value.tolist()]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if value is None or isinstance(value, (bool, numbers.Number, str)):
            return value
        return str(value)
    return convert(options)


def _hash_options(options, length=8):
    """Return a short SHA-256 hash of canonicalized options."""
    def canonical(obj):
        if isinstance(obj, dict):
            return sorted((canonical(key), canonical(value)) for key, value in obj.items())
        if isinstance(obj, list):
            return [canonical(value) for value in obj]
        return obj
    payload = json.dumps(canonical(_base_type_options(options)), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def _short_stat_name(stat):
    translate_stat_name = {
        'S2': ['mesh2'],
        'S3': ['mesh3'],
        'BAOR': ['bao', 'recon'],
        'C2R': ['particle2_correlation', 'recon'],
    }
    for name, tags in translate_stat_name.items():
        if all(tag in stat for tag in tags):
            return name
    raise ValueError(f'could not find short name for {stat}')


def _damping_label_from_observables(observables):
    damping = None
    for observable in observables:
        theory = observable.get('theory', {}) or {}
        if damping is None:
            damping = theory.get('damping', None)
    zerr = _zerr_label_from_observables(observables)
    items = [item for item in [damping, zerr] if item is not None]
    return '-'.join(str(item) for item in items)


def _zerr_label_from_observables(observables):
    """Return the compact redshift-error label for a shared observable catalog."""
    from cat_tools import _unzip_catalog_options

    labels = []
    for observable in observables:
        for catalog_options in _unzip_catalog_options(observable.get('catalog', {})).values():
            use_dv = catalog_options.get('use_dv', None)
            if use_dv in [None, False, 'None', 'False']:
                label = 'None'
            else:
                label = str(use_dv)
                if catalog_options.get('z_evol', False):
                    label += '_zevol'
            labels.append(label)
    if not labels or any(label != labels[0] for label in labels[1:]):
        return None
    return labels[0]


def _kmax_label_from_observables(observables):
    """Return compact kmax values ordered as mesh2 ell0/ell2, mesh3 000/202."""
    values = {}
    for observable in observables:
        stat = observable.get('stat', {}).get('kind', '')
        for item in observable.get('stat', {}).get('select', []) or []:
            if callable(item):
                return None
            item = dict(item)
            ell = item.get('ells', None)
            k = item.get('k', None)
            if k is None or len(k) < 2:
                continue
            if 'mesh2' in stat:
                if ell == 0:
                    values['mesh2_0'] = float(k[1])
                elif ell == 2:
                    values['mesh2_2'] = float(k[1])
            elif 'mesh3' in stat:
                ell = tuple(ell) if isinstance(ell, list) else ell
                if ell == (0, 0, 0):
                    values['mesh3_000'] = float(k[1])
                elif ell == (2, 0, 2):
                    values['mesh3_202'] = float(k[1])
    ordered = [
        ('mesh2_0', '%.3f'),
        ('mesh2_2', '%.3f'),
        ('mesh3_000', '%.2f'),
        ('mesh3_202', '%.2f'),
    ]
    labels = [fmt % values[name] for name, fmt in ordered if name in values]
    return '-'.join(labels) if labels else None


def str_from_observable_options(options: dict, level: int | dict = None) -> str:
    """Return a compact identifier for one observable options dictionary."""
    from cat_tools import _unzip_catalog_options
    from utils import float2str

    level = _get_level(level)
    out_str = []
    catalog = _unzip_catalog_options(options['catalog'])

    def str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    def str_mock_mean(catalog_options):
        catalog_options = normalize_catalog_mock_ids(catalog_options)
        mock_ids = catalog_options.get('mock_ids', None)
        if mock_ids is None:
            return None
        return f'mean{len(mock_ids)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer, catalog_options in catalog.items():
            stracer = str(get_tracerz(tracer, catalog_options.get('zrange', None)))
            found = stracer != str(tracer) or stracer in translate_tracerz
            tracer_catalog_str = [stracer]
            if 'zrange' in catalog_options:
                if not found or level['catalog'] >= 2:
                    tracer_catalog_str.append(str_zrange(catalog_options['zrange']))
            elif 'zsnap' in catalog_options:
                tracer_catalog_str.append(f'z{float2str(catalog_options["zsnap"], prec_min=1, prec_max=5)}')
            if level['catalog'] >= 3 and 'region' in catalog_options:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4 and 'weight' in catalog_options:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            mean_label = str_mock_mean(options['catalog'])
            if mean_label is not None:
                tracer_catalog_str.append(mean_label)
            catalog_str.append('-'.join(item for item in tracer_catalog_str if item))
        out_str.append('x'.join(catalog_str))

    stat_options = options['stat']
    stat = stat_options['kind']
    short_name = None
    if level['stat'] >= 1:
        short_name = _short_stat_name(stat)
        out_str.append(short_name)
    if level['stat'] >= 2:
        select_str = []
        select = stat_options.get('select', [])
        if callable(select):
            select_str.append(getattr(select, 'name', 'custom'))
        else:
            def str_ell(ell):
                if isinstance(ell, (list, tuple)):
                    return ''.join(str(item) for item in ell)
                return str(ell)

            for item in select:
                item = dict(item)
                label = []
                if 'ells' in item:
                    label.append('ell' + str_ell(item.pop('ells')))
                for coord_name, limits in item.items():
                    prec = {'prec_min': 2, 'prec_max': 3} if short_name and short_name.startswith('S') else {'prec_min': 0, 'prec_max': 0}
                    label.append(coord_name + '-'.join(float2str(limit, **prec) for limit in limits))
                select_str.append('-'.join(label))
        out_str.append('-'.join(select_str))

    if level['theory'] > 0:
        out_str.extend(['th', options['theory']['model']])
        if level['theory'] >= 2:
            theory = options['theory']
            for name in ['damping', 'damping_method']:
                if name in theory and theory[name] is not None:
                    out_str.append(f'{name}-{theory[name]}')
    smearing = options.get('theory', {}).get('redshift_smearing')
    if smearing and (level['stat'] > 0 or level['theory'] > 0):
        mode = smearing if isinstance(smearing, str) else smearing['mode']
        out_str.append('rs-' + mode.replace('+', '-'))
    return '-'.join(item for item in out_str if item)


def str_from_likelihood_options(likelihood_options, level: int | dict = None, fit_output=False):
    """Return a compact identifier, optionally using the fit-directory convention."""
    level = _get_level(level)
    observables = likelihood_options['observables']
    catalogs = [_base_type_options(observable.get('catalog', {})) for observable in observables]
    if fit_output and level['catalog'] < 3:
        # Region is hidden at this verbosity; mesh2 GCcomb and mesh3 ALL share a label.
        catalogs = [{name: value for name, value in catalog.items() if name != 'region'}
                    for catalog in catalogs]
    same_catalog = (
        len(observables) > 1
        and all(catalog == catalogs[0] for catalog in catalogs[1:])
    )
    if same_catalog and level['catalog'] >= 1 and level['stat'] == 1 and level['theory'] == 0:
        catalog_label = str_from_observable_options(observables[0], level=level | {'stat': 0, 'theory': 0})
        stat_label = ('-' if fit_output else '+').join(
            _short_stat_name(observable['stat']['kind']) for observable in observables)
        if fit_output:
            joint_items = [stat_label, _zerr_label_from_observables(observables)]
        else:
            kmax_label = _kmax_label_from_observables(observables)
            damping_label = _damping_label_from_observables(observables)
            joint_items = [stat_label, kmax_label, damping_label]
        joint_label = '-'.join(item for item in joint_items if item)
        modes = [obs.get('theory', {}).get('redshift_smearing') for obs in observables]
        if any(modes):
            mode_labels = [
                (mode if isinstance(mode, str) else mode['mode']).replace('+', '-') if mode else 'none'
                for mode in modes]
            # One correction label when all jointly fitted statistics share the mode.
            if len(set(mode_labels)) == 1:
                mode_labels = mode_labels[:1]
            joint_label += '-rs-' + '-'.join(mode_labels)
        out_str = ['-'.join(item for item in [catalog_label, joint_label] if item)]
    else:
        out_str = [str_from_observable_options(options, level=level)
                   for options in observables]
        if fit_output:
            out_str = ['-'.join(item for item in [label, _zerr_label_from_observables([observable])] if item)
                       for label, observable in zip(out_str, observables)]
    if level['covariance'] > 0:
        covariance = likelihood_options.get('covariance', {}) or {}
        covariance_str = ['cov-' + '-'.join([covariance.get('source', 'none'),
                                             covariance.get('version', 'none')])]
        if level['covariance'] >= 3:
            corrections = covariance.get('corrections', None)
            if isinstance(corrections, str):
                corrections = [corrections]
            corrections = sorted(str(correction).lower() for correction in (corrections or []))
            if corrections:
                covariance_str.append('corr-' + '-'.join(corrections))
            nparams = covariance.get('nparams', None)
            if nparams is not None:
                covariance_str.append(f'nparams-{int(nparams)}')
        out_str.append('-'.join(covariance_str))
    return '+'.join(out_str)


def str_from_cosmology_options(cosmology_options: dict, level: int | dict = None):
    """Return a compact identifier for cosmology options."""
    level = _get_level(level)
    if level['cosmology'] < 1:
        return ''
    model, template = cosmology_options['model'], cosmology_options['template']
    return f'cosmo-{model}' if template.lower() == 'direct' else f'template-{template}'


def str_from_options(options: dict, level: int | dict = None, fit_output=False):
    """Return a compact identifier for full fitting options."""
    level = _get_level(level)
    out_str = [str_from_cosmology_options(options['cosmology'], level=level)]
    out_str += [str_from_likelihood_options(likelihood_options, level=level, fit_output=fit_output)
                for likelihood_options in options['likelihoods']]
    return '_'.join(out_str)
