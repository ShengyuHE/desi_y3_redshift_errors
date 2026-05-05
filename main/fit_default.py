import os
import numpy as np
import lsstypes as types

def propose_fiducial_covariance_options():
    """Return fiducial covariance options."""
    return {'source': 'mock', 'version': 'holi-v3-altmtl', 'corrections': ['hartlap', 'percival']}

def propose_fiducial_cosmology_options():
    """Return fiducial cosmology options."""
    return {'model': 'base_ns-fixed', 'template': 'direct'}

def propose_fiducial_sampler_options(sampler=None):
    """Return fiducial sampler options."""
    if sampler is None:
        sampler = 'emcee'
    init = {}
    if sampler == 'mcmc':
        init['oversample_power'] = 0
    return {'sampler': sampler, 'init': init, 'run': {'check': {'max_eigen_gr': 0.03}}, 'nchains': 1}

def propose_fiducial_profiler_options(profiler=None):
    """Return fiducial profiler options."""
    if profiler is None: profiler = 'minuit'
    return {'profiler': profiler, 'init': {}, 'maximize': {}}

def propose_fiducial_observable_options(stat, tracer=None, zrange=None):
    """Return fiducial observable options for a statistic."""
    propose_fiducial = {'stat': {'kind': stat},
                        'catalog': {'weight': 'default-FKP'},
                        'theory': {},
                        'emulator': {'name': 'taylor', 'order': 3},
                        'window': {}}
    propose_stat = {
        'mesh2_spectrum': {'select': [{'ells': ell, 'k': [0.02, 0.2, 0.005]} for ell in [0, 2]]},
        'mesh3_spectrum': {'select': [{'ells': (0, 0, 0), 'k': [0.02, 0.12, 0.005]},
                                      {'ells': (2, 0, 2), 'k': [0.02, 0.08, 0.005]}],
                           'basis': 'sugiyama-diagonal'},
    }
    base_full_shape_theory = {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'lor', 'marg': True}
    base_bao_theory = {'model': 'bao', 'broadband': 'pcs2', 'marg': True}
    propose_theory = {
        'mesh2_spectrum': base_full_shape_theory | {'b3_coev': True, 'A_full': False},
        'mesh3_spectrum': base_full_shape_theory | {'A_full': False},
    }
    for name in propose_stat:
        if name in stat:
            propose_fiducial['stat'].update(propose_stat[name])
            propose_fiducial['theory'].update(propose_theory[name])
    return propose_fiducial

def fill_fiducial_observable_options(options):
    """Fill missing observable options with fiducial values."""
    options = dict(options)
    stat = options['stat']['kind']
    tracer = options.get('catalog', {}).get('tracer', None)
    zrange = options.get('catalog', {}).get('zrange', None)
    fiducial_options = propose_fiducial_observable_options(stat, tracer, zrange)
    options = fiducial_options | options
    for key, value in fiducial_options.items():
        options[key] = value | options[key]
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
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory']['cosmology'] = options['cosmology']
    for name in ['sampler', 'profiler']:
        options.setdefault(name, {})
        options[name] = globals()[f'propose_fiducial_{name}_options'](options[name].get(name)) | options[name]
    return options

_fiducial = None
def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial

def _get_effective_redshift_from_window(window, fn=None):
    mono = window.theory.get(ells=0)
    zeff = getattr(mono, 'z', None)
    if zeff is None:
        zeff = getattr(mono, '_meta', {}).get('z', None)
    if zeff is None:
        location = f' {fn}' if fn is not None else ''
        raise AttributeError(f'No z_eff found in window function{location}')
    return float(zeff)

def get_effective_redshift(args=None, kind='window_mesh2_spectrum_poles',
                           get_measurement_fn=None, **kwargs):
    """Read the effective redshift from a window-function measurement file."""
    from pathlib import Path
    if get_measurement_fn is None:
        from cat_tools import get_measurement_fn
    args = dict(args or {})
    args.update(kwargs)
    kind = args.pop('kind', kind)
    fn = Path(get_measurement_fn(**args).format(kind))
    window = types.read(fn)
    return _get_effective_redshift_from_window(window, fn=fn)

def _observable_labels(observable_options):
    stat = observable_options['stat']['kind']
    catalog = observable_options.get('catalog', {})
    tracer = catalog.get('tracer', None)
    if isinstance(tracer, str):
        tracer = (tracer,)
    elif tracer is None:
        tracer = ()
    else:
        tracer = tuple(tracer)
    nfields = 3 if 'mesh3' in stat else 2
    if tracer:
        tracer = tracer + (tracer[-1],) * (nfields - len(tracer))
    return {'observables': stat, 'tracers': tracer}

def _apply_select(observable, select=None):
    """Apply lsstypes-style ell/k selections to an observable tree."""
    if select is None:
        return observable
    if callable(select):
        return select(observable)
    labels = []
    for item in select:
        item = dict(item)
        keys = observable.labels(return_type='keys')
        label = {}
        for key in keys:
            if key in item:
                label[key] = item.pop(key)
        labels.append(label)
        pole = observable.get(**label)
        for coord_name, limits in item.items():
            if len(limits) == 3:
                step = limits[2]
                edge = pole.edges(coord_name)[0]
                rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                pole = pole.select(**{coord_name: slice(0, None, rebin)})
            pole = pole.select(**{coord_name: tuple(limits[:2])})
        observable = observable.at(**label).replace(pole)
    return observable.get(labels)

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

def get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=True, tracer=None, sigma8_fid=1.):
    """
    Build a dictionary of parameter priors.

    Parameters
    ----------
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2_spectrum', 'mesh2_spectrum'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float, optional
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        :meth:`Parameter.update` (e.g. ``{'fixed': True}`` or ``{'prior': {...}}``).
    """
    params = {}
    scale_eft = 12.5
    scale_sn0 = 2.0
    scale_sn2 = 5.0

    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7. * sigma8_fid**2, 'scale': 5}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3p'] = {'fixed': True}
            else:
                params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23. / 42. * sigma8_fid**4, 'scale': sigma8_fid**4},
                                 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_pp'] = {'fixed': True}
            else:
                params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3'] = {'fixed': True}
            else:
                params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}, 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_p'] = {'fixed': True}
            else:
                params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            shotnoise = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    for config in params.values():
        if config.get('fixed', False):
            continue
        ref = _get_default_ref_from_prior(config.get('prior', None), value=config.get('value', None))
        if ref is not None and 'ref' not in config:
            config['ref'] = ref
    return params


def get_covariance_correction_factor(covariance: types.CovarianceMatrix,
                                      observables: list[dict],
                                      covariance_options: dict,
                                      default_corrections=('hartlap', 'percival')):
    """Return multiplicative covariance correction factor and per-term metadata."""
    from lsstypes.utils import get_hartlap2007_factor, get_percival2014_factor, mkdir
    corrections = covariance_options.get('corrections', default_corrections)
    if isinstance(corrections, str):
        corrections = [corrections]
    corrections = [str(corr).lower() for corr in (corrections or [])]

    factor = 1.
    nbins = int(covariance.value().shape[0])
    nobs = covariance.attrs.get('nobs', None)
    metadata = {'nbins': nbins, 'corrections': tuple(corrections)}
    if nobs is None:  # analytic covariance matrix
        return factor, metadata | dict(corrections=tuple())

    nobs = int(nobs)
    metadata.update(nobs=nobs)

    if 'hartlap' in corrections:
        hartlap = get_hartlap2007_factor(nobs, nbins)
        factor /= hartlap
        metadata['hartlap_factor'] = float(hartlap)

    if 'percival' in corrections:
        def _infer_effective_nparams(observables: list[dict]) -> int:
            """Infer effective free-parameter count for covariance corrections.

            Uses a fixed effective count by fit content:
            - 7 for single-stat fits (mesh2-only or mesh3-only)
            - 9 for joint mesh2+mesh3 fits
            """
            stats = {obs['stat']['kind'] for obs in observables}
            has_mesh2 = any('mesh2_spectrum' in stat for stat in stats)
            has_mesh3 = any('mesh3_spectrum' in stat for stat in stats)
            return 9 if (has_mesh2 and has_mesh3) else 7
        nparams = covariance_options.get('nparams', None)
        if nparams is None:
            nparams = _infer_effective_nparams(observables)
        percival = get_percival2014_factor(nobs, nbins, nparams)
        factor *= percival
        metadata['percival_factor'] = float(percival)
        metadata['nparams'] = int(nparams)
    return factor, metadata


def _get_level(level: int | dict=None):
    """Normalize verbosity level dictionaries used by string helpers."""
    default_level = {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}
    if level is None:
        level = {}
    if not isinstance(level, dict):
        level = {name: level for name in default_level}
    return default_level | level

def _base_type_options(options):
    """
    Recursively cast objects of input dictionary ``d`` to Python base types
    so they can be serialized by standard YAML.
    """
    import numbers
    import numpy as np
    def convert(v):
        if isinstance(v, dict):
            return {k: convert(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set, frozenset)):
            return [convert(vv) for vv in v]
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return convert(v.item())
            return [convert(vv) for vv in v.tolist()]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if v is None or isinstance(v, (bool, numbers.Number, str)):
            return v
        return str(v)
    return convert(options)

def _hash_options(options, length=8):
    """Return a short SHA-256 hash of a canonicalized options dict."""
    import json
    import hashlib
    def _canonical(obj):
        if isinstance(obj, dict):
            return sorted((_canonical(k), _canonical(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [_canonical(x) for x in obj]
        return obj
    s = json.dumps(_canonical(_base_type_options(options)), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:length]

def _str_from_observable_options(options: dict, level: int | dict=None) -> str:
    """Return a compact string identifier for one observable options dictionary."""
    import numpy as np
    from cat_tools import get_full_tracer_zrange, get_simple_tracer, _unzip_catalog_options
    from utils import float2str

    level = _get_level(level)
    out_str = []
    catalog = _unzip_catalog_options(options['catalog'])

    def _str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer, catalog_options in catalog.items():
            stracer = get_simple_tracer(tracer)
            found = False
            if 'zrange' in catalog_options:
                for tracerz, zrange in translate_tracerz.items():
                    if tracerz.startswith(stracer) and np.allclose(catalog_options['zrange'], zrange):
                        stracer = tracerz
                        found = True
                        break
            tracer_catalog_str = [stracer]
            if 'zrange' in catalog_options:
                if not found or level['catalog'] >= 2:
                    tracer_catalog_str.append(_str_zrange(catalog_options['zrange']))
            elif 'zsnap' in catalog_options:
                tracer_catalog_str.append(f'z{float2str(catalog_options["zsnap"], prec_min=1, prec_max=5)}')
            if level['catalog'] >= 3 and 'region' in catalog_options:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4 and 'weight' in catalog_options:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            catalog_str.append('-'.join(item for item in tracer_catalog_str if item))
        out_str.append('x'.join(catalog_str))

    translate_stat_name = {'S2': ['mesh2_spectrum'],
                           'S3': ['mesh3_spectrum'],
                           'BAOR': ['bao', 'recon'],
                           'C2R': ['particle2_correlation', 'recon']}
    stat_options = options['stat']
    stat = stat_options['kind']
    if level['stat'] >= 1:
        short_name = None
        for name, tags in translate_stat_name.items():
            if all(tag in stat for tag in tags):
                short_name = name
                break
        if short_name is None:
            raise ValueError(f'could not find short name for {stat}')
        out_str.append(short_name)
    if level['stat'] >= 2:
        select_str = []
        select = stat_options.get('select', [])
        if callable(select):
            select_str.append(getattr(select, 'name', 'custom'))
        else:
            def _str_ell(ell):
                if isinstance(ell, (list, tuple)):
                    return ''.join(str(item) for item in ell)
                return str(ell)

            for item in select:
                item = dict(item)
                label = []
                if 'ells' in item:
                    label.append('ell' + _str_ell(item.pop('ells')))
                for coord_name, limits in item.items():
                    prec = dict(prec_min=2, prec_max=3) if short_name and short_name.startswith('S') else dict(prec_min=0, prec_max=0)
                    label.append(coord_name + '-'.join(float2str(lim, **prec) for lim in limits))
                select_str.append('-'.join(label))
        out_str.append('-'.join(select_str))

    if level['theory'] > 0:
        out_str.extend(['th', options['theory']['model']])
    return '-'.join(item for item in out_str if item)


def str_from_likelihood_options(likelihood_options, level: int | dict=None):
    """Return a compact string identifier for likelihood options."""
    level = _get_level(level)
    out_str = [_str_from_observable_options(options, level=level)
               for options in likelihood_options['observables']]
    if level['covariance'] > 0:
        covariance = likelihood_options.get('covariance', {}) or {}
        covariance_str = ['cov-' + '-'.join([covariance.get('source', 'none'),
                                             covariance.get('version', 'none')])]
        if level['covariance'] >= 3:
            corrections = covariance.get('corrections', None)
            if isinstance(corrections, str):
                corrections = [corrections]
            corrections = sorted(str(corr).lower() for corr in (corrections or []))
            if corrections:
                covariance_str.append('corr-' + '-'.join(corrections))
            nparams = covariance.get('nparams', None)
            if nparams is not None:
                covariance_str.append(f'nparams-{int(nparams)}')
        out_str.append('-'.join(covariance_str))
    return '+'.join(out_str)


def str_from_cosmology_options(cosmology_options: dict, level: int | dict=None):
    """Return a compact string identifier for cosmology options."""
    level = _get_level(level)
    out_str = []
    if level['cosmology'] >= 1:
        model, template = cosmology_options['model'], cosmology_options['template']
        out_str.append(f'cosmo-{model}' if template.lower() == 'direct' else f'template-{template}')
    return '-'.join(out_str)


def str_from_options(options: dict, level: int | dict=None):
    """Return a compact string identifier for full fitting options."""
    level = _get_level(level)
    out_str = [str_from_cosmology_options(options['cosmology'], level=level)]
    out_str += [str_from_likelihood_options(likelihood_options, level=level)
                for likelihood_options in options['likelihoods']]
    return '_'.join(out_str)
