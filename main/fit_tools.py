import os
import sys
import logging
import copy
from pathlib import Path
import numpy as np
import lsstypes as types

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from fit_support import (fill_fiducial_options, get_default_theory_nuisance_priors, _hash_options,
                         str_from_cosmology_options, str_from_likelihood_options,
                         str_from_observable_options, str_from_options)
from cat_tools import get_measurement_fn

logger = logging.getLogger(__name__)


def default_mpicomm(func):
    """Attach MPI.COMM_WORLD when a caller does not provide ``mpicomm``."""
    def wrapper(*args, **kwargs):
        if kwargs.get('mpicomm', None) is None:
            try:
                from mpi4py import MPI
                kwargs['mpicomm'] = MPI.COMM_WORLD
            except ImportError:
                kwargs['mpicomm'] = None
        return func(*args, **kwargs)
    return wrapper


def load_bins(corr_type, bins_type='test'):
    """Return default scale ranges and binning used in diagnostic readers."""
    if corr_type == 'xi':
        if bins_type in ['test', 'y3_sys']:
            return (20, 200, 4, 45)
        if bins_type == 'y3_bao':
            return (60, 150, 4, 23)
    elif corr_type in ['pk', 'mesh2']:
        if bins_type in ['y3_bao', 'test', 'y3_sys']:
            return (0.02, 0.3, 0.005, 56)
        if bins_type == 'y3_fs':
            return (0.02, 0.2, 0.005, 36)
        if bins_type == 'test_covbox':
            return (0.03, 0.2, 0.005, 34)
    elif corr_type in ['mpslog', 'wplog']:
        if bins_type in ['test', 'y3_sys']:
            return (0.10, 30, None, None)
    elif corr_type == 'mesh3_sugiyama':
        if bins_type in ['test', 'y3_sys']:
            return (0, 0.2, 0.01, 20)
    elif corr_type == 'mesh3_scoccimarro':
        return (None, None, None, None)
    raise ValueError(f"Invalid corr_type {corr_type!r} or bins_type {bins_type!r}.")


def read_data_from_fn(fn, corr_type, bin_type='test', ells=(0, 2), remove_bins=None, verbose=False):
    """Read one clustering measurement for notebook diagnostics."""
    minimum, maximum, step, length = load_bins(corr_type, bin_type)
    bin_set = (minimum, maximum, step, length)
    if corr_type in ['xi', 'mpslog', 'wplog']:
        from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp
        file_kind = 'xipoles' if corr_type == 'xi' else corr_type
        result = TwoPointCorrelationFunction.load(fn.format(file_kind))
        result = result[::step, :]
        result.select((minimum, maximum))
        if file_kind in ['xipoles', 'mpslog']:
            separation, values = project_to_multipoles(result, ells=ells)
        else:
            separation, values = project_to_wp(result)
        if remove_bins is not None:
            values = np.atleast_2d(values)
            keep = np.ones(len(separation), dtype=bool)
            keep[:remove_bins] = False
            separation, values = separation[keep], values[:, keep]
        return (separation, values), bin_set
    if corr_type == 'pk':
        from pypower import PowerSpectrumMultipoles
        result = PowerSpectrumMultipoles.load(fn.format('pkpoles')).select((minimum, maximum, step))
        return (result.kavg, np.real(result.get_power())), bin_set
    if corr_type == 'mesh2':
        try:
            result = types.read(fn.format('mesh2_spectrum_poles'))
        except Exception as exc:
            logger.warning(f"Failed to read {fn.format('mesh2_spectrum_poles')}: {exc}")
            return (None, None), bin_set
        if verbose:
            logger.info(f"Read {fn.format('mesh2_spectrum_poles')}")
        result = result.select(k=slice(0, None, 5)).select(k=(minimum, maximum))
        k = result.get(ells=0).coords('k')
        return (k, [result.get(ells=ell).values()['value'] for ell in result.ells]), bin_set
    if corr_type == 'mesh3_scoccimarro':
        result = types.read(fn.format('mesh3_spectrum_poles_scoccimarro'))
        k1, k2, k3 = np.asarray(result.get(ells=0).coords('k')).T
        return ((k1, k2, k3), [result.get(ells=ell).values()['value'] for ell in result.ells]), bin_set
    if corr_type == 'mesh3_sugiyama':
        result = types.read(fn.format('mesh3_spectrum_poles_sugiyama'))
        if verbose:
            logger.info(f"Read {fn.format('mesh3_spectrum_poles_sugiyama')}")
        result = result.select(k=slice(0, None, 1)).select(k=(minimum, maximum))
        k = result.get(ells=(0, 0, 0)).coords('k')[:, 1]
        return (k, [result.get(ells=ell).values()['value'] for ell in result.ells]), bin_set
    raise ValueError(f'{corr_type} not available')

class LikelihoodBuilder:
    """
    Organize DESI likelihood options and build data/theory/likelihood objects.

    The canonical option shape is
    ``options -> likelihoods -> observables -> {stat, catalog, theory, emulator, window}``.
    This builder accepts shorter inputs, fills the defaults from ``fit_support``,
    and keeps the unfilled options in ``raw_options`` for easy editing.
    """
    def __init__(self, options=None, likelihoods=None, observables=None, stats=None, catalog=None,
                 covariance=None, cache_dir=None, cosmology=None, sampler=None, profiler=None,
                 fill=True):
        self.fill = fill
        self.raw_options = self._build_options(options=options, likelihoods=likelihoods,
                                               observables=observables, stats=stats,
                                               catalog=catalog, covariance=covariance,
                                               cosmology=cosmology, sampler=sampler,
                                               profiler=profiler)
        self.cache_dir = cache_dir
        self.refill()

    def __repr__(self):
        stats = [obs['stat']['kind'] for obs in self.observables]
        return f'{type(self).__name__}(nlikelihoods={len(self.likelihoods)}, stats={stats})'

    def __len__(self):
        return len(self.likelihoods)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _deep_update(base, updates):
        base = copy.deepcopy(base)
        for name, value in (updates or {}).items():
            if isinstance(value, dict) and isinstance(base.get(name, None), dict):
                base[name] = LikelihoodBuilder._deep_update(base[name], value)
            else:
                base[name] = copy.deepcopy(value)
        return base

    @classmethod
    def _normalize_observable(cls, observable=None, stat='mesh2', catalog=None,
                              theory=None, emulator=None, window=None):
        if observable is None:
            observable = {}
        elif isinstance(observable, str):
            observable = {'stat': {'kind': observable}}
        else:
            observable = copy.deepcopy(observable)

        stat_options = observable.get('stat', stat)
        if isinstance(stat_options, str):
            stat_options = {'kind': stat_options}
        observable['stat'] = copy.deepcopy(stat_options)
        observable.setdefault('catalog', {})
        for name, value in [('catalog', catalog), ('theory', theory),
                            ('emulator', emulator), ('window', window)]:
            if value is not None:
                observable[name] = cls._deep_update(observable.get(name, {}), value)
        return observable

    @classmethod
    def _normalize_likelihood(cls, likelihood=None, observables=None, stats=None,
                              catalog=None, covariance=None):
        likelihood = copy.deepcopy(likelihood or {})
        if observables is None:
            observables = likelihood.get('observables', None)
        if observables is None:
            observables = [cls._normalize_observable(stat=stat, catalog=catalog)
                           for stat in cls._as_list(stats or 'mesh2')]
        else:
            observables = [cls._normalize_observable(observable=observable, catalog=catalog)
                           for observable in cls._as_list(observables)]
        likelihood['observables'] = observables
        if covariance is not None:
            likelihood['covariance'] = cls._deep_update(likelihood.get('covariance', {}), covariance)
        return likelihood

    @classmethod
    def _normalize_options(cls, options):
        options = copy.deepcopy(options or {})
        likelihoods = options.get('likelihoods', [])
        likelihoods = [cls._normalize_likelihood(likelihood)
                       for likelihood in cls._as_list(likelihoods)]
        if likelihoods:
            options['likelihoods'] = likelihoods
        return options

    def _build_options(self, options=None, likelihoods=None, observables=None, stats=None,
                       catalog=None, covariance=None, cosmology=None, sampler=None, profiler=None):
        if options is not None:
            base = self._normalize_options(options)
        elif likelihoods is not None:
            base = {'likelihoods': [self._normalize_likelihood(likelihood)
                                    for likelihood in self._as_list(likelihoods)]}
        else:
            base = {'likelihoods': [self._normalize_likelihood(observables=observables, stats=stats,
                                                               catalog=catalog, covariance=covariance)]}
        for name, value in [('cosmology', cosmology), ('sampler', sampler), ('profiler', profiler)]:
            if value is not None:
                base[name] = copy.deepcopy(value)
        return base

    @property
    def likelihoods(self):
        return self.options.get('likelihoods', [])

    @property
    def observables(self):
        return [observable for likelihood in self.likelihoods
                for observable in likelihood.get('observables', [])]

    @property
    def cosmology_options(self):
        return self.options.get('cosmology', {})

    def refill(self):
        """Re-apply fiducial defaults after edits to ``raw_options``."""
        self.options = fill_fiducial_options(self.raw_options) if self.fill else copy.deepcopy(self.raw_options)
        return self

    def update(self, **updates):
        """Deep-update top-level raw options, then re-fill defaults."""
        self.raw_options = self._deep_update(self.raw_options, updates)
        return self.refill()

    def add_likelihood(self, observables=None, stats=None, catalog=None, covariance=None):
        """Append one likelihood block to the raw options."""
        likelihood = self._normalize_likelihood(observables=observables, stats=stats,
                                                catalog=catalog, covariance=covariance)
        self.raw_options.setdefault('likelihoods', []).append(likelihood)
        return self.refill()

    def add_observable(self, stat='mesh2', catalog=None, theory=None,
                       emulator=None, window=None, likelihood_index=0):
        """Append one observable to an existing likelihood block."""
        while len(self.raw_options.setdefault('likelihoods', [])) <= likelihood_index:
            self.raw_options['likelihoods'].append({'observables': []})
        observable = self._normalize_observable(stat=stat, catalog=catalog, theory=theory,
                                                emulator=emulator, window=window)
        self.raw_options['likelihoods'][likelihood_index].setdefault('observables', []).append(observable)
        return self.refill()

    def get_likelihood_options(self, likelihood=0, filled=True):
        """Return one likelihood options dictionary."""
        options = self.options if filled else self.raw_options
        return options['likelihoods'][likelihood]

    def get_observable_options(self, observable=0, likelihood=0, filled=True):
        """Return one observable options dictionary."""
        return self.get_likelihood_options(likelihood=likelihood, filled=filled)['observables'][observable]

    def get_cosmology(self):
        """Build the desilike cosmology calculator from organized options."""
        return get_cosmology(self.cosmology_options)

    def get_redshift(self, observable=0, likelihood=0, data_attrs=None):
        """Return the effective redshift for one observable."""
        data_attrs = dict(data_attrs or {})
        z = data_attrs.get('z', data_attrs.get('zeff', None))
        if z is not None:
            return float(z)
        observable_options = self.get_observable_options(observable=observable, likelihood=likelihood)
        stat_options = observable_options['stat']
        window_kind = stat_to_kind(stat_options['kind'], window=True, basis=stat_options.get('basis', None))
        return get_effective_redshift(observable_options.get('catalog', {}), kind=window_kind)

    def get_theory(self, observable=0, likelihood=0, cosmology=None, data_attrs=None, data=None):
        """Build the theory calculator for one organized observable."""
        observable_options = self.get_observable_options(observable=observable, likelihood=likelihood)
        if cosmology is None:
            cosmology = self.get_cosmology()
        if data_attrs is None:
            data_attrs = {}
        else:
            data_attrs = dict(data_attrs)
        if data_attrs.get('z', None) is None:
            data_attrs['z'] = self.get_redshift(observable=observable, likelihood=likelihood,
                                                data_attrs=data_attrs)
        return get_theory(observable_options['stat']['kind'],
                          theory_options=observable_options['theory'],
                          cosmology=cosmology, data_attrs=data_attrs, data=data)

    def get_data_stats(self, likelihood=0, covariance_options=None, unpack=False, cache_mode='rw'):
        """Load data/window/covariance for one likelihood block."""
        likelihood_options = self.get_likelihood_options(likelihood=likelihood)
        covariance = copy.deepcopy(likelihood_options.get('covariance', {}))
        if covariance_options is not None:
            covariance = self._deep_update(covariance, covariance_options)
        return get_data_stats(likelihood_options['observables'], covariance_options=covariance,
                              unpack=unpack, cache_dir=self.cache_dir, cache_mode=cache_mode)

    def get_all_data_stats(self, covariance_options=None, unpack=False,cache_mode='rw'):
        """Load data/window/covariance for every likelihood block."""
        return [self.get_data_stats(likelihood=i, covariance_options=covariance_options,
                                    unpack=unpack, cache_mode=cache_mode)
                for i in range(len(self.likelihoods))]

    def get_likelihood(self, stats=None, cache_mode='rw'):
        """Build one desilike likelihood, or a list when multiple blocks exist."""
        cosmology = self.get_cosmology()
        if len(self.likelihoods) == 1:
            return get_likelihood(self.likelihoods[0], stats=stats, cosmology_options=cosmology,
                                  cache_dir=self.cache_dir, cache_mode=cache_mode)
        return [get_likelihood(likelihood, stats=None if stats is None else stats[i],
                               cosmology_options=cosmology, cache_dir=self.cache_dir, cache_mode=cache_mode)
                for i, likelihood in enumerate(self.likelihoods)]

    def get_fits_fn(self, likelihood=None, filled=True, **kwargs):
        """Return the default filename for saving fits results."""
        options = copy.deepcopy(self.options if filled else self.raw_options)
        if likelihood is not None:
            options['likelihoods'] = [options['likelihoods'][likelihood]]
        return get_fits_fn(options=options, fill=False, **kwargs)

DESIlikelihood = LikelihoodBuilder


############ Internal functions for building likelihood components from options ############

_fiducial = None
def get_fiducial():
    """Return the cached fiducial cosmology used by the fit models."""
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial

def get_cosmology(cosmology_options: dict=None):
    """
    Construct and return a :mod:`desilike` :class:`Cosmoprimo` calculator.

    Returns
    -------
    cosmo : :class:`desilike.theories.Cosmoprimo`
        Instance with configured priors.
    """
    from desilike.theories import Cosmoprimo
    if isinstance(cosmology_options, Cosmoprimo):
        return cosmology_options
    cosmology_options = cosmology_options or {}
    model = cosmology_options.get('model', 'base_ns-fixed')
    cosmo = Cosmoprimo(engine='class', fiducial=get_fiducial())
    is_fixed_model = model == 'fixed'
    # Free parameters h, omega_cdm, omega_b, logA with uniform priors
    # n_s and tau_reio are fixed
    # A Gaussian prior on omega_b.
    params = {
        'H0':       {'derived': True},
        'Omega_m':  {'derived': True},
        'sigma8_m': {'derived': True},
        'tau_reio': {'fixed': True},
        'n_s':      {'fixed': is_fixed_model or 'ns-fixed' in model},
        'omega_b':  {'fixed': is_fixed_model, 'prior': {'dist': 'norm', 'loc': 0.02237,  'scale': 0.00037}},
        'h':        {'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [0.5,  0.9]}},
        'omega_cdm':{'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [0.05, 0.2]}},
        'logA':     {'fixed': is_fixed_model, 'prior': {'dist': 'uniform', 'limits': [2.0,  4.0]}},
    }
    if 'w0wa' in model:
        params['w0_fld'] = {'fixed': is_fixed_model}
        params['wa_fld'] = {'fixed': is_fixed_model}
    for name, config in params.items():
        if config.get('fixed', False):
            config = {key: value for key, value in config.items() if key != 'prior'}
        if name in cosmo.init.params:
            cosmo.init.params[name].update(**config)
        else:
            cosmo.init.params[name] = config
    return cosmo

def _drop_prior_limits(config):
    """Return a copy of config with prior limits removed."""
    config = copy.deepcopy(config)
    prior = config.get('prior', None)
    if isinstance(prior, dict):
        prior.pop('limits', None)
    return config

def get_theory(stat: str, theory_options: dict, cosmology: object=None, data_attrs: dict=None, data=None):
    """
    Return a configured theory desilike calculator for the requested statistic.

    Parameters
    ----------
    stat : str
        Statistic name, e.g. 'mesh2' or 'mesh3_sugiyama'.
    theory_options : dict
        Theory options dict containing at least 'model' and possibly other keys.
    cosmology : Cosmoprimo
        Cosmology calculator.
    data_attrs : dict
        Data attributes ('z', 'recon_mode', 'recon_smoothing_radius', 'tracers', ...).
    Returns
    -------
    theory : BaseCalculator
        Initialized theory object from desilike for the requested statistic.
    """
    from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, BAOPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)
    theory_options = dict(theory_options)
    fiducial = get_fiducial()
    template = None
    theory_options.setdefault('cosmology', {'template': 'direct'})
    if theory_options.get('model') in ['folpsD', 'folpsEFT']:
        theory_options.setdefault('damping', 'lor')
        theory_options.setdefault('prior_basis', 'physical_aap')
        theory_options.setdefault('marg', True)
        if 'mesh2' in stat:
            theory_options.setdefault('b3_coev', True)
            theory_options.setdefault('A_full', False)
        elif 'mesh3' in stat:
            theory_options.setdefault('A_full', False)
    cosmology_options = theory_options['cosmology']
    z = data_attrs.get('z', data_attrs.get('zeff', None))
    if z is None:
        raise KeyError('No effective redshift found; pass data_attrs["z"] or read it from the window with get_effective_redshift().')
    if cosmology_options['template'] == 'direct':
        template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmology, z=z)
    elif cosmology_options['template'] == 'shapefit':
        template = ShapeFitPowerSpectrumTemplate(fiducial=fiducial, z=z)
    elif cosmology_options['template'] == 'bao':
        kw = {name: cosmology_options[name] for name in ['apmode', 'now'] if name in cosmology_options}
        template = BAOPowerSpectrumTemplate(fiducial=fiducial, z=z, **kw)
    if template is None:
        raise ValueError(f'template not found for {stat} and {repr(cosmology_options["template"])}')
    theory = None
    if 'mesh2' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis', 'b3_coev', 'A_full']}
            theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                if theory_options['marg'] and (name.startswith('alpha') or name.startswith('sn')):
                    config = _drop_prior_limits(config)
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
            if theory_options['marg']:
                for param in theory.init.params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
    elif 'mesh3' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis']}
            theory = FOLPSv2TracerBispectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory

def get_effective_redshift_from_window(window, fn=None):
    """Return effective redshift metadata encoded in a window function."""
    mono = window.theory.get(ells=0)
    zeff = getattr(mono, 'z', None)
    if zeff is None:
        zeff = getattr(mono, '_meta', {}).get('z', None)
    if zeff is None:
        location = f' {fn}' if fn is not None else ''
        raise AttributeError(f'No z_eff found in window function{location}')
    return float(zeff)


def get_effective_redshift(args=None, kind='window_mesh2_spectrum_poles',
                           get_measurement_fn=get_measurement_fn, **kwargs):
    """Read effective redshift from a window-function measurement file."""
    args = dict(args or {})
    args.update(kwargs)
    kind = args.pop('kind', kind)
    fn = Path(get_measurement_fn(**args).format(kind))
    return get_effective_redshift_from_window(types.read(fn), fn=fn)

def observable_labels(observable_options):
    """Return labels for joining one observable into a multi-statistics tree."""
    stat = observable_options['stat']['kind']
    tracer = observable_options.get('catalog', {}).get('tracer', None)
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


def parameter_namespace(observable_options):
    """Return a tracer namespace independent of the data realizations averaged."""
    options = copy.deepcopy(observable_options)
    catalog = options.get('catalog', {})
    catalog.pop('mock_ids', None)
    if not np.isscalar(catalog.get('mock_id', None)):
        catalog.pop('mock_id', None)
    return str_from_observable_options(
        options, level={'catalog': 1, 'stat': 0, 'theory': 0, 'covariance': 0})


def apply_select(observable, select=None):
    """Apply lsstypes-style ell/k selections to an observable tree."""
    if select is None:
        return observable
    if callable(select):
        return select(observable)
    labels = []
    for item in select:
        item = dict(item)
        label = {key: item.pop(key) for key in observable.labels(return_type='keys') if key in item}
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

def get_covariance_correction_factor(covariance: types.CovarianceMatrix, observables: list[dict],
                                     covariance_options: dict,
                                     default_corrections=('hartlap', 'percival')):
    """Return multiplicative covariance correction factor and correction metadata."""
    from lsstypes.utils import get_hartlap2007_factor, get_percival2014_factor

    corrections = covariance_options.get('corrections', default_corrections)
    if isinstance(corrections, str):
        corrections = [corrections]
    corrections = [str(correction).lower() for correction in (corrections or [])]
    nbins = int(covariance.value().shape[0])
    nobs = covariance.attrs.get('nobs', None)
    metadata = {'nbins': nbins, 'corrections': tuple(corrections)}
    if nobs is None:
        return 1., metadata | {'corrections': tuple()}
    nobs = int(nobs)
    metadata['nobs'] = nobs
    factor = 1.
    if 'hartlap' in corrections:
        hartlap = get_hartlap2007_factor(nobs, nbins)
        factor /= hartlap
        metadata['hartlap_factor'] = float(hartlap)
    if 'percival' in corrections:
        stats = {observable['stat']['kind'] for observable in observables}
        has_mesh2 = any('mesh2' in stat for stat in stats)
        has_mesh3 = any('mesh3' in stat for stat in stats)
        nparams = covariance_options.get('nparams', 9 if has_mesh2 and has_mesh3 else 7)
        percival = get_percival2014_factor(nobs, nbins, nparams)
        factor *= percival
        metadata.update(percival_factor=float(percival), nparams=int(nparams))
    return factor, metadata

def stat_to_kind(stat, window=False, basis=None):
    """Translate an observable stat name to the stored measurement key."""
    prefix = 'window_' if window else ''
    if 'mesh2' in stat:
        return f'{prefix}mesh2_spectrum_poles'
    if 'mesh3' in stat:
        basis = 'sugiyama' if basis is None else basis
        suffix = 'sugiyama' if 'sugiyama' in basis else ('scoccimarro' if 'scoccimarro' in basis else basis)
        return f'{prefix}mesh3_spectrum_poles_{suffix}'
    return f'{prefix}{stat}'

def resolve_measurement_path(observable_options, kind, get_measurement_fn=get_measurement_fn, **overrides):
    """Resolve a stored measurement path from observable and runtime options."""
    catalog = dict(observable_options.get('catalog', {}))
    catalog.update(overrides)
    for name in ['source', 'corrections', 'covariance', 'fn', 'mock_ids', 'mockid', 'nparams',
                 'rescale', 'scale_mean_covariance']:
        catalog.pop(name, None)
    if catalog.get('version') == 'holi-v3-altmtl':
        catalog['version'] = 'holi-v3'
        catalog.setdefault('domain', 'altmtl')
    if catalog.get('version') == 'abacus-hf-dr2-v2-altmtl':
        catalog['version'] = 'AbacusHF-v2'
        catalog.setdefault('domain', 'altmtl')
    return Path(get_measurement_fn(**catalog).format(kind))

def _get_data_mock_ids(observable_options):
    """Return mock realizations to average, or ``None`` for one scalar realization."""
    catalog = observable_options.get('catalog', {})
    mock_ids = catalog.get('mock_ids', None)
    if mock_ids is None:
        mock_id = catalog.get('mock_id', None)
        if mock_id is not None and not np.isscalar(mock_id):
            mock_ids = mock_id
    if mock_ids is None:
        return None
    mock_ids = [int(mock_ids)] if isinstance(mock_ids, (int, np.integer)) else list(mock_ids)
    if not mock_ids:
        raise ValueError('catalog["mock_ids"] must contain at least one mock realization')
    return mock_ids

def _get_mean_data_mock_ids(observables_options):
    """Validate and return the common mock set used in a joint mean-data fit."""
    mock_id_sets = [_get_data_mock_ids(observable_options) for observable_options in observables_options]
    nonempty = [mock_ids for mock_ids in mock_id_sets if mock_ids is not None]
    if not nonempty:
        return None
    reference = nonempty[0]
    if any(mock_ids is None or mock_ids != reference for mock_ids in mock_id_sets):
        raise ValueError('All observables in a joint mean-data fit must use the same catalog["mock_ids"]')
    return reference


def _read_mean_observable(observable_options, kind, select=None,
                          get_measurement_fn=get_measurement_fn, rank=0):
    """Read one measurement or average measurements over ``catalog['mock_ids']``."""
    mock_ids = _get_data_mock_ids(observable_options)
    ids_to_read = mock_ids if mock_ids is not None else [None]
    measurements = []
    for mock_id in ids_to_read:
        overrides = {} if mock_id is None else {'mock_id': mock_id}
        fn = resolve_measurement_path(observable_options, kind, get_measurement_fn=get_measurement_fn, **overrides)
        if not fn.exists():
            raise FileNotFoundError(f'No data file for {observable_options["stat"]["kind"]}: {fn}')
        if rank == 0:
            logger.info(f'Reading data for {observable_options["stat"]["kind"]} from {fn}')
        measurements.append(apply_select(types.read(fn), select=select))
    datum = types.mean(measurements) if len(measurements) > 1 else measurements[0]
    if mock_ids is not None:
        datum = datum.clone(attrs=dict(datum.attrs) | {'mock_ids': np.asarray(mock_ids), 'nmocks': len(mock_ids)})
        if rank == 0 and len(mock_ids) > 1:
            logger.info(f'Averaged data for {observable_options["stat"]["kind"]} over {len(mock_ids)} mock realizations')
    return datum

def _read_window_for_data(observable_options, kind, datum,
                          get_measurement_fn=get_measurement_fn, rank=0):
    """Read one window or arithmetic-average mock-specific window matrices."""
    mock_ids = _get_data_mock_ids(observable_options)
    ids_to_read = mock_ids if mock_ids is not None else [None]
    windows = []
    for mock_id in ids_to_read:
        overrides = {} if mock_id is None else {'mock_id': mock_id}
        fn = resolve_measurement_path(observable_options, kind, get_measurement_fn=get_measurement_fn, **overrides)
        if not fn.exists():
            raise FileNotFoundError(f'No window file for {observable_options["stat"]["kind"]}: {fn}')
        if rank == 0:
            logger.info(f'Reading window for {observable_options["stat"]["kind"]} from {fn}')
        windows.append(types.read(fn).at.observable.match(datum))
    if len(windows) == 1:
        return windows[0]
    return windows[0].clone(value=np.mean([window.value() for window in windows], axis=0))


def pack_stats(stats, **labels):
    """Pack observables or windows into joint lsstypes containers."""
    if isinstance(stats[0], types.ObservableLike):
        return types.ObservableTree(stats, **labels)
    if isinstance(stats[0], types.WindowMatrix):
        import scipy as sp
        return types.WindowMatrix(
            value=sp.linalg.block_diag(*[window.value() for window in stats]),
            observable=pack_stats([window.observable for window in stats], **labels),
            theory=pack_stats([window.theory for window in stats], **labels),
        )
    raise ValueError(f'unrecognized stats type {type(stats[0])}')


def unpack_stats(stats):
    """Unpack joint lsstypes data/window/likelihood objects."""
    if isinstance(stats, types.ObservableLike):
        return stats.flatten(level=1)
    if isinstance(stats, types.WindowMatrix):
        return [stats.at.observable.get(**label).at.theory.get(**label)
                for label in stats.observable.labels(level=1)]
    if isinstance(stats, types.GaussianLikelihood):
        return unpack_stats(stats.observable), unpack_stats(stats.window), stats.covariance
    if isinstance(stats, dict):
        return stats['data'], stats['window'], stats['covariance']
    raise ValueError(f'unrecognized stats type {type(stats)}')


def _rescale_covariance_for_mean_data(covariance, mean_data_mock_ids, covariance_options, rank=0):
    """Optionally divide covariance by the number of mocks averaged in the data."""
    if (mean_data_mock_ids is None or len(mean_data_mock_ids) <= 1
            or not covariance_options.get('rescale', False)):
        return covariance
    nmocks = len(mean_data_mock_ids)
    covariance = covariance.clone(value=covariance.value() / nmocks)
    covariance.attrs.update(data_nmocks=nmocks, mean_data_covariance_factor=1. / nmocks)
    if rank == 0:
        logger.info(f'Scaled covariance by 1 / {nmocks} for mean mock data')
    return covariance


@default_mpicomm
def get_data_stats(observables_options, covariance_options=None, unpack=False,
                   get_measurement_fn=get_measurement_fn, cache_dir=None,
                   cache_mode='rw', mpicomm=None):
    """Load data, window, and covariance products for one likelihood block."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / 'prepared_stats'
    read_cache = cache_dir is not None and 'r' in cache_mode
    write_cache = cache_dir is not None and 'w' in cache_mode
    rank = getattr(mpicomm, 'rank', 0)
    covariance_options = covariance_options or {}

    def get_cache_fn(kind, kwargs):
        if cache_dir is None:
            return None
        full_options = {'observables': [{name: dict(options[name]) for name in ['stat', 'catalog']}
                                        for options in observables_options]}
        level = {'stat': 1, 'catalog': 2, 'covariance': 0}
        if kind == 'covariance':
            full_options['covariance'] = covariance_options
            level['covariance'] = 1
        label = str_from_likelihood_options(full_options, level=level)
        digest = _hash_options(full_options | kwargs | {'prepared_stats_version': 'mean-mocks-v2'})
        return cache_dir / f'{kind}_{label}-{digest}.h5'

    def get_from_cache(cache_fn):
        if cache_fn is None or not read_cache:
            return None
        exists = cache_fn.exists()
        if mpicomm is not None:
            exists = all(mpicomm.allgather(exists))
        stats = None
        if exists:
            if rank == 0:
                logger.info(f'Reading cached stats {cache_fn}.')
            stats = types.read(cache_fn)
        return stats if mpicomm is None else mpicomm.bcast(stats, root=0)

    def save_to_cache(stats, cache_fn):
        if cache_fn is None or not write_cache:
            return
        if rank == 0:
            cache_fn.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Writing cached stats {cache_fn}.')
            stats.write(cache_fn)
        if mpicomm is not None:
            mpicomm.Barrier()

    mean_data_mock_ids = _get_mean_data_mock_ids(observables_options)
    joint_labels = {'observables': [], 'tracers': []}
    for observable_options in observables_options:
        labels = observable_labels(observable_options)
        for name in joint_labels:
            joint_labels[name].append(labels[name])

    data_cache_fn, window_cache_fn = get_cache_fn('data', {}), get_cache_fn('window', {})
    data, window = get_from_cache(data_cache_fn), get_from_cache(window_cache_fn)
    if data is None or window is None:
        data_items, window_items = [], []
        cached_data_items = None if data is None else unpack_stats(data)
        for index, observable_options in enumerate(observables_options):
            stat_options = observable_options['stat']
            kind = stat_to_kind(stat_options['kind'], basis=stat_options.get('basis'))
            window_kind = stat_to_kind(stat_options['kind'], window=True, basis=stat_options.get('basis'))
            datum = (_read_mean_observable(observable_options, kind, select=stat_options.get('select'),
                                           get_measurement_fn=get_measurement_fn, rank=rank)
                     if cached_data_items is None else cached_data_items[index])
            data_items.append(datum)
            if window is None:
                window_items.append(_read_window_for_data(observable_options, window_kind, datum,
                                                          get_measurement_fn=get_measurement_fn, rank=rank))
        if data is None:
            data = pack_stats(data_items, **joint_labels)
            save_to_cache(data, data_cache_fn)
        if window is None:
            window = pack_stats(window_items, **joint_labels)
            save_to_cache(window, window_cache_fn)

    covariance_cache_fn = get_cache_fn('covariance', {})
    covariance = get_from_cache(covariance_cache_fn)
    if covariance is not None:
        likelihood = types.GaussianLikelihood(observable=data, window=window, covariance=covariance)
        return unpack_stats(likelihood) if unpack else likelihood

    covariance = covariance_options.get('covariance', None)
    if covariance is not None and rank == 0:
        logger.info(f'Using provided covariance object of type {type(covariance).__name__}')
    if covariance is None and 'fn' in covariance_options:
        covariance_fn = Path(covariance_options['fn'])
        if rank == 0:
            logger.info(f'Reading covariance from {covariance_fn}')
        covariance = types.read(covariance_fn)
    if covariance is None and covariance_options.get('source', 'mock') == 'mock':
        mock_ids = covariance_options.get('mock_ids', covariance_options.get('mockid', range(200)))
        mock_ids = range(mock_ids) if isinstance(mock_ids, int) else mock_ids
        mocks, nmissing, first_logged = [], 0, False
        for mock_id in mock_ids:
            observables, ok = [], True
            for observable_options in observables_options:
                stat_options = observable_options['stat']
                kind = stat_to_kind(stat_options['kind'], basis=stat_options.get('basis'))
                fn = resolve_measurement_path(observable_options, kind, get_measurement_fn=get_measurement_fn,
                                              mock_id=mock_id, **covariance_options)
                if not fn.exists():
                    nmissing += 1
                    ok = False
                    break
                if rank == 0 and not first_logged:
                    logger.info(f'Reading covariance, first matched: {fn}')
                    first_logged = True
                observables.append(apply_select(types.read(fn), select=stat_options.get('select')))
            if ok:
                mocks.append(types.ObservableTree(observables, **joint_labels))
        if not mocks:
            raise FileNotFoundError('No covariance mock files found with get_measurement_fn.')
        covariance = types.cov(mocks)
        covariance.attrs['nobs'] = len(mocks)
        if rank == 0:
            logger.info(f'Built covariance from {len(mocks)} mock realizations; shape={covariance.value().shape}')
            if nmissing:
                logger.warning(f'Skipped {nmissing} missing covariance mock files')
    if covariance is None:
        raise ValueError('No covariance could be constructed; provide covariance_options["fn"] or mock files.')

    covariance = covariance.at.observable.match(data)
    factor, metadata = get_covariance_correction_factor(covariance, observables_options, covariance_options)
    if factor != 1.:
        covariance = covariance.clone(value=covariance.value() * factor)
    covariance = _rescale_covariance_for_mean_data(covariance, mean_data_mock_ids, covariance_options, rank=rank)
    covariance.attrs['covariance_correction_factor'] = float(factor)
    covariance.attrs.update(metadata)
    if rank == 0 and metadata['corrections']:
        info = f"Applied covariance corrections {metadata['corrections']} with factor {factor:.6f}"
        if 'hartlap_factor' in metadata:
            info += f", hartlap={metadata['hartlap_factor']:.6f}"
        if 'percival_factor' in metadata:
            info += f", percival={metadata['percival_factor']:.6f}, nparams={metadata['nparams']}"
        logger.info(info)
    save_to_cache(covariance, covariance_cache_fn)
    likelihood = types.GaussianLikelihood(observable=data, window=window, covariance=covariance)
    return unpack_stats(likelihood) if unpack else likelihood


@default_mpicomm
def get_likelihood(likelihood_options,  stats: types.GaussianLikelihood=None,
                   cosmology_options: dict=None, get_measurement_fn=get_measurement_fn,
                   get_theory=get_theory, cache_dir:str | Path=None, cache_mode: str='rw', mpicomm=None):
    """
    Build a single :mod:`desilike` Gaussian likelihood from provided options.

    Parameters
    ----------
    likelihood_options : dict
        Options containing 'observables' list and 'covariance' dict.
    stats : dict or None
        Pre-loaded data, window, covariance dict. If None, will be loaded from files using get_measurement_fn.
    cosmology_options : optional
        Cosmology options or object or :class:`desilike.theories.Cosmoprimo`.
    get_measurement_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    ObservablesGaussianLikelihood
    """
    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, TracerSpectrum3PolesObservable, TracerCorrelation2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance_options = likelihood_options.get('covariance', {})
    cosmology = get_cosmology(cosmology_options)
    if stats is None:
        stats = get_data_stats(observables_options, covariance_options=covariance_options, unpack=False, get_measurement_fn=get_measurement_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    data, window, covariance = unpack_stats(stats)
    labels = covariance.observable.labels(level=1)
    observables = []
    for observable_options, data, window, label in zip(observables_options, data, window, labels, strict=True):
        stat = observable_options['stat']['kind']
        if 'mesh2' in stat:
            cls = TracerSpectrum2PolesObservable
        elif 'mesh3' in stat:
            cls = TracerSpectrum3PolesObservable
        elif 'particle2_correlation' in stat:
            cls = TracerCorrelation2PolesObservable
        else:
            raise NotImplementedError(stat)
        data_attrs = dict(data.attrs) | label
        if data_attrs.get('z', None) is None:
            zeff = get_effective_redshift_from_window(window)
            data_attrs['z'] = zeff
            logger.warning(f'No redshift found in data attributes; read z={zeff:.3f} from the window.')
        theory = get_theory(stat, theory_options=observable_options['theory'], cosmology=cosmology, data_attrs=data_attrs, data=data)
        namespace = parameter_namespace(observable_options)
        theory_params = theory.init.params
        observable = cls(data=data, window=window, theory=theory)
        observable()
        if observable_options['emulator']['name']:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            read_cache = cache_dir is not None and 'r' in cache_mode
            write_cache = cache_dir is not None and 'w' in cache_mode
            cache_dir = Path(cache_dir)
            _hash = _hash_options({name: observable_options[name] for name in ['theory', 'catalog']})
            _str_cosmology = str_from_cosmology_options(observable_options['theory']['cosmology'], level=100)
            _str_cosmology += '_' + observable_options['emulator']['name']
            _str_theory = str_from_observable_options(observable_options, level={'theory': 100, 'catalog': 2})
            cache_fn = cache_dir / f'emulator_{_str_cosmology}' / f'emulator_{_str_theory}_{_hash}.npy'
            from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
            emulated_pt = None
            if read_cache and cache_fn.exists():
                logger.info(f'Reading cached emulator {cache_fn}')
                emulated_pt = EmulatedCalculator.load(cache_fn)
            else:
                logger.info(f'Fitting emulator {cache_fn}')
                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=observable_options['emulator'].get('order', 3)),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                if write_cache:
                    emulated_pt.save(cache_fn)
            theory.init.update(pt=emulated_pt)
            theory.init.params.update(theory_params)
            for param in theory.init.params:
                param.update(namespace=namespace)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables, covariance=covariance.value())


def get_fits_fn(fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits', project='', kind='chain',
                options: dict=None, likelihoods: list=None, observables=None, stats=None,
                catalog=None, covariance=None, sampler: dict=None, profiler: dict=None,
                cosmology: dict=None, ichain: int=None, level=None, extra='', ext='npy',
                fill=True):
    """
    Construct a file path for fit outputs based on likelihood and run options.
    Returns
    -------
    fn : Path
        Fit file name.
    """
    fits_dir = Path(fits_dir)
    if project:
        fits_dir = fits_dir / project
    if options is None:
        options = LikelihoodBuilder._build_options(options=None, likelihoods=likelihoods,
                                                  observables=observables, stats=stats,
                                                  catalog=catalog, covariance=covariance,
                                                  cosmology=cosmology, sampler=sampler,
                                                  profiler=profiler)
    else:
        options = LikelihoodBuilder._normalize_options(options)
        for name, value in [('cosmology', cosmology), ('sampler', sampler), ('profiler', profiler)]:
            if value is not None:
                options[name] = copy.deepcopy(value)
    if fill:
        options = fill_fiducial_options(options)

    _str_from_options = str_from_options(options, level=level)
    _hash = _hash_options(options)
    max_component_len = 180
    if len(_str_from_options) > max_component_len:
        _str_from_options = _str_from_options[:max_component_len].rstrip('_+-')
    if not _str_from_options:
        _str_from_options = 'fit'
    extra = f'_{extra}' if extra else ''
    ichain = f'_{ichain:d}' if ichain is not None else ''
    return fits_dir / f'{_str_from_options}-{_hash}{extra}' / f'{kind}{ichain}.{ext}'
