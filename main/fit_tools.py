import os
import sys
import logging
import copy
from pathlib import Path
import numpy as np
import lsstypes as types

sys.path.append(str(Path(__file__).resolve().parent))
from fit_support import (fill_fiducial_options, get_default_cosmology_priors, update_theory_nuisance_priors, _hash_options,
                         str_from_cosmology_options, str_from_likelihood_options, str_from_observable_options, str_from_options,
                         normalize_catalog_options, propose_fiducial_sampler_options, make_redshift_smearing)
from cat_tools import get_measurement_fn, get_measurement_ready_fn

logger = logging.getLogger(__name__)

def get_sampler_cls(name='emcee'):
    """Return a desilike sampler class from a compact sampler name."""
    name = name.lower()
    if name == 'emcee':
        from desilike.samplers.emcee import EmceeSampler
        return EmceeSampler
    if name == 'mcmc':
        from desilike.samplers.mcmc import MCMCSampler
        return MCMCSampler
    if name == 'nautilus':
        from desilike.samplers.nautilus import NautilusSampler
        return NautilusSampler
    if name == 'pocomc':
        try:
            from desilike.samplers.pocomc import PocoMCSampler
            return PocoMCSampler
        except ImportError:
            from desilike.base import Posterior, build, copy as desilike_copy, get_params
            from desilike.samples import Profiles
            from desilike.samplers import AffineConditioner, PocoMC, Sampler

            class PocoMCSampler:
                """Compatibility wrapper for desilike refactor-jax PocoMC."""

                @staticmethod
                def check_resume(save_fn):
                    """Require the population checkpoints used by the current API."""
                    for index, fn in enumerate(save_fn):
                        path = Path(fn)
                        suffix = path.stem.rsplit('_', 1)[-1]
                        sample_id = int(suffix) if suffix.isdigit() else index
                        checkpoint_dir = path.parent / 'samples' / f'samples_{sample_id}'
                        if not any(checkpoint_dir.glob('pmc_*.state')):
                            raise FileNotFoundError(f'Cannot resume: no pmc_*.state in {checkpoint_dir}')

                def __init__(self, likelihood, save_fn=None, chains=None, rng=None,
                             seed=None, batch_size=None, rescale=None, profiles_fn=None,
                             **kernel_options):
                    if save_fn is None:
                        save_fn = chains
                    if rng is None:
                        rng = seed
                    nparallel, output_dir = 1, None
                    if save_fn is not None:
                        save_paths = [Path(fn) for fn in save_fn] if isinstance(save_fn, (list, tuple)) else [Path(save_fn)]
                        sample_ids = []
                        for index, save_path in enumerate(save_paths):
                            suffix = save_path.stem.rsplit('_', 1)[-1]
                            sample_ids.append(int(suffix) if suffix.isdigit() else index)
                        nparallel = sample_ids
                        output_dir = save_paths[0].parent / 'samples'
                    covariance = None
                    likelihood_sampler = desilike_copy(likelihood)
                    if rescale and profiles_fn is not None:
                        profiles_fn = Path(profiles_fn)
                        if profiles_fn.exists():
                            profiles = Profiles.read(profiles_fn).choice(index='argmax', squeeze=True)
                            if profiles.error is None:
                                logger.warning(f'Error is not provided in {profiles_fn}; skipping profile rescaling.')
                            else:
                                best, error, covariance = profiles.best, profiles.error, profiles.covariance
                                for param in get_params(likelihood_sampler):
                                    if param.name in error:
                                        param.update(ref=dict(dist='norm', loc=best[param.name], scale=error[param.name]))
                        else:
                            logger.warning(f'Profile file {profiles_fn} not found; sampling without profile rescaling.')
                    conditioner = None if rescale is None else AffineConditioner(rescale=rescale, covariance=covariance)
                    posterior = build(Posterior(likelihood_sampler, prior=get_prior(likelihood_sampler)))
                    self._sampler = Sampler(posterior, kernel=PocoMC(**kernel_options),
                                            nparallel=nparallel, rng=rng,
                                            output_dir=output_dir, conditioner=conditioner,
                                            batch_size=batch_size)

                def run(self, **kwargs):
                    return self._sampler.run(**kwargs)

            return PocoMCSampler
    raise ValueError(f'Unknown sampler {name!r}')


def get_prior(likelihood):
    """Return the desilike prior object used to build a refactor-jax posterior."""
    import jax.numpy as jnp
    from desilike import Prior, get_params

    class CustomPrior(Prior):
        """Hard constraint w0 + wa < 0, on top of individual parameter priors."""

        def __call__(self):
            self.logpdf = super().__call__()
            if 'w0_fld' in self.params:
                w0, wa = self.params['w0_fld'], self.params['wa_fld']
                self.logpdf = jnp.where(w0.value + wa.value < 0., self.logpdf, -jnp.inf)
            return self.logpdf

    return CustomPrior(get_params(likelihood))


def get_chain_sampler_options(sampler='emcee', stats=None, overrides=None):
    """
    Return sampler init/run options for chain production.

    The defaults are kept in ``fit_support.propose_fiducial_sampler_options`` so
    notebook builders and command-line runs use the same chain settings.
    """
    options = propose_fiducial_sampler_options(sampler=sampler)
    if overrides:
        options = LikelihoodBuilder._deep_update(options, overrides)
    init_options = dict(options.get('init', {}))
    run_options = dict(options.get('run', {}))
    if sampler == 'pocomc' and stats is not None:
        run_options.setdefault('progress', True)
    if sampler == 'emcee':
        run_options.setdefault('max_iterations', 20000)
    return options, init_options, run_options

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

def load_bins(corr_type, bin_type='test'):
    """Return default scale ranges and binning used in diagnostic readers."""
    if corr_type == 'xi':
        if bin_type in ['test', 'y3_sys']:
            return (20, 200, 4, 45)
        if bin_type == 'y3_bao':
            return (60, 150, 4, 23)
    elif corr_type in ['pk', 'mesh2']:
        if bin_type in ['y3_bao', 'y3_fs']:
            return (0.02, 0.3, 0.005, 56)
        if bin_type == 'test':
            return (0.02, 0.35, 0.01, None)
        if bin_type in ['folpsD', 'y3_sys']:
            return (0.02, 0.35, 0.01, 32)
        if bin_type == 'test_covbox':
            return (0.03, 0.2, 0.005, 34)
        if bin_type == 'kmax0.4':
            return (0.02, 0.4, 0.005, 38)
    elif corr_type in ['mpslog']:
        if bin_type in ['test', 'y3_sys']:
            return (0.10, 30, None, None)
    elif corr_type in ['wplog']:
        if bin_type in ['test', 'y3_sys']:
            return (0.01, 30, None, None)
    elif corr_type in ['mesh3_sugiyama', 'mesh3']:
        if bin_type in ['y3_sys', 'folpsD']:
            return (0.02, 0.2, 0.01, 18)
        if bin_type in ['test']:
            return (0.02, 0.2, 0.005, 20)
        
    elif corr_type == 'mesh3_scoccimarro':
        return (None, None, None, None)
    raise ValueError(f"Invalid corr_type {corr_type!r} or bin_type {bin_type!r}.")

def read_data_from_fn(fn, corr_type, bin_type='test', ells=(0, 2, 4), verbose=False, **kwargs):
    """Read one clustering measurement for notebook diagnostics.

    For mesh statistics, return_observable=True returns (lsstypes object, bins),
    preserving bin edges and weights for native covariance matching.
    """
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
        remove_bins = kwargs.get('remove_bins', None)
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
        bin = kwargs.get('bin', int(step*1000))
        result = result.select(k=slice(0, None, bin)).select(k=(minimum, maximum))
        if kwargs.get('return_observable', False):
            return result, bin_set
        k = result.get(ells=0).coords('k')
        return (k, [result.get(ells=ell).values()['value'] for ell in result.ells]), bin_set
    if corr_type == 'mesh3_scoccimarro':
        result = types.read(fn.format('mesh3_spectrum_poles_scoccimarro'))
        k1, k2, k3 = np.asarray(result.get(ells=0).coords('k')).T
        return ((k1, k2, k3), [result.get(ells=ell).values()['value'] for ell in result.ells]), bin_set
    if corr_type in ['mesh3_sugiyama', 'mesh3']:
        result = types.read(fn.format('mesh3_spectrum_poles_sugiyama'))
        if verbose:
            logger.info(f"Read {fn.format('mesh3_spectrum_poles_sugiyama')}")
        kbin = kwargs.get('kbin', 1)
        result = result.select(k=slice(0, None, kbin)).select(k=(minimum, maximum))
        if kwargs.get('return_observable', False):
            return result, bin_set
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
        observable['catalog'] = normalize_catalog_options(observable['catalog'])
        # Mock bispectra use ALL; observed P and B both use GCcomb.
        if ('mesh3' in stat_options['kind']
                and observable['catalog'].get('region') == 'GCcomb'
                and observable['catalog'].get('domain') not in ['cubic', 'obs']):
            observable['catalog']['region'] = 'ALL'
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
        catalog = observable_options.get('catalog', {})
        z = get_catalog_redshift(catalog)
        if z is not None:
            return z
        stat_options = observable_options['stat']
        window_kind = stat_to_kind(stat_options['kind'], window=True, basis=stat_options.get('basis', None),
                                   version=catalog.get('version'))
        return get_effective_redshift(catalog, kind=window_kind)

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
    from desilike import Parameter
    from desilike.theories import CosmoprimoCosmology
    if isinstance(cosmology_options, CosmoprimoCosmology):
        return cosmology_options
    cosmology_options = cosmology_options or {}
    engine = cosmology_options.get('engine', 'class')
    fiducial = get_fiducial()
    params = CosmoprimoCosmology.propose_params(engine=engine, fiducial=fiducial)
    for name, config in get_default_cosmology_priors(cosmology_options).items():
        if name in params:
            params[name].update(**config)
        else:
            params.set(Parameter(name, **config))
    cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial, params=params)
    return cosmo

def check_comet_data():
    """Check COMET's tables without triggering its import-time download."""
    from importlib.util import find_spec

    spec = find_spec('comet')
    if spec is None:
        raise ImportError('COMET fits require comet-emu in the fit environment.')
    data_dir = Path(os.environ.get('COMET_DATA_DIR') or Path(spec.origin).parent / 'data_dir')
    missing = [name for name in ['tables', 'models'] if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f'COMET data missing in {data_dir}: {", ".join(missing)}. '
            'Set COMET_DATA_DIR to the directory containing the downloaded tables/ and models/.')
    return data_dir


def get_comet_theory(stat, theory_options, cosmology, data_attrs):
    """Build native COMET calculators with priors restricted to their training range."""
    from desilike.base import get_params
    from desilike.theories.galaxy_clustering import COMETTracerSpectrum2Poles, COMETTracerSpectrum3Poles

    cosmology_options = theory_options.get('cosmology', {'template': 'direct'})
    if cosmology_options.get('template', 'direct') != 'direct':
        raise ValueError('COMET requires a direct cosmology, not a power-spectrum template.')
    prior_basis = theory_options.get('prior_basis', 'physical_aap')
    if prior_basis not in ['physical', 'physical_aap']:
        raise ValueError('COMET fits support prior_basis physical or physical_aap.')
    model = theory_options.get('comet_model', 'VDG_infty')
    if model not in ['VDG_infty', 'EFT']:
        raise ValueError('comet_model must be VDG_infty or EFT.')
    if 'mesh2' in stat:
        cls = COMETTracerSpectrum2Poles
    elif 'mesh3' in stat:
        cls = COMETTracerSpectrum3Poles
    else:
        raise ValueError(f'COMET is not implemented for {stat}.')
    z = data_attrs.get('z', data_attrs.get('zeff'))
    if z is None:
        raise KeyError('No effective redshift found for COMET; pass data_attrs["z"].')
    check_comet_data()
    if cosmology is None:
        cosmology = get_cosmology(cosmology_options)
    # Update the shared cosmology before attaching it to either PT calculator.
    params = cls.truncate_priors(get_params(cosmology), model=model)
    cosmology.update(params=params)
    tracers = data_attrs.get('tracers')
    if isinstance(tracers, (list, tuple)) and tracers and all(t == tracers[0] for t in tracers):
        tracers = tracers[0]
    kw = dict(theory_options.get('options', {}))
    if theory_options.get('redshift_smearing') is not None:
        if 'redshift_smearing' in kw:
            raise ValueError('Specify redshift_smearing once, under theory or theory.options.')
        kw['redshift_smearing'] = make_redshift_smearing(theory_options['redshift_smearing'], z, get_fiducial())
    nbar = theory_options.get('nbar', data_attrs.get('nbar'))
    if nbar is not None:
        kw['nbar'] = nbar
    theory = cls(cosmo=cosmology, fiducial=get_fiducial(), z=z, tracers=tracers,
                 model=model, prior_basis=prior_basis, **kw)
    theory.update(params=update_theory_nuisance_priors(
        get_params(theory, level=1), 'comet', stat, prior_basis,
        b3_coev=theory_options.get('b3_coev', False), marg=theory_options.get('marg', True),
        user_params=theory_options.get('params')))
    return theory


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
    if str(theory_options.get('model')).lower() == 'comet':
        return get_comet_theory(stat, theory_options, cosmology, data_attrs or {})
    from desilike.theories.galaxy_clustering import (
        DirectSpectrum2Template, ShapeFitSpectrum2Template, BAOSpectrum2Template,
        REPTVelocileptorsTracerSpectrum2Poles,
        FOLPSPTSpectrum2Poles, FOLPSPTSpectrum3Poles,
        FOLPSTracerSpectrum2Poles, FOLPSTracerSpectrum3Poles,
        DampedBAOWigglesTracerCorrelation2Poles,
    )
    from desilike.base import get_params
    theory_options = dict(theory_options)
    fiducial = get_fiducial()
    template = None
    theory_options.setdefault('cosmology', {'template': 'direct'})
    if theory_options.get('model') in ['folpsD', 'folpsEFT']:
        theory_options.setdefault('damping', 'vdg')
        theory_options.setdefault('prior_basis', 'physical_aap')
        theory_options.setdefault('marg', True)
        no_fog_damping = (
            theory_options['model'] == 'folpsEFT'
            or str(theory_options.get('damping')).lower() in ['none', 'null', 'off', 'false', '0']
        )
        if no_fog_damping:
            theory_options['damping'] = 'lor'
            theory_options['params'] = dict(theory_options.get('params') or {})
            theory_options['params']['X_FoG'] = {'value': 0., 'fixed': True}
            if 'mesh2' in stat:
                theory_options['damping_method'] = 'loop+ctr'
        if 'mesh2' in stat:
            if theory_options['model'] == 'folpsD':
                theory_options.setdefault('damping_method', 'tree')
            theory_options.setdefault('b3_coev', False)
            theory_options.setdefault('A_full', False)
        elif 'mesh3' in stat:
            theory_options.setdefault('A_full', False)
    cosmology_options = theory_options['cosmology']
    z = data_attrs.get('z', data_attrs.get('zeff', None))
    if z is None:
        raise KeyError('No effective redshift found; pass data_attrs["z"] or read it from the window with get_effective_redshift().')
    if cosmology_options['template'] == 'direct':
        template = DirectSpectrum2Template(fiducial=fiducial, cosmo=cosmology, z=z)
    elif cosmology_options['template'] == 'shapefit':
        template = ShapeFitSpectrum2Template(fiducial=fiducial, z=z)
    elif cosmology_options['template'] == 'bao':
        kw = {name: cosmology_options[name] for name in ['apmode', 'now'] if name in cosmology_options}
        if 'now' in kw:
            kw['with_now'] = kw.pop('now')
        template = BAOSpectrum2Template(fiducial=fiducial, z=z, **kw)
    if template is None:
        raise ValueError(f'template not found for {stat} and {repr(cosmology_options["template"])}')
    theory = None
    tracers = data_attrs.get('tracers', None)
    if isinstance(tracers, (list, tuple)) and tracers and all(tracer == tracers[0] for tracer in tracers):
        tracers = tracers[0]
    tracer_kwargs = {}
    if theory_options.get('redshift_smearing') is not None:
        if theory_options['model'] not in ['folpsD', 'folpsEFT']:
            raise ValueError('Redshift smearing is supported for COMET and FOLPS theories only.')
        if 'redshift_smearing' in theory_options.get('options', {}):
            raise ValueError('Specify redshift_smearing once, under theory or theory.options.')
        tracer_kwargs['redshift_smearing'] = make_redshift_smearing(theory_options['redshift_smearing'], z, fiducial)
    if tracers is not None:
        tracer_kwargs['tracers'] = tracers
    nbar = theory_options.get('nbar', data_attrs.get('nbar', None))
    if nbar is not None:
        tracer_kwargs['nbar'] = nbar
    if 'mesh2' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerSpectrum2Poles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            A_full = theory_options.get('A_full', True)
            pt = FOLPSPTSpectrum2Poles(A_full=A_full)
            kw = {name: theory_options[name] for name in ['damping', 'damping_method', 'prior_basis', 'use_GTNS'] if name in theory_options}
            kw.update(theory_options.get('options', {}))
            if theory_options['model'] == 'folpsEFT':
                kw['use_GTNS'] = True
            theory = FOLPSTracerSpectrum2Poles(template=template, pt=pt, **tracer_kwargs, **kw)
            theory.update(params=update_theory_nuisance_priors(
                get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'],
                b3_coev=theory_options['b3_coev'], marg=theory_options['marg'],
                user_params=theory_options.get('params')))
    elif 'mesh3' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            A_full = theory_options.get('A_full', True)
            # B needs its own emulator, which restores physical linear-spectrum
            # coordinates after interpolation (P handles this in its assembly).
            pt = FOLPSPTSpectrum3Poles(A_full=A_full)
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis'] if name in theory_options}
            theory = FOLPSTracerSpectrum3Poles(template=template, pt=pt, **tracer_kwargs, **kw, **theory_options.get('options', {}))
            theory.update(params=update_theory_nuisance_priors(
                get_params(theory, level=1), theory_options['model'], stat, kw['prior_basis'],
                marg=theory_options['marg'], user_params=theory_options.get('params')))
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory

def get_effective_redshift_from_window(window, fn=None):
    """Read legacy theory z or observed-window monopole zeff metadata."""
    found_monopole = False
    # Older products attach z to the theory; DR2 data attach zeff to the
    # observable monopole's attrs. Preserve the legacy priority when both exist.
    for part in [window.theory, window.observable]:
        for ells in [0, (0, 0, 0)]:
            try:
                mono = part.get(ells=ells)
            except (ValueError, KeyError):
                continue
            found_monopole = True
            for name in ['z', 'zeff', 'z_eff']:
                zeff = getattr(mono, name, None)
                if zeff is None:
                    zeff = getattr(mono, '_meta', {}).get(name, None)
                if zeff is None:
                    zeff = getattr(mono, 'attrs', {}).get(name, None)
                if zeff is not None:
                    return float(zeff)
    location = f' {fn}' if fn is not None else ''
    if not found_monopole:
        raise ValueError(f'No monopole found in window function{location}')
    raise AttributeError(f'No z_eff found in window function{location}')


def get_effective_redshift(args=None, kind='window_mesh2_spectrum_poles',
                           get_measurement_fn=get_measurement_fn, **kwargs):
    """Read effective redshift from a window-function measurement file."""
    args = dict(args or {})
    args.update(kwargs)
    kind = args.pop('kind', kind)
    get_measurement_fn = select_measurement_fn(args, get_measurement_fn=get_measurement_fn)
    fn = Path(get_measurement_fn(**args).format(kind))
    return get_effective_redshift_from_window(types.read(fn), fn=fn)

def _get_spectrum3_theory_rebin(ostep, tstep, theory_dk=None):
    """Return the first-stage mesh3 theory-grid stride."""
    ostep, tstep = float(ostep), float(tstep)
    if not np.isfinite(ostep) or ostep <= 0.:
        raise ValueError(f'Observable mesh3 spacing must be positive and finite; found {ostep}.')
    if not np.isfinite(tstep) or tstep <= 0.:
        raise ValueError(f'Native mesh3 theory spacing must be positive and finite; found {tstep}.')
    if theory_dk is None:
        rebin = int(ostep / tstep)
    else:
        theory_dk = float(theory_dk)
        if not np.isfinite(theory_dk) or theory_dk <= 0.:
            raise ValueError(f'Fixed mesh3 theory spacing must be positive and finite; found {theory_dk}.')
        ratio = theory_dk / tstep
        rebin = int(np.rint(ratio))
        if ratio < 1. or rebin < 1:
            raise ValueError(
                f'Fixed mesh3 theory spacing {theory_dk} cannot be finer than the native spacing {tstep}.'
            )
        if not np.isclose(ratio, rebin, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f'Fixed mesh3 theory spacing {theory_dk} must be an integer multiple of the native spacing {tstep}.'
            )
    if rebin < 1:
        raise ValueError(f'Observable mesh3 spacing {ostep} cannot be finer than the native theory spacing {tstep}.')
    return rebin

def rebin_spectrum3_window(window, data=None, theory_dk=None):
    """Compact a mesh3 window, optionally using a fixed first-stage theory spacing."""
    from clustering_statistics.tools import rebinning_matrix

    if data is None:
        data = window.observable
    ostep = min(np.diff(pole.edges('k'), axis=-1).min() for pole in data)
    with_templates = hasattr(window.theory, 'types')
    window_theory = window.at.theory.get(types='theory') if with_templates else window

    tstep = min(np.diff(pole.edges('k'), axis=-1).min() for pole in window_theory.theory)
    rebin = _get_spectrum3_theory_rebin(ostep, tstep, theory_dk=theory_dk)
    requested = 'dynamic' if theory_dk is None else f'{float(theory_dk):g}'
    logger.info(
        'Compactifying mesh3 window theory: observable dk=%g, native theory dk=%g, '
        'requested theory dk=%s, first-stage stride=%d, effective theory dk=%g',
        ostep, tstep, requested, rebin, rebin * tstep,
    )
    window_theory = window_theory.at.theory.select(k=slice(0, None, rebin))
    rebin = rebinning_matrix(window_theory.theory, new_coords=window_theory.theory.select(k=slice(0, None, 2)),
                             interp_order=3, diag='separate')
    window_theory = window_theory.clone(value=window_theory.value() @ rebin.value(), theory=rebin.theory)

    if with_templates:
        window_templates = window.at.theory.get(types=[item for item in window.theory.types if item != 'theory'])
        window = window.clone(value=np.concatenate([window_theory.value(), window_templates.value()], axis=-1),
                              theory=window.theory.at(types='theory').replace(window_theory.theory))
    else:
        window = window_theory
    return window

def select_window_theory(window, data):
    """Restrict window theory to a range close to observed data."""
    coord_name = list(next(iter(data)).coords())
    assert len(coord_name) == 1
    coord_name = coord_name[0]
    data_limits = min(pole.edges(coord_name).min() for pole in data), max(pole.edges(coord_name).max() for pole in data)
    return window.at.theory.select(**{coord_name: (data_limits[0] / 1.5, data_limits[1] * 1.5)})

def get_catalog_redshift(catalog):
    """Return catalog redshift metadata, using zsnap only for cubic boxes."""
    z = catalog.get('z', catalog.get('zeff', None))
    if z is None and catalog.get('domain', None) == 'cubic':
        z = catalog.get('zsnap', None)
    return None if z is None else float(z)

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
    """Return the catalog tracer namespace for theory nuisance parameters."""
    from cat_tools import _unzip_catalog_options

    catalog = observable_options.get('catalog', {})
    tracers = [str(tracer) for tracer in _unzip_catalog_options(catalog)]
    return 'x'.join(tracers)

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

def stat_to_kind(stat, window=False, basis=None, version=None):
    """Translate an observable stat name to the stored measurement key."""
    prefix = 'window_' if window else ''
    if 'mesh2' in stat:
        return f'{prefix}mesh2_spectrum_poles'
    if 'mesh3' in stat:
        basis = 'sugiyama' if basis is None else basis
        if version is not None and ('test' in str(version) or version in ['AbacusHF-lor', 'blinded-data-dr2']):
            if 'sugiyama' in basis:
                return f'{prefix}mesh3_spectrum_sugiyama-diagonal_poles'
            if 'scoccimarro' in basis:
                return f'{prefix}mesh3_spectrum_scoccimarro_poles'
        suffix = 'sugiyama' if 'sugiyama' in basis else ('scoccimarro' if 'scoccimarro' in basis else basis)
        return f'{prefix}mesh3_spectrum_poles_{suffix}'
    return f'{prefix}{stat}'

def select_measurement_fn(options, get_measurement_fn=get_measurement_fn):
    """Use ready-made measurement paths for externally supplied versions."""
    version = options.get('version', None)
    if 'test' in str(version) or 'RY' in str(version) or version in ['AbacusHF-lor', 'blinded-data-dr2']:
        return get_measurement_ready_fn
    return get_measurement_fn

def resolve_measurement_path(observable_options, kind, get_measurement_fn=get_measurement_fn, **overrides):
    """Resolve a stored measurement path from observable and runtime options."""
    catalog = dict(observable_options.get('catalog', {}))
    catalog.update(overrides)
    if catalog.get('version') == 'AbacusHF-lor' and kind.startswith('window_'):
        # These external measurements provide shared windows only in mock0.
        catalog['mock_id'] = 0
    for name in ['source', 'corrections', 'covariance', 'fn', 'mock_ids', 'mockid', 'nparams',
                 'rescale', 'scale', 'scale_mean_covariance']:
        catalog.pop(name, None)
    if catalog.get('version') == 'holi-v3-altmtl':
        catalog['version'] = 'holi-v3'
        catalog.setdefault('domain', 'altmtl')
    if catalog.get('version') == 'abacus-hf-dr2-v2-altmtl':
        catalog['version'] = 'AbacusHF-v2'
        catalog.setdefault('domain', 'altmtl')
    get_measurement_fn = select_measurement_fn(catalog, get_measurement_fn=get_measurement_fn)
    return Path(get_measurement_fn(**catalog).format(kind))

def _get_data_mock_ids(observable_options):
    """Return mock realizations to average, or ``None`` for one scalar realization."""
    catalog = observable_options.get('catalog', {})
    catalog = normalize_catalog_options(catalog)
    if 'mock_ids' in catalog:
        return list(catalog['mock_ids'])
    if 'mock_id' in catalog:
        return None
    return None

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
    if observable_options.get('catalog', {}).get('domain', None) == 'cubic':
        return None
    mock_ids = _get_data_mock_ids(observable_options)
    ids_to_read = mock_ids if mock_ids is not None else [None]
    windows = []
    if observable_options.get('catalog', {}).get('version') == 'AbacusHF-lor':
        ids_to_read = [0]  # Read the shared window once, also for mean-mock data.
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
    if all(stat is None for stat in stats):
        return None
    if any(stat is None for stat in stats):
        raise ValueError('Cannot pack a mix of window matrices and None windows.')
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
    if stats is None:
        return None
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
        full_options = {'observables': []}
        for options in observables_options:
            observable_options = {name: dict(options[name]) for name in ['stat', 'catalog']}
            observable_options['catalog'] = normalize_catalog_options(observable_options['catalog'])
            if kind == 'covariance' and not covariance_options.get('rescale', False):
                for name in ['mock_id', 'mock_ids']:
                    observable_options['catalog'].pop(name, None)
            full_options['observables'].append(observable_options)
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
            try:
                stats.write(cache_fn)
            except OSError as exc:
                logger.warning(f'Could not write cached stats {cache_fn}: {exc}')
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
            catalog = observable_options.get('catalog', {})
            kind = stat_to_kind(stat_options['kind'], basis=stat_options.get('basis'), version=catalog.get('version'))
            window_kind = stat_to_kind(stat_options['kind'], window=True, basis=stat_options.get('basis'),
                                       version=catalog.get('version'))
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
            if window is not None:
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
                version = covariance_options.get('version', observable_options.get('catalog', {}).get('version'))
                kind = stat_to_kind(stat_options['kind'], basis=stat_options.get('basis'), version=version)
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

    covariance_scale = float(covariance_options.get('scale', 1.))
    if not np.isfinite(covariance_scale) or covariance_scale <= 0.:
        raise ValueError(f'covariance scale must be positive and finite, got {covariance_scale!r}')
    if covariance_scale != 1.:
        covariance = covariance.clone(value=covariance.value() * covariance_scale)
        if rank == 0:
            logger.info(f'Applied covariance scale factor {covariance_scale:.6f}.')
    covariance.attrs['covariance_scale'] = covariance_scale

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


def get_emulator_space(calculator, options):
    """Select the example's declared Chebyshev bounds and resolution for varied axes."""
    from desilike import get_params
    from desilike.emulators import Space

    params = get_params(calculator).select(varied=True, derived=False)
    if not params:
        return None
    space_options = options.get('space', {})
    declared = space_options.get('limits', {})
    missing = sorted(set(params.names()) - set(declared))
    if missing:
        raise ValueError(f'Emulator space limits missing for {missing}; declare --emulator-bound NAME LOW HIGH.')
    bounds = {param.name: declared[param.name] for param in params}
    for name, limits in bounds.items():
        if len(limits) != 2 or not np.isfinite(limits).all() or not limits[0] < limits[1]:
            raise ValueError(f'Invalid emulator bounds for {name}: {limits}; expected finite LOW < HIGH.')
    return Space(bounds=bounds,
                 levels={name: level for name, level in options.get('levels', {}).items() if name in bounds},
                 transforms={name: transform for name, transform in space_options.get('transforms', {}).items()
                             if name in bounds})


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
    from desilike.observables.galaxy_clustering import Spectrum2PolesObservable, Spectrum3PolesObservable, Correlation2PolesObservable
    from desilike.base import get_params, replace
    from desilike.likelihoods import ObservablesGaussianLikelihood
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance_options = likelihood_options.get('covariance', {})
    cosmology = get_cosmology(cosmology_options)
    if stats is None:
        stats = get_data_stats(observables_options, covariance_options=covariance_options, unpack=False, get_measurement_fn=get_measurement_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    data_items, window_items, covariance = unpack_stats(stats)
    if window_items is None:
        window_items = [None] * len(data_items)
    labels = covariance.observable.labels(level=1)
    zeff_mesh2 = None
    observables = []
    for observable_options, data, window, label in zip(observables_options, data_items, window_items, labels, strict=True):
        stat = observable_options['stat']['kind']
        if 'mesh2' in stat:
            cls = Spectrum2PolesObservable
        elif 'mesh3' in stat:
            cls = Spectrum3PolesObservable
        elif 'particle2_correlation' in stat:
            cls = Correlation2PolesObservable
        else:
            raise NotImplementedError(stat)
        data_attrs = dict(data.attrs) | label
        if data_attrs.get('z', None) is None:
            window_kind = stat_to_kind(stat, window=True, basis=observable_options['stat'].get('basis'))
            catalog = observable_options.get('catalog', {})
            catalog_z = get_catalog_redshift(catalog)
            if catalog_z is not None:
                data_attrs['z'] = catalog_z
            elif window is not None and window_kind == 'window_mesh2_spectrum_poles':
                zeff_mesh2 = get_effective_redshift_from_window(window)
                data_attrs['z'] = zeff_mesh2
                logger.warning(f'No redshift found in data attributes; read z={zeff_mesh2:.3f} from the mesh2 window.')
            elif 'mesh3' in stat and zeff_mesh2 is not None:
                data_attrs['z'] = zeff_mesh2
        theory = get_theory(stat, theory_options=observable_options['theory'], cosmology=cosmology, data_attrs=data_attrs, data=data)
        if window is not None and cls == Spectrum3PolesObservable:
            theory_dk = observable_options.get('window', {}).get('theory_dk', None)
            window = rebin_spectrum3_window(window, data=data, theory_dk=theory_dk)
        if window is not None:
            window = select_window_theory(window, data)
        observable = cls(data=data, window=window, theory=theory)
        if observable_options['emulator']['name']:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            read_cache = cache_dir is not None and 'r' in cache_mode
            write_cache = cache_dir is not None and 'w' in cache_mode
            cache_dir = Path(cache_dir)
            emulator_options = observable_options['emulator']
            if emulator_options['name'] != 'chebyshev':
                raise ValueError(f'Unsupported emulator: {emulator_options["name"]!r}')
            emulator_space = get_emulator_space(theory.pt, emulator_options)
            if emulator_space is None:
                observables.append(observable)
                continue
            emulator_center = {param.name: float(param.value)
                               for param in get_params(theory.pt).select(varied=True, derived=False)}
            cache_fn = get_emulator_cache_fn(observable_options, theory.pt, data_attrs['z'], cache_dir)
            from desilike.emulators import Emulator
            cached = mpicomm.bcast(read_cache and cache_fn.exists() if mpicomm.rank == 0 else None, root=0)
            if cached:
                logger.info(f'Reading cached emulator {cache_fn}')
                emulator = Emulator.read(str(cache_fn))
            else:
                logger.info(f'Fitting emulator {cache_fn}')
                # Let the theory choose its emulator (FOLPS includes its scaling
                # treatment and trains the associated scalar provider).
                emulator = Emulator(theory.pt, emulator_space,
                                    engine='chebyshev', coverage='raise')
                if mpicomm.rank == 0:
                    logger.info('Chebyshev emulator %s: bounds=%s, budget=%s',
                                type(emulator).__name__, emulator_space.limits,
                                emulator_options.get('budget', 3))
                if write_cache and mpicomm.rank == 0:
                    cache_fn.parent.mkdir(parents=True, exist_ok=True)
                mpicomm.Barrier()
                checkpoint = str(cache_fn) + '.ckpt.npz' if write_cache else None
                try:
                    emulator.train(budget=emulator_options.get('budget', 3), checkpoint=checkpoint,
                                   chunk=emulator_options.get('chunk'),
                                   mpicomm=mpicomm if mpicomm.size > 1 else None)
                finally:
                    for param in get_params(theory.pt).select(varied=True, derived=False):
                        param.update(value=emulator_center[param.name])
                if write_cache and mpicomm.rank == 0:
                    cache_fn.parent.mkdir(parents=True, exist_ok=True)
                    emulator.write(str(cache_fn))
                mpicomm.Barrier()
                # A fresh deserialized object avoids carrying training tracers into
                # the posterior, matching example_emulator.py's tools implementation.
                if write_cache:
                    emulator = Emulator.read(str(cache_fn))
            current_params = get_params(theory.pt)
            for param in emulator.graph_params:
                current = current_params.get(param.name)
                if getattr(current, 'prior', None) is not None:
                    param.update(prior=current.prior, ref=current.ref)
            emulated_pt = emulator.to_calculator(calculator=theory.pt, center=emulator_center)
            replace(observable, theory.pt, emulated_pt)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables, covariance=covariance.value())


def get_emulator_revision(observable_options):
    """Separate corrected FOLPS B caches/results from the old P-emulated B fits.

    P-only, direct-theory and COMET identities retain their existing hashes.
    """
    if ('mesh3' in observable_options.get('stat', {}).get('kind', '')
            and observable_options.get('theory', {}).get('model') in ['folpsD', 'folpsEFT']
            and observable_options.get('emulator', {}).get('name')):
        return {'folps_bispectrum_pt': 'FOLPSPTSpectrum3Poles-v1'}
    return {}


def get_emulator_cache_fn(observable_options, pt, z, cache_dir):
    """Return the emulator cache path for the actual PT grid and training space."""
    space = get_emulator_space(pt, observable_options['emulator'])
    identity = {name: observable_options[name] for name in ['stat', 'theory', 'catalog', 'emulator']}
    identity.update(get_emulator_revision(observable_options))
    identity.update(emulator_api='theory-chebyshev-v1', bounds=space.limits,
                    z=z, k=np.asarray(pt.k).tolist())
    cosmology = str_from_cosmology_options(observable_options['theory']['cosmology'], level=100)
    cosmology += '_' + observable_options['emulator']['name']
    theory = str_from_observable_options(observable_options, level={'stat': 2, 'theory': 100, 'catalog': 2})
    return Path(cache_dir) / f'emulator_{cosmology}' / f'emulator_{theory}_{_hash_options(identity)}.h5'


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

    if level is None:
        has_test_catalog = any(
            'test' in str(observable.get('catalog', {}).get('version', ''))
            for likelihood in options.get('likelihoods', [])
            for observable in likelihood.get('observables', [])
        )
        output_level = {'catalog': 2, 'stat': 2} if has_test_catalog else {'catalog': 2}
    else:
        output_level = level
    _str_from_options = str_from_options(options, level=output_level, fit_output=True)
    fit_identity_options = {name: options[name] for name in ['cosmology', 'likelihoods'] if name in options}
    for likelihood in options.get('likelihoods', []):
        for observable in likelihood.get('observables', []):
            fit_identity_options.update(get_emulator_revision(observable))
    _hash = _hash_options(fit_identity_options)
    max_component_len = 180
    if len(_str_from_options) > max_component_len:
        _str_from_options = _str_from_options[:max_component_len].rstrip('_+-')
    if not _str_from_options:
        _str_from_options = 'fit'
    extra = f'_{extra}' if extra else ''
    ichain = f'_{ichain:d}' if ichain is not None else ''
    return fits_dir / f'{_str_from_options}_{_hash}{extra}' / f'{kind}{ichain}.{ext}'
