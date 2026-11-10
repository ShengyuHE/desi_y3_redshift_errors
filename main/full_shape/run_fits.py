import os, sys
import logging
import argparse
import itertools
import warnings
import inspect
import math
import copy
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))
from utils import setup_logging
from cat_tools import parse_zerr_name
from fit_tools import LikelihoodBuilder, get_chain_sampler_options, get_prior, get_sampler_cls
from fit_support import propose_fiducial_emulator_options, REDSHIFT_SMEARING_MODES, normalize_smearing
from helper import REDSHIFT_BIN_TRACER, REDSHIFT_TEST_TRACER

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_fits')

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors', 'COMET']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'mcmc', 'nautilus', 'pocomc']
DAMPING='lor' #vdg, lor
DAMPING_METHOD='tree+loop' # tree+loop, tree+loop+ctr, tree+loop+ctr+sn, all
PROFILE_ITERATIONS = 4
GELMAN_RUBIN = 1.03
ESS = 700
SAMPLER_INIT = {'batch_size': 8, 'n_active': 512, 'n_effective': 1024}

def validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3' in stats:
        raise ValueError('reptvelocileptors is only implemented for mesh2 fits.')

def parse_theory_model(value):
    return 'COMET' if value.lower() == 'comet' else value

def kmax_labels_for_stats(stats):
    """Return the expected kmax labels for the requested statistics."""
    labels = []
    if 'mesh2' in stats:
        labels.extend(('mesh2_0', 'mesh2_2'))
    if 'mesh3' in stats:
        labels.extend(('mesh3_000', 'mesh3_202'))
    return tuple(labels)

def parse_kmax(kmax, stats):
    """Accept selected-statistic values or the full P0-P2-B000-B202 list."""
    labels = kmax_labels_for_stats(stats)
    try:
        values = [float(value) for value in kmax.replace('/', '-').split('-')]
    except ValueError as exc:
        raise ValueError(f'--kmax must be dash- or slash-separated floats for {labels}; got {kmax!r}') from exc
    all_labels = kmax_labels_for_stats(['mesh2', 'mesh3'])
    if len(values) == len(all_labels):
        full_kmax = dict(zip(all_labels, values))
        return {label: full_kmax[label] for label in labels}
    if len(values) != len(labels):
        raise ValueError(f'--kmax must have {len(labels)} values for {labels}, or 4 values '
                         f'for P0-P2-B000-B202; got {len(values)} from {kmax!r}')
    return dict(zip(labels, values))
    
def build_kranges(kmax, stats):
    kmax = parse_kmax(kmax, stats)
    kmin, dk = 0.02, 0.01
    def select(ells, kmax_value):
        if kmax_value <= kmin:
            return None
        return {'ells': ells, 'k': [kmin, kmax_value, dk]}
    kranges = {}
    if 'mesh2' in stats:
        kranges['mesh2'] = [
            select(0, kmax['mesh2_0']),
            select(2, kmax['mesh2_2']),
        ]
    if 'mesh3' in stats:
        kranges['mesh3'] = [
            select((0, 0, 0), kmax['mesh3_000']),
            select((2, 0, 2), kmax['mesh3_202']),
        ]
    for stat, items in kranges.items():
        kranges[stat] = [item for item in items if item is not None]
        if not kranges[stat]:
            raise ValueError(f'No {stat} multipoles selected by --kmax={kmax!r}; at least one kmax must be > {kmin}.')
    return kranges

def parse_optional_method(value):
    """Convert CLI spellings for disabled optional methods to Python None."""
    if value is None:
        return None
    value = str(value)
    return None if value.lower() in ['none', 'null'] else value

def parse_damping(value):
    """Normalize FoG damping names, preserving an explicit no-damping request."""
    if value is None:
        return None
    value = str(value).lower()
    return 'none' if value in ['none', 'null', 'off', 'false', '0'] else value

def build_emulator_options(budget=3, bounds=None):
    """Build Chebyshev settings from repeated CLI NAME LOW HIGH bounds."""
    if budget < 1:
        raise ValueError('--emulator-budget must be a positive integer.')
    options = propose_fiducial_emulator_options()
    options['budget'] = budget
    for name, low, high in bounds or []:
        low, high = float(low), float(high)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f'Invalid --emulator-bound {name}: expected finite LOW < HIGH.')
        if name not in options['space']['limits']:
            raise ValueError(f'Unknown emulator parameter {name!r}.')
        options['space']['limits'][name] = (low, high)
    return options


def build_observables(stats, theory_model, prior_basis, kmax, damping=None, damping_method=None,
                      mesh3_theory_dk=None, emulator_options=None, comet_model='VDG_infty',
                      redshift_smearing=None):
    theory_model = parse_theory_model(theory_model)
    validate_theory_model(stats, theory_model)
    if redshift_smearing is not None and theory_model not in ['COMET', 'folpsD', 'folpsEFT']:
        raise ValueError('Redshift smearing is supported for COMET and FOLPS theories only.')
    if theory_model == 'COMET':
        if prior_basis not in ['physical', 'physical_aap']:
            raise ValueError('COMET fits support prior_basis physical or physical_aap.')
        if comet_model not in ['VDG_infty', 'EFT']:
            raise ValueError('comet_model must be VDG_infty or EFT.')
    if mesh3_theory_dk is not None and mesh3_theory_dk <= 0.:
        raise ValueError('mesh3_theory_dk must be positive.')
    kranges = build_kranges(kmax, stats)
    observables = []
    for stat in stats:
        theory = {'model': theory_model, 'prior_basis': prior_basis}
        if redshift_smearing is not None:
            theory['redshift_smearing'] = copy.deepcopy(redshift_smearing)
        if theory_model == 'COMET':
            theory['comet_model'] = comet_model
        else:
            no_fog_damping = theory_model == 'folpsEFT' or damping == 'none'
            if no_fog_damping:
                theory['damping'] = 'none'
                theory['params'] = {'X_FoG': {'value': 0., 'fixed': True}}
                theory['use_GTNS'] = True
            elif damping is not None:
                theory['damping'] = damping
            if 'mesh2' in stat:
                theory['damping_method'] = 'loop+ctr' if no_fog_damping else damping_method
                if theory_model == 'folpsEFT':
                    # Isotropic k^2 stochastic term; retain desilike's native prior.
                    # Keep this explicit so fit identities distinguish SN1-enabled runs.
                    theory['params']['sn1'] = {'fixed': False}
        observable = {'stat': {'kind': stat, 'select': kranges[stat]},
                      'theory': theory,
                      'emulator': dict(emulator_options if emulator_options is not None
                                       else propose_fiducial_emulator_options())}
        if theory_model == 'COMET':
            observable['emulator'] = {'name': None}
        if 'mesh3' in stat and mesh3_theory_dk is not None:
            observable['window'] = {'theory_dk': float(mesh3_theory_dk)}
        observables.append(observable)
    return observables


def resolve_smearing_observables(observables, catalog):
    """Resolve per-tracer PDF templates without changing the common options."""
    observables = copy.deepcopy(observables)
    for observable in observables:
        spec = observable['theory'].get('redshift_smearing')
        if spec is None:
            continue
        spec = {'mode': spec} if isinstance(spec, str) else dict(spec)
        if spec.get('pdf_file'):
            labels = dict(catalog)
            labels['tracer'] = str(catalog['tracer']).rstrip('0123456789')
            labels['tracer_bin'] = catalog['tracer']
            if catalog.get('zrange') is not None:
                labels['zmin'], labels['zmax'] = catalog['zrange']
            try:
                spec['pdf_file'] = spec['pdf_file'].format(**labels)
            except (KeyError, ValueError) as exc:
                raise ValueError(f'Cannot resolve redshift-error PDF template {spec["pdf_file"]!r} for {catalog}') from exc
        observable['theory']['redshift_smearing'] = normalize_smearing(spec)
    return observables

def filter_callable_kwargs(func, kwargs):
    """Keep kwargs accepted by ``func``; pass all through when **kwargs exists."""
    kwargs = dict(kwargs)
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in signature.parameters}

def adapt_sampler_init_options(sampler_cls, init_options):
    """Translate shared sampler defaults to the installed sampler API when needed."""
    init_options = dict(init_options)
    signature = inspect.signature(sampler_cls.__init__)
    names = set(signature.parameters)
    if 'rng' in init_options and 'rng' not in names and 'seed' in names:
        init_options['seed'] = init_options.pop('rng')
    if 'n_effective' in init_options and 'n_effective' not in names and 'n_ess' in names:
        init_options['n_ess'] = init_options.pop('n_effective')
    return filter_callable_kwargs(sampler_cls.__init__, init_options)

def build_sampler_args(sampler, nchains, resume, stats):
    """Build sampler options, tightening MCMC convergence settings when supported."""
    sampler_args = {'sampler': sampler, 'nchains': nchains, 'resume': resume}
    if sampler == 'pocomc':
        sampler_args['init'] = dict(SAMPLER_INIT)
    sampler_defaults, _, _ = get_chain_sampler_options(sampler=sampler, stats=stats)
    sampler_kw = {'nparallel': nchains, 'gelman_rubin': GELMAN_RUBIN, 'ess': ESS}
    for section in ['init', 'run']:
        for name in sampler_defaults.get(section, {}):
            if name in sampler_kw:
                sampler_args.setdefault(section, {})[name] = sampler_kw[name]
    return sampler_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--todos", nargs = '+', type=str, default=['test'], choices=['build', 'test', 'profile', 'sample'], help="todo types (build trains and caches emulators)")
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="data or mock version", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3', 'AbacusHF-test', 'AbacusHF-RY', 'blinded-data-dr2'])
    parser.add_argument("--cov_version", type = str,  default='holi-v3', help="mock types", choices=['holi-v3', 'EZmocks-test',])
    parser.add_argument("--cov_scale", type = float,  default=1.0, help="covariance scale factor")
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl', 'obs'], help="data or mock domain (blinded-data-dr2 requires obs)")
    parser.add_argument("--hod", type = str, default='base', choices=['base', 'base_dv'], help="HOD model")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO1'], choices=['LRG1', 'LRG2', 'LRG3', 'QSO1'], help="tracer types")
    parser.add_argument("--mockid", type = str, default="0", help="Mock ID, or a range/list to average into one data vector (e.g. 0-24). Ignored for obs.")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, e.g. 'None', 'repeat', 'verr_empirical', 'verr_nonparam' with '_zevol' for redshift evolution")
    parser.add_argument('--fits_dir', type=str, default =os.getenv('SCRATCH', '.') + '/fits',
                        help='Directory to save fit results (chains, logs, etc.)')
    parser.add_argument("--stats", nargs = '+', type=str, default=['mesh2'], choices=['mesh2', 'mesh3'], help="statistics to fit, e.g. 'mesh2', 'mesh3'")
    parser.add_argument("--kmax", type=str, default='0.350-0.250',
                        help='Dash- or slash-separated kmax values for selected --stats (mesh2 ell0/ell2, mesh3 000/202), '
                             'or all four P0-P2-B000-B202 values; unselected statistics are ignored. '
                             'Default: 0.350-0.250. Theory-specific presets are set in srun_fit.sh.')
    parser.add_argument('--mesh3_theory_dk', '--mesh3-theory-dk', type=float, default=0.005, metavar='DK',
                        help='Fixed first-stage mesh3 window-theory spacing. Defaults to 0.005.')
    parser.add_argument("--regions", nargs = '+', type=str, default=['ALL'], help="Region labels, e.g. ALL NGC SGC GCcomb; obs maps ALL to GCcomb")
    parser.add_argument('--theory_model', type=parse_theory_model, default='folpsD', choices=THEORY_MODELS, help='Theory model to fit (comet is accepted as an alias for COMET). Defaults to folpsD.')
    parser.add_argument('--comet-model', choices=['VDG_infty', 'EFT'], default='VDG_infty',
                        help='COMET internal model; replaces the FOLPS damping options for COMET.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap', choices=PRIOR_BASES, help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--damping', type=parse_damping, default=DAMPING,
                        help=f"FOLPS FoG damping model: 'vdg', 'lor', 'exp', or 'none'. Defaults to {DAMPING}; unused by COMET.")
    parser.add_argument('--damping_method', type=parse_optional_method, default=DAMPING_METHOD,
                        help=f"FoG damping method for mesh2: 'loop+ctr', 'tree+loop', 'tree+loop+ctr', or 'all'. Defaults to {DAMPING_METHOD}.")
    parser.add_argument('--cosmo_params', type=str, default='base', choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='pocomc', choices=SAMPLERS, help='desilike sampler backend to use. Defaults to pocomc.')
    parser.add_argument('--nchains', type=int, default=1, help='Number of independent chains to run for sampling. Defaults to 1.')
    parser.add_argument('--emulator-budget', type=int, default=3,
                        help='Chebyshev sparse-grid resolution budget. Defaults to 3; unused by COMET, which has its own emulator.')
    parser.add_argument('--emulator-bound', nargs=3, action='append', default=[],
                        metavar=('NAME', 'LOW', 'HIGH'),
                        help='Declare a finite Chebyshev training range; repeat for each cosmological parameter.')
    parser.add_argument("--resume", action="store_true", help="Resume sampling from existing chain files.")
    parser.add_argument('--corr_dv', choices=REDSHIFT_SMEARING_MODES,
                        help='Theory correction, independent of --zerrs: gauss, lor, pdf, pdf+gauss, pdf+lor. Disabled by default.')
    parser.add_argument('--redshift-error-pdf',
                        help='Single-galaxy velocity PDF: NPZ grid/pdf or lsstypes HDF5 dv/pdf. '
                             'Templates may use {tracer}, {tracer_bin}, {zmin:.1f}, {zmax:.1f}, {zsnap}.')

    args = parser.parse_args()
    observed_data = args.version == 'blinded-data-dr2'
    if observed_data != (args.domain == 'obs'):
        parser.error('--version blinded-data-dr2 must be paired with --domain obs.')
    if observed_data:
        if any(parse_zerr_name(zerr) != ('None', False) for zerr in args.zerrs):
            parser.error('Observed data require --zerrs None; use --corr_dv for a theory correction.')
        # Combined observed measurements and windows use GCcomb for both P and B.
        args.regions = list(dict.fromkeys('GCcomb' if region == 'ALL' else region
                                         for region in args.regions))
    smearing = None
    if args.corr_dv:
        smearing = {'mode': args.corr_dv}
        if args.corr_dv.startswith('pdf') and not args.redshift_error_pdf:
            parser.error('--redshift-error-pdf is required for PDF smearing.')
        if args.redshift_error_pdf:
            if not args.corr_dv.startswith('pdf'):
                parser.error('--redshift-error-pdf requires a PDF smearing mode.')
            smearing['pdf_file'] = args.redshift_error_pdf
    elif args.redshift_error_pdf:
        parser.error('--redshift-error-pdf requires --corr_dv.')
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")
    todos = args.todos
    domain = args.domain
    stats = args.stats
    try:
        emulator_options = build_emulator_options(args.emulator_budget, args.emulator_bound)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        observables = build_observables(stats, theory_model=args.theory_model, prior_basis=args.prior_basis,
                                    kmax=args.kmax, damping=args.damping, damping_method=args.damping_method,
                                    mesh3_theory_dk=args.mesh3_theory_dk, emulator_options=emulator_options,
                                    comet_model=args.comet_model, redshift_smearing=smearing)
    except ValueError as exc:
        parser.error(str(exc))
    cosmology_args = {'model': args.cosmo_params, 'template': 'direct'}
    if args.theory_model in ['folpsD', 'folpsEFT']:
        logger.info('FOLPS emulator: %s, budget=%s; mesh3 uses FOLPSPTSpectrum3Poles '
                    'with FOLPSD3PolesEmulator and the corrected cache identity.',
                    emulator_options['name'], emulator_options['budget'])
    if args.theory_model == 'COMET':
        from fit_tools import check_comet_data
        check_comet_data()
        logger.info('COMET %s: kmax=%s; using its native emulator and training-range priors. '
                    'Cosmology provider: eisenstein_hu; derived outputs: H0, Omega_m, sigma8_m, sigma8_cb, rs_drag. '
                    'Minuit precision: 1e-4.',
                    args.comet_model, args.kmax)
    regions = [None] if domain == 'cubic' else args.regions

    fits_dir = Path(args.fits_dir)
    cache_dir = fits_dir / '_cache'

    # Convert mockid string input to a list
    if observed_data:
        mockids = None  # One observed vector; never average over --mockid.
    elif '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    for tracer, zerr, region in itertools.product(args.tracers, args.zerrs, regions):
        use_dv, z_evol = parse_zerr_name(zerr)
        data_args = {'version': args.version, 'domain':args.domain, 'tracer':tracer,
                     'region': region, "use_dv": use_dv, "z_evol": z_evol, 'hod': args.hod}
        cov_mock_ids = range(1, 1001) if args.cov_version == 'EZmocks-test' else range(1000)
        covariance_args = {'source': 'mock', 'version': args.cov_version, 'mock_ids': cov_mock_ids,
                           'corrections': ['hartlap', 'percival'], 'rescale': False}
        # GCcomb data/windows use the ALL-region mock covariance by convention.
        if region == 'GCcomb':
            covariance_args['region'] = 'ALL'

        if domain == 'cubic':
            zsnap = REDSHIFT_TEST_TRACER[tracer]
            data_args['zsnap'] = zsnap
            covariance_args['domain'] = 'cubic'
        elif domain == 'cutsky':
            zrange = REDSHIFT_BIN_TRACER[tracer]
            data_args['zrange'] = zrange
            data_args['domain'] = 'cutsky'
            covariance_args['domain'] = 'altmtl'
        elif domain == 'altmtl':
            zrange = REDSHIFT_BIN_TRACER[tracer]
            data_args['zrange'] = zrange
            data_args['domain'] = 'altmtl'
            covariance_args['domain'] = 'altmtl'
        elif domain == 'obs':
            data_args['zrange'] = REDSHIFT_BIN_TRACER[tracer]
            covariance_args['domain'] = 'altmtl'
        # QSO uses base_dv measurements; altmtl inputs map to cutsky.
        if 'QSO' in tracer and domain == 'altmtl':
            data_args['domain'] = 'cutsky'
        if 'QSO' in tracer:
            data_args['hod'] = 'base_dv'
        if args.cov_scale != 1.0:
            covariance_args['scale'] = args.cov_scale
        if not observed_data:
            if len(mockids) > 1:
                data_args['mock_ids'] = mockids
            else:
                data_args['mock_id'] = mockids[0]
            
        sampler_args = build_sampler_args(args.sampler, args.nchains, args.resume, stats)
        profiler_args = {'maximize': {'niterations': PROFILE_ITERATIONS}}
        fit_observables = resolve_smearing_observables(observables, data_args)
        fit_args = dict(observables=fit_observables, catalog=data_args, covariance=covariance_args,
                        cosmology=cosmology_args, sampler=sampler_args, profiler=profiler_args,
                        cache_dir=cache_dir)
        builder = LikelihoodBuilder(**fit_args)
        for todo in todos:
            if todo == 'build':
                builder.get_likelihood()
            elif todo == 'test':
                import numpy as np
                from desilike import Posterior, build
                likelihood = builder.get_likelihood()
                posterior = build(Posterior(likelihood, prior=get_prior(likelihood)))
                value = float(posterior())
                if not np.isfinite(value):
                    raise ValueError(f'Non-finite fiducial log posterior: {value}')
                if mpicomm.rank == mpiroot:
                    logger.info('Likelihood check passed: log posterior = %g', value)
            elif todo == 'profile':
                from desilike.base import Posterior, build, copy as desilike_copy, get_params
                from desilike.conditioning import AffineConditioner
                from desilike.profilers import Profiler
                from profile_tools import FiniteMinuit
                likelihood = builder.get_likelihood()
                profile_fn = builder.get_fits_fn(fits_dir=fits_dir, kind='profiles', ext='h5')
                Path(profile_fn).parent.mkdir(parents=True, exist_ok=True)
                likelihood_profiler = desilike_copy(likelihood)
                for param in get_params(likelihood_profiler).select(solved=True):
                    param.update(derived='best')
                posterior = build(Posterior(likelihood_profiler, prior=get_prior(likelihood_profiler)))
                profiler_options = dict(builder.options['profiler'])
                profiler_options.pop('profiler', None)
                init_options = dict(profiler_options.get('init', {}))
                kernel_options = {name: init_options.pop(name) for name in list(init_options)
                                  if name not in ['rng', 'rescale', 'covariance']}
                conditioner = AffineConditioner(**{name: init_options.pop(name, None)
                                                   for name in ['rescale', 'covariance']})
                profiler = Profiler(posterior, kernel=FiniteMinuit(**kernel_options),
                                    output_fn=profile_fn, conditioner=conditioner,
                                    **init_options)
                profiler.maximize(**profiler_options.get('maximize', {}))
                if mpicomm.rank == mpiroot:
                    print(profiler.profiles.to_stats(tablefmt='pretty'))
            elif todo == 'sample':
                sampler_options = builder.options['sampler']
                sampler_cls = get_sampler_cls(sampler_options['sampler'])
                _, sampler_init_options, sampler_run_options = get_chain_sampler_options(
                    sampler=sampler_options['sampler'], stats=stats, overrides=sampler_options)
                sampler_init_options = adapt_sampler_init_options(sampler_cls, sampler_init_options)
                sampler_run_options = filter_callable_kwargs(sampler_cls.run, sampler_run_options)
                likelihood = builder.get_likelihood()
                save_fn = [builder.get_fits_fn(fits_dir=fits_dir, kind='chain', ichain=ichain)
                           for ichain in range(sampler_options['nchains'])]
                for fn in save_fn:
                    Path(fn).parent.mkdir(parents=True, exist_ok=True)
                if sampler_options.get('resume', False):
                    if hasattr(sampler_cls, 'check_resume'):
                        sampler_cls.check_resume(save_fn)
                    else:
                        missing = [fn for fn in save_fn if not Path(fn).exists()]
                        if missing:
                            missing_str = ', '.join(str(fn) for fn in missing)
                            raise FileNotFoundError(f'cannot resume sampling; missing chain file(s): {missing_str}')
                    sampler_init_options['chains'] = save_fn
                sampler_init_options['profiles_fn'] = builder.get_fits_fn(fits_dir=fits_dir, kind='profiles', ext='h5')
                sampler = sampler_cls(likelihood, save_fn=save_fn, **sampler_init_options)
                sampler.run(**sampler_run_options)
