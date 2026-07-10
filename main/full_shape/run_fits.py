import os, sys
import logging
import argparse
import itertools
import warnings
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))
from utils import setup_logging
from cat_tools import parse_zerr_name
from fit_tools import LikelihoodBuilder
from helper import REDSHIFT_BIN_TRACER, REDSHIFT_TEST_TRACER

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compute_mesh')
warnings.filterwarnings(
    'ignore',
    message=r'loglikelihood is NaN for .*',
    category=UserWarning,
    module=r'desilike\.samplers\.base',
)

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'mcmc', 'nautilus', 'pocomc']

def validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3' in stats:
        raise ValueError('reptvelocileptors is only implemented for mesh2 fits.')

def parse_kmax(kmax):
    """Return kmax values for mesh2 ell0, mesh2 ell2, mesh3 000."""
    labels = ('mesh2_0', 'mesh2_2', 'mesh3_000')
    # labels = ('mesh2_0', 'mesh2_2', 'mesh3_000')
    try:
        values = [float(value) for value in kmax.split('-')]
    except ValueError as exc:
        raise ValueError(f'--kmax must be three dash-separated floats for {labels}; got {kmax!r}') from exc
    if len(values) != len(labels):
        raise ValueError(f'--kmax must have 3 values for {labels}; got {len(values)} from {kmax!r}')
    return dict(zip(labels, values))
    
def build_kranges(kmax):
    kmax = parse_kmax(kmax)
    kmin, dk = 0.02, 0.01
    def select(ells, kmax_value):
        if kmax_value <= kmin:
            return None
        return {'ells': ells, 'k': [kmin, kmax_value, dk]}
    kranges = {
        'mesh2': [
            select(0, kmax['mesh2_0']),
            select(2, kmax['mesh2_2']),
        ],
        'mesh3': [
            select((0, 0, 0), kmax['mesh3_000']),
            # select((2, 0, 2), kmax['mesh3_202']),
        ],
    }
    for stat, items in kranges.items():
        kranges[stat] = [item for item in items if item is not None]
        if not kranges[stat]:
            raise ValueError(f'No {stat} multipoles selected by --kmax={kmax!r}; at least one kmax must be > {kmin}.')
    return {
        stat: items for stat, items in kranges.items()
    }

def build_observables(stats, theory_model, prior_basis, kmax):
    validate_theory_model(stats, theory_model)
    kranges = build_kranges(kmax)
    return [{'stat': {'kind': stat, 'select': kranges[stat]},
             'theory': {'model': theory_model, 'prior_basis': prior_basis},
            } for stat in stats]

def get_sampler_cls(name='emcee', stats=None):
    """Return desilike sampler class from a short command-line name."""
    if name == 'emcee':
        from desilike.samplers.emcee import EmceeSampler
        init_options = {}
        run_options = {'max_iterations': 20000, 'check_every': 200, 'check': {'max_eigen_gr': 0.03}}
        return EmceeSampler, init_options, run_options
    if name == 'mcmc':
        from desilike.samplers.mcmc import MCMCSampler
        init_options = {}
        run_options = {}
        return MCMCSampler, init_options, run_options
    if name == 'nautilus':
        from desilike.samplers.nautilus import NautilusSampler
        init_options = {}
        run_options = {}
        return NautilusSampler, init_options, run_options
    if name == 'pocomc':
        from desilike.samplers.pocomc import PocoMCSampler
        max_gr = 0.03 if 'mesh3' in stats else 0.02
        # init_options = {'n_active': 128, 'n_ess': 512}
        init_options = {'n_active': 96, 'n_ess': 256}
        run_options = {'min_iterations': 200, 'check_every': 20,
                       'check': {'max_eigen_gr': max_gr}, 'progress': False}
        return PocoMCSampler, init_options, run_options
    raise ValueError(f'Unknown sampler {name!r}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--todos", nargs = '+', type=str, default=['test'], choices=['test', 'profile', 'sample'], help="todo types")
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3', 'AbacusHF-test',])
    parser.add_argument("--cov_version", type = str,  default='holi-v3', help="mock types", choices=['holi-v3', 'EZmocks-test',])
    parser.add_argument("--cov_scale", type = float,  default=1.0, help="covariance scale factor")
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl'], help="mock domain")
    parser.add_argument("--hod", type = str, default='base', choices=['base', 'base_dv'], help="HOD model")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO1'], choices=['LRG1', 'LRG2', 'LRG3', 'QSO1'], help="tracer types")
    parser.add_argument("--mockid", type = str, default="0", help="Mock ID, or a range/list to average into one data vector (e.g. 0-24).")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, e.g. 'None', 'repeat', 'verr_empirical', 'verr_nonparam' with '_zevol' for redshift evolution")
    parser.add_argument('--fits_dir', type=str, default =os.getenv('SCRATCH', '.') + '/fits',
                        help='Directory to save fit results (chains, logs, etc.)')
    parser.add_argument("--stats", nargs = '+', type=str, default=['mesh2'], choices=['mesh2', 'mesh3'], help="statistics to fit, e.g. 'mesh2', 'mesh3'")
    parser.add_argument("--kmax", type=str, default='0.350-0.250-0.20-0.08', help='kmax values for mesh2 ell0, mesh2 ell2, mesh3 000, mesh3 202')
    parser.add_argument("--regions", nargs = '+', type=str, default=['ALL'], help="Region labels for cutsky/altmtl runs, e.g. ALL NGC SGC GCcomb")
    parser.add_argument('--theory_model', type=str, default='folpsD', choices=THEORY_MODELS, help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap', choices=PRIOR_BASES, help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--cosmo_params', type=str, default='base', choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='emcee', choices=SAMPLERS, help='desilike sampler backend to use. Defaults to emcee.')
    parser.add_argument('--nchains', type=int, default=1, help='Number of independent chains to run for sampling. Defaults to 1.')
    parser.add_argument("--resume", action="store_true", help="Resume sampling from existing chain files.")

    args = parser.parse_args()
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")
    todos = args.todos
    domain = args.domain
    stats = args.stats
    observables = build_observables(stats, theory_model=args.theory_model, prior_basis=args.prior_basis, kmax=args.kmax)
    cosmology_args = {'model': args.cosmo_params, 'template': 'direct'}
    regions = [None] if domain == 'cubic' else args.regions

    fits_dir = Path(args.fits_dir)
    cache_dir = fits_dir / '_cache'

    # Convert mockid string input to a list
    if '-' in args.mockid:
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
        if 'QSO' in tracer:
            data_args['domain'] = 'cutsky'
            data_args['hod'] = 'base_dv'

        if args.cov_scale != 1.0:
            covariance_args['scale'] = args.cov_scale
        if len(mockids) > 1:
            data_args['mock_ids'] = mockids
        else:
            data_args['mock_id'] = mockids[0]
            
        fit_args = dict(observables=observables, catalog=data_args, covariance=covariance_args,
                        cosmology=cosmology_args, cache_dir=cache_dir)
        builder = LikelihoodBuilder(**fit_args)
        for todo in todos:
            if todo == 'test':
                builder.get_likelihood()
                builder.get_all_data_stats()
            elif todo == 'profile':
                from desilike.profilers import MinuitProfiler
                likelihood = builder.get_likelihood()
                profiler = MinuitProfiler(likelihood, seed=42)
                profiler.maximize()
            elif todo == 'sample':
                sampler_cls, sampler_init_options, sampler_run_options = get_sampler_cls(name=args.sampler, stats=stats)
                likelihood = builder.get_likelihood()
                save_fn = [builder.get_fits_fn(fits_dir=fits_dir, kind='chain', ichain=ichain)
                           for ichain in range(args.nchains)]
                for fn in save_fn:
                    Path(fn).parent.mkdir(parents=True, exist_ok=True)
                if args.resume:
                    missing = [fn for fn in save_fn if not Path(fn).exists()]
                    if missing:
                        missing_str = ', '.join(str(fn) for fn in missing)
                        raise FileNotFoundError(f'cannot resume sampling; missing chain file(s): {missing_str}')
                    sampler_init_options['chains'] = save_fn
                sampler = sampler_cls(likelihood, save_fn=save_fn, **sampler_init_options)
                sampler.run(**sampler_run_options)
