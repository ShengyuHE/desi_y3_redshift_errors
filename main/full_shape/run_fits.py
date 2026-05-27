import os, sys
import logging
import argparse
import itertools
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))
from utils import setup_logging
from cat_tools import parse_zerr_name
from fit_tools import LikelihoodBuilder
from helper import REDSHIFT_BIN_TRACER

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compute_mesh')

THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors']
COSMO_MODELS = ['base', 'base_ns-fixed', 'fixed']
PRIOR_BASES = ['physical', 'physical_aap', 'tcm_chudaykin_aap', 'standard']
SAMPLERS = ['emcee', 'mcmc', 'nautilus', 'pocomc']

def validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3' in stats:
        raise ValueError('reptvelocileptors is only implemented for mesh2 fits.')

def build_observables(stats, theory_model, prior_basis):
    validate_theory_model(stats, theory_model)
    return [{'stat': {'kind': stat},
            'theory': { 'model': theory_model, 'prior_basis': prior_basis,
            },} for stat in stats]

def get_sampler_cls(name='emcee'):
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
        init_options = {'n_active': 128, 'n_ess': 512}
        run_options = {'min_iterations': 400, 'check_every': 20,
                       'check': {'max_eigen_gr': 0.02}, 'progress': False}
        return PocoMCSampler, init_options, run_options
    raise ValueError(f'Unknown sampler {name!r}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3'])
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl'], help="mock domain")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'])
    parser.add_argument("--mockid", type = str, default="0", help="Mock ID, or a range/list to average into one data vector (e.g. 0-24).")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, e.g. 'None', 'repeat', 'verr_empirical', 'verr_nonparam' with '_zevol' for redshift evolution")
    parser.add_argument("--todos", nargs = '+', type=str, default=['test'], choices=['test', 'profile', 'sample'], help="todo types")
    parser.add_argument("--stats", nargs = '+', type=str, default=['mesh2'], choices=['mesh2', 'mesh3'], help="statistics to fit, e.g. 'mesh2', 'mesh3'")
    parser.add_argument("--regions", nargs = '+', type=str, default=['ALL'], help="Region labels for cutsky/altmtl runs, e.g. ALL NGC SGC GCcomb")
    parser.add_argument('--theory_model', type=str, default='folpsD', choices=THEORY_MODELS, help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--prior_basis', type=str, default='physical_aap', choices=PRIOR_BASES, help='Nuisance-parameter prior basis. Defaults to physical_aap.')
    parser.add_argument('--cosmo_params', type=str, default='base_ns-fixed', choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA; '
                             'fixed varies only nuisance parameters. Defaults to base.')
    parser.add_argument('--sampler', type=str, default='emcee', choices=SAMPLERS, help='desilike sampler backend to use. Defaults to emcee.')
    parser.add_argument('--nchains', type=int, default=1, help='Number of independent chains to run for sampling. Defaults to 1.')
    parser.add_argument("--resume", action="store_true", help="Resume sampling from existing chain files.")
    args = parser.parse_args()
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")
    todos = args.todos
    version = args.version
    domain = args.domain
    stats = args.stats
    observables = build_observables(stats, theory_model=args.theory_model, prior_basis=args.prior_basis)
    cosmology_args = {'model': args.cosmo_params, 'template': 'direct'}
    regions = [None] if domain == 'cubic' else args.regions
    tracer_redshifts = []

    fits_dir = Path(os.getenv('SCRATCH', '.'))/ 'Y3' / 'redshift_errors' / 'fits'
    cache_dir = fits_dir / '_cache'

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    for tracer, zerr, region in itertools.product(args.tracers, args.zerrs, regions):
        zrange = REDSHIFT_BIN_TRACER[tracer]
        use_dv, z_evol = parse_zerr_name(zerr)
        data_args = {'version':version, 'domain':domain, 'tracer':tracer[:3], 'zrange':zrange,
                     'mock_id': mockids[0], 'region': region, "use_dv": use_dv, "z_evol": z_evol}
        if len(mockids) > 1:
            data_args['mock_ids'] = mockids
        covariance_args = {'source': 'mock', 'version': 'holi-v3', 'mock_ids': range(1000), 
                           'corrections': ['hartlap', 'percival'], 'rescale': False}
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
            elif todo == 'sample':
                sampler_cls, sampler_init_options, sampler_run_options = get_sampler_cls(name=args.sampler)
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
