"""Precompute the holi-v3 covariances used by plot_clustering.ipynb."""
import argparse
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clustering.clustering_covariance import DEFAULT_COV_DIR, get_mock_covariance
from helper import REDSHIFT_BIN_TRACER
from cat_tools import parse_zerr_name
from mpi4py import MPI


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracers', nargs='+', default=list(REDSHIFT_BIN_TRACER),
                        choices=list(REDSHIFT_BIN_TRACER))
    parser.add_argument('--stats', nargs='+', default=['mesh2', 'mesh3_sugiyama', 'mpslog'],
                        choices=['mesh2', 'mesh3_sugiyama', 'mpslog'])
    parser.add_argument('--num', type=int, default=None,
                        help='Mock ID upper bound (default: 120 for mpslog, 1000 otherwise).')
    parser.add_argument('--zerrs', nargs='+', default=['None', 'verr_nonparam'])
    parser.add_argument('--regions', nargs='+', default=['ALL'])
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_COV_DIR)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    tasks = [task for task in itertools.product(args.tracers, args.regions, args.zerrs, args.stats)
             if not (task[0].startswith(('BGS', 'ELG')) and
                     (parse_zerr_name(task[2])[0] != 'None' or task[3] == 'mpslog'))]
    unavailable = []
    for tracer, region, zerr, stat in tasks[comm.rank::comm.size]:
        use_dv, z_evol = parse_zerr_name(zerr)
        catalog = dict(version='holi-v3', domain='altmtl', tracer=tracer,
                       zrange=REDSHIFT_BIN_TRACER[tracer], use_jax=stat != 'mpslog',
                       use_dv=use_dv, z_evol=z_evol, region=region)
        try:
            get_mock_covariance(catalog, stat, num=args.num,
                                bin_type='test' if stat == 'mesh2' else 'y3_sys',
                                cache_dir=args.cache_dir, rebuild=args.rebuild)
        except FileNotFoundError as exc:
            unavailable.append(str(exc))
            print(f'Unavailable: {exc}', flush=True)
    unavailable = [message for messages in comm.allgather(unavailable) for message in messages]
    if comm.rank == 0 and unavailable:
        print('\nUnavailable measurement sets:', *unavailable, sep='\n', flush=True)
