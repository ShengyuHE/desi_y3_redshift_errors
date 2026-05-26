#!/usr/bin/env python

'''
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/homes/s/shengyu/env.sh 2pt_env
srun -n 4 python compute_2pt.py
'''

import os
import sys
import fitsio
import argparse
import logging
import itertools
import numpy as np
import lsstypes as types
from pathlib import Path
from astropy.table import Table, vstack
# from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
from pypower import CatalogFFTPower,mpi, setup_logging
from pycorr import TwoPointCorrelationFunction, setup_logging
setup_logging()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compute_2pt') 

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_ABACUSHF, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO, GET_REDSHIFT_SET, NRAN_Y3, SKIP_HOLI_ID
from cat_tools import get_proposal_mattrs, read_positions_weights, get_measurement_fn

SKIP_HOLI_ID_SET = np.loadtxt('./dubious_holi-v3-altmtl.txt', dtype=int)

def _parse_zerr_name(zerr):
    zerr = str(zerr)
    z_evol = zerr.endswith('_zevol')
    use_dv = zerr[:-6] if z_evol else zerr
    valid = {'None', 'False', 'repeat', 'verr_empirical', 'verr_nonparam'}
    if use_dv not in valid:
        raise ValueError(f"Unsupported zerr label {zerr!r}")
    if use_dv in {'None', 'False'} and z_evol:
        raise ValueError(f"z_evol is not valid with zerr={zerr!r}")
    if use_dv in {'None', 'False'}:
        use_dv = 'None'
        z_evol = False
    return use_dv, z_evol

# basic settings
BOXSIZE = 2000
ells = (0, 2, 4)
kedges   = np.arange(0.,0.4001,0.001) # for PK
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201)) # for 2PCF
slogedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for small scale 2PCF
rlogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for Projected CF

def compute_box_2pt(fn, get_data, overwrite=False, **args):
    """
    Compute a set of two-point statistics (configuration- and Fourier-space) for a cubic mock using pycorr / pypower.

    Parameters
    ----------
    fn : str
        Filename template for saving/loading results. Must contain one
        '{}' placeholder that will be formatted with:
            - 'xipoles' : ξℓ(s) from smu-binned correlation function
            - 'pkpoles' : Pℓ(k) from FFT-based power spectrum
            - 'mpslog'  : ξℓ(s) on logarithmic s-bins
            - 'wplog'   : projected correlation function w_p(r_p)
    """
    boxsize = args.get('boxsize', 2000)
    los = args.get('los', 'z')
    data_positions, _ = tuple(x.T for x in get_data())
    # compute mps
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions,
                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                 gpu=True, nthreads=4, mpiroot=None, mpicomm=mpicomm)
        result_mps.save(fn_mps)
        if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_mps}')
    else:
        if mpicomm.rank == mpiroot: result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    # fn_pk = fn.format('pkpoles')
    # if not os.path.exists(fn_pk) or overwrite==True:
    #     result_pk = CatalogFFTPower(edges=kedges, data_positions1=data_positions, ells=ells,
    #                                 boxsize=boxsize, resampler='tsc',los=los, position_type='xyz',
    #                                 interlacing=3, nmesh=512, mpiroot=None, mpicomm=mpicomm)
    #     result_pk.save(fn_pk)
    #     if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_pk}')
    # else:
    #     if mpicomm.rank == mpiroot: result_pk = CatalogFFTPower.load(fn_pk)
    # compute mps log scales
    fn_mpslog = fn.format('mpslog')
    if not os.path.exists(fn_mpslog) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', slogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                gpu=True, nthreads = 4, mpiroot=None, mpicomm=mpicomm)
        result_mps.save(fn_mpslog)
        if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_mpslog}')
    else:
        if mpicomm.rank == mpiroot: result_mps = TwoPointCorrelationFunction.load(fn_mpslog)
    # compute projected correlation function wp
    fn_wplog = fn.format('wplog')
    if not os.path.exists(fn_wplog) or overwrite==True:
        result_wp = TwoPointCorrelationFunction('rppi', rlogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                nthreads = 4, mpiroot=None, mpicomm=mpicomm)
        result_wp.save(fn_wplog)
        if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_wplog}')
    else:
        if mpicomm.rank == mpiroot: result_wp = TwoPointCorrelationFunction.load(fn_wplog)

def compute_cutsky_2pt(fn, get_data, get_randoms, overwrite=False, **args):
    tracer = args.get('tracer', 'LRG')
    los = args.get('los', 'firstpoint')
    gpu = args.get('gpu', False)
    nthreads = args.get('nthreads', 64)
    # fn_mps = fn.format('xipoles')
    fn_mpslog = fn.format('mpslog')
    fn_wplog = fn.format('wplog')
    need_catalogs = overwrite or any(not os.path.exists(tmp) for tmp in [fn_mpslog, fn_wplog])
    if need_catalogs:
        data_positions, data_weights = get_data()
        random_positions, randoms_weights = get_randoms()
        # read_positions_weights(..., use_jax=False) returns RA/DEC in radians;
        # Corrfunc's rdd mode expects angular coordinates in degrees.
        data_positions = np.asarray(data_positions, dtype='f8').copy()
        random_positions = np.asarray(random_positions, dtype='f8').copy()
        data_positions[:, :2] = np.degrees(data_positions[:, :2])
        random_positions[:, :2] = np.degrees(random_positions[:, :2])
        data_positions = data_positions.T
        random_positions = random_positions.T

    # if not os.path.exists(fn_mps) or overwrite==True:
    #     result_mps = TwoPointCorrelationFunction('smu', smuedges, 
    #                                              data_positions1=data_positions, data_weights1=data_weights,
    #                                              randoms_positions1=random_positions, randoms_weights1=randoms_weights,
    #                                              engine='corrfunc', position_type = 'rdd', los=los,
    #                                              gpu=gpu, nthreads = nthreads, mpiroot=None, mpicomm=mpicomm)
    #     result_mps.save(fn_mps)
    #     if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_mps}')
    # else:
    #     result_mps = TwoPointCorrelationFunction.load(fn_mps)

    if not os.path.exists(fn_mpslog) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', slogedges, 
                                                 data_positions1=data_positions, data_weights1=data_weights,
                                                 randoms_positions1=random_positions, randoms_weights1=randoms_weights,
                                                 engine='corrfunc', position_type = 'rdd', los=los,
                                                 gpu=gpu, nthreads = nthreads,mpiroot=None, mpicomm=mpicomm)
        result_mps.save(fn_mpslog)
        if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_mpslog}')
    else:
        if mpicomm.rank == mpiroot: result_mps = TwoPointCorrelationFunction.load(fn_mpslog)
    # compute projected correlation function wp
    if not os.path.exists(fn_wplog) or overwrite==True:
        result_wp = TwoPointCorrelationFunction('rppi', rlogedges, 
                                                data_positions1=data_positions, data_weights1=data_weights,
                                                randoms_positions1=random_positions, randoms_weights1=randoms_weights,
                                                engine='corrfunc', position_type = 'rdd', los=los,
                                                nthreads = nthreads, mpiroot=None, mpicomm=mpicomm)
        result_wp.save(fn_wplog)
        if mpicomm.rank == mpiroot: logger.info(f'Save to {fn_wplog}')
    else:
        if mpicomm.rank == mpiroot: result_wp = TwoPointCorrelationFunction.load(fn_wplog)

    '''
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk) or overwrite==True:
        data_positions, data_weights = tuple(x.T for x in get_data())
        random_positions, randoms_weights = tuple(x.T for x in get_randoms())
        mat = get_proposal_mattrs(domain='cutsky', tracer=tracer)
        result_pk = CatalogFFTPower(edges=kedges, ells=ells,
                                    data_positions1=data_positions, data_weights1=data_weights,
                                    randoms_positions1=random_positions, randoms_weights1=randoms_weights,
                                    resampler='tsc', position_type = 'rdd', los=los,
                                    interlacing=3, nmesh=mat['meshsize'],
                                    mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
        if mpicomm.rank == mpiroot: logger.info(f'Writing to {fn_pk}')
    else:
        result_pk = CatalogFFTPower.load(fn_pk)
    '''

def combine_regions(output_fn, fns, labels=('xipoles', 'mpslog', 'wplog'), overwrite=False):
    ok = True
    for label in labels:
        output_label_fn = output_fn.format(label)
        input_label_fns = [fn.format(label) for fn in fns]
        missing = [fn for fn in input_label_fns if not os.path.exists(fn)]
        if missing:
            if mpicomm.rank == mpiroot:
                logger.warning(f"Cannot combine {label}; missing input files: {missing}")
            ok = False
            continue
        if os.path.exists(output_label_fn) and not overwrite:
            continue
        error = None
        if mpicomm.rank == mpiroot:
            try:
                results = [TwoPointCorrelationFunction.load(fn) for fn in input_label_fns]
                combined = results[0]
                for result in results[1:]:
                    combined = combined + result
                combined.save(output_label_fn)
                logger.info(f'Save to {output_label_fn}')
            except Exception as exc:
                error = str(exc)
        error = mpicomm.bcast(error, root=mpiroot)
        if error is not None:
            if mpicomm.rank == mpiroot:
                logger.warning(f"Failed to combine {label}: {error}")
            ok = False
    mpicomm.Barrier()
    return ok

########################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3'])
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl'], help="mock domain")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--regions", nargs = '+', type=str, default=['ALL'], help="Region labels for cutsky/altmtl runs, e.g. ALL NGC SGC GCcomb")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, choices ['None/False', 'repeat', 'verr_empirical', 'verr_nonparam']")
    parser.add_argument("--task", nargs = '+', type=str, default=['xi'], choices=['xi'], help="task types")
    parser.add_argument("--nran", type=int, default=None, help="Optional number of random catalogs to use for cutsky/altmtl")
    parser.add_argument("--nthreads", type=int, default=32, help="Number of Corrfunc CPU threads per MPI rank")
    parser.add_argument("--gpu", action="store_true", help="Use Corrfunc GPU pair counts for xi")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing measurements")
    args = parser.parse_args()
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")
    version = args.version
    domain = args.domain
    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
    use_jax=False
    z_snaps, z_ranges = GET_REDSHIFT_SET(version, domain) if domain == 'altmtl' else REDSHIFT_ABACUSHF[version]
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(z_snaps[tracer][:], z_ranges[tracer][:]):
            tracer_redshifts.append((tracer, zp, zr))

    tracer_redshifts = tracer_redshifts[:]
    regions = [None] if domain == 'cubic' else args.regions
    for (tracer, zsnap, zrange), mock_id, zerr, region in itertools.product(tracer_redshifts, mockids, args.zerrs, regions):
        if version == 'holi-v3' and domain == 'altmtl' and mock_id in SKIP_HOLI_ID_SET:
            if mpicomm.rank == mpiroot: logger.warning(f'Skipping holi-v3 altmtl mock_id={mock_id}')
            continue
        mock_id03 =  f"{mock_id:03}"
        use_dv, z_evol = _parse_zerr_name(zerr)
        data_args = {'version':version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id, 'region': region, "use_dv": use_dv, "z_evol": z_evol}
        fn_2pt = get_measurement_fn(**data_args, use_jax=use_jax)
        if mpicomm.rank == mpiroot: logger.info(f'Proceed {data_args}')
        if region in ['GCcomb', 'ALL']:
            region_fns = [get_measurement_fn(**(data_args | {'region': r}), use_jax=use_jax) for r in ['NGC', 'SGC']]
            combine_regions(fn_2pt, region_fns, overwrite=args.overwrite)
            continue
        io_cache = {}
        if domain == 'cubic':
            def get_data():
                if 'data' not in io_cache:
                    io_cache['data'] = read_positions_weights(**data_args)
                return io_cache['data']
            compute_box_2pt(fn_2pt, get_data, overwrite = args.overwrite)
        elif domain in ['cutsky', 'altmtl']:
            def get_data():
                if 'data' not in io_cache:
                    io_cache['data'] = read_positions_weights(**data_args, use_jax=False)
                return io_cache['data']
            def get_random():
                if 'random' not in io_cache:
                    io_cache['random'] = read_positions_weights(**data_args, random=True, nran=args.nran or NRAN_Y3[tracer], use_jax=False)
                return io_cache['random']
            if 'xi' in args.task:
                compute_cutsky_2pt(fn_2pt, get_data, get_random, overwrite=args.overwrite, tracer=tracer, los='firstpoint', gpu=args.gpu, nthreads=args.nthreads)
            if 'pk' in args.task and mpicomm.rank == mpiroot:
                logger.warning('pk task is requested, but cutsky/altmtl pk computation is currently disabled in compute_2pt.py')
        else:
            raise ValueError(f"Unsupported domain {domain!r}")