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
# from pycorr import TwoPointCorrelationFunction, setup_logging
setup_logging()

logger = logging.getLogger('compute_2pt') 

mpicomm = mpi.COMM_WORLD
mpiroot = 0

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS, TRACER_CUTSKY_INFO
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR
from cat_tools import get_proposed_mattrs, read_positions_weights, get_measurement_fn

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

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
    data_positions, _ = get_data()
    # compute mps
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions,
                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                 gpu=True, nthreads=4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
        logger.info(f'Writing to {fn_mps}')
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk) or overwrite==True:
        result_pk = CatalogFFTPower(edges=kedges, data_positions1=data_positions, ells=ells,
                                    boxsize=boxsize, resampler='tsc',los=los, position_type='xyz',
                                    interlacing=3, nmesh=512, mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
        logger.info(f'Writing to {fn_pk}')
    else:
        result_pk = CatalogFFTPower.load(fn_pk)
    # compute mps log scales
    fn_mpslog = fn.format('mpslog')
    if not os.path.exists(fn_mpslog):
        result_mps = TwoPointCorrelationFunction('smu', slogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                gpu=True, nthreads = 4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mpslog)
        logger.info(f'Writing to {fn_mpslog}')
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mpslog)
    # compute projected correlation function wp
    fn_wplog = fn.format('wplog')
    if not os.path.exists(fn_wplog) or overwrite==True:
        result_wp = TwoPointCorrelationFunction('rppi', rlogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                nthreads = 4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_wp.save(fn_wplog)
        logger.info(f'Writing to {fn_wplog}')
    else:
        result_wp = TwoPointCorrelationFunction.load(fn_wplog)

def compute_cutsky_2pt(fn, get_data, get_randoms, overwrite=False, **args):
    tracer = args.get('tracer', 'LRG')
    # fn_mps = fn.format('xipoles')
    # if not os.path.exists(fn_mps) or overwrite==True:
    #     data_positions, data_weights = get_data()
    #     random_positions, _randoms_weights = get_randoms()
    #     result_mps = TwoPointCorrelationFunction('smu', smuedges, 
    #                                              data_positions1=data_positions, data_weights1=None,
    #                                              randoms_positions1=random_positions, randoms_weights1=None,
    #                                              engine='corrfunc', position_type = 'rdd', #los = 'firstpoint',
    #                                              # D1D2 = D1D2, R1R2 = R1R2,
    #                                              gpu=True, nthreads = 256,mpiroot=mpiroot, mpicomm=mpicomm)
    #     result_mps.save(fn_mps)
    #     logger.info(f'Writing to {fn_mps}')
    # else:
    #     result_mps = TwoPointCorrelationFunction.load(fn_mps)
    fn_pk = fn.format('pkpoles')
    output_fn = './notebooks/tests/LRG_pkpoles.npy'
    if not os.path.exists(fn_pk) or overwrite==True:
        data_positions, data_weights = get_data()
        random_positions, _randoms_weights = get_randoms()
        mat = get_proposed_mattrs(tracer)
        isplit = 9 
        randoms_positions1 = np.array_plit()
        for random in randoms_positions1():
            result_pk = CatalogFFTPower(edges=kedges, ells=ells,
                                        data_positions1=data_positions, data_weights1=None,
                                        randoms_positions1=random_positions, randoms_weights1=None,
                                        position_type='rdd', resampler='tsc',
                                        interlacing=3, boxsize = mat['boxsize'], cellsize = mat['cellsize'],
                                        mpiroot=mpiroot, mpicomm=mpicomm)


        result_pk.save(fn_pk)
        logger.info(f'Writing to {fn_pk}')
    else:
        result_pk = CatalogFFTPower.load(fn_pk)

def compute_pk_by_jaxpower(fn, get_data, get_random,  overwrite=True, **args):
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    jax.distributed.initialize()
    from jaxpower.mesh import create_sharding_mesh
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)
    # output_fn = fn.format('mesh2_spectrum_poles')
    output_fn = './notebooks/tests/LRG_mesh2_spectrum_poles.h5'
    if not os.path.exists(output_fn) or overwrite==True:
        los = 'firstpoint'
        data = tuple(x.T for x in get_data())
        randoms = tuple(x.T for x in get_random())
        with create_sharding_mesh() as sharding_mesh:
            mattrs = get_mesh_attrs(data[0], randoms[0], check=True)
            data1 = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
            randoms1 = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
            fkp = FKPField(data1, randoms1)
            # Paint FKP field to mesh
            mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
            bin = BinMesh2SpectrumPoles(mattrs,  edges={'step': 0.001}, ells=ells)
            norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
            num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
            wsum_data1 = data1.sum()
            jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
            spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
            jax.block_until_ready(spectrum)
            mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
            spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
            if output_fn is not None and jax.process_index() == 0:
                logger.info(f'Writing to {output_fn}')
                spectrum.write(output_fn)
    else:
        types.read(output_fn)

########################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs = '+', type = str,  default=['AbacusHF-v1'], help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2'])
    parser.add_argument("--domains", nargs = '+', type = str, default=['cutsky'], choices=['cubic', 'cutsky', 'cutsky_QSO'], help="mock domain: cubic box or cut-sky survey footprint")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--zerrs", nargs = '+', type = str, default=[False], help="redshift error input, bins choices ['LSS', 'global', 'bin']")
    parser.add_argument("--task", nargs = '+', type=str, default=['pk'], choices=['xi', 'pk'], help="task types")
    args = parser.parse_args()
    if mpicomm.rank == mpiroot:
        logger.info(f"Received arguments: {args}")
    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    use_jax=True

    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(REDSHIFT_ABACUSHF_v1[tracer][:1], REDSHIFT_BIN_LSS[tracer][:1]):
            tracer_redshifts.append((tracer, zp, zr))

    weight_type = 'default' 
    for version, domain, (tracer, zsnap, zrange), mock_id, use_zerr in itertools.product(args.versions, args.domains, tracer_redshifts, mockids, args.zerrs):
        mock_id03 =  f"{mock_id:03}"
        data_args = {'version':version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id, 'use_zrr': use_zerr}
        fn_2pt = get_measurement_fn(**data_args, use_jax=True)
        if mpicomm.rank == mpiroot:
            logger.info(f'Procceed {data_args}')
        if domain == 'cubic':
            get_data = lambda: read_positions_weights(**data_args)
            compute_box_2pt(fn_2pt, get_data)
        elif domain == 'cutsky':
            get_data = lambda: read_positions_weights(**data_args)
            get_random = lambda:  read_positions_weights(**data_args, use_random = True)
            # if use_jax == True:
            compute_pk_by_jaxpower(fn_2pt, get_data, get_random, **data_args)
            # else:
            # compute_cutsky_2pt(fn_2pt, get_data, get_random, **data_args)
        continue

        '''
        elif domain == 'cutsky_QSO':
            for (zmin, zmax) in [(0.8, 1.4), (1.4, 2.1)]:
                mock_dir = BASE_DIR+ f'/Cutsky/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering'   
                fn_path = mock_dir+ '/mpspk'
                for use_zerr in args.mzrr:
                    if use_zerr == 'bin':
                        (z1, z2) = REDSHIFT_BIN_LSS[tracer][indz]
                        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_zbin{zmin}-{zmax}_DR2_v1.0_zobs{z1:.1f}-{z2:.1f}_bin.npy'
                    else:
                        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_zbin{zmin}-{zmax}_DR2_v1.0.npy'
                    data_args = {'tracer':tracer, 'indz':indz, 'mock_id03':mock_id03, 'domain':domain, 'zmin': zmin, 'zmax': zmax}
                    if mpicomm.rank == mpiroot:
                        print(data_args, flush=True)
                    data_positions, _ = read_positions_weights(data_args, use_zerr = use_zerr)
                    random_positions, _ =  read_positions_weights(data_args, use_random = True)
                    # if 'xi' in args.corr:
                    #     compute_cutsky_2pcf(data_positions, random_positions, fn_2pt, **data_args)
                    # if 'pk' in args.corr:
                    #     compute_cutsky_pk(data_positions, random_positions, fn_2pt, **data_args)
            '''