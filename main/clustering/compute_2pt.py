#!/usr/bin/env python

import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
from pypower import CatalogFFTPower,mpi, setup_logging
from pycorr import TwoPointCorrelationFunction, setup_logging
setup_logging()

mpicomm = mpi.COMM_WORLD
mpiroot = 0

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS, TRACER_CUTSKY_INFO
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR
from cat_tools import get_proposed_mattrs, read_positions_weights

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

# basic settings
BOXSIZE = 2000
ells = (0, 2, 4)
kedges   = np.arange(0.,0.4001,0.001) # for PK
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201)) # for 2PCF
slogedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for small scale 2PCF
rlogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for Projected CF

ABACUS_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1'

def compute_box_2pt(data_positions, fn, overwrite=False, **args):
    """
    Compute a set of two-point statistics (configuration- and Fourier-space) for a cubic mock using pycorr / pypower.

    Parameters
    ----------
    data_positions : ndarray of shape (3, N)
        Cartesian coordinates of the tracer sample in the box:
        data_positions = [X, Y, Z], in units consistent with `boxsize`.
        Produced by `read_data_positions(..., domain='cubic')`.

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
    # compute mps
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions,
                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                 gpu=True, nthreads=4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk) or overwrite==True:
        result_pk = CatalogFFTPower(edges=kedges, data_positions1=data_positions, ells=ells,
                                    boxsize=boxsize, resampler='tsc',los=los, position_type='xyz',
                                    interlacing=3, nmesh=512, mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
    else:
        result_pk = CatalogFFTPower.load(fn_pk)
    # compute mps log scales
    fn_mpslog = fn.format('mpslog')
    if not os.path.exists(fn_mpslog):
        result_mps = TwoPointCorrelationFunction('smu', slogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                gpu=True, nthreads = 4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mpslog)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mpslog)
    # compute projected correlation function wp
    fn_wplog = fn.format('wplog')
    if not os.path.exists(fn_wplog) or overwrite==True:
        result_wp = TwoPointCorrelationFunction('rppi', rlogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                nthreads = 4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_wp.save(fn_wplog)
    else:
        result_wp = TwoPointCorrelationFunction.load(fn_wplog)

def compute_cutsky_2pcf(data_positions, random_positions, fn, overwrite=False, **args):
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps) or overwrite==True:
        result_mps = TwoPointCorrelationFunction('smu', smuedges, 
                                                 data_positions1=data_positions, data_weights1=None,
                                                 randoms_positions1=random_positions, randoms_weights1=None,
                                                 engine='corrfunc', position_type = 'rdd', #los = 'firstpoint',
                                                 # D1D2 = D1D2, R1R2 = R1R2,
                                                 gpu=True, nthreads = 256,mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)

def compute_cutsky_pk(data_positions, random_positions, fn, overwrite=True, **args):
    fn_pk = fn.format('pkpoles')
    tracer = args.get('tracer', 'LRG')
    mat = get_proposed_mattrs(tracer)
    if not os.path.exists(fn_pk) or overwrite==True:
        result_pk = CatalogFFTPower(edges=kedges, ells=ells,
                                    data_positions1=data_positions, data_weights1=None,
                                    randoms_positions1=random_positions, randoms_weights1=None,
                                    position_type='rdd', resampler='tsc',
                                    interlacing=3, boxsize = mat['boxsize'], cellsize = mat['cellsize'],
                                    mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
    else:
        result_pk = CatalogFFTPower.load(fn_pk)

####################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzrr", help="redshift error input, bins choices ['LSS', 'global', 'bin']", default=[False], nargs = '+')
    parser.add_argument("--version", help="mock types", choices=['AbacusHF_v1'], default='AbacusHF_v1')
    parser.add_argument("--domain", help="mock domain: cubic box or cut-sky survey footprint", choices=['cubic', 'cutsky', 'cutsky_QSO'], default='cutsky')
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['QSO'], nargs = '+')
    parser.add_argument("--mockid", help="Mock ID range or list", type=str, default="0-24")
    parser.add_argument("--corr", help="correlation types", type=str, choices=['xi', 'pk'], default=['pk'],  nargs = '+')
    args = parser.parse_args()
    if mpicomm.rank == mpiroot:
        print(f"Received arguments: {args}", flush=True)
    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
    if args.version == 'AbacusHF_v1':
        BASE_DIR = ABACUS_DIR
    domain = args.domain

    for tracer in args.tracers:
        for indz, zsnap in enumerate(REDSHIFT_ABACUSHF_v1[tracer][:1]):
            for mock_id in mockids:
                mock_id03 =  f"{mock_id:03}"
                if 'cubic' in domain:
                    mock_dir = BASE_DIR+ f'/Boxes/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}'
                    fn_path = mock_dir+ '/mpspk'
                    if not os.path.exists(fn_path): os.makedirs(fn_path)
                    # print(fn_path, flush = True)
                    for use_zerr in args.mzrr:
                        if use_zerr == 'LSS':
                            (z1, z2) = REDSHIFT_BIN_LSS[tracer][indz]
                            fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0_zobs{z1:.1f}-{z2:.1f}.npy'
                        else:
                            fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0.npy'
                        recon = False
                        data_args = {'tracer':tracer, 'indz':indz, 'mock_id03':mock_id03, 'domain':domain, 
                                    'boxsize':BOXSIZE, 'loc':'z', 'recon':recon}
                        if mpicomm.rank == mpiroot:
                            print(data_args, flush=True)
                        data_positions = read_positions_weights(data_args, use_zerr)
                        compute_box_2pt(data_positions, fn_2pt)
                elif 'cutsky' in domain:
                    (zmin, zmin) = REDSHIFT_BIN_LSS[tracer][indz]
                    tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
                    fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
                    mock_dir = BASE_DIR+ f'/Cutsky/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering'
                    fn_path = mock_dir+ '/mpspk'
                    if not os.path.exists(fn_path): os.makedirs(fn_path)
                    for use_zerr in args.mzrr:
                        if use_zerr == 'global':
                            (z1, z2) = REDSHIFT_BIN_LSS[tracer][indz]
                            fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0_zobs{z1:.1f}-{z2:.1f}_global.npy'
                        elif use_zerr == 'bin':
                            (z1, z2) = REDSHIFT_BIN_LSS[tracer][indz]
                            fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0_zobs{z1:.1f}-{z2:.1f}_bin.npy'
                        else:
                            fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0.npy'
                        # if not os.path.exists(fn_2pt.format('xipoles')) or not os.path.exists(fn_2pt.format('pkpoles')):
                        data_args = {'tracer':tracer, 'indz':indz, 'mock_id03':mock_id03, 'domain':domain}
                        if mpicomm.rank == mpiroot:
                            print(data_args, flush=True)
                        data_positions, _ = read_positions_weights(data_args, use_zerr = use_zerr)
                        random_positions, _ =  read_positions_weights(data_args, use_random = True)
                        if 'xi' in args.corr:
                            compute_cutsky_2pcf(data_positions, random_positions, fn_2pt, **data_args)
                        if 'pk' in args.corr:
                            compute_cutsky_pk(data_positions, random_positions, fn_2pt, **data_args)
                        # else:
                            # print(f'[EXISTS]:{fn_2pt}', flush=True)
                elif 'cutsky_QSO' in domain:
                    for (zmin, zmax) in [(0.8, 1.4), (1.4, 2.1)]:
                        tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
                        fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
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