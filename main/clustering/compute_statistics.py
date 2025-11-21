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
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

# basic settings
BOXSIZE = 2000
ells = (0, 2, 4)
kedges   = np.arange(0.,0.4001,0.001) # for PK
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201)) # for 2PCF
slogedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for small scale 2PCF
rlogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for Projected CF

ABACUSHF_DIR = f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0'

def read_data_positions(args, use_zerr = False):
    """
    Return the 3D positions of tracer galaxies for either cubic-box or 
    light-cone mocks, formatted for pycorr / Corrfunc two-point estimators.

    Parameters
    ----------
    args : dict
        Dictionary containing the following keys:
            - 'tracer' : str
                Tracer name, e.g. 'LRG', 'ELG', 'QSO'.
            - 'z' : float
                Snapshot redshift of the mock (used for file naming).
            - 'mock_id03' : str or int
                Mock identifier formatted as a 3-digit string, e.g. '003'.
            - 'mock_dir' : str
                Directory where mock files are stored.
            - 'domain' : str
                Either 'cubic' or 'cutsky'. Determines how coordinates are read.
            - 'boxsize' : float, optional
                Size of the cubic box (default: 2000).
            - 'los' : str, optional
                Line-of-sight definition; default 'z'.

    use_zerr : bool, optional
        If False (default), use Z_RSD (true RSD displacement).
        If True, use Z_OBS (includes modeled redshift errors).

    Returns
    -------
    data_positions : ndarray of shape (3, N)
    """
    boxsize = args.get('boxsize', 2000)
    los = args.get('los', 'z')
    (tracer, zsnap, mock_id03, mock_dir, domain) = (args[key] for key in ["tracer", "z", "mock_id03", "mock_dir", "domain"])
    if domain == 'cubic':
        data_fn = mock_dir+ f'/abacus_HF_{tracer}_{zfmt(zsnap)}_DR2_v1.0_AbacusSummit_base_c000_ph{mock_id03}_clustering.dat.fits'
        data = Table.read(data_fn)
        if los == 'z':
            if use_zerr == False:
                data_positions = np.array([data['X'], data['Y'], data['Z_RSD']])%boxsize
            elif use_zerr == True:
                data_positions = np.array([data['X'], data['Y'], data['Z_OBS']])%boxsize
    return data_positions


def compute_box_2pt(data_positions, fn, **args):
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
    if not os.path.exists(fn_mps):
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions,
                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                 gpu=True, nthreads=4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk):
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
    if not os.path.exists(fn_wplog):
        result_wp = TwoPointCorrelationFunction('rppi', rlogedges, data_positions1=data_positions,
                                                engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
                                                nthreads = 4, mpiroot=mpiroot, mpicomm=mpicomm)
        result_wp.save(fn_wplog)
    else:
        result_wp = TwoPointCorrelationFunction.load(fn_wplog)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='2pt', help="statistics to compute")
    parser.add_argument("--version", help="mock types", choices=['AbacusHF_v1'], default='AbacusHF_v1')
    parser.add_argument("--domain", help="mock domain: cubic box or cut-sky survey footprint", choices=['cubic', 'cutsky'], default='cubic')
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['QSO'], nargs = '+')
    parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    args = parser.parse_args()
    print(f"Received arguments: {args}", flush=True)

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    if args.version == 'AbacusHF_v1':
        BASE_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1'

    for tracer in args.tracers:
        for zind, z in enumerate(REDSHIFT_ABACUSHF_v1[tracer]):
            for mock_id in mockids:
                mock_id03 =  f"{mock_id:03}"
                mock_dir = BASE_DIR+ f'/Boxes/{tracer}/z{z:.3f}/AbacusSummit_base_c000_ph{mock_id03}'
                fn_path = mock_dir+ '/mpspk'
                if not os.path.exists(fn_path): os.makedirs(fn_path)
                # print(fn_path, flush = True)
                for use_zerr in [True, False]:
                    if use_zerr is True:
                        (z1, z2) = REDSHIFT_BIN_LSS[tracer][zind]
                        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{z:.3f}_DR2_v1.0_dv_zobs{z1:.1f}-{z2:.1f}.npy'
                    else:
                        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{z:.3f}_DR2_v1.0.npy'
                    if '2pt' in args.task:
                        print(fn_2pt)
                        recon = False
                        data_args = {'tracer':tracer, 'z':z, 'mock_id03':mock_id03, 'mock_dir':mock_dir, 'domain':args.domain, 
                                     'boxsize':BOXSIZE, 'loc':'z', 'recon':recon}
                        data_positions = read_data_positions(data_args, use_zerr)
                        compute_box_2pt(data_positions, fn_2pt)