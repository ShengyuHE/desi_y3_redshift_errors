#!/usr/bin/env python

import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit
from mockfactory import utils, DistanceToRedshift, Catalog, RandomBoxCatalog
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
from pypower import CatalogFFTPower,mpi, setup_logging
from pycorr import TwoPointCorrelationFunction, setup_logging
setup_logging()

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_OVERALL, COLOR_OVERALL
from helper import REDSHIFT_VSMEAR, REDSHIFT_LSS_VSMEAR, REDSHIFT_CUBICBOX, EDGES, COLOR_TRACERS, GET_RECON_BIAS

# basic settings
kedges   = np.arange(0.,0.4001,0.001); ells = (0, 2, 4) # for PK
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201)) # for 2PCF
slogedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for small scale 2PCF
rlogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201)) # for Projected CF

ABACUSHF_DIR = f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0'

def compute_2pt(data_positions, shifted_positions, fn, **args):
    boxsize = args.get('boxsize', 2000)
    los = args.get('los', 'z')
    recon = args.get('recon', False)
    # compute mps
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps):
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions, 
                                                    shifted_positions1 = shifted_positions, 
                                                    engine='corrfunc', 
                                                    boxsize=boxsize, los=los, position_type='xyz',
                                                    gpu=True, nthreads = 4)
                                                #  mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk):
        result_pk = CatalogFFTPower(data_positions1=data_positions, 
                                    shifted_positions1 = shifted_positions, 
                                    edges=kedges, ells=ells, interlacing=3, 
                                    boxsize=boxsize, nmesh=512, resampler='tsc',los=los, position_type='xyz',)
                                    # mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
    else:
        result_pk = CatalogFFTPower.load(fn_pk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['BGS','LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    args = parser.parse_args()
    print(f"Received arguments: {args}", flush=True)

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))


