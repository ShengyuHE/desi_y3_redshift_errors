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
logger = logging.getLogger('compute_cutsky') 

mpicomm = mpi.COMM_WORLD
mpiroot = 0

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_ABACUSHF, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO
from cat_tools import get_proposed_mattrs, read_positions_weights, get_measurement_fn

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

# basic settings
use_jax=True

def compute_mesh2_cutsky(fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint',  overwrite=True, **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, 
                          compute_mesh2_spectrum, BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    individual_weight = data[1]
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)
    norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    del data, randoms
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    jax.block_until_ready(spectrum)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    jax.block_until_ready(spectrum)
    if mpicomm.rank == mpiroot: 
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    mpicomm.Barrier()
    return spectrum

########################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2'])
    parser.add_argument("--domains", nargs = '+', type = str, default=['cubic'], choices=['cubic'], help="mock domain: cubic box in this script")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['LRG'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['False'], help="redshift error input, choices ['False', 'repeat', 'verr_empirical']")
    parser.add_argument("--todo", nargs = '+', type=str, default=['mesh3_scoccimarro', 'mesh3_sugiyama'], choices=['mesh2', 'mesh3_scoccimarro', 'mesh3_sugiyama'], help="todo types")
    parser.add_argument("--overwrite", type=bool, default=True, choices=[True, False])
    args = parser.parse_args()
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")

    # jax configuration
    import jax
    from jax import config
    from jaxpower.mesh import create_sharding_mesh
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    jax.distributed.initialize()

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
    z_snaps, z_ranges = REDSHIFT_ABACUSHF[args.version]
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(z_snaps[tracer][:], z_ranges[tracer][:]):
            tracer_redshifts.append((tracer, zp, zr))
    tracer_redshifts = tracer_redshifts[:1]

    weight_type = 'default'
    for domain, (tracer, zsnap, zrange), mock_id, use_dv, todo in itertools.product(args.domains, tracer_redshifts, mockids, args.zerrs, args.todo):
        mock_id03 =  f"{mock_id:03}"
        data_args = {'version':args.version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id, "use_dv": use_dv}
        if mpicomm.rank == mpiroot: logger.info(f'Procceed {data_args}')
        get_data = lambda: read_positions_weights(**data_args)
        # output_fn = get_measurement_fn(**data_args, use_jax=use_jax)
        output_fn = f'/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering/notebooks/tests/{{}}.h5'
        spectrum_args = dict(boxsize=2000., cellsize=5., ells=(0, 2, 4))
        cache = {}
        if 'mesh2' in todo:
            pk_fn = output_fn.format('mesh2_spectrum_poles')
            if not os.path.exists(pk_fn) or args.overwrite==True:
                compute_mesh2_cutsky(pk_fn, get_data, **spectrum_args)
                jax.clear_caches()
        # if 'mesh3' in todo:
        #     if 'scoccimarro' in todo:
        #         basis = 'scoccimarro',
        #         bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2], cellsize=10)
        #     elif 'sugiyama' in todo:
        #         basis = 'sugiyama'
        #         bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], cellsize=10)
        #         bispectrum_args.pop('cellsize')
        #         bispectrum_args.update(meshsize=512, buffer_size=8)
        #     bk_fn = output_fn.format(f'mesh3_spectrum_poles_{basis}')
        #     if not os.path.exists(bk_fn) or args.overwrite==True:
        #         with create_sharding_mesh() as sharding_mesh:
        #             compute_mesh3_box_by_jaxpower(pk_fn, get_data, **spectrum_args)
        #             jax.clear_caches()
