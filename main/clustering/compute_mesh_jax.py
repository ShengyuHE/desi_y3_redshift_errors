#!/usr/bin/env python

'''
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/homes/s/shengyu/env.sh 2pt_env
srun -n 4 python compute_mesh_jax.py
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
logger = logging.getLogger('compute_box') 

mpicomm = mpi.COMM_WORLD
mpiroot = 0

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_ABACUSHF, REDSHIFT_LSS, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO, NRAN_Y3, NRAN_TEST
from helper import GET_REDSHIFT_SET
from cat_tools import get_proposal_mattrs, read_positions_weights, get_measurement_fn

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

# basic settings
use_jax=True
BOXSIZE = 2000

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

def compute_mesh2_box(output_fn, get_data, ells=(0, 2, 4), los='z', cache=None, **attrs):
    import jax
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_box2_normalization, compute_fkp2_shotnoise)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        mattrs = get_mesh_attrs(**attrs)
        if cache is None: cache = {}
        bin = cache.get('bin_mesh2_spectrum', None)
        if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
        cache.setdefault('bin_mesh2_spectrum', bin)
        data = ParticleField(*get_data(), attrs=mattrs, exchange=True)
        norm = compute_box2_normalization(data, bin=bin)
        num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
        mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        mesh = mesh - mesh.mean()
        del data
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
        spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
        mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
        spectrum = spectrum.clone(attrs=dict(los=los, **mattrs))
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot: 
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_mesh3_box(output_fn, get_data, get_shifted=None, basis='scoccimarro', ells=(0, 2), los='z', mask_edges=None, cache=None, buffer_size=16, **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_box3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        mattrs = get_mesh_attrs(**attrs)
        data = ParticleField(*get_data(), attrs=mattrs, exchange=True)
        edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005} #, 'max': 0.4}
        if cache is None: cache = {}
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size, mask_edges=mask_edges)
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
        norm = compute_box3_normalization(data, bin=bin)
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
        mesh = data.paint(**kw, out='real')
        mesh = mesh - mesh.mean()
        del data
        spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot: 
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_mesh2_cutsky(output_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', cache=None, **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs,
                          compute_mesh2_spectrum)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        data, randoms = get_data(), get_randoms()
        mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
        # Force MPI exchange for the distributed cutsky/altmtl workflow.
        data = ParticleField(*data, attrs=mattrs, exchange=True, backend='mpi')
        randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='mpi')
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
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3'])
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl'], help="mock domain")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, e.g. 'None', 'repeat', 'verr_empirical', 'verr_nonparam' with '_zevol' for redshift evolution")
    parser.add_argument("--todo", nargs = '+', type=str, default=['mesh2'], choices=['mesh2', 'mesh3_scoccimarro', 'mesh3_sugiyama'], help="todo types")
    parser.add_argument("--meshsize", type=int, default=None, help="Optional meshsize override for mesh runs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
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
    version = args.version
    domain = args.domain
    z_snaps, z_ranges = GET_REDSHIFT_SET(version, domain)
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(z_snaps[tracer][:1], z_ranges[tracer][:1]):
            tracer_redshifts.append((tracer, zp, zr))
    for (tracer, zsnap, zrange), mock_id, zerr, todo in itertools.product(tracer_redshifts, mockids, args.zerrs, args.todo[:]):
        mock_id03 =  f"{mock_id:03}"
        use_dv, z_evol = _parse_zerr_name(zerr)
        data_args = {'version':version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id, "use_dv": use_dv, "z_evol": z_evol, "overwrite":args.overwrite}
        if mpicomm.rank == mpiroot: logger.info(f'Procceed {data_args}')
        if domain == 'cubic':
            get_data = lambda: read_positions_weights(**data_args)
            spectrum_args = dict(boxcenter=0., boxsize=2000., cellsize=5., ells=(0, 2, 4), los='z')
        elif domain in ['cutsky', 'altmtl']:
            get_data = lambda: read_positions_weights(**data_args)
            get_random = lambda: read_positions_weights(**data_args, random=True, nran=NRAN_Y3[tracer])
            spectrum_args = dict(**get_proposal_mattrs(domain=domain, tracer=tracer[:3]), ells=(0, 2, 4), los='firstpoint')
        else:
            raise ValueError(f"Unsupported domain {domain!r}")
        if args.meshsize is not None:
            spectrum_args['meshsize'] = args.meshsize
        output_fn = get_measurement_fn(**data_args, use_jax=use_jax)
        cache = {}
        if 'mesh2' in todo:
            pk_fn = output_fn.format('mesh2_spectrum_poles')
            if not os.path.exists(pk_fn) or args.overwrite:
                if domain == 'cubic': compute_mesh2_box(pk_fn, get_data, **spectrum_args)
                if domain in ['cutsky', 'altmtl']: compute_mesh2_cutsky(pk_fn, get_data, get_random,  **spectrum_args)
            else:
                types.read(pk_fn)
            jax.clear_caches()
        if 'mesh3' in todo:
            if domain != 'cubic':
                raise ValueError(f"mesh3 is only implemented for cubic catalogs, got domain {domain!r}")
            if 'scoccimarro' in todo:
                basis = 'scoccimarro'
                bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2], cellsize=10)
            elif 'sugiyama' in todo:
                basis = 'sugiyama'
                bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 2, 0), (2, 0, 2), (2, 2, 2)], cellsize=5, buffer_size=8)
            bk_fn = output_fn.format(f'mesh3_spectrum_poles_{basis}')
            if not os.path.exists(bk_fn) or args.overwrite:
                compute_mesh3_box(bk_fn, get_data, **bispectrum_args)
            else:
                types.read(bk_fn)
            jax.clear_caches()
