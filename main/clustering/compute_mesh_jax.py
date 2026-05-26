#!/usr/bin/env python

'''
salloc -N 1 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos debug --account desi_g
source /global/homes/s/shengyu/env.sh 2pt_env
srun -n 4 python compute_mesh_jax.py --tracers QSO --mockid 0 --region NGC --todos mesh3_sugiyama_window --overwrite
'''

import os
import sys
import jax
import time
import fitsio
import argparse
import logging
import itertools
import numpy as np
import lsstypes as types
from jax import numpy as jnp
from pathlib import Path
from astropy.table import Table, vstack

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))
from utils import setup_logging
from helper import REDSHIFT_ABACUSHF, REDSHIFT_LSS, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO, NRAN_Y3, NRAN_TEST
from helper import GET_REDSHIFT_SET, SKIP_HOLI_ID
from cat_tools import get_proposal_mattrs, read_positions_weights, get_measurement_fn, parse_zerr_name
from jax_support import get_interpolator_1d, initialize_jax_distributed

setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compute_mesh') 

# basic settings
use_jax=True
BOXSIZE = 2000
SKIP_HOLI_ID_SET = np.loadtxt('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/clustering/dubious_holi-v3-altmtl.txt', dtype=int)

def _parse_todo(todo, basis=None):
    valid = {'mesh2', 'mesh2_window', 'mesh3_scoccimarro', 'mesh3_sugiyama', 'mesh3_scoccimarro_window', 'mesh3_sugiyama_window'}
    if todo not in valid:
        raise ValueError(f"Unsupported todo item {todo!r}")
    w = 'window_' if 'window' in todo else ''
    if 'mesh2' in todo:
        return f'{w}mesh2_spectrum_poles'
    elif 'mesh3' in todo:
        basis = basis or todo.split('_')[1]
        return f'{w}mesh3_spectrum_poles_{basis}'

def compute_fkp_effective_redshift(*fkps, cellsize=10., order=2, split=None, fields=None, func_of_z=lambda x: x,
                                   resampler='cic', return_fraction=False):
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from jaxpower import split_particles, FKPField
    from jaxpower.mesh import _iter_meshes
    fiducial = DESI()
    zstep = 0.005
    zgrid = np.arange(0., 1100 + zstep, zstep)
    rgrid = fiducial.comoving_radial_distance(zgrid)
    d2z = get_interpolator_1d(rgrid, func_of_z(zgrid), order=1)
    fkps_none =  list(fkps) + [None] * (order - len(fkps))
    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp
    randoms = [get_randoms(fkp) for fkp in fkps_none]
    def compute_fkp_normalization_z(*particles, cellsize=cellsize, split=split, fields=fields):
        if split is not None or any(particle is None for particle in particles):
            particles = split_particles(*particles, seed=42 if split is None else split, fields=fields)
        reduce = 1
        for mesh in _iter_meshes(*particles, resampler=resampler, cellsize=cellsize, compensate=False, interlacing=0):
            reduce *= mesh
        rsum = reduce.sum()
        if not return_fraction: reduce /= rsum
        distance = jnp.sqrt(sum(xx**2 for xx in mesh.attrs.xcoords(kind='position', sparse=True)))
        reduce *= d2z(distance)
        if not return_fraction: return reduce.sum()
        return reduce.sum(), rsum
    return compute_fkp_normalization_z(*randoms)

def compute_mesh2_box(output_fn, get_data, ells=(0, 2, 4), los='z', cache=None, **attrs):
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
        if mpicomm.rank == mpiroot and output_fn is not None:
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_mesh3_box(output_fn, get_data, get_shifted=None, basis='scoccimarro', ells=(0, 2), los='z', mask_edges=None, cache=None, buffer_size=2, **attrs):
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
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
        spectrum = jitted_compute_mesh3_spectrum(mesh, los=los, bin=bin)
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot and output_fn is not None:
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_mesh2_cutsky(output_fn, get_data, get_random, ells=(0, 2, 4), los='firstpoint', cache=None, **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs,
                          compute_mesh2_spectrum)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        # Load and prepare catalogs (data, randoms and shifted if available), and create mesh binning object
        data, randoms = get_data(), get_random()
        mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
        # Set default k-space binning
        edges={'step': 0.001}
        # Create particle fileds with MPI exchange for the distributed workflow.
        data = ParticleField(*data, attrs=mattrs, exchange=True, backend='mpi')
        randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='mpi')
        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        bin = cache.get('bin_mesh2_spectrum', None)
        if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        cache.setdefault(f'bin_mesh2_spectrum', bin)
        # Create FKP fields for shot noise
        fkp = FKPField(data, randoms)
        wsum_data1 = data.sum()
        del data, randoms
        # Compute FKP normalization: integral of n^3(x) 
        norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
        # Compute short noise
        num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
        # Paint FKP fields onto mesh grids (stored as real-valued arrays to save memory)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        del fkp
        # Compute power spectrum (2-point spectrum) from mesh grids
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
        spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
        mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
        spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
        # Wait for spectrum computation to complete on all devices
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot: 
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_mesh3_cutsky(output_fn, get_data, get_random, get_shifted=None, basis='scoccimarro-diagonal', ells=[(0, 0, 0), (2, 0, 2)], los='firstpoint', mask_edges=None, cache=None, buffer_size=1, **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs,
                          compute_mesh3_spectrum)
    from jaxpower.mesh import create_sharding_mesh
    def get_particle_field(particles, mattrs):
        extra = particles[2] if len(particles) > 2 else {}
        return ParticleField(particles[0], particles[1], attrs=mattrs, exchange=True, backend='mpi', extra=extra)
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        # Load and prepare catalogs (data, randoms and shifted if available), and create mesh binning object
        data, randoms = get_data(), get_random()
        mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
        # Set default k-space binning (finer for sugiyama)
        edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005} #, 'max': 0.4}
        # Create particle fileds with MPI exchange for the distributed workflow.
        data = get_particle_field(data, mattrs)
        randoms = get_particle_field(randoms, mattrs)
        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if bin is None:  bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size, mask_edges=mask_edges)
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
        # Create FKP fields for shot noise
        fkp = FKPField(data, randoms)
        wsum_data1 = data.sum()
        del data, randoms
        # Compute FKP normalization: integral of n^3(x) 
        split = [(42, fkp.randoms.extra['IDS'])] if 'IDS' in fkp.randoms.extra else None
        norm = compute_fkp3_normalization(fkp, bin=bin, split=split, cellsize=10)
        # Compute short noise
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        num_shotnoise = compute_fkp3_shotnoise(fkp, bin=bin, los=los, **kw)
        # Paint FKP fields onto mesh grids (stored as real-valued arrays to save memory)
        mesh = fkp.paint(**kw, out='real')
        del fkp
        # Compute bispectrum (3-point spectrum) from mesh grids
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
        spectrum = jitted_compute_mesh3_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
        mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
        spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
        # Wait for spectrum computation to complete on all devices
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot: 
            logger.info(f'Writing to {output_fn}')
            spectrum.write(output_fn)
        mpicomm.Barrier()
        return spectrum

def compute_window_mesh2_spectrum(output_fn, get_spectrum=None, get_data=None, get_random=None, **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, 
                          compute_smooth2_spectrum_window, MeshAttrs, get_smooth2_window_bin_attrs, interpolate_window_function, compute_mesh2_spectrum, split_particles)
    spectrum = get_spectrum()
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, edges = spectrum.ells, pole.edges('k')
    norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells], axis=0)
    mean_norm = jnp.mean(norm)
    bin = BinMesh2SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells) | kwargs))
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    ellsin = [0, 2, 4]
    output_fn = str(output_fn) if output_fn is not None else None
    with create_sharding_mesh(meshsize=getattr(mattrs, 'meshsize', None)):
        randoms = get_random()
        randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='mpi')
        zeff, norm_zeff = compute_fkp_effective_redshift(randoms, order=2, return_fraction=True)
        #randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms
        correlations = []
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        # Window computed in configuration space, summing Bessel over the Fourier-space mesh
        coords = jnp.logspace(-3, 5, 4 * 1024)
        list_edges = []
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize) #, meshsize=800)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='mpi'), None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            distmax, cellsize = mattrs2.boxsize.min() / 4., mattrs2.cellsize.min()
            edges = np.arange(0., distmax + cellsize, cellsize)
            list_edges.append(edges)
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw, basis='bessel') #, kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp2_shotnoise(randoms, bin=sbin)
            correlation = compute_mesh2_correlation(*meshes, bin=sbin, los=los).clone(norm=[mean_norm] * len(sbin.ells)) #, num_shotnoise=num_shotnoise)
            del meshes
            correlation = interpolate_window_function(correlation, coords=coords, order=3)
            correlations.append(correlation)
        masks = [coords < edges[-3] for edges in list_edges[:-1]]
        masks.append(coords < np.inf)
        weights = []
        for mask in masks:
            if len(weights):
                weights.append(mask & (~weights[-1]))
            else:
                weights.append(mask)
        weights = [jnp.maximum(mask, 1e-6) for mask in weights]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
        window = window.clone(value=window.value() / (norm[..., None] / mean_norm))
        # Save norm and shotnoise here
        num_shotnoise = next(iter(spectrum)).values('num_shotnoise')[0]
        # Set shotnoise and norm of input spectrum
        observable = window.observable.map(lambda pole, label: pole.clone(value=0. * pole.value(), num_shotnoise=num_shotnoise * (label['ells'] == 0) * np.ones_like(pole.values('num_shotnoise')), norm=spectrum.get(ells=label['ells']).values('norm')), input_label=True)
        window = window.clone(observable=observable)
        window.attrs.update(spectrum.attrs)
        for pole in window.theory: pole._meta['z'] = zeff / norm_zeff
        if output_fn is not None and jax.process_index() == 0:
            logger.info(f'Writing to {output_fn}')
            window.write(output_fn)
        return window

def _get_window_edges(mattrs, scales=(1, 4)):
    distmax, cellmin = mattrs.boxsize.min() / 4., mattrs.cellsize.min()
    nsizes, cellsizes = [6] * 5 + [None], [cellmin * 2**i for i in range(6)]
    edges = []
    for scale in scales:
        edges_scale = []
        start = 0.
        for nsize, cellsize in zip(nsizes, cellsizes):
            cellsize = cellsize * scale
            if nsize is None:
                tmp = np.arange(start, distmax * scale / scales[0] + cellsize, cellsize)
            else:
                tmp = start + np.arange(nsize) * cellsize
            if tmp.size:
                start = tmp[-1] + cellsize
                edges_scale.append(tmp)
        edges_scale = np.concatenate(edges_scale, axis=0)
        edges_scale = edges_scale[edges_scale < distmax * scale / scales[0] + cellsize]
        edges.append(edges_scale)
    return edges

def compute_window_mesh3_spectrum(output_fn, get_spectrum=None, get_data=None, get_random=None, ibatch=None, computed_batches=None, buffer_size=1, **kwargs):
    from jax import numpy as jnp
    from jaxpower import (MeshAttrs, ParticleField, create_sharding_mesh, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles,
                          compute_mesh3_correlation, compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs,
                          interpolate_window_function, split_particles)
    def get_particle_field(particles, mattrs):
        extra = particles[2] if len(particles) > 2 else {}
        return ParticleField(particles[0], particles[1], attrs=mattrs, exchange=True, backend='mpi', extra=extra)
    spectrum = get_spectrum()
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, basis = spectrum.ells, pole.basis
    norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in spectrum.ells])
    edges = pole.edges('k')
    _, index = np.unique(pole.coords('k', center='mid_if_edges')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])
    output_fn = str(output_fn) if output_fn is not None else None
    with create_sharding_mesh(meshsize=getattr(mattrs, 'meshsize', None)):
        randoms = get_random()
        randoms = get_particle_field(randoms, mattrs)
        mattrs = randoms.attrs
        fields = [0, 0, 0]
        seed = [(42, randoms.extra['IDS'])] if 'IDS' in randoms.extra else 42
        zeff, norm_zeff = compute_fkp_effective_redshift(randoms, order=3, split=seed, fields=fields, return_fraction=True)
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')
        stop = bin.edges1d[0].max()
        step = np.diff(bin.edges1d[0], axis=-1).min()
        edgesin = np.arange(0., 1.5 * stop, step / 2.)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=fields, return_ellsin=True, basis=basis)
        kw['ells'] = [ell for ell in kw['ells'] if all(value <= 2 for value in ell)]
        kw['ells'] = kw['ells'][:1]
        jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])
        coords = jnp.logspace(-3, 5, 1024)
        list_scales = [1, 4]
        list_edges = _get_window_edges(mattrs, scales=list_scales)

        def pad_strictly_increasing_coords(coords, label=None):
            if isinstance(coords, list):
                return [pad_strictly_increasing_coords(coord, label=label) for coord in coords]
            step_first = coords[1] - coords[0]
            step_last = coords[-1] - coords[-2]
            return jnp.pad(coords, (1, 1), mode='constant', constant_values=(coords[0] - step_first, coords[-1] + step_last))

        all_ells = kw['ells']
        if ibatch is not None:
            start = ibatch[0] * len(all_ells) // ibatch[1]
            stop = (ibatch[0] + 1) * len(all_ells) // ibatch[1]
            kw['ells'] = all_ells[start:stop]

        if kw['ells'] and not bool(computed_batches):
            correlations = []
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            for scale, corr_edges in zip(list_scales, list_edges):
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                sbin = BinMesh3CorrelationPoles(mattrs2, edges=corr_edges, **kw, buffer_size=buffer_size)
                meshes = []
                split_randoms = split_particles(randoms, None, None, seed=seed, fields=fields)
                for split_random in split_randoms:
                    split_random = split_random.clone(attrs=mattrs2).exchange(backend='mpi')
                    alpha = spectrum.attrs['wsum_data1'] / split_random.sum()
                    meshes.append(alpha * split_random.paint(**kw_paint, out='real'))
                t0 = time.time()
                correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los)
                correlation = correlation.clone(norm=[np.mean(np.asarray(norm))] * len(sbin.ells))
                jax.block_until_ready(correlation)
                if jax.process_index() == 0:
                    logger.info(f"Computed windows {kw['ells']}, scale {scale}, in {time.time() - t0:.2f} s.")
                correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3, pad_coords=pad_strictly_increasing_coords)
                jax.block_until_ready(correlation)
                correlation = jax.device_get(correlation)
                correlations.append(correlation)
                del correlation, meshes, split_randoms, sbin
                jax.clear_caches()

            coords = list(next(iter(correlations[0])).coords().values())
            masks = [(coords[0] < corr_edges[-3])[:, None] * (coords[1] < corr_edges[-3])[None, :] for corr_edges in list_edges[:-1]]
            masks.append((coords[0] < np.inf)[:, None] * (coords[1] < np.inf)[None, :])
            weights = []
            for mask in masks:
                if weights:
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            correlation = correlations[0].sum(correlations, weights=weights)
        elif computed_batches:
            correlation = types.join(computed_batches)
            correlation = types.join([correlation.get(ells=[ell]) for ell in all_ells])
        else:
            raise ValueError('No window multipoles selected for compute_window_mesh3_spectrum.')

        jax.block_until_ready(correlation)
        if jax.process_index() == 0:
            logger.info('Window functions computed.')
        if ibatch is not None:
            return {'window_mesh3_correlation_raw': correlation}
        window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',), batch_size=4)
        observable = window.observable.map(
            lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)),
            input_label=True
        )
        window = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))
        window.attrs.update(spectrum.attrs)
        if output_fn is not None and jax.process_index() == 0:
            logger.info(f'Writing to {output_fn}')
            window.write(output_fn)
        return window

def combine_regions(output_fn, fns):
    missing = [fn for fn in fns if not os.path.exists(fn)]
    if missing:
        if mpicomm.rank == mpiroot:
            logger.warning(f"Cannot combine regions; missing input files: {missing}")
            if 'mesh3' in fns[0]:
                raise ValueError(f"Do not compute mesh3 spectra directly; use NGC and SGC instead and combine.")
        mpicomm.Barrier()
        return False
    if mpicomm.rank == mpiroot:
        combined = types.sum([types.read(fn) for fn in fns])
        if output_fn is not None:
            logger.info(f'Writing to {output_fn}')
            combined.write(output_fn)
    mpicomm.Barrier()
    return True

########################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2', 'holi-v3'])
    parser.add_argument("--domain", type = str, default='altmtl', choices=['cubic', 'cutsky', 'altmtl'], help="mock domain")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--zerrs", nargs = '+', type = str, default= ['None'], help="redshift error input, e.g. 'None', 'repeat', 'verr_empirical', 'verr_nonparam' with '_zevol' for redshift evolution")
    parser.add_argument("--todos", nargs = '+', type=str, default=['mesh2'], choices=['mesh2', 'mesh2_window', 'mesh3_scoccimarro', 'mesh3_sugiyama', 'mesh3_scoccimarro_window', 'mesh3_sugiyama_window'], help="todo types")
    parser.add_argument("--regions", nargs = '+', type=str, default=['ALL'], help="Region labels for cutsky/altmtl runs, e.g. ALL NGC SGC GCcomb")
    parser.add_argument("--meshsize", type=int, default=None, help="Optional meshsize override for mesh runs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file")
    args = parser.parse_args()
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")
    # jax configuration
    from jax import config
    from jaxpower.mesh import create_sharding_mesh
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    initialize_jax_distributed()

    postprocess = 'combine_regions' if args.domain == 'altmtl' else None

    version = args.version
    domain = args.domain
    z_snaps, z_ranges = GET_REDSHIFT_SET(version, domain)
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(z_snaps[tracer][:], z_ranges[tracer][:]):
            tracer_redshifts.append((tracer, zp, zr))

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
        
    regions = [None] if domain == 'cubic' else args.regions
    for (tracer, zsnap, zrange), mock_id, zerr, region in itertools.product(tracer_redshifts, mockids, args.zerrs, regions):
        if version == 'holi-v3' and domain == 'altmtl' and mock_id in SKIP_HOLI_ID_SET:
            if mpicomm.rank == mpiroot: logger.warning(f'Skipping holi-v3 altmtl mock_id={mock_id}')
            continue
        mock_id03 =  f"{mock_id:03}"
        use_dv, z_evol = parse_zerr_name(zerr)
        data_args = {'version':version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id, 'region': region, "use_dv": use_dv, "z_evol": z_evol, "overwrite":args.overwrite}
        io_cache = {}
        if domain == 'cubic':
            def get_data():
                if 'data' not in io_cache:
                    io_cache['data'] = read_positions_weights(**data_args)
                return io_cache['data']
            base_spectrum_args = dict(boxcenter=0., boxsize=2000., cellsize=5., ells=(0, 2, 4))
        elif domain in ['cutsky', 'altmtl']:
            def get_data():
                if 'data' not in io_cache:
                    io_cache['data'] = read_positions_weights(**data_args, use_jax = True)
                return io_cache['data']
            def get_random():
                if 'random' not in io_cache:
                    io_cache['random'] = read_positions_weights(**data_args, use_jax = True, random=True, nran=NRAN_Y3[tracer])
                return io_cache['random']
            def get_data_mesh3():
                if 'data_mesh3' not in io_cache:
                    io_cache['data_mesh3'] = read_positions_weights(**data_args, use_jax=True, weight_type='WEIGHT_FKP_NX13')
                return io_cache['data_mesh3']
            def get_random_mesh3():
                if 'random_mesh3' not in io_cache:
                    io_cache['random_mesh3'] = read_positions_weights(**data_args, use_jax=True, random=True, nran=NRAN_Y3[tracer], weight_type='WEIGHT_FKP_NX13', extra_columns=('IDS',))
                return io_cache['random_mesh3']
            base_spectrum_args = dict(**get_proposal_mattrs(domain=domain, tracer=tracer[:3]), ells=(0, 2, 4))
        else:
            raise ValueError(f"Unsupported domain {domain!r}")
        if args.meshsize is not None:
            base_spectrum_args['meshsize'] = args.meshsize
        output_fn = get_measurement_fn(**data_args, use_jax=use_jax)
        for todo in args.todos:
            if mpicomm.rank == mpiroot: logger.info(f'** {todo} ** {data_args}')
            spectrum_args = base_spectrum_args | dict(los='z' if domain == 'cubic' else ('firstpoint' if 'mesh2' in todo else 'local'))
            cache = {}
            if region in ['GCcomb', 'ALL']:
                region_fns = [get_measurement_fn(**(data_args | {'region': r}), use_jax=use_jax).format(_parse_todo(todo)) for r in ['NGC', 'SGC']]
                combine_regions(get_measurement_fn(**(data_args | {'region': region}), use_jax=use_jax).format(_parse_todo(todo)), region_fns)
                if region == 'GCcomb': continue
                if 'mesh3' in todo: continue

            if 'mesh2' in todo and 'window' not in todo:
                pk_fn = output_fn.format(_parse_todo(todo))
                if not os.path.exists(pk_fn) or args.overwrite:
                    if domain == 'cubic': compute_mesh2_box(pk_fn, get_data, **spectrum_args)
                    if domain in ['cutsky', 'altmtl']: compute_mesh2_cutsky(pk_fn, get_data, get_random, **spectrum_args)
                else:
                    types.read(pk_fn)
                jax.clear_caches()

            if 'mesh2' in todo and 'window' in todo:
                win_fn = output_fn.format(_parse_todo(todo))
                if not os.path.exists(win_fn) or args.overwrite==True:
                    pk_fn = output_fn.format(_parse_todo(todo.replace('_window', '')))
                    if not os.path.exists(pk_fn):
                        get_spectrum = lambda: compute_mesh2_cutsky(None, get_data, get_random, **spectrum_args)
                    else:
                        get_spectrum = lambda: types.read(pk_fn)
                    compute_window_mesh2_spectrum(win_fn, get_spectrum=get_spectrum, get_data=get_data, get_random=get_random)
                jax.clear_caches()

            if 'mesh3' in todo and 'window' not in todo:
                mesh3_buffer_size = {'BGS': 0, 'LRG': 0, 'ELG': 0, 'QSO': 0}[tracer[:3]]
                if 'scoccimarro' in todo:
                    basis = 'scoccimarro'
                    bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2])
                elif 'sugiyama' in todo:
                    basis = 'sugiyama'
                    bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], buffer_size=mesh3_buffer_size)
                else:
                    raise ValueError(f"Specify bispectrum basis in todo {todo!r}")
                bk_fn = output_fn.format(_parse_todo(todo, basis=basis))
                if not os.path.exists(bk_fn) or args.overwrite:
                    if domain == 'cubic': compute_mesh3_box(bk_fn, get_data, **bispectrum_args)
                    if domain in ['cutsky', 'altmtl']: compute_mesh3_cutsky(bk_fn, get_data_mesh3, get_random_mesh3, **bispectrum_args) 
                else:
                    types.read(bk_fn)
                jax.clear_caches()

            if 'mesh3' in todo and 'window' in todo:
                window_mesh3_buffer_size = {'BGS': 3, 'LRG': 3, 'ELG': 0, 'QSO': 0}[tracer[:3]]
                if 'scoccimarro' in todo:
                    basis = 'scoccimarro'
                    bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2])
                elif 'sugiyama' in todo:
                    basis = 'sugiyama'
                    bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], buffer_size=window_mesh3_buffer_size)
                else:
                    raise ValueError(f"Specify bispectrum basis in todo {todo!r}")
                win_fn = output_fn.format(_parse_todo(todo, basis=basis))
                if not os.path.exists(win_fn) or args.overwrite:
                    bk_fn = output_fn.format(_parse_todo(todo.replace('_window', ''), basis=basis))
                    if not os.path.exists(bk_fn):
                        get_spectrum = lambda: compute_mesh3_cutsky(None, get_data_mesh3, get_random_mesh3, **bispectrum_args)
                    else:
                        get_spectrum = lambda: types.read(bk_fn)
                    compute_window_mesh3_spectrum(win_fn, get_spectrum=get_spectrum, get_data=get_data_mesh3, get_random=get_random_mesh3, **bispectrum_args)
                jax.clear_caches()
