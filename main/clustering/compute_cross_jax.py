#!/usr/bin/env python

import os
import sys
import jax
import argparse
import logging
import itertools
import ast
import json
import math
import numpy as np
import lsstypes as types
from pathlib import Path

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

MAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_DIR))
from utils import setup_logging
from helper import NRAN_Y3, Y3_MESHSIZES
from cat_tools import read_positions_weights, get_measurement_fn, get_catalog_fn, parse_zerr_name
from jax_support import initialize_jax_distributed

setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compute_cross')

# Basic settings. Each --zerrs setting is a separate run applied to both tracers.
use_jax = True
BOXSIZE = 2000.
SKIP_HOLI_ID_SET = np.atleast_1d(np.loadtxt(Path(__file__).with_name('dubious_holi-v3-altmtl.txt'), dtype=int))
DEFAULT_ZRANGES = {
    frozenset(('LRG', 'ELG')): [(0.8, 1.1)],
    frozenset(('LRG', 'QSO')): [(0.8, 1.1)],
    frozenset(('ELG', 'QSO')): [(0.8, 1.1), (1.1, 1.6)],
}


def parse_tracers(text):
    tracers = text.split('x')
    if len(tracers) != 2 or any(tracer not in ['BGS', 'LRG', 'ELG', 'QSO'] for tracer in tracers):
        raise argparse.ArgumentTypeError('Expected a tracer pair such as LRGxELG or ELGxLRG')
    if tracers[0] == tracers[1]:
        raise argparse.ArgumentTypeError('Cross-power requires two distinct tracers')
    return tracers


def parse_zrange(text):
    try:
        values = ast.literal_eval(text) if text.startswith(('(', '[')) else text.split('-')
        lo, hi = map(float, values)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise ValueError
        return lo, hi
    except (ValueError, TypeError, SyntaxError):
        raise argparse.ArgumentTypeError('Expected an increasing redshift range, e.g. 0.8-1.1') from None


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return value


def parse_mockids(text):
    try:
        if '-' in text:
            start, stop = map(int, text.split('-'))
            if start < 0 or stop < start:
                raise ValueError
            return list(range(start, stop + 1))
        values = list(dict.fromkeys(map(int, text.split(','))))
        if not values or min(values) < 0:
            raise ValueError
        return values
    except ValueError:
        raise argparse.ArgumentTypeError('Expected mock IDs such as 0, 0-24, or 0,2,4') from None


def catalog_options(args, mock_id, zrange, region, zerr):
    """Use one error setting for both tracers, keeping their catalog options separate."""
    result = []
    for i, tracer in enumerate(args.tracers):
        zevol = zerr.endswith('_zevol')
        data = dict(version=args.version, domain=args.domain, tracer=tracer, mock_id=mock_id,
                    region=region, zrange=zrange, zsnap=args.zsnap, hod=args.hod,
                    use_dv=zerr[:-6] if zevol else zerr, z_evol=zevol)
        if args.ntile_cut is not None:
            data['cut_ntile'] = args.ntile_cut
        if args.domain == 'cubic':
            result.append({'data': data})
            continue
        nran = NRAN_Y3[tracer]
        if (args.version == 'AbacusHF-v2' and args.domain == 'cutsky') or args.version == 'AbacusHF-4snap':
            nran = None
        if args.nran:
            nran = args.nran[i]
        randoms = data | dict(random=True, use_dv='None', z_evol=False, nran=nran,
                              expand=args.domain == 'altmtl' and args.version in ['AbacusHF-v2', 'holi-v3'])
        result.append({'data': data, 'randoms': randoms})
    return result


def output_filename(args, mock_id, zrange, region, zerr, window=False):
    use_dv, z_evol = parse_zerr_name(zerr)
    suffix = ''
    if args.domain == 'cutsky':
        suffix += f'_zp{args.zsnap:g}'
    if args.ntile_cut is not None:
        suffix += f'ntile_{args.ntile_cut}'
    if args.shotnoise:
        suffix += f'_sn-{args.shotnoise:g}'
    filename = get_measurement_fn(
        version=args.version, domain=args.domain, tracer='x'.join(args.tracers),
        zrange=zrange, zsnap=args.zsnap, mock_id=mock_id, region=region,
        hod=args.hod, use_dv=use_dv, z_evol=z_evol, suffix=suffix,
        use_jax=True, base_dir=args.output_dir, mkdir=False,
    )
    if filename is None:
        raise ValueError(f'No measurement filename for {args.version} mock {mock_id}')
    kind = 'window_mesh2_spectrum_poles' if window else 'mesh2_spectrum_poles'
    return Path(filename.format(kind))


def _check_catalog(catalog, name):
    positions, weights = catalog
    valid = (positions.ndim == 2 and positions.shape[1] == 3 and weights.shape == (len(positions),)
             and np.isfinite(positions).all() and np.isfinite(weights).all())
    if not mpicomm.allreduce(bool(valid), op=MPI.LAND):
        raise ValueError(f'Invalid positions or weights for {name}')
    if mpicomm.allreduce(float(np.sum(weights))) <= 0:
        raise ValueError(f'Empty or nonpositive-weight {name}')


def _read_box_data(data_args):
    # Preserve HOD selection and distribute catalog rows over MPI ranks.
    from mockfactory import Catalog
    cat = Catalog.read(get_catalog_fn(**data_args), mpicomm=mpicomm)
    zcol = {'None': 'Z_RSD', 'repeat': 'Z_REP', 'verr_empirical': 'Z_ERR_V1',
            'verr_nonparam': 'Z_ERR_V2'}[data_args['use_dv']]
    positions = np.column_stack([cat['X'], cat['Y'], cat[zcol]]) % BOXSIZE - BOXSIZE / 2.
    return positions, np.ones(len(positions), dtype='f8')


def compute_mesh2_box(output_fn, get_data1, get_data2, ells=(0, 2, 4), los='z',
                      edges=None, shotnoise=0., metadata=None, **attrs):
    """Cross spectrum of two periodic catalogs; each loader returns local positions and weights."""
    from jaxpower import (ParticleField, BinMesh2SpectrumPoles, get_mesh_attrs,
                          compute_mesh2_spectrum, compute_box2_normalization, compute_fkp2_shotnoise)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        # Load both tracers and define their shared mesh.
        data1, data2 = get_data1(), get_data2()
        _check_catalog(data1, 'data1')
        _check_catalog(data2, 'data2')
        mattrs = get_mesh_attrs(data1[0], data2[0], check=True, **attrs)
        if edges is None: edges = {'step': 0.001}
        bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        data1 = ParticleField(*data1, attrs=mattrs, exchange=True, backend='mpi')
        data2 = ParticleField(*data2, attrs=mattrs, exchange=True, backend='mpi')
        wsum_data1, wsum_data2 = data1.sum(), data2.sum()

        # Cross normalization and shot noise (zero by default for disjoint tracers).
        norm = compute_box2_normalization(data1, data2, bin=bin)
        if not all(np.isfinite(np.asarray(value)).all() and np.all(np.asarray(value) > 0) for value in norm):
            raise ValueError('Cross normalization is nonpositive/nonfinite')
        num_shotnoise = compute_fkp2_shotnoise(data1, data2, bin=bin, fields=(0, 1))
        if shotnoise:
            num_shotnoise = [value + shotnoise * normalization * (ell == 0)
                             for value, normalization, ell in zip(num_shotnoise, norm, ells)]
        if mpicomm.rank == mpiroot:
            logger.info(f'Shared mesh: {mattrs}; cross normalization: {norm[0]}')

        # Paint each tracer separately, then cross the two Fourier fields.
        mesh1 = data1.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        mesh2 = data2.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        mesh1, mesh2 = mesh1 - mesh1.mean(), mesh2 - mesh2.mean()
        del data1, data2
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0, 1])
        spectrum = jitted_compute_mesh2_spectrum(mesh1, mesh2, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
        mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
        spectrum_attrs = dict(metadata or {})
        spectrum_attrs.update(los=los, periodic=True, shotnoise=shotnoise,
                              shotnoise_method='constant' if shotnoise else 'disjoint',
                              wsum_data1=wsum_data1, wsum_data2=wsum_data2,
                              wsum_data=[wsum_data1, wsum_data2], **mattrs)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=spectrum_attrs)).clone(attrs=spectrum_attrs)
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot and output_fn is not None:
            Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Writing to {output_fn}')
            spectrum.write(str(output_fn))
        mpicomm.Barrier()
        return spectrum


def compute_mesh2_cutsky(output_fn, get_data1, get_random1, get_data2, get_random2,
                         ells=(0, 2, 4), los='firstpoint', edges=None, norm_cellsize=10.,
                         shotnoise=0., metadata=None, **attrs):
    """Cross two FKP fields on one mesh using each tracer's data and randoms.

    Loaders return MPI-local (positions, weights). The default shot noise assumes
    disjoint objects and independent random points; shared TARGETIDs are not matched.
    shotnoise supplies a normalized constant monopole subtraction in (Mpc/h)^3.
    """
    from jaxpower import (ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs,
                          compute_mesh2_spectrum, compute_fkp2_normalization, compute_fkp2_shotnoise)
    from jaxpower.mesh import create_sharding_mesh
    with create_sharding_mesh(meshsize=attrs.get('meshsize', None)):
        # Load both tracers and construct one geometry from all four catalogs.
        data1, randoms1 = get_data1(), get_random1()
        data2, randoms2 = get_data2(), get_random2()
        for name, catalog in [('data1', data1), ('randoms1', randoms1), ('data2', data2), ('randoms2', randoms2)]:
            _check_catalog(catalog, name)
        mattrs = get_mesh_attrs(data1[0], randoms1[0], data2[0], randoms2[0], check=True, **attrs)
        if edges is None: edges = {'step': 0.001}
        bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        data1 = ParticleField(*data1, attrs=mattrs, exchange=True, backend='mpi')
        randoms1 = ParticleField(*randoms1, attrs=mattrs, exchange=True, backend='mpi')
        data2 = ParticleField(*data2, attrs=mattrs, exchange=True, backend='mpi')
        randoms2 = ParticleField(*randoms2, attrs=mattrs, exchange=True, backend='mpi')

        # Each FKP field keeps its own weights and data/random normalization.
        fkp1 = FKPField(data1, randoms1)
        fkp2 = FKPField(data2, randoms2)
        wsum_data1, wsum_data2 = data1.sum(), data2.sum()
        wsum_randoms1, wsum_randoms2 = randoms1.sum(), randoms2.sum()
        del data1, randoms1, data2, randoms2
        norm = compute_fkp2_normalization(fkp1, fkp2, bin=bin, cellsize=norm_cellsize)
        if not all(np.isfinite(np.asarray(value)).all() and np.all(np.asarray(value) > 0) for value in norm):
            raise ValueError('Cross normalization is nonpositive/nonfinite; check the survey overlap')
        num_shotnoise = compute_fkp2_shotnoise(fkp1, fkp2, bin=bin, fields=(0, 1))
        if shotnoise:
            num_shotnoise = [value + shotnoise * normalization * (ell == 0)
                             for value, normalization, ell in zip(num_shotnoise, norm, ells)]
        if mpicomm.rank == mpiroot:
            logger.info(f'Shared mesh: {mattrs}; cross normalization: {norm[0]}')

        # Paint both FKP fields and measure their cross-power multipoles.
        mesh1 = fkp1.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        mesh2 = fkp2.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        del fkp1, fkp2
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0, 1])
        spectrum = jitted_compute_mesh2_spectrum(mesh1, mesh2, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
        mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
        spectrum_attrs = dict(metadata or {})
        spectrum_attrs.update(los=los, periodic=False, shotnoise=shotnoise,
                              shotnoise_method='constant' if shotnoise else 'disjoint',
                              wsum_data1=wsum_data1, wsum_data2=wsum_data2,
                              wsum_data=[wsum_data1, wsum_data2],
                              wsum_randoms=[wsum_randoms1, wsum_randoms2], **mattrs)
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=spectrum_attrs)).clone(attrs=spectrum_attrs)
        jax.block_until_ready(spectrum)
        if mpicomm.rank == mpiroot and output_fn is not None:
            Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Writing to {output_fn}')
            spectrum.write(str(output_fn))
        mpicomm.Barrier()
        return spectrum


def compute_window_mesh2_spectrum(output_fn, get_spectrum, get_random1, get_random2, metadata=None, **kwargs):
    """Cross survey window from two independent random catalogs in tracer order.

    Use the saved cross spectrum's geometry, bins, normalizations and data weight
    sums. As for the cross spectrum, the random catalogs must be independent;
    shared random points are not matched or subtracted.
    """
    from jax import numpy as jnp
    from jaxpower import (ParticleField, MeshAttrs, BinMesh2SpectrumPoles,
                          BinMesh2CorrelationPoles, compute_mesh2_correlation,
                          get_smooth2_window_bin_attrs, interpolate_window_function,
                          compute_smooth2_spectrum_window)
    from jaxpower.mesh import create_sharding_mesh
    from clustering.compute_mesh_jax import compute_fkp_effective_redshift

    spectrum = get_spectrum()
    if spectrum.attrs.get('periodic', False) or spectrum.attrs.get('region') == 'GCcomb':
        raise ValueError('Compute cross windows for individual survey regions; combine them for GCcomb.')
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    ells = spectrum.ells
    norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells])
    if not np.isfinite(np.asarray(norm)).all() or not np.all(np.asarray(norm) > 0):
        raise ValueError('Cross-spectrum normalization must be finite and positive.')
    wsum_data = [spectrum.attrs[f'wsum_data{i}'] for i in (1, 2)]
    if not np.isfinite(wsum_data).all() or not np.all(np.asarray(wsum_data) > 0):
        raise ValueError('Both cross-spectrum data weight sums must be finite and positive.')
    mean_norm = jnp.mean(norm)
    bin = BinMesh2SpectrumPoles(mattrs, **(dict(edges=next(iter(spectrum)).edges('k'), ells=ells) | kwargs))
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    ellsin = [0, 2, 4]

    with create_sharding_mesh(meshsize=mattrs.meshsize):
        randoms = []
        for i, get_random in enumerate((get_random1, get_random2), start=1):
            if mpicomm.rank == mpiroot:
                logger.info(f'Preparing cross-window random catalog {i}/2')
            catalog = get_random()
            _check_catalog(catalog, f'randoms{i}')
            randoms.append(ParticleField(*catalog, attrs=mattrs, exchange=True, backend='mpi'))
        if mpicomm.rank == mpiroot:
            logger.info('Computing cross-window effective redshift')
        zeff, norm_zeff = compute_fkp_effective_redshift(*randoms, order=2, return_fraction=True)
        if not np.isfinite(float(norm_zeff)) or float(norm_zeff) <= 0:
            raise ValueError('Cross random catalogs have no positive overlap for the effective redshift.')
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        coords = jnp.logspace(-3, 5, 4 * 1024)
        correlations, list_edges = [], []
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
            if mpicomm.rank == mpiroot:
                logger.info(f'Computing cross-window correlation at box scale {scale}: {mattrs2}')
            meshes = []
            # Unlike an auto window, cross two full, independent random fields.
            for weight_sum, random in zip(wsum_data, randoms, strict=True):
                random = random.clone(attrs=mattrs2, exchange=True, backend='mpi')
                alpha = weight_sum / random.sum()
                meshes.append(alpha * random.paint(resampler='tsc', interlacing=3, compensate=True, out='real'))
            distmax, cellsize = mattrs2.boxsize.min() / 4., mattrs2.cellsize.min()
            edges = np.arange(0., distmax + cellsize, cellsize)
            list_edges.append(edges)
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw, basis='bessel')
            correlation = compute_correlation(*meshes, bin=sbin, los=los).clone(norm=[mean_norm] * len(sbin.ells))
            del meshes
            correlations.append(interpolate_window_function(correlation, coords=coords, order=3))
        masks = [coords < edges[-3] for edges in list_edges[:-1]] + [coords < np.inf]
        weights = []
        for mask in masks:
            weights.append(mask & (~weights[-1]) if weights else mask)
        weights = [jnp.maximum(mask, 1e-6) for mask in weights]
        correlation = correlations[0].sum(correlations, weights=weights)
        if mpicomm.rank == mpiroot:
            logger.info('Transforming cross-window correlation into the window matrix')
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin,
                                                bin=bin, flags=('fftlog',))
        window = window.clone(value=window.value() / (norm[..., None] / mean_norm))

        def copy_normalization(pole, label):
            source = spectrum.get(ells=label['ells'])
            return pole.clone(value=0. * pole.value(), norm=source.values('norm'),
                              num_shotnoise=source.values('num_shotnoise'))

        window = window.clone(observable=window.observable.map(copy_normalization, input_label=True))
        window.attrs.update(spectrum.attrs)
        window.attrs.update(metadata or {})
        window.attrs['window_wsum_randoms'] = [float(random.sum()) for random in randoms]
        for pole in window.theory:
            pole._meta['z'] = zeff / norm_zeff
        jax.block_until_ready(window)
        if output_fn is not None and mpicomm.rank == mpiroot:
            Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Writing to {output_fn}')
            window.write(str(output_fn))
        mpicomm.Barrier()
        return window


def combine_regions(output_fn, fns):
    missing = mpicomm.bcast([str(fn) for fn in fns if not Path(fn).exists()] if mpicomm.rank == mpiroot else None, root=mpiroot)
    if missing:
        raise FileNotFoundError(f'Compute NGC and SGC before GCcomb: {missing}')
    if mpicomm.rank == mpiroot:
        spectra = [types.read(str(fn)) for fn in fns]
        combined = types.sum(spectra)
        combined.attrs.update(region='GCcomb', input_files=list(map(str, fns)))
        for name in ['wsum_data', 'wsum_randoms', 'window_wsum_randoms']:
            if name in spectra[0].attrs:
                combined.attrs[name] = [sum(item.attrs[name][i] for item in spectra) for i in range(2)]
        combined.attrs['wsum_data1'], combined.attrs['wsum_data2'] = combined.attrs['wsum_data']
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing to {output_fn}')
        combined.write(str(output_fn))
    mpicomm.Barrier()
    return True


# Command-line setup
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', default='AbacusHF-v2', choices=['AbacusHF-v1', 'AbacusHF-v2', 'AbacusHF-4snap', 'holi-v3', 'data-dr1-v1.5', 'AbacusSecondGen'])
    parser.add_argument('--domain', default='altmtl', choices=['cubic', 'cutsky', 'altmtl', 'lightcone'])
    parser.add_argument('--tracers', type=parse_tracers, required=True, metavar='TRACER1xTRACER2', help='Cross pair, e.g. LRGxELG or ELGxLRG; order is preserved for the firstpoint LOS and per-tracer options')
    parser.add_argument('--zranges', nargs='+', type=parse_zrange, help='Common bins; defaults to the standard overlap bins for the pair')
    parser.add_argument('--zsnap', type=float, help='Common snapshot, required for cubic/cutsky catalogs')
    parser.add_argument('--mockid', type=parse_mockids, default=[0], help='0, 0-24, or 0,2,4')
    parser.add_argument('--hod', default='base', choices=['base', 'base_dv'])
    errors = ['None', 'repeat', 'verr_empirical', 'verr_nonparam']
    parser.add_argument('--zerrs', nargs='+', default=['None'], choices=errors + [name + '_zevol' for name in errors[1:]], help='Redshift-error settings to run separately; each setting is applied to both tracers')
    parser.add_argument('--todos', nargs='+', type=str, default=['cross2'], choices=['cross2', 'cross2_window'],
                        help='Cross-power spectrum and/or its survey window; a missing spectrum is saved before computing its window')
    parser.add_argument('--regions', nargs='+', default=['NGC', 'SGC'], choices=['ALL', 'NGC', 'SGC', 'GCcomb'])
    parser.add_argument('--nran', nargs=2, type=positive_int, help='Number of random files per tracer for spectra and windows; defaults to NRAN_Y3 for altmtl catalogs')
    parser.add_argument('--meshsize', type=positive_int, help='Shared mesh size; surveys default to the reference proposal for the first tracer')
    parser.add_argument('--boxsize', type=float, help='Shared box size in Mpc/h; surveys default to meshsize times 7.5 (cubic boxes require 2000)')
    parser.add_argument('--boxcenter', nargs=3, type=float, help='Shared center; inferred jointly for surveys')
    parser.add_argument('--dk', type=float, default=0.001, help='k bin width in h/Mpc')
    parser.add_argument('--kmax', type=float, help='Upper k bin edge (otherwise jaxpower default)')
    parser.add_argument('--norm-cellsize', type=float, default=10., help='Cross-normalization cell size in Mpc/h')
    parser.add_argument('--shotnoise', type=float, default=0., help='Normalized constant cross-monopole subtraction in (Mpc/h)^3; default assumes disjoint samples')
    parser.add_argument('--ntile_cut', type=int, help='For data, require NTILE > this value')
    parser.add_argument('--output-dir', type=Path, help='Override the root for cross measurements')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.zerrs = list(dict.fromkeys(args.zerrs))
    if args.domain == 'cubic':
        if 'cross2_window' in args.todos:
            parser.error('cross2_window requires survey random catalogs; use a survey domain')
        args.zranges, args.regions = [None], [None]
    else:
        if args.zranges is None:
            args.zranges = DEFAULT_ZRANGES[frozenset(args.tracers)]
        # Combine caps after their individual measurements, regardless of input order.
        args.regions = sorted(dict.fromkeys(args.regions), key=lambda region: region == 'GCcomb')
    args.todos = sorted(dict.fromkeys(args.todos), key=lambda todo: todo == 'cross2_window')
    return args


# Keep main callable for notebook debugging and configuration checks.
def main(argv=None):
    args = parse_args(argv)
    if mpicomm.rank == mpiroot:
        logger.info(f'Received arguments: {args}')

    # JAX configuration; dry runs only construct options and filenames.
    if not args.dry_run:
        jax.config.update('jax_enable_x64', True)
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')
        initialize_jax_distributed()

    version = args.version
    domain = args.domain
    tracer1, tracer2 = args.tracers
    tracer = 'x'.join(args.tracers)
    mockids = args.mockid
    zranges = args.zranges
    regions = args.regions

    # Match the reference full-shape mesh convention for the first tracer.
    if domain == 'cubic':
        base_spectrum_args = dict(boxcenter=0., boxsize=BOXSIZE, meshsize=args.meshsize or 400)
    else:
        meshsize = args.meshsize or Y3_MESHSIZES[tracer1]
        base_spectrum_args = dict(boxsize=args.boxsize or 7.5 * meshsize, meshsize=meshsize)
        if args.boxcenter is not None:
            base_spectrum_args['boxcenter'] = args.boxcenter
    edges = {'step': args.dk}
    if args.kmax is not None:
        edges['stop'] = args.kmax
    base_spectrum_args.update(ells=(0, 2, 4), edges=edges, shotnoise=args.shotnoise)

    for zerr, zrange, mock_id, region in itertools.product(args.zerrs, zranges, mockids, regions):
        if version == 'holi-v3' and domain == 'altmtl' and mock_id in SKIP_HOLI_ID_SET:
            if mpicomm.rank == mpiroot:
                logger.warning(f'Skipping holi-v3 altmtl mock_id={mock_id}')
            continue
        all_options = catalog_options(args, mock_id, zrange, region, zerr)
        data_args1 = all_options[0]['data']
        data_args2 = all_options[1]['data']
        random_args1 = all_options[0].get('randoms')
        random_args2 = all_options[1].get('randoms')
        for todo in args.todos:
            pk_fn = output_filename(args, mock_id, zrange, region, zerr)
            output_fn = output_filename(args, mock_id, zrange, region, zerr, window=todo == 'cross2_window')
            if args.dry_run:
                if mpicomm.rank == mpiroot:
                    print(json.dumps(dict(todo=todo, output=str(output_fn), catalogs=all_options, combine=region == 'GCcomb'), indent=2))
                continue
            if mpicomm.rank == mpiroot:
                logger.info(f'** {todo} ** {tracer} zrange={zrange} mock_id={mock_id} region={region} zerr={zerr}')
            exists = mpicomm.bcast(output_fn.exists() if mpicomm.rank == mpiroot else None, root=mpiroot)
            if exists and not args.overwrite:
                if mpicomm.rank == mpiroot:
                    logger.info(f'Skipping existing {output_fn}')
                continue
            if region == 'GCcomb':
                region_fns = [output_filename(args, mock_id, zrange, cap, zerr, window=todo == 'cross2_window')
                              for cap in ['NGC', 'SGC']]
                combine_regions(output_fn, region_fns)
                jax.clear_caches()
                continue

            # Separate callbacks make each tracer's catalog easy to inspect.
            io_cache = {}
            def get_data1():
                if 'data1' not in io_cache:
                    io_cache['data1'] = (_read_box_data(data_args1) if domain == 'cubic' else
                                         read_positions_weights(**data_args1, use_jax=use_jax))
                return io_cache['data1']

            def get_data2():
                if 'data2' not in io_cache:
                    io_cache['data2'] = (_read_box_data(data_args2) if domain == 'cubic' else
                                         read_positions_weights(**data_args2, use_jax=use_jax))
                return io_cache['data2']

            def get_random1():
                if 'randoms1' not in io_cache:
                    io_cache['randoms1'] = read_positions_weights(**random_args1, use_jax=use_jax)
                return io_cache['randoms1']

            def get_random2():
                if 'randoms2' not in io_cache:
                    io_cache['randoms2'] = read_positions_weights(**random_args2, use_jax=use_jax)
                return io_cache['randoms2']

            metadata = dict(tracers=args.tracers, zerrs=[zerr, zerr], version=version, domain=domain,
                            hod=args.hod, mock_id=mock_id, region=region or 'box',
                            zrange=list(zrange) if zrange is not None else [],
                            zsnap=args.zsnap if args.zsnap is not None else -1.,
                            catalog_options=json.dumps(all_options), norm_cellsize=args.norm_cellsize)
            spectrum_args = base_spectrum_args | dict(metadata=metadata)
            if todo == 'cross2':
                if domain == 'cubic':
                    compute_mesh2_box(pk_fn, get_data1, get_data2, los='z', **spectrum_args)
                elif domain in ['cutsky', 'altmtl', 'lightcone']:
                    compute_mesh2_cutsky(pk_fn, get_data1, get_random1, get_data2, get_random2,
                                         los='firstpoint', norm_cellsize=args.norm_cellsize, **spectrum_args)
            elif todo == 'cross2_window':
                def get_spectrum():
                    exists = mpicomm.bcast(pk_fn.exists() if mpicomm.rank == mpiroot else None, root=mpiroot)
                    if exists:
                        return types.read(str(pk_fn))
                    return compute_mesh2_cutsky(pk_fn, get_data1, get_random1, get_data2, get_random2,
                                                los='firstpoint', norm_cellsize=args.norm_cellsize, **spectrum_args)

                compute_window_mesh2_spectrum(output_fn, get_spectrum, get_random1, get_random2,
                                             metadata=dict(window_nran=[random_args1['nran'], random_args2['nran']],
                                                           window_catalog_options=json.dumps(all_options)))
            io_cache.clear()
            jax.clear_caches()


if __name__ == '__main__':
    main()
