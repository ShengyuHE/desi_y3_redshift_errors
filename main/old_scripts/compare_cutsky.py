"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
srun -n 4 python compare_cutsky.py
"""
import os
import time
import logging
import itertools
from pathlib import Path

import numpy as np

import lsstypes as types
from mockfactory import Catalog, sky_to_cartesian, setup_logging


logger = logging.getLogger('compare_cutsky')


def select_region(ra, dec, region=None):
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if footprint is None: load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError('unknown region {}'.format(region))


def get_proposal_mattrs(tracer):
    if 'BGS' in tracer:
        mattrs = dict(boxsize=4000., cellsize=7)
    elif 'LRG+ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'LRG' in tracer:
        mattrs = dict(boxsize=7000., cellsize=7)
    elif 'ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'QSO' in tracer:
        mattrs = dict(boxsize=10000., cellsize=10)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    mattrs.update(cellsize=10)
    return mattrs

def get_proposal_recattrs(tracer):
    if 'BGS' in tracer:
        recattrs = dict(bias=1.5, smoothing_radius=15.)
    elif 'LRG+ELG' in tracer:
        recttrs = dict(bias=1.6, smoothing_radius=15.)
    elif 'LRG' in tracer:
        recattrs = dict(bias=2.0, smoothing_radius=15.)
    elif 'ELG' in tracer:
        recattrs = dict(bias=1.2, smoothing_radius=15.)
    elif 'QSO' in tracer:
        recattrs = dict(bias=2.1, smoothing_radius=30.)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    return recattrs   


def get_clustering_rdzw(*fns, zrange=None, region=None, tracer=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            for name in ['WEIGHT', 'WEIGHT_FKP']:
                if name not in catalog: catalog[name] = catalog.ones()
            if tracer is not None and 'Z' not in catalog:
                catalog['Z'] = catalog[f'Z_{tracer}']
            catalog = catalog[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        weight = catalog['WEIGHT'] #* catalog['WEIGHT_FKP']
        rdzw.append([catalog['RA'], catalog['DEC'], catalog['Z'], weight])
    return [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(4)]


def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    ra, dec, z, weights = get_clustering_rdzw(*fns, **kwargs)
    weights = np.asarray(weights, dtype='f8')
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec, dtype='f8')
    return positions, weights


def compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, get_shifted=None, cache=None, ells=(0, 2, 4), los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)
    norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
    if get_shifted is not None:
        del fkp, randoms
        randoms = ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='jax')
        fkp = FKPField(data, randoms)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    #t0 = time.time()
    wsum_data1 = data.sum()
    del fkp, data, randoms
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    jax.block_until_ready(spectrum)
    #logger.info(f'Elapsed time: {time.time() - t0:.2f}.')
    return spectrum, bin


def compute_fkp_effective_redshift(fkp, cellsize=10., order=2):
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from cosmoprimo.utils import DistanceToRedshift
    from jaxpower import compute_fkp2_normalization, compute_fkp3_normalization, FKPField
    fiducial = TabulatedDESI()
    d2z = DistanceToRedshift(lambda z: jnp.array(fiducial.comoving_radial_distance(z)))

    compute_fkp_normalization = {2: compute_fkp2_normalization, 3: compute_fkp3_normalization}[order]

    def compute_z(positions):
        return d2z(jnp.sqrt(jnp.sum(positions**2, axis=-1)))

    if isinstance(fkp, FKPField):
        norm = compute_fkp_normalization(fkp, cellsize=cellsize)
        fkp = fkp.clone(data=fkp.data.clone(weights=data.weights * compute_z(fkp.data.positions)), randoms=randoms.clone(weights=fkp.randoms.weights  * compute_z(fkp.randoms.positions)))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize)
    else:  # fkp is randoms
        norm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
        fkp = fkp.clone(weights=fkp.weights * compute_z(fkp.positions))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
    return znorm / norm


def compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=None, spectrum_fn=None, kind='smooth', **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, compute_smooth2_spectrum_window, MeshAttrs, get_smooth2_window_bin_attrs, interpolate_window_function, compute_mesh2_spectrum, split_particles, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges = spectrum.ells, pole.values('norm')[0], pole.edges('k')
    bin = BinMesh2SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells) | kwargs))
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]]) 
    ellsin = [0, 2, 4]
    output_fn = str(output_fn)

    randoms = ParticleField(*get_randoms(), attrs=mattrs, exchange=True, backend='jax')
    zeff = compute_fkp_effective_redshift(randoms, order=2)
    #if get_data is not None:
    #    from jaxpower import FKPField
    #    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='jax')
    #    zeff = compute_fkp_effective_redshift(FKPField(data=data, randoms=randoms))
    randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms

    kind = 'smooth'

    if kind == 'smooth':
        correlations = []
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        # Window computed in configuration space, summing Bessel over the Fourier-space mesh
        coords = jnp.logspace(-3, 5, 4 * 1024)
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize) #, meshsize=800)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=np.arange(0., mattrs2.boxsize.min() / 2., mattrs2.cellsize.min()), **kw, basis='bessel') #, kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp2_shotnoise(randoms, bin=sbin)
            correlation = compute_mesh2_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells)) #, num_shotnoise=num_shotnoise)
            del meshes
            correlation_fn = output_fn.replace('window_mesh2_spectrum', f'window_correlation{scale:d}_bessel_mesh2_spectrum')
            if jax.process_index() == 0:
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation, coords=coords, order=3)
            correlations.append(correlation)
        limits = [0, 0.4 * mattrs.boxsize.min(), 2. * mattrs.boxsize.max()]
        weights = [jnp.maximum((coords >= limits[i]) & (coords < limits[i + 1]), 1e-10) for i in range(len(limits) - 1)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh2_spectrum', 'window_correlation_bessel_mesh2_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        mesh = randoms.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        window = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=ellsin, los=los, bin=bin, pbar=True, flags=('infinite',), norm=norm)
    window = window.clone(observable=window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')))))
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window


def compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=None, cache=None, basis='scoccimarro', ells=[0, 2], los='local', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    wsum_data1 = data.sum()
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    # buffer_size = 4 is too much for QSO, 1 standard GPU node
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01 if 'scoccimarro' in basis else 0.005}, basis=basis, ells=ells, buffer_size=2)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
    #norm = compute_fkp3_normalization(fkp, bin=bin, cellsize=None)
    norm = compute_fkp3_normalization(fkp, split=42, bin=bin, cellsize=10)
    if get_shifted is not None:
        del fkp, randoms
        randoms = ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='jax')
        fkp = FKPField(data, randoms)
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(fkp, los=los, bin=bin, **kw)
    mesh = fkp.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum


def compute_jaxpower_window_mesh3_spectrum(output_fn, get_randoms, spectrum_fn=None, kind='smooth', **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation, compute_fkp3_shotnoise, compute_smooth3_spectrum_window, MeshAttrs, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    #mattrs = mattrs.clone(meshsize=mattrs.meshsize // 4)
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges, basis = spectrum.ells, pole.values('norm')[0], pole.edges('k'), pole.basis
    _, index = np.unique(pole.coords('k', center='mid')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])
    bin = BinMesh3SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells, basis=basis) | kwargs), mask_edges='')
    #bin = BinMesh3SpectrumPoles(mattrs, **(dict(edges={'step': 0.005}, ells=ells, basis=basis) | kwargs), mask_edges='')
    step = bin.edges1d[0][-1, 1] - bin.edges1d[0][-1, 0]
    edgesin = np.arange(0., 1.5 * bin.edges1d[0].max(), step) # / 2.)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]]) 
    output_fn = str(output_fn)
    
    randoms = ParticleField(*get_randoms(), attrs=mattrs, exchange=True, backend='jax')
    zeff = compute_fkp_effective_redshift(randoms, order=3)

    kind = 'smooth'
    if kind == 'smooth':
        correlations = []
        kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=[1] * 3, return_ellsin=True)
        compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0, 1])

        coords = jnp.logspace(-3, 5, 4 * 1024)
        scales = [1, 4]
        b, c = mattrs.boxsize.min(), mattrs.cellsize.min()
        edges = [np.concatenate([np.arange(11) * c, np.arange(11 * c, 0.3 * b, 4 * c)]),
                np.concatenate([np.arange(11) * scales[1] * c, np.arange(11 * scales[1] * c, 2 * b, 4 * scales[1] * c)])]

        for scale, edges in zip(scales, edges):

            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=40)  # kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp3_shotnoise(randoms2, bin=sbin, **kw_paint)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            correlation = compute_mesh3_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
            if output_fn is not None and jax.process_index() == 0:
                correlation_fn = output_fn.replace('window_mesh3_spectrum', f'window_correlation{scale:d}_bessel_mesh3_spectrum')
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
            correlations.append(correlation)

        coords = list(next(iter(correlations[0])).coords().values())
        limit = 0.25 * mattrs.boxsize.min()
        mask = (coords[0] < limit)[:, None] * (coords[1] < limit)[None, :]
        weights = [jnp.maximum(mask, 1e-6), jnp.maximum(~mask, 1e-6)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh3_spectrum', 'window_correlation_bessel_mesh3_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)

        window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        raise NotImplementedError
    window = window.clone(observable=window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')))))
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        #output_fn = output_fn.replace('sugiyama-diagonal', 'sugiyama-diagonal_damped')
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window


def compute_theory_for_covariance_mesh2_spectrum(output_fn, spectrum_fns, window_fn):
    import lsstypes as types
    from jaxpower import (ParticleField, MeshAttrs, compute_spectrum2_covariance)
    mean = types.mean([types.read(fn) for fn in spectrum_fns])
    smooth = mean
    window = types.read(window_fn)

    mattrs = MeshAttrs(**{name: mean.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    covariance = compute_spectrum2_covariance(mattrs, mean)

    sl = slice(0, None, 5)  # rebin to dk = 0.001 h/Mpc
    mean = mean.select(k=sl)
    window = window.at.observable.select(k=sl)
    covariance = covariance.at.observable.select(k=sl)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=window.theory.get(ells=0).z)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(data=mean.value(concatenate=True), wmatrix=window.value(), ells=mean.ells, k=[pole.coords('k') for pole in mean], kin=window.theory.get(ells=0).coords('k'), ellsin=window.theory.ells, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    theory.init.update(k=smooth.get(0).coords('k'))
    poles = theory(**profiles.bestfit.choice(index='argmax', input=True))
    smooth = smooth.clone(value=poles.ravel())
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        smooth.write(output_fn)
    return smooth


def compute_jaxpower_covariance_mesh2_spectrum(output_fn, get_data, get_randoms, get_theory):
    import jax
    from jaxpower import (ParticleField, get_mesh_attrs, MeshAttrs, compute_fkp2_covariance_window, compute_spectrum2_covariance, interpolate_window_function)
    theory = get_theory()
    data, randoms = get_data(), get_randoms()
    mattrs = MeshAttrs(**{name: theory.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fftlog = False
    kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
    windows = compute_fkp2_covariance_window(randoms, alpha=data.sum() / randoms.sum(),
                                             interlacing=3, resampler='tsc', los='local', **kw)
    if output_fn is not None and jax.process_index() == 0:
        for name, window in zip(['WW', 'WS', 'SS'], windows):
            fn = Path(output_fn)
            fn = fn.parent / f'{name}_{fn.name}'
            logger.info(f'Writing to {fn}')
            window.write(fn)

    if fftlog:
        coords = np.logspace(-2, 8, 8 * 1024)
        windows = [interpolate_window_function(window, coords=coords) for window in windows]

    # delta is the maximum abs(k1 - k2) where the covariance will be computed (to speed up calculation)
    covs_analytical = compute_spectrum2_covariance(windows, get_theory(), flags=['smooth'] + (['fftlog'] if fftlog else []), delta=0.4)
    
    # Sum all contributions (WW, WS, SS), with W = standard window (multiplying delta), S = shotnoise
    # Here we assumed randoms have a negligible contribution to the shot noise in the measurements
    cov = covs_analytical[0].clone(value=sum(cov.value() for cov in covs_analytical))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        cov.write(output_fn)
    return cov


def compute_reconstruction(get_data, get_randoms, mode='recsym', z=0.8, bias=2.0, smoothing_radius=15., cellsize=10.):
    # cellsize = 10 vs 4 in the CPU version
    from jaxpower import ParticleField, FKPField, get_mesh_attrs
    from jaxrecon.zeldovich import IterativeFFTReconstruction
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], boxpad=1.2, cellsize=cellsize)

    # Define FKP field = data - randoms
    data = ParticleField(*data, attrs=mattrs, exchange=True, return_inverse=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, return_inverse=True, backend='jax')
    fkp = FKPField(data, randoms)
    # Line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
    # In case of IterativeFFTParticleReconstruction, and multi-GPU computation, provide the size of halo regions in cell units. E.g., maximum displacement is ~ 40 Mpc/h => 4 * chosen cell size => provide halo_add=2
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    z = np.mean(zrange)  # FIXME: to improve with catalog z
    growth_rate = cosmo.growth_rate(z)
    recon = IterativeFFTReconstruction(fkp, growth_rate=growth_rate, bias=bias, los=None, smoothing_radius=smoothing_radius, halo_add=0)
    data_positions_rec = recon.read_shifted_positions(data.positions)
    assert mode in ['recsym', 'reciso']
    # RecSym = remove large scale RSD from randoms
    if mode == 'recsym':
        randoms_positions_rec = recon.read_shifted_positions(randoms.positions)
    else:
        randoms_positions_rec = recon.read_shifted_positions(randoms.positions, field='disp')

    def get_data():
        return data_positions_rec, data.weights

    def get_randoms():
        return randoms_positions_rec, randoms.weights

    return get_data, get_randoms


def compute_triumvirate_mesh3_spectrum(output_fn, get_data, get_randoms, ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama', los='local', boxsize=10000., cellsize=10.):
    from lsstypes.external import from_triumvirate
    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec
    from triumvirate.parameters import ParameterSet
    from triumvirate.logger import setup_logger

    logger = setup_logger(20)
    data_positions, data_weights = get_data()
    data = ParticleCatalogue(*np.array(data_positions.T), ws=data_weights, nz=np.ones_like(data_weights))
    randoms_positions, randoms_weights = get_randoms()
    randoms = ParticleCatalogue(*np.array(randoms_positions.T), ws=randoms_weights, nz=np.ones_like(randoms_weights))

    boxsize = boxsize * np.ones(3, dtype=float)
    cellsize = cellsize * np.ones(3, dtype=float)
    meshsize = np.ceil(boxsize / cellsize).astype(int)
    boxsize = meshsize * cellsize
    edges = np.arange(0., np.pi / cellsize.max(), 0.005)

    results = []
    for ell in ells:
        paramset = dict(norm_convention='mesh', form='diag' if 'diagonal' in basis else 'full', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None), range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='tsc', interlace='off', alignment='centre', padfactor=0., boxsize=dict(zip('xyz', boxsize)), ngrid=dict(zip('xyz', meshsize)), verbose=20)

        paramset = ParameterSet(param_dict=paramset)
        results.append(compute_bispec(data, randoms, paramset=paramset, logger=logger))

    spectrum = from_triumvirate(results, ells=ells)
    if output_fn is not None:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum


def combine_regions(output_fn, fns):
    combined = types.sum([types.read(fn) for fn in fns])  # for the covariance matrix, assumes observables are independent
    if output_fn is not None:
        logger.info(f'Writing to {output_fn}')
        combined.write(output_fn)
    return combined


def get_catalog_fn(version='abacus-2ndgen-complete', kind='data', tracer='LRG', imock=0, zrange=(0.8, 1.1), nran=1, **kwargs):
    desi_dir = Path('/dvs_ro/cfs/cdirs/desi/')
    if version == 'abacus-2ndgen-complete':
        if 'BGS' in tracer:
            base_dir = desi_dir / f'survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummitBGS_v2/mock{imock:d}'
        else:
            base_dir = desi_dir / f'survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{imock:d}'
        if kind == 'data':
            return base_dir / f'{tracer}_complete_clustering.dat.fits'
        if kind == 'randoms':
            return [base_dir / f'{tracer}_complete_{iran:d}_clustering.ran.fits' for iran in range(nran)]
    if version == 'abacus-hf':
        zsnap = {(0.1, 0.4): 0.2, (0.4, 0.6): 0.5, (0.6, 0.8): 0.725, (0.8, 1.1): 0.950, (1.1, 1.6): 0.950, (0.8, 2.1): 1.4}[zrange]
        sznap = f'{zsnap:.3f}'.replace('.', 'p')
        stracer = tracer
        if tracer == 'QSO':
            szrange = '0p8to3p5'
        if tracer == 'ELG_LOP':
            szrange = '0p8to1p6'
            stracer = 'ELG_v5'
        if tracer == 'LRG':
            szrange = '0p4to1p1'
        if tracer == 'BGS':
            szrange = '0p1to0p4'
        base_dir = desi_dir / f'mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{stracer}/z{zsnap:.3f}/forclustering'
        print(base_dir)
        if kind == 'data':
            return base_dir / f'cutsky_abacusHF_DR2_{tracer}_z{sznap}_zcut_{szrange}_clustering.dat.fits'
        if kind == 'randoms':
            #return [desi_dir / f'mocks/cai/abacus_HF/DR2_v1.0/randoms/rands_intiles_DARK_nomask_{iran:d}_v2.fits' for iran in range(nran)]
            return [desi_dir / f'mocks/cai/abacus_HF/DR2_v1.0/randoms/raw/rands_intiles_DARK_{iran:d}_NO_imagingmask_withz.fits' for iran in range(nran)]
    if 'holi' in version:
        logger.info(f"Attempting to load {version}")
        version = version.replace('holi-', '')
        tracer = {'ELG_LOP': 'ELG'}.get(tracer, tracer)
        base_dir = desi_dir / f'mocks/cai/holi/{version}/seed{imock:04d}'
        if catalog == 'data':
            return base_dir / f'holi_{tracer}_{version.replace("i","")}_GCcomb_clustering.dat.h5'
        if catalog == 'randoms':
            return [base_dir / f'holi_{tracer}_{version.replace("i","")}_GCcomb_{iran:d}_clustering.ran.h5' for iran in range(nran)]
    if version == 'glam-uchuu-final':
        base_dir = Path('/pscratch/sd/e/efdez/Uchuu-GLAM/GLAM/final_mocks/')
        if kind == 'data':
            return base_dir / f'GLAM-Uchuu_{tracer}_{imock:02d}_Y3_cut_sky_clustering.dat.fits'
        if kind == 'randoms':
            return [base_dir / f'randoms/GLAM-Uchuu_{tracer}_{imock:02d}_Y3_cut_sky_clustering.ran.fits']
    if 'uchuu-hf' in version:
        if 'altmtl' in version:
            base_dir =  Path(desi_dir / f'mocks/cai/Uchuu-SHAM/Y3-v2.0/{imock:04d}/altmtl/')
        else:
            base_dir =  Path(desi_dir / f'mocks/cai/Uchuu-SHAM/Y3-v2.0/{imock:04d}/complete/')
        if kind == 'data':
            return Path(base_dir / f'Uchuu-SHAM_{tracer.upper()}_Y3-v2.0_0000_clustering.dat.fits')
        if kind == 'randoms':
            return [base_dir / f'Uchuu-SHAM_{tracer.upper()}_Y3-v2.0_0000_{iran}_clustering.ran.fits' for iran in range(nran)]
    raise ValueError('issue with input args')


def get_measurement_fn(kind='mesh2_spectrum_poles', version='abacus-2ndgen-complete', recon=None, tracer='LRG', region='NGC', zrange=(0.8, 1.1), imock=0, **kwargs):
    if imock is None:
        import glob
        return sorted(glob.glob(get_measurement_fn(kind=kind, version=version, recon=recon, tracer=tracer, region=region, zrange=zrange, imock='*')))
    #base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-benchmark-dr2/summary_statistics')
    base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/summary_statistics')
    base_dir = base_dir / (f'cutsky_{recon}' if recon else 'cutsky')
    base_name = f'{version}/{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}'
    base_name = f'{base_name}_{imock}.h5' if not any(name in kind for name in ['window', 'covariance']) else f'{base_name}.h5'
    return str(base_dir / base_name)


def get_imocks(nmocks=5, version='abacus-2ndgen-complete'):
    if 'holi' in version:
        return list(range(201, 201 + nmocks))
    if 'uchuu-hf' in version:
        return [0]
    if version == 'glam-uchuu-final':
        return list(range(10, 10 + nmocks))
    return list(range(nmocks))


if __name__ == '__main__':

    tracers = [('BGS', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG', (0.8, 1.1)), ('ELG_LOP', (0.8, 1.1)), ('ELG_LOP', (1.1, 1.6)), ('QSO', (0.8, 2.1))][-3:] #[-3:]
    regions = ['NGC', 'SGC']
    versions = ['abacus-2ndgen-complete', 'abacus-hf', 'holi-v3.00', 'glam-uchuu-final', 'uchuu-hf-complete'][:1]
    nmocks = 25
    todo = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'mesh3_spectrum_sugiyama', 'window_mesh3_spectrum_sugiyama', 'mesh3_spectrum_scoccimarro', 'mesh3_spectrum_sugiyama_triumvirate'][1:2] #[1:3] #[3:4]
    #todo += ['recsym']
    todo = ['combine']
    setup_logging()

    with_jax = any(td in ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'mesh3_spectrum_scoccimarro', 'mesh3_spectrum_sugiyama', 'window_mesh3_spectrum_sugiyama', 'recsym'] for td in todo)
    import jax
    from jax import config
    config.update('jax_enable_x64', True)

    if with_jax:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
        jax.distributed.initialize()
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
    from jaxpower.mesh import create_sharding_mesh

    for (tracer, zrange), region, version in itertools.product(tracers, regions, versions):

        catalog_args = dict(version=version, region=region, tracer=tracer, zrange=zrange, nran=4)
        spectrum_args = dict(**get_proposal_mattrs(catalog_args['tracer']), ells=(0, 2, 4))
        iimock = 0
        cache = {}
        for imock in get_imocks(nmocks=nmocks, version=catalog_args['version']):
            data_fn = get_catalog_fn(imock=imock, kind='data', **catalog_args)
            all_randoms_fn = get_catalog_fn(imock=imock, kind='randoms', **catalog_args)
            if not data_fn.exists() or not all(fn.exists() for fn in all_randoms_fn): continue
            iimock += 1
            if iimock > nmocks: continue
            get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
            get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)
            get_shifted = None

            for mode in ['recsym', 'reciso']:
                if mode in todo:
                    with create_sharding_mesh() as sharding_mesh:
                        get_data, get_shifted = compute_reconstruction(get_data, get_randoms, mode=mode, z=catalog_args['zrange'], **get_proposal_recattrs(catalog_args['tracer']))
                    catalog_args.update(recon=mode)

            if 'combine' in todo and region == regions[0]:
                for kind in ['mesh2_spectrum_poles', 'window_mesh2_spectrum_poles', 'covariance_mesh2_spectrum_poles']:
                    kw = dict(imock=imock, kind=kind, **catalog_args)
                    fns = [get_measurement_fn(**(kw | dict(region=region))) for region in regions]
                    #for fn in fns: types.read(fn).write(fn)
                    output_fn = get_measurement_fn(**(kw | dict(region='GCcomb')))
                    combine_regions(output_fn, fns)

            if 'mesh2_spectrum' in todo:
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2_spectrum_poles')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, get_shifted=get_shifted, cache=cache, **spectrum_args)

            if 'window_mesh2_spectrum' in todo and iimock == 1:
                jax.experimental.multihost_utils.sync_global_devices("spectrum")
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2_spectrum_poles')
                output_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=get_data, spectrum_fn=spectrum_fn)
                    jax.clear_caches()

            if 'covariance_mesh2_spectrum' in todo and iimock == 1:
                jax.experimental.multihost_utils.sync_global_devices("spectrum")

                def get_theory():
                    spectrum_fns = get_measurement_fn(imock=None, **catalog_args, kind='mesh2_spectrum_poles')
                    window_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
                    return compute_theory_for_covariance_mesh2_spectrum(None, spectrum_fns, window_fn)
                   
                output_fn = get_measurement_fn(**catalog_args, kind='covariance_mesh2_spectrum_poles')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_covariance_mesh2_spectrum(output_fn, get_data, get_randoms, get_theory)
                    jax.clear_caches()
                jax.experimental.multihost_utils.sync_global_devices("spectrum")

            if 'mesh3_spectrum_scoccimarro' in todo:
                bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2], cellsize=12)
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=get_shifted, cache=cache, **bispectrum_args)
                    jax.clear_caches()  # free-up some memory (why is that necessary?)

            if 'mesh3_spectrum_sugiyama' in todo:
                bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], cellsize=12)
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=get_shifted, cache=cache, **bispectrum_args)
                    jax.clear_caches()

            if 'window_mesh3_spectrum_sugiyama' in todo and iimock == 1:
                jax.experimental.multihost_utils.sync_global_devices("spectrum")
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_sugiyama-diagonal')
                output_fn = get_measurement_fn(**catalog_args, kind=f'window_mesh3_spectrum_poles_sugiyama-diagonal')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_window_mesh3_spectrum(output_fn, get_randoms, spectrum_fn=spectrum_fn)
                    jax.clear_caches()

            if 'mesh3_spectrum_sugiyama_triumvirate' in todo:
                bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], cellsize=12)
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}_triumvirate')
                compute_triumvirate_mesh3_spectrum(output_fn, get_data, get_randoms, **bispectrum_args)

    if with_jax:
        jax.distributed.shutdown()
