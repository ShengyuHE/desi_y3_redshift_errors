"""Persist the mock covariances used by clustering diagnostic notebooks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import lsstypes as types

from cat_tools import get_cov_fn, get_measurement_fn, get_simple_tracer
from fit_tools import load_bins, read_data_from_fn


DEFAULT_COV_DIR = Path('/global/cfs/cdirs/desi/users/shengyu/galaxies/catalogs/Y3/holi-v3/altmtl/cov')
HOLI_SKIP_FN = Path(__file__).parent / 'dubious_holi-v3-altmtl.txt'
MEASUREMENT_KINDS = {
    'mesh2': 'mesh2_spectrum_poles', 'mesh3': 'mesh3_spectrum_poles_sugiyama',
    'mesh3_sugiyama': 'mesh3_spectrum_poles_sugiyama',
    'mesh3_scoccimarro': 'mesh3_spectrum_poles_scoccimarro',
    'pk': 'pkpoles', 'xi': 'xipoles', 'mpslog': 'mpslog', 'wplog': 'wplog',
}


def load_plot_covariance(cov_args, corr, data, ells=None, bin_type='y3_sys', cache_dir=None):
    """Load a saved covariance and select bins/multipoles in plotting order.

    Native lsstypes observables are passed straight to at.observable.match.
    For array-based readers, ``data`` is the actual k or s coordinate array.
    Effective bin centers may differ between catalogs by up to 5% of the local
    bin spacing. They are matched one-to-one to saved bins, never interpolated.
    Array matching supports selection only; native observables also support
    rebinning using the saved edges and weights.
    Missing files or incompatible bins raise; this function never reads mocks.
    """
    fn = get_cov_fn(**cov_args, cache_dir=cache_dir).format(MEASUREMENT_KINDS[corr])
    if not Path(fn).is_file():
        raise FileNotFoundError(f'No saved covariance: {fn}. Prepare this covariance separately.')
    covariance = types.read(fn)
    if isinstance(data, (types.ObservableTree, types.ObservableLeaf)):
        if ells is not None:
            data = data.get(ells=list(ells))
        return covariance.at.observable.match(data)
    coords = data
    config = json.loads(covariance.attrs['config_json'])
    saved_corr = config['corr']
    if saved_corr != corr and {saved_corr, corr} != {'mesh3', 'mesh3_sugiyama'}:
        raise ValueError(f'Saved covariance statistic {saved_corr!r} does not match {corr!r}.')
    # mesh3's plotting reader always uses native kbin=1, irrespective of bin_type.
    if corr not in ['mesh3', 'mesh3_sugiyama']:
        step = load_bins(corr, bin_type)[2]
        if step != config['bins'][2]:
            raise ValueError(f'Observable bin width {step} differs from saved covariance '
                             f'bin width {config["bins"][2]}; prepare a covariance with matching bins.')
    if ells is None:
        ells = ([(0, 0, 0), (2, 0, 2)] if corr in ['mesh3', 'mesh3_sugiyama']
                else [0] if corr == 'wplog' else [0, 2, 4])
    axis = 'k' if corr in ['pk', 'mesh2', 'mesh3', 'mesh3_sugiyama'] else 's'
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or not coords.size or not np.isfinite(coords).all():
        raise ValueError('Expected a nonempty, finite, one-dimensional observable coordinate array.')
    leaves = []
    for ell in ells:
        pole = covariance.observable.get(ells=ell)
        saved = np.asarray(pole.coords(axis))
        if len(saved) < 2 or np.any(np.diff(saved) <= 0):
            raise ValueError(f'Cannot match {corr} coordinates: saved bins must be strictly increasing.')
        distance = np.abs(coords[:, None] - saved[None, :])
        indices = distance.argmin(axis=1)
        spacing = np.diff(saved)
        tolerance = 0.05 * np.minimum(np.r_[spacing[0], spacing], np.r_[spacing, spacing[-1]])
        if (len(np.unique(indices)) != len(indices) or
                np.any(distance[np.arange(len(coords)), indices] > tolerance[indices])):
            raise ValueError(f'Observable {axis} bins do not match saved {corr} covariance bins in {fn}. '
                             'Use covered bins with the same binning, or prepare a matching covariance.')
        # Snap only for the native exact-coordinate selection. Plot coordinates
        # remain those measured in the data; both axes of the covariance follow.
        leaves.append(types.ObservableLeaf(coords=[axis], **{axis: saved[indices]},
                                           value=np.zeros(len(indices))))
    observable = types.ObservableTree(leaves, ells=list(ells))
    return covariance.at.observable.match(observable)


def get_saved_error_bars(cov_args, corr, data, ells=None, **kwargs):
    """Read and align a saved covariance; return single-realization errors."""
    covariance = load_plot_covariance(cov_args, corr, data, ells=ells, **kwargs)
    if isinstance(data, (types.ObservableTree, types.ObservableLeaf)):
        shape = np.shape(covariance.observable.value(concatenate=False)) if isinstance(
            covariance.observable, types.ObservableTree) else (1, covariance.observable.size)
        errors = np.asarray(covariance.std()).reshape(shape)
    else:
        errors = np.asarray(covariance.std()).reshape(-1, len(data))
    if not np.isfinite(errors).all() or np.any(errors <= 0):
        raise ValueError('Matched covariance has invalid or zero diagonal errors.')
    return errors


def _compatible_config(config):
    """Compare the actual reader settings, including equivalent binning aliases."""
    config = dict(config)
    config.pop('bin_type', None)
    config['bins'] = list(config['bins'][:3])  # The reader does not use length.
    if config['corr'] in ['mesh3', 'mesh3_sugiyama']:
        config['corr'] = 'mesh3_sugiyama'
        config['bins'][2] = 1  # The diagnostic reader uses kbin=1 for both aliases.
    return config


def _save_covariance(covariance, fn):
    fn.parent.mkdir(parents=True, exist_ok=True)
    covariance.attrs['cache_path'] = str(fn)
    fd, temporary = tempfile.mkstemp(prefix=f'.{fn.stem}-', suffix='.h5', dir=fn.parent)
    os.close(fd)
    try:
        covariance.write(temporary)
        os.replace(temporary, fn)
    finally:
        Path(temporary).unlink(missing_ok=True)


def get_mock_covariance(cov_args, corr, num=None, bin_type='y3_sys', remove_bins=None,
                        cache_dir=DEFAULT_COV_DIR, rebuild=False, skip_mock_ids=None):
    """Read a cached lsstypes covariance, or build it using the plotting reader.

    The covariance describes one realization (sample covariance with ddof=1),
    including correlations between bins and multipoles. No division by the
    number of mocks or inverse-covariance correction is applied.
    Filenames prepend ``covariance_`` to the measurement name, retaining its
    region and redshift-error suffix. ``cov_args['use_dv']`` is respected.
    Binning and mock selection are validated against the saved metadata.
    Set rebuild=True to refresh after adding/replacing measurement files.
    holi-v3 mpslog defaults to mock IDs 0--119 and drops the first 11 LRG
    or 16 QSO separation bins, matching the small-scale diagnostic plots.
    """
    catalog = dict(cov_args, mkdir=False)
    catalog.setdefault('use_dv', False)
    catalog.setdefault('z_evol', False)
    catalog.setdefault('use_jax', corr != 'mpslog')
    catalog.pop('mock_id', None)
    if num is None:
        num = (120 if corr == 'mpslog' else 1000) if catalog['version'] == 'holi-v3' else 25
    if corr == 'mpslog' and catalog['version'] == 'holi-v3' and remove_bins is None:
        remove_bins = 16 if get_simple_tracer(catalog['tracer']) == 'QSO' else 11
    if skip_mock_ids is None:
        skip_mock_ids = (np.atleast_1d(np.loadtxt(HOLI_SKIP_FN, dtype=int))
                         if catalog['version'] == 'holi-v3' else [])
    excluded = set(map(int, skip_mock_ids))
    requested = [i for i in range(num) if i not in excluded]
    template = get_measurement_fn(**catalog, mock_id=0)
    config = dict(schema=1, template=template, corr=corr, bin_type=bin_type,
                  bins=load_bins(corr, bin_type), remove_bins=remove_bins,
                  requested_mock_ids=requested)
    config_json = json.dumps(config, sort_keys=True)
    digest = hashlib.sha256(config_json.encode()).hexdigest()[:12]
    tracer = get_simple_tracer(catalog['tracer'])
    zrange = catalog.get('zrange')
    zlabel = (f'z{zrange[0]}-{zrange[1]}' if catalog.get('domain') != 'cubic'
              else f'z{catalog["zsnap"]}')
    region = catalog.get('region', 'ALL')
    kinds = MEASUREMENT_KINDS
    fn = Path(get_cov_fn(**catalog, cache_dir=cache_dir).format(kinds[corr]))
    if fn.is_file() and not rebuild:
        covariance = types.read(fn)
        if _compatible_config(json.loads(covariance.attrs['config_json'])) != _compatible_config(config):
            raise ValueError(f'{fn} has different binning or mock selection. '
                             'Use rebuild=True to replace it, or choose another cache_dir.')
        print(f'Loaded covariance ({covariance.attrs["nobs"]} mocks): {fn}', flush=True)
        return covariance

    # Rename matching caches from the old hashed convention without rereading mocks.
    legacy = Path(cache_dir) / f'covariance_{corr}_{tracer}_{zlabel}_{region}_{bin_type}_{digest}.h5'
    if legacy.is_file() and not rebuild:
        covariance = types.read(legacy)
        if _compatible_config(json.loads(covariance.attrs['config_json'])) == _compatible_config(config):
            _save_covariance(covariance, fn)
            legacy.unlink()
            print(f'Renamed covariance ({covariance.attrs["nobs"]} mocks): {fn}', flush=True)
            return covariance

    print(f'Building {corr} covariance for {tracer} {zlabel} {region}, use_dv={catalog["use_dv"]!r}: '
          f'{len(requested)} requested mocks', flush=True)
    samples, used, missing = [], [], []
    coordinates = None
    for index, mock_id in enumerate(requested):
        template = get_measurement_fn(**catalog, mock_id=mock_id)
        if not Path(template.format(kinds[corr])).is_file():
            missing.append(mock_id)
            continue
        (coords, values), _ = read_data_from_fn(template, corr, bin_type=bin_type,
                                               remove_bins=remove_bins, verbose=False)
        if values is None:
            missing.append(mock_id)
            continue
        if isinstance(values, dict):
            values = list(values.values())
        values = np.atleast_2d(np.asarray(values))
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Mock {mock_id} contains nonfinite {corr} values after bin selection: '
                             f'{template.format(kinds[corr])}')
        if coordinates is None:
            coordinates = np.asarray(coords)
            shape = values.shape
        if values.shape != shape:
            raise ValueError(f'Mock {mock_id} has shape {values.shape}, expected {shape}.')
        # Effective coordinates can vary slightly between mocks, but bin counts
        # must agree. The reader applies the same binning to every realization.
        if np.shape(coords) != coordinates.shape:
            raise ValueError(f'Mock {mock_id} has inconsistent coordinate dimensions.')
        samples.append(values)
        used.append(mock_id)
        if (index + 1) % 100 == 0:
            print(f'  Read {len(used)} mocks', flush=True)
    if len(samples) < 2:
        raise FileNotFoundError(f'Need at least two readable mocks for {fn.name}; found {len(samples)}.')
    samples = np.asarray(samples)
    nells = samples.shape[1]
    if corr in ['mesh3', 'mesh3_sugiyama']:
        ells = [(0, 0, 0), (2, 0, 2)][:nells]
    elif corr == 'wplog':
        ells = [0]
    else:
        ells = [0, 2, 4][:nells]
    if len(ells) != nells:
        raise ValueError(f'Unexpected number of multipoles for {corr}: {nells}')
    coord_name = 'k' if corr in ['pk', 'mesh2', 'mesh3', 'mesh3_sugiyama'] else 's'
    observable = types.ObservableTree([
        types.ObservableLeaf(coords=[coord_name], value=mean, **{coord_name: coordinates})
        for mean in samples.mean(axis=0)
    ], ells=ells)
    if corr in ['mesh2', 'mesh3', 'mesh3_sugiyama']:
        # Retain native bin edges, weights and (for mesh3) all three k sides.
        first_template = get_measurement_fn(**catalog, mock_id=used[0])
        observable, _ = read_data_from_fn(first_template, corr, bin_type=bin_type,
                                         return_observable=True, verbose=False)
        observable = observable.clone(value=samples.mean(axis=0).ravel())
    covariance = types.CovarianceMatrix(
        np.atleast_2d(np.cov(samples.reshape(len(samples), -1), rowvar=False, ddof=1)),
        observable=observable,
        attrs=dict(nobs=len(used), mock_ids=used, missing_mock_ids=missing,
                   sample_shape=list(samples.shape[1:]), config_json=config_json,
                   normalization='single realization; ddof=1', cache_path=str(fn)))
    # Publish only a complete HDF5 file, so an interrupted build cannot be reused.
    _save_covariance(covariance, fn)
    print(f'Saved covariance ({len(used)} mocks, {len(missing)} missing): {fn}', flush=True)
    return covariance


def get_mock_error_bars(cov_args, corr, **kwargs):
    """Return cached standard deviations in the notebook's (multipole, bin) order."""
    covariance = get_mock_covariance(cov_args, corr, **kwargs)
    return covariance.std().reshape(tuple(covariance.attrs['sample_shape']))
