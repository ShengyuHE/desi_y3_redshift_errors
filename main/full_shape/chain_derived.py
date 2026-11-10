"""Post-process base-LCDM chains without evaluating their fitted likelihood."""

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version

import numpy as np


COSMO_PARAMS = ('h', 'omega_cdm', 'omega_b', 'logA', 'n_s')


def _class_sigma8(row):
    from cosmoprimo.fiducial import DESI

    cosmology = DESI(engine='class').clone(**dict(zip(COSMO_PARAMS, row)))
    return float(cosmology.get_fourier().sigma8_z(0., of='delta_m'))


def base_sigma8_m(samples, cache_dir, workers=1):
    """Return total-matter sigma8 at z=0 for the project's base-LCDM fits.

    ``samples`` maps the five COSMO_PARAMS to arrays (or desilike Parameters).
    All other cosmological settings are fixed to DESI, as in these base fits.
    Use this only for that cosmology, not extended models. Values are evaluated
    with CLASS, never with the COMET amplitude or Eisenstein-Hu spectrum.

    Cache keys include the inputs, fiducial settings and solver versions. Partial
    results are saved every ten samples and can be resumed after interruption.
    ``workers > 1`` uses spawned processes to isolate the Boltzmann solver.
    Call from a notebook or a script protected by ``if __name__ == '__main__'``.
    """
    from cosmoprimo.fiducial import DESI

    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    arrays = [np.asarray(getattr(samples[p], 'value', samples[p]), dtype='f8')
              for p in COSMO_PARAMS]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Cosmological parameter arrays must have matching shapes')
    rows = np.stack(arrays, axis=-1).reshape(-1, len(COSMO_PARAMS))
    if not np.isfinite(rows).all():
        raise ValueError('Cosmological parameters must be finite')
    metadata = dict(schema=1, definition='CLASS sigma8_z(0, of=delta_m)',
                    cosmoprimo=version('cosmoprimo'), pyclass=version('pyclass'),
                    fiducial=DESI().get_params(), params=COSMO_PARAMS)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True,
                                      default=lambda a: np.asarray(a).tolist()).encode())
    digest.update(rows.tobytes())
    cache_fn = Path(cache_dir) / f'sigma8_m_{digest.hexdigest()}.npz'
    values = np.full(len(rows), np.nan)
    if cache_fn.exists():
        with np.load(cache_fn, allow_pickle=False) as cached:
            if not np.array_equal(cached['inputs'], rows):
                raise ValueError(f'Input mismatch in {cache_fn}')
            values[:] = cached['sigma8_m']
    missing = np.flatnonzero(~np.isfinite(values))
    if not len(missing):
        print(f'Loaded cached sigma8_m: {len(rows)} samples', flush=True)
        return values.reshape(arrays[0].shape)

    cache_fn.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        with tempfile.NamedTemporaryFile(dir=cache_fn.parent, suffix='.npz', delete=False) as tmp:
            try:
                np.savez(tmp, inputs=rows, sigma8_m=values)
                tmp.close()
                os.replace(tmp.name, cache_fn)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    print(f'Deriving sigma8_m with CLASS: {len(missing)}/{len(rows)} samples '
          f'remain, {workers} worker(s). First pass can take hours; cache is resumable.', flush=True)
    executor = None
    started = time.monotonic()
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=multiprocessing.get_context('spawn'))
            results = executor.map(_class_sigma8, rows[missing])
        else:
            results = map(_class_sigma8, rows[missing])
        for done, (index, result) in enumerate(zip(missing, results), 1):
            if not np.isfinite(result) or result <= 0:
                raise ValueError(f'Invalid sigma8_m at sample {index}: {result}')
            values[index] = result
            if done % 10 == 0 or done == len(missing):
                checkpoint()
                print(f'sigma8_m: {done}/{len(missing)} new samples, '
                      f'{time.monotonic() - started:.0f}s elapsed', flush=True)
    finally:
        checkpoint()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return values.reshape(arrays[0].shape)
