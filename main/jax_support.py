import os
import jax
import logging
from jax import numpy as jnp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('jax_support') 

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpiroot = 0

def _get_jax_coordinator_address():
    import socket
    host = mpicomm.bcast(os.environ.get('SLURMD_NODENAME', socket.gethostname()) if mpicomm.rank == mpiroot else None, root=mpiroot)
    if 'JAX_COORDINATOR_PORT' in os.environ:
        port = int(os.environ['JAX_COORDINATOR_PORT'])
    else:
        job_id = os.environ.get('SLURM_JOB_ID', None)
        port = int(job_id) % 2**12 + (65535 - 2**12 + 1) if job_id is not None else 12355
    return f'{host}:{port}'

def _get_local_process_id():
    for name in ['SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'PMI_LOCAL_RANK', 'MPI_LOCALRANKID']:
        value = os.environ.get(name, None)
        if value is not None:
            return int(value)
    return None

def _get_local_device_ids(local_process_id):
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(',') if device.strip()]
        if len(devices) == 1:
            return [0]
    return [local_process_id]


def initialize_jax_distributed():
    if jax.distributed.is_initialized():
        return
    if mpicomm.size <= 1:
        if mpicomm.rank == mpiroot:
            if int(os.environ.get('SLURM_NTASKS', '1')) > 1:
                logger.warning('SLURM allocated multiple tasks but MPI world size is 1; launch this script with srun to use all ranks.')
            logger.info('Skipping jax.distributed.initialize(); running with a single process.')
        return
    init_kwargs = {
        'coordinator_address': _get_jax_coordinator_address(),
        'num_processes': mpicomm.size,
        'process_id': mpicomm.rank,
        'cluster_detection_method': 'deactivate',
    }
    local_process_id = _get_local_process_id()
    if local_process_id is not None:
        init_kwargs['local_device_ids'] = _get_local_device_ids(local_process_id)
    if mpicomm.rank == mpiroot:
        logger.info(f'Initializing JAX distributed with {mpicomm.size} MPI ranks via {init_kwargs["coordinator_address"]}')
    jax.distributed.initialize(**init_kwargs)


def get_interpolator_1d(x: jax.Array, y: jax.Array, order: int=1):
    """
    Return a 1D interpolator function for arrays x, y.
    """
    xmin, xmax = x[0], x[-1]
    step = (xmax - xmin) / (len(x) - 1)

    if order == 1:
        def interp(xp):
            return jnp.interp(xp, x, y)
    else:
        from interpax import Interpolator1D
        interpolator = Interpolator1D(x, jnp.asarray(y), method={1: 'linear', 3: 'cubic2'}[order], extrap=False, period=None)
        @jax.jit
        def interp(xp):
            # clip
            toret = interpolator(xp)
            #return jnp.where((xp >= xmin) & (xp <= xmax), toret, 0.)
            toret = jnp.where(xp < xmin, y[0], toret)
            toret = jnp.where(xp > xmax, y[-1], toret)
            return toret
    return interp
