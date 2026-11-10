import os, sys
import time
import logging
logger = logging.getLogger('Utils')


def setup_fit_env_notebook_paths(cosmodesi_root=None, cpu_jax=True, clear_modules=True):
    """Mirror the fit_env Python path setup inside notebooks."""
    from pathlib import Path

    cosmodesi_root = cosmodesi_root or '/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0'
    cosmodesi_paths = [
        f'{cosmodesi_root}/code/desilike/refactor-jax/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/folpsD/adematti-damping/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/folpsax/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/folps/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/lsstypes/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/jaxpower/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/mockfactory/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/mpytools/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/cosmoprimo/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/code/desi-clustering/main/lib/python3.12/site-packages',
        f'{cosmodesi_root}/conda/lib/python3.12/site-packages',
    ]
    cosmodesi_paths = [path for path in cosmodesi_paths if Path(path).is_dir()]
    sys.path = cosmodesi_paths + [
        path for path in sys.path
        if path not in cosmodesi_paths and '/global/u1/s/shengyu/packages/desilike' not in path
    ]

    if clear_modules:
        for module_name in list(sys.modules):
            if module_name in {'desilike', 'folps'} or module_name.startswith(('desilike.', 'folps.')):
                del sys.modules[module_name]

    fit_env_runtime_dir = Path(os.environ.get('FIT_ENV_RUNTIME_DIR', f"{os.environ.get('TMPDIR', '/tmp')}/fit_env_{os.environ.get('USER', 'shengyu')}"))
    os.environ['FIT_ENV_RUNTIME_DIR'] = str(fit_env_runtime_dir)
    os.environ.setdefault('DESILIKE_CONFIG_DIR', str(fit_env_runtime_dir / 'desilike'))
    os.environ.setdefault('MPLCONFIGDIR', str(fit_env_runtime_dir / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(fit_env_runtime_dir / 'cache'))
    for env_name in ['DESILIKE_CONFIG_DIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

    if cpu_jax:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['MPICH_GPU_SUPPORT_ENABLED'] = '0'

    return cosmodesi_paths


def float2str(value, prec_min=0, prec_max=None):
    """Format floats compactly while keeping a minimum number of decimals."""
    if prec_max is None:
        prec_max = prec_min
    text = f'{float(value):.{prec_max}f}'
    if prec_max > prec_min:
        head, _, tail = text.partition('.')
        tail = tail.rstrip('0')
        if len(tail) < prec_min:
            tail = tail + '0' * (prec_min - len(tail))
        text = head if not tail else f'{head}.{tail}'
    return text

def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    import traceback
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    def mkdir(dirname):
        """Try to create ``dirname`` and catch :class:`OSError`."""
        try:
            os.makedirs(dirname)  # MPI...
        except OSError:
            return
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Print exception with a logger."""
        # Do not print traceback if the exception has been handled and logged
        _logger_name = 'Exception'
        log = logging.getLogger(_logger_name)
        line = '=' * 100
        # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
        log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
        if exc_type is KeyboardInterrupt:
            log.critical('Interrupted by the user.')
        else:
            log.critical('An error occured.')
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    t0 = time.time()
    class MyFormatter(logging.Formatter):
        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)
    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler
