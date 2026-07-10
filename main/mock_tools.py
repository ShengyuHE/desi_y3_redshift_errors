import sys
import logging
import numpy as np
from pathlib import Path
import warnings
import time
from mpi4py import MPI

sys.path.append(str(Path(__file__).resolve().parent))
from utils import setup_logging
setup_logging()
logger = logging.getLogger('mock_tools')

mpicomm = MPI.COMM_WORLD
mpiroot = 0
mpirank = mpicomm.rank
mpisize = mpicomm.size

ASCII_MOCK_COLUMNS = ("RA", "DEC", "Z", "Z_COSMO", "NZ", "STATUS", "RAN_NUM_0_1")
ASCII_MOCK_DV_COLUMNS = ("RA", "DEC", "Z", "Z_COSMO", "Z_VSMEAR", "NZ", "STATUS", "RAN_NUM_0_1")

ASCII_MOCK_COLUMNS_BY_HOD = {
    "base": ASCII_MOCK_COLUMNS,
    "base_dv": ASCII_MOCK_DV_COLUMNS,
}


def get_ascii_mock_columns(hod="base", columns=None):
    if columns is not None:
        return tuple(columns)
    try:
        return ASCII_MOCK_COLUMNS_BY_HOD[hod]
    except KeyError as exc:
        valid = ", ".join(sorted(ASCII_MOCK_COLUMNS_BY_HOD))
        raise ValueError(f"Unknown HOD model {hod!r}; expected one of: {valid}") from exc

def _normalize_array(cat, ncols):
    cat = np.asarray(cat)
    if cat.size == 0:
        return np.empty((0, ncols), dtype="f8")
    if cat.ndim == 1:
        cat = cat[None, :]
    return cat

def _skip_ascii_data_rows(handle, nrows):
    skipped = 0
    while skipped < nrows:
        line = handle.readline()
        if line == "":
            break
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            skipped += 1

def _split_rows(nrows, rank, size):
    start = nrows * rank // size
    stop = nrows * (rank + 1) // size
    return start, stop

def _split_bytes(nbytes, rank, size):
    start = nbytes * rank // size
    stop = nbytes * (rank + 1) // size
    return start, stop

def _iter_ascii_byte_range(fn, byte_start, byte_stop):
    with open(fn, "rb") as handle:
        if byte_start > 0:
            handle.seek(byte_start - 1)
            previous = handle.read(1)
            if previous != b"\n":
                handle.readline()
            else:
                handle.seek(byte_start)
        else:
            handle.seek(byte_start)
        while True:
            line_start = handle.tell()
            if line_start >= byte_stop:
                break
            line = handle.readline()
            if not line:
                break
            yield line

def select_by_status(cat, status_col=5, keep_status="nonzero", return_mask=False):
    """
    Select catalog rows by the STATUS column.

    The QSO cutsky ASCII catalogs have columns:
    RA, DEC, Z, Z_COSMO, NZ, STATUS, RAN_NUM_0_1.
    The base_dv HOD catalogs add Z_VSMEAR before NZ.
    By default, this keeps rows with STATUS != 0.
    """
    status = np.asarray(cat[:, status_col])
    ok = np.isfinite(status)
    if keep_status == "nonzero":
        ok &= status != 0
    elif keep_status is not None:
        ok &= np.isin(status, np.atleast_1d(keep_status))
    selected = cat[ok]
    if return_mask:
        return selected, ok
    return selected

def array_to_catalog(cat, columns=ASCII_MOCK_COLUMNS, mpicomm=None, add_weight=True):
    """Convert an ASCII mock array to a mockfactory.Catalog."""
    from mockfactory import Catalog
    cat = _normalize_array(cat, len(columns))
    if cat.shape[1] < len(columns):
        raise ValueError(f"Expected at least {len(columns)} columns, got {cat.shape[1]}")
    data = {name: cat[:, icol] for icol, name in enumerate(columns)}
    data["STATUS"] = data["STATUS"].astype("i4")
    if add_weight and "WEIGHT" not in data:
        data["WEIGHT"] = np.ones(len(cat), dtype="f8")
    return Catalog(data, mpicomm=mpicomm)

def read_ascii_mock_catalog(fn, columns=None, select_status=False, keep_status="nonzero",
                            max_rows=None, mpicomm=None, add_weight=True, hod="base"):
    """Read an ASCII mock by assigning one contiguous byte block to each MPI rank."""
    mpicomm = mpicomm or globals()["mpicomm"]
    rank = mpicomm.rank
    size = mpicomm.size
    fn = Path(fn)
    columns = get_ascii_mock_columns(hod=hod, columns=columns)
    status_col = columns.index("STATUS")
    t_total = time.perf_counter()
    if max_rows is None:
        file_size = fn.stat().st_size
        byte_start, byte_stop = _split_bytes(file_size, rank, size)
        if rank == mpiroot:
            logger.info(f"Reading all rows from {fn} across {size} MPI rank(s), file_size={file_size} bytes")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Input line .* contained no data")
            warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
            cat = np.loadtxt(_iter_ascii_byte_range(fn, byte_start, byte_stop))
        cat = _normalize_array(cat, len(columns))
    else:
        row_start, row_stop = _split_rows(max_rows, rank, size)
        with open(fn, "r") as handle:
            _skip_ascii_data_rows(handle, row_start)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Input line .* contained no data")
                warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
                cat = np.loadtxt(handle, max_rows=row_stop - row_start)
        cat = _normalize_array(cat, len(columns))
    if select_status:
        cat = select_by_status(cat, status_col=status_col, keep_status=keep_status)
        logger.info(f"Rank {rank} selected {len(cat)} rows with STATUS {keep_status}")
    mpicomm.Barrier()
    if rank == mpiroot:
        logger.info(f"Finished reading in {time.perf_counter() - t_total:.2f}s")
    return array_to_catalog(cat, columns=columns, mpicomm=mpicomm, add_weight=add_weight)
