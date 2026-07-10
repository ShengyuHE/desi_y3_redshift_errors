#!/usr/bin/env python
import os, sys
import argparse
import logging
import itertools
import time
from pathlib import Path
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
from mock_tools import read_ascii_mock_catalog
from cat_tools import get_catalog_fn
from utils import setup_logging
setup_logging()

logger = logging.getLogger('QSO_lightcone')
MOCK_DIR = Path("/pscratch/sd/x/xryang/QSO_cutsky/")
OUTPUT_DIR = Path("/global/cfs/cdirs/desi/users/shengyu/galaxies/catalogs/Y3")

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

valid_version_domain = {
    "AbacusHF-4snap": ['lightcone'],
    "AbacusHF-v1": ['cubic', 'cutsky'],
    "AbacusHF-v2": ['cubic', 'cutsky'],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='AbacusHF-4snap', help="Version name for the output files.")
    parser.add_argument("--domain", type=str, default='cutsky', help="Domain: cubic, cutsky, lightcone, altmtl.")
    parser.add_argument("--tracers", nargs = '+', default=['QSO'], choices=['QSO'], help="Tracer type for the output files.")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--hod", type=str, default='base', help="HOD variant for AbacusHF")
    parser.add_argument("--input_dir", type=Path, default=MOCK_DIR, help="Directory to the default mocks.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save the FITS catalogs.")
    parser.add_argument("--max_rows", type=int, default=None, help="Read only this many rows for smoke tests.")
    parser.add_argument("--do_random", action="store_true", help="Whether to also convert the random catalog.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()
    mpicomm = MPI.COMM_WORLD
    mpiroot = 0
    if mpicomm.rank == mpiroot: logger.info(f"Received arguments: {args}")

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    version = args.version
    domain = args.domain
    mock_dir = args.input_dir or MOCK_DIR

    if domain not in valid_version_domain.get(version, []):
        if mpicomm.rank == mpiroot:
            logger.error(f"Invalid version {version} for domain {domain}.")
        sys.exit(1)

    for tracer, mock_id in itertools.product(args.tracers, mockids):
        data_args = dict(version=version, domain=domain, tracer=tracer, zsnap=1.4, zrange=(0.8, 2.1), mock_id=mock_id, hod=args.hod)
        if not args.do_random:
            mock_id03 = f"{mock_id:03}"
            if domain == 'lightcone':
                if args.hod == 'base':
                    data_fn = mock_dir / f"ph{mock_id03}" / 'merged_cutsky.dat'
            elif domain == 'cutsky':
                if args.hod == 'base':
                    data_fn = mock_dir / f"ph{mock_id03}" / f'cutsky_{zfmt(data_args["zsnap"])}.dat'
                elif args.hod == 'base_dv':
                    data_fn = mock_dir / f"ph{mock_id03}" / f'cutsky_{zfmt(data_args["zsnap"])}_dv.dat'
            output_fn = get_catalog_fn(**data_args)
            if os.path.exists(output_fn) and not args.overwrite:
                if mpicomm.rank == mpiroot: 
                    logger.info(f"{output_fn} already exists. Skipping.")
                continue
            cat = read_ascii_mock_catalog(data_fn, select_status=True, hod=args.hod, max_rows=args.max_rows, mpicomm=mpicomm,)
            mpicomm.Barrier()
            cat.write(output_fn)
        elif args.do_random:
            if domain == 'lightcone':
                random_fn = mock_dir  / 'merged_random.dat'
            elif domain == 'cutsky':
                random_fn = mock_dir / f'random_{zfmt(data_args["zsnap"])}.dat'
            output_fn = get_catalog_fn(**data_args, random=True)
            if os.path.exists(output_fn) and not args.overwrite:
                if mpicomm.rank == mpiroot: 
                    logger.info(f"{output_fn} already exists. Skipping.")
                continue
            cat = read_ascii_mock_catalog(random_fn, select_status=True, max_rows=args.max_rows, mpicomm=mpicomm,)
            mpicomm.Barrier()
            cat.write(output_fn)

        # random_fn = mock_dir / 'merged_random.dat'