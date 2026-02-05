#!/usr/bin/env python

import os
import sys
import logging
import argparse
import fitsio
import itertools
import numpy as np
from mockfactory import Catalog
from cosmoprimo.fiducial import DESI, AbacusSummit

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_ABACUSHF, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO
from utils import setup_logging
from dv_tools import get_repeats_dv, get_cthr, model_dv_from_cdf
from cat_tools import get_catalog_fn
setup_logging()
logger = logging.getLogger('build_catalogue')

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

BOXSIZE = 2000

####################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--version",type = str,  default='AbacusHF-v2', help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2'])
    parser.add_argument("--domains", nargs = '+', type = str, default=['cubic'], choices=['cubic', 'cutsky', 'cutsky_QSO'], help="mock domain: cubic box or cut-sky survey footprint")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--zerrs",  nargs = '+',  type = str, default= ['verr_empirical'], choices=['None', 'repeat', 'verr_empirical'], help="dv profiles consider" )
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    # parser.add_argument("--outputdir",  default= '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1' , help="output directory for results")    
    args = parser.parse_args()

    logger.info(f"Received arguments: {args}")

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))

    cosmo = DESI() # c000 cosmology
    z_snaps, z_ranges = REDSHIFT_ABACUSHF[args.version]
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(z_snaps[tracer], z_ranges[tracer]):
            tracer_redshifts.append((tracer, zp, zr))

    for domain, (tracer, zsnap, zrange), mock_id in itertools.product(args.domains, tracer_redshifts, mockids):
        Hz = cosmo.H0 * cosmo.efunc(zsnap)/cosmo.h # in km/s/(Mpc/h)
        fac = (1+zsnap)/Hz
        data_args = {'version':args.version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id}
        if domain == 'cubic':
            cat_fn = get_catalog_fn(**data_args)
            logger.info(f"Load {cat_fn}")
            data = Catalog.read(cat_fn)
            # Build redshift-space coordinates from positions + peculiar velocities
            if 'Z_RSD' not in data.columns():
                for pos_RSD, pos, vel in zip(['X_RSD', 'Y_RSD', 'Z_RSD'],
                                            ['X',      'Y',      'Z'     ],
                                            ['VX',     'VY',     'VZ'    ]):
                    if pos_RSD not in data.columns():
                        data[pos_RSD] = (data[pos] + data[vel]*fac)%BOXSIZE
                data.write(cat_fn)
            for arg in ['Z_DV_REP', 'VZ_DV_REP', 'Z_DV_ERR', 'VZ_DV_ERR', 'Z_DV_ERR_V1', 'VZ_DV_ERR_V1']:
                if arg in data.columns(): del data[arg]
            for dv_mode in args.zerrs:
                if dv_mode == 'repeat': 
                    dv_label = '_REP'
                    cdf_mode = 'HCDF'
                elif dv_mode == 'verr_empirical': 
                    dv_label = '_ERR_V1'
                    cdf_mode = 'CDF'
                else:
                    ValueError(f"not valid {dv_mode}")
                if f'Z{dv_label}' not in data.columns():
                    (zmin, zmax) = (zrange[0], zrange[1])
                    ##### assume Z-direction is the LOS #####
                    dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), dv_mode = dv_mode, cdf_mode = cdf_mode)
                    data[f'VZ{dv_label}'] = data['VZ'] + dv
                    data[f'Z{dv_label}']=(data['Z_RSD'] + dv*fac)%BOXSIZE
                    data.write(cat_fn)
        elif domain == 'cutsky':
            cat_fn = get_catalog_fn(**data_args)
            logger.info(f"[LOAD] {cat_fn}")
            data = Catalog.read(cat_fn)
            (zmin, zmax) = z_ranges[tracer]
            for dv_mode in args.zerrs:
                if dv_mode == 'repeat': 
                    dv_label = '_REP'
                    cdf_mode = 'HCDF'
                elif dv_mode == 'verr_empirical': 
                    dv_label = '_ERR_V1'
                    cdf_mode = 'CDF'
                else:
                    ValueError(f"not valid {dv_mode}")
            if f'Z{dv_label}' not in data.columns():
                ##### add redshift errors on Z #####
                dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), cdf_mode=cdf_mode, vmode=args.vmode,)
                data[f'Z{dv_label}'] = data['Z']+dv/CSPEED*(1+data['Z'])
                data.write(cat_fn)
                ##### add redshift errors on Z with (0.1) bin#####
                step = 0.1
                zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
                zbins = list(zip(zrange[:-1], zrange[1:]))
                for indz, (z1, z2) in enumerate(zbins):
                    sel = (data['Z'] >= z1) & (data['Z'] < z2)
                    if not np.any(sel): continue
                    z_sel = data['Z'][sel]
                    dv_bin = model_dv_from_cdf(tracer, z1, z2, len(z_sel), cdf_mode=cdf_mode, vmode=args.vmode,)
                    data[f'Z{dv_label}_BIN'][sel] = z_sel + dv_bin / CSPEED * (1.0 + z_sel)
                data.write(cat_fn)
