#!/usr/bin/env python

import os
import sys
import logging
import argparse
import fitsio
import itertools
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR
from cat_tools import model_dv_from_cdf, get_catalog_fn

logger = logging.getLogger('build_catalogue')

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

BOXSIZE = 2000

####################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--versions", nargs = '+', type = str,  default=['AbacusHF-v1'], help="mock types", choices=['AbacusHF-v1', 'AbacusHF-v2'])
    parser.add_argument("--domains", nargs = '+', type = str, default=['cutsky'], choices=['cubic', 'cutsky', 'cutsky_QSO'], help="mock domain: cubic box or cut-sky survey footprint")
    parser.add_argument("--tracers", nargs = '+', type = str, default=['QSO'], choices=['BGS','LRG','ELG','QSO'], help="tracer type to be selected")
    parser.add_argument("--vmode",  nargs = '+',  type = str, default= ['log_abs'], choices=['log_signed', 'log_abs', 'linear'], help="dv bin mode" )
    parser.add_argument("--cdfmode", nargs = '+',  type = str, default=['obs'], choices=['obs', 'kcdf'], help="CDF modeling mode")
    parser.add_argument("--mockid", type = str, default="0-24", help="Mock ID range or list (0-24)")
    parser.add_argument("--outputdir",  default= '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1' , help="output directory for results")    

    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
    if args.cdfmode == "obs":
        cdf_kind = "obsCDF"
    elif args.cdfmode == "kcdf":
        cdf_kind = "KCDF"

    if args.version == 'AbacusHF_v1':
        base_dir = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1'

    cosmo = DESI()
    tracer_redshifts = []
    for tracer in args.tracers:
        for zp, zr in zip(REDSHIFT_ABACUSHF_v1[tracer], REDSHIFT_BIN_LSS[tracer]):
            tracer_redshifts.append((tracer, zp, zr))
    
    for version, domain, (tracer, zsnap, zrange), mock_id in itertools.product(args.versions, args.domains, tracer_redshifts, mockids):
        Hz = cosmo.H0 * cosmo.efunc(zsnap)/cosmo.h # in km/s/(Mpc/h)
        fac = (1+zsnap)/Hz
        data_args = {'version':version, 'domain':domain, 'tracer':tracer, 'zsnap': zsnap, 'zrange':zrange, 'mock_id': mock_id}
        if domain == 'cubic':
            cat_fn = get_catalog_fn(**data_args)
            data = Table.read(cat_fn)
            logger.info(f"[LOAD] {cat_fn}")
            # Build redshift-space coordinates from positions + peculiar velocities
            if 'Z_RSD' not in data.colnames:
                for pos_RSD, pos, vel in zip(['X_RSD', 'Y_RSD', 'Z_RSD'],
                                            ['X',      'Y',      'Z'     ],
                                            ['VX',     'VY',     'VZ'    ]):
                    if pos_RSD not in data.colnames:
                        data[pos_RSD] = (data[pos] + data[vel]*fac)%BOXSIZE
                data.write(cat_fn, overwrite=True)
            if 'Z_OBS' not in data.colnames:
                (zmin, zmax) = (zrange[0], zrange[1])
                if mock_id == '0':
                    logger.info(f"[BUILD CUBIC] {tracer} snapshot z={zsnap:.3f} dv z={zmin:.1f}-{zmax:.1f}")
                ##### assume Z-direction is the LOS #####
                dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), 
                                    cdf_kind=cdf_kind, vmode=args.vmode,)
                data['VZ_OBS'] = data['VZ'] + dv
                data['Z_OBS']=(data['Z_RSD'] + dv*fac)%BOXSIZE
                data.write(cat_fn, overwrite=True)
        elif domain == 'cutsky':
            cat_fn = get_catalog_fn(**data_args)
            data = Table.read(cat_fn)
            logger.info(f"[LOAD] {cat_fn}")
            ##### add redshift errors on Z #####
            (zmin, zmax) = REDSHIFT_BIN_OVERALL[tracer]
            if 'Z_OBS_GLOBAL' not in data.colnames:
                if mock_id == '0':
                    logger.info(f'Build cutsky for {tracer} snapshort z{z:.3f} using repeats {tracer} dv global z{zmin:.1f}-{zmax:.1f}')
                dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), 
                                    cdf_kind=cdf_kind, vmode=args.vmode,)
                data['Z_OBS_GLOBAL'] = data['Z']+dv/CSPEED*(1+data['Z'])
                data.write(cat_fn, overwrite=True)
            if 'Z_OBS_BIN' not in data.colnames:
                data['Z_OBS_BIN'] = data['Z'].copy()
                step = 0.1
                zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
                zbins = list(zip(zrange[:-1], zrange[1:]))
                for indz, (z1, z2) in enumerate(zbins):
                    if mock_id == '0':
                        logger.info(f'Build cutsky for {tracer} snapshort z{z:.3f} using repeats {tracer} dv zbin{z1:.1f}-{z2:.1f}')
                    sel = (data['Z'] >= z1) & (data['Z'] < z2)
                    if not np.any(sel):
                        continue
                    z_sel = data['Z'][sel]
                    dv_bin = model_dv_from_cdf(tracer, z1, z2, len(z_sel), 
                                        cdf_kind=cdf_kind, vmode=args.vmode,)
                    data['Z_OBS_BIN'][sel] = z_sel + dv_bin / CSPEED * (1.0 + z_sel)
                data.write(cat_fn, overwrite=True)
                