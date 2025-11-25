#!/usr/bin/env python

import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS, CSPEED, TRACER_CUTSKY_INFO
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR
from cat_tools import model_dv_from_cdf

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

BOXSIZE = 2000

####################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--version", help="mock types", choices=['AbacusHF_v1'], default='AbacusHF_v1')
    parser.add_argument("--domain", help="mock domain: cubic box or cut-sky survey footprint", choices=['cubic', 'cutsky'], default='cutsky')
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    parser.add_argument("--vmode", help="dv bin mode", choices=['log_signed', 'log_abs', 'linear'], default='log_abs')
    parser.add_argument("--cdfmode", help="CDF modeling mode", choices=['obs', 'kcdf'], default='obs')
    parser.add_argument("--outputdir", help="output directory for results", default= '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1' )    
    args = parser.parse_args()
    print(f"Received arguments: {args}", flush=True)

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
    for tracer in args.tracers:
        for zind, z in enumerate(REDSHIFT_ABACUSHF_v1[tracer]):
            Hz = cosmo.H0 * cosmo.efunc(z)/cosmo.h # in km/s/(Mpc/h)
            fac = (1+z)/Hz
            for mock_id in mockids:
                mock_id03 =  f"{mock_id:03}"
                if args.domain == 'cubic':
                    cubic_name = f'/abacus_HF_{tracer}_{zfmt(z)}_DR2_v1.0_AbacusSummit_base_c000_ph{mock_id03}_clustering.dat.fits'
                    data_fn = base_dir+ f'/Boxes/{tracer}/z{z:.3f}/AbacusSummit_base_c000_ph{mock_id03}'+cubic_name
                    data = Table.read(data_fn)
                    # Build redshift-space coordinates from positions + peculiar velocities
                    if 'Z_RSD' not in data.colnames:
                        for pos_RSD, pos, vel in zip(['X_RSD', 'Y_RSD', 'Z_RSD'],
                                                    ['X',      'Y',      'Z'     ],
                                                    ['VX',     'VY',     'VZ'    ]):
                            if pos_RSD not in data.colnames:
                                data[pos_RSD] = (data[pos] + data[vel]*fac)%BOXSIZE
                        data.write(data_fn, overwrite=True)
                    if 'Z_OBS' not in data.colnames:
                        (zmin, zmax) = REDSHIFT_BIN_LSS[tracer][zind]
                        if mock_id03 == '000':
                            print(f'[BUILD CUBIC CAT:] {tracer} snapshort z{z:.3f} using repeats {tracer} dv z{zmin:.1f}-{zmax:.1f}', flush=True)
                        ##### assume Z-direction is the LOS #####
                        dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), 
                                            cdf_kind=cdf_kind, vmode=args.vmode,)
                        data['VZ_OBS'] = data['VZ'] + dv
                        data['Z_OBS']=(data['Z_RSD'] + dv*fac)%BOXSIZE
                        data.write(data_fn, overwrite=True)
                elif args.domain == 'cutsky':
                    tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
                    fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
                    cutsky_name = f'cutsky_abacusHF_DR2_{tracer_type}_z{zfmt(z)}_zcut_{fit_range}_clustering.dat.fits'
                    data_fn = base_dir+ f'/Cutsky/{tracer_type[:3]}/z{z:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering/'+cutsky_name
                    data = Table.read(data_fn)
                    ##### add redshift errors on Z #####
                    (zmin, zmax) = REDSHIFT_BIN_OVERALL[tracer]
                    if 'Z_OBS_GLOBAL' not in data.colnames:
                        if mock_id03 == '000':
                            print(f'[BUILD CUTSKY CAT:] {tracer} snapshort z{z:.3f} using repeats {tracer} dv global z{zmin:.1f}-{zmax:.1f}', flush=True)
                        dv = model_dv_from_cdf(tracer, zmin, zmax, len(data), 
                                            cdf_kind=cdf_kind, vmode=args.vmode,)
                        data['Z_OBS_GLOBAL'] = data['Z']+dv/CSPEED*(1+data['Z'])
                        data.write(data_fn, overwrite=True)
                    if 'Z_OBS_BIN' not in data.colnames:
                        data['Z_OBS_BIN'] = data['Z'].copy()
                        step = 0.1
                        zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
                        zbins = list(zip(zrange[:-1], zrange[1:]))
                        for indz, (z1, z2) in enumerate(zbins):
                            if mock_id03 == '000':
                                print(f'[BUILD CUTSKY CAT:] {tracer} snapshort z{z:.3f} using repeats {tracer} dv zbin{z1:.1f}-{z2:.1f}', flush=True)
                            sel = (data['Z'] >= z1) & (data['Z'] < z2)
                            if not np.any(sel):
                                continue
                            z_sel = data['Z'][sel]
                            dv_bin = model_dv_from_cdf(tracer, z1, z2, len(z_sel), 
                                                cdf_kind=cdf_kind, vmode=args.vmode,)
                            data['Z_OBS_BIN'][sel] = z_sel + dv_bin / CSPEED * (1.0 + z_sel)
                        data.write(data_fn, overwrite=True)