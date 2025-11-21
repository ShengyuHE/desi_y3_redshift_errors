#!/usr/bin/env python

import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit
from scipy.interpolate import interp1d
from mockfactory import utils, DistanceToRedshift, Catalog, RandomBoxCatalog

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_ABACUSHF_v1, REDSHIFT_BIN_LSS, CSPEED
from helper import GET_REPEATS_DV, GET_REPEATS_NUMBER, GET_CTHR

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

GLOBAL_SEED = 123
BOXSIZE = 2000
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'
TRACER_CUTSKY_INFO = {
    'LRG': {'tracer_type': 'LRG', 'fit_range': '0p4to1p1'},
    'ELG': {'tracer_type': 'ELG_LOP','fit_range': '0p8to1p6'},
    'QSO': {'tracer_type': 'QSO','fit_range': '0p8to3p5'},
}

def sample_from_cdf(cdf_fn, Ngal, vmode, seed=1234):
    """
    Sample Δv from a stored |Δv| CDF.

    Parameters
    ----------
    cdf_fn : str
        Path to the CDF file (npz) containing arrays {grid, cdf}.
    Ngal : int
        Number of Δv samples to generate.
    vmode : {"log_abs", "log_signed", "linear"}
        Sampling mode
    seed : int, optional
        Random seed.

    Returns
    -------
    dv : ndarray
        Sampled Δv array of length Ngal.
    inv_cdf : function
        Inverse-CDF interpolator used for sampling.
    """
    np.random.seed(seed)
    data = np.load(cdf_fn, allow_pickle=True)
    grid = data["grid"]
    cdf  = data["cdf"]
    cdf_unique, ind = np.unique(cdf, return_index=True)
    grid_unique = grid[ind]
    inv_cdf = interp1d(
        cdf_unique / cdf_unique[-1],
        grid_unique,
        bounds_error=False,
        fill_value=(grid_unique[0], grid_unique[-1]),
        kind='linear'
    )
    if 'log' in vmode:
        if 'abs' in vmode:
            u = np.random.uniform(0, 1, int(Ngal / 2))
            y = inv_cdf(u)
            dv = np.append(10**y, -10**y)
            if Ngal % 2 == 1:
                dv = np.append([0.0], dv)
            np.random.shuffle(dv)
        elif 'signed' in vmode:
            u = np.random.uniform(0, 1, int(Ngal))
            dv = 10**inv_cdf(u)
    elif 'linear' in vmode:
        u = np.random.uniform(0, 1, int(Ngal))
        dv = inv_cdf(u)
    return dv, inv_cdf

def model_dv_from_cdf(tracer, z1, z2, N, cdf_kind = 'obsCDF', vmode = 'log_signed', seed=GLOBAL_SEED):
    """
    Generate model Δv samples for a given tracer and redshift bin.

    Parameters
    ----------
    tracer : str
        Tracer name (e.g., "LRG", "ELG", "QSO").
    z1, z2 : float
        Lower and upper redshift bounds of the bin.
    N : int
        Number of Δv values to sample.
    cdf_kind : {"KCDF", "obsCDF"}
        Type of CDF used for sampling.
    vmode : {"log_signed", "log_abs", "linear"}
        Modeling mode:
        - "log_abs"    : sample |Δv| from log-CDF.
        - "log_signed" : sample positive/negative Δv separately using observed N_p/N_n fractions.
        - "linear"     : sample Δv directly.
    seed : int
        Random seed.

    Returns
    -------
    dv_model : ndarray
        Model Δv samples of length N.
    """
    if vmode == "log_abs":
        fn = f"{REPEAT_DIR}/vmode/{cdf_kind}_{tracer}_z{z1:.1f}-{z2:.1f}_{vmode}.npz"
        dv_model, _ = sample_from_cdf(fn, N, vmode, seed)
        return np.asarray(dv_model, float)
    elif vmode == "log_signed":
        (_N, _p, _n) = GET_REPEATS_NUMBER(tracer, z1, z2)
        N_p = int(N*float(_p/_N))
        N_n = N-N_p
        dv_model_list = []
        for sign, Num in [('+', N_p), ('-', N_n)]:
            fn = f"{REPEAT_DIR}/vmode/{cdf_kind}_{tracer}_z{z1:.1f}-{z2:.1f}_{vmode}_{sign}.npz"
            sample, _ = sample_from_cdf(fn, Num, vmode, seed)
            sample = np.asarray(sample, float)
            dv_model_list.append(sample if sign=='+' else -sample)
        dv_model = np.concatenate(dv_model_list)
        np.random.shuffle(dv_model)
        return np.asarray(dv_model, float)
    elif vmode == "linear":
        fn = f"{REPEAT_DIR}/vmode/{cdf_kind}_{tracer}_z{z1:.1f}-{z2:.1f}_{vmode}.npz"
        dv_model, _ = sample_from_cdf(fn, N, vmode, seed)
        return np.asarray(dv_model, float)
    else:
        raise ValueError(f"Unknown mode: {vmode}")

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
                    data_fn = base_dir+ f'/Cutsky/{tracer[:3]}/z{z:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering/'+cutsky_name
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
