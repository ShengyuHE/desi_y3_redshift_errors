#!/usr/bin/env python

import os
import sys
import fitsio
import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_OVERALL, COLOR_OVERALL
from helper import REDSHIFT_VSMEAR, REDSHIFT_LSS_VSMEAR, REDSHIFT_CUBICBOX, COLOR_TRACERS, GET_RECON_BIAS

# save the figure to the overleaf or not
c = 299792.458
overwrite = True
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'

def generate_dv(repeat_dir, tracer, zmin, zmax):
    '''
    Generate the dv distribution from repeats
    '''
    d = Table.read(f'{repeat_dir}/{tracer}repeats.fits', hdu=1)
    sel      = np.full(len(d),True)
    sel = np.isfinite(d['Z1']) & np.isfinite(d['Z2'])
    selz = ((zmin<d["Z1"])&(d["Z1"]<zmax))
    # d_zbin = d[sel]
    d_zbin = d[sel & selz]
    dv_zbin = (d_zbin['Z2']-d_zbin['Z1'])/(1+d_zbin['Z1'])*c
    return dv_zbin

def suggest_vbin(dv, vmode='linear', bw_method='scott', points_per_sigma=3):
    """
    Suggest an optimal linear dv bin width for evaluating the KDE model.

    The recommended bin width is computed as:
        vbin ≈ σ_KDE / points_per_sigma,
    where σ_KDE = kde.factor * std(v) (or |v| if use_abs=True).
    """
    v = np.asarray(dv, float)
    v = v[np.isfinite(v)]
    if 'log' in vmode:
        y = np.log10(np.abs(v))
    elif 'linear' in vmode:
        y = v
    kde = gaussian_kde(y, bw_method=bw_method)
    bw  = kde.factor * y.std()  # σ_KDE in linear dv space
    vbin = bw / points_per_sigma
    return vbin, bw

def get_kcdf(dv, vmode='linear', vbin=None,  bw='scott', extend_sigma=0.5, nmax=None):
    """
    Kernel-smoothed CDF (KCDF) for dv.

    Parameters
    ----------
    dv : array-like
        Input dv values (can be signed).
    vmode: 
        Controls how the Δv distribution is modeled.
    vbin : float
        Step size in log scale
    bw : str or float
        Bandwidth method for Gaussian KDE ('scott', 'silverman', or scalar factor).
    extend_sigma : float
        Range padding in σ for the linear grid; in log case, padding in log units.
    nmax : int or None
        for the linear case; in log case, padding in log units.

    Returns
    -------
    grid : ndarray
        Grid points (dv for linear case, log10(|dv|) for log case).
    pdf  : ndarray
        KCDF PDF values on the grid.
    cdf  : ndarray
        KCDF values on the grid (monotonic from ~0 to ~1).
    F    : callable
        Interpolator F(x): returns CDF at arbitrary x (same domain as grid).
    """
    v = np.asarray(dv, float)
    v = v[np.isfinite(v)]
    # Optional subsampling for speed
    if nmax is not None and len(v) > nmax :
        v = np.random.choice(v, size=nmax, replace=False)
    if vbin == None:
        vbin, _ = suggest_vbin(v, vmode=vmode, bw_method=bw, points_per_sigma=3)
    if 'log' in vmode:
        y = np.log10(abs(v))
    elif 'linear' in vmode:
        y = v
    ymu, ysig = y.mean(), y.std()
    vmin = y.min() - extend_sigma*ysig
    vmax = y.max() + extend_sigma*ysig
    edges = np.arange(vmin, vmax + vbin, vbin)
    # caluculate the CDF
    grid = 0.5 * (edges[1:] + edges[:-1])
    kde   = gaussian_kde(y, bw_method=bw)
    pdf   = kde(grid)
    # Normalize and integrate
    w = np.diff(edges)
    pdf /= np.sum(pdf * w)
    cdf = np.cumsum(pdf * w)
    # handy interpolator for F_y(y0)
    F = lambda y: np.clip(np.interp(y, grid, cdf, left=0, right=1), 0, 1)
    return grid, pdf, cdf, vbin, F


def save_KCDF(tracer, zmin, zmax, vmode):
    '''
    Save the Kernel-smoothed CDF (KCDF) for dv.

    Parameters
    ----------
    tracer : str
        Tracer name (e.g. 'QSO', 'LRG', 'ELG'). The first 3 letters select the tracer type.
    zmin, zmax : float
        Redshift interval of the Δv sample.
    vmode : str
        Controls how the Δv distribution is modeled. Supported options:

            - "log-signed":
                Model positive and negative Δv separately in log10|Δv| space.
                We split the data into
                    Δv_pos = {Δv > 0}
                    Δv_neg = {Δv < 0}
                and fit two independent KCDFs:
                    KCDF_pos  for log10(Δv_pos)
                    KCDF_neg  for log10(|Δv_neg|)
                This preserves any asymmetry between positive and negative tails.

            - "log-abs":
                Model only the absolute Δv distribution using
                    y = log10(|Δv|)
                A single KCDF is saved. Useful when only the magnitude of Δv
                matters (e.g., modeling catastrophic contamination rates or
                RMS-like statistics).
    '''
    dv = generate_dv(REPEAT_DIR, tracer[:3], zmin, zmax)
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    if 'log' in vmode:
        dv = dv[dv!=0]
        # log_vbin = 0.02 # set the bins
        if 'signed' in vmode: # model the negative and positive part separately
            cdf_p_fn = REPEAT_DIR+f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_+.npz'
            cdf_n_fn = REPEAT_DIR+f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_-.npz'
            if not os.path.exists(cdf_p_fn) or overwrite == True:
                dv_pos = dv[dv > 0]
                grid_p, pdf_p, cdf_p, log_vbin, F_p = get_kcdf(dv_pos, vmode)
                np.savez(cdf_p_fn, grid=grid_p, pdf=pdf_p, cdf=cdf_p, log_vbin=log_vbin)
                print('[SAVE KCDF]', cdf_p_fn, flush=True)
            if not os.path.exists(cdf_n_fn) or overwrite == True:
                dv_neg = dv[dv < 0]
                grid_n, pdf_n, cdf_n, log_vbin, F_n = get_kcdf(-dv_neg, vmode)
                np.savez(cdf_n_fn, grid=grid_n, pdf=pdf_n, cdf=cdf_n, log_vbin=log_vbin)
                print('[SAVE KCDF]', cdf_n_fn, flush=True)
        if 'abs' in vmode: # model the absolute dv
            cdf_fn = REPEAT_DIR+f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
            if not os.path.exists(cdf_fn) or overwrite == True:
                dv = abs(dv)
                grid, pdf, cdf, log_vbin, F = get_kcdf(dv, vmode)
                np.savez(cdf_fn, grid=grid, pdf=pdf, cdf=cdf, log_vbin=log_vbin)
                print('[SAVE KCDF]', cdf_fn, flush=True)
    elif 'linear' in vmode:
        cdf_fn = REPEAT_DIR+f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
        grid, pdf, cdf, vbin, F = get_kcdf(dv, vmode)
        np.savez(cdf_fn, grid=grid, pdf=pdf, cdf=cdf, vbin=vbin)
        print('[SAVE KCDF]', cdf_fn, flush=True)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--vmode", help="dv modeling mode", choices=['log_signed', 'log_abs', 'linear'], default='linear')
    parser.add_argument("--outputdir", help="output directory for results", default= '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1' )
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--ztype", help="z bins type", type = str, choices=['LSS','bin'], default='LSS')

    args = parser.parse_args()
    vmode = args.vmode

    for tracer in args.tracers:
        zmin, zmax = REDSHIFT_OVERALL[tracer[:3]]
        if args.ztype == 'LSS':
            print(f'Calculate KCDF for repeats {tracer} z{zmin}-{zmax}',  flush=True)
            save_KCDF(tracer, zmin, zmax, vmode)
        elif args.ztype == 'bin':
            step = 0.1
            zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
            zbins = list(zip(zrange[:-1], zrange[1:]))
            for indz, (z1, z2) in enumerate(zbins):
                print(f'Calculate KCDF for repeats {tracer} z{z1}-{z2}', flush=True)
                save_KCDF(tracer, z1, z2, vmode)
