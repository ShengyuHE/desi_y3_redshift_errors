#!/usr/bin/env python

import os
import sys
import fitsio
import logging
import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

logger = logging.getLogger('repeats_variance') 

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_OVERALL, REDSHIFT_LSS
from helper import GET_REPEATS_DV, GET_CTHR

# save the figure to the overleaf or not
c = 299792.458
overwrite = False
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'

def suggest_vbin(dv, vmode='log_abs', bw_method='scott', points_per_sigma=5):
    """
    Suggest an optimal linear dv bin width for evaluating the KDE model.

    The recommended bin width is computed as:
        vbin ≈ σ_KDE / points_per_sigma,
    where σ_KDE = kde.factor * std(v) (or |v| if use_abs=True).
    """
    v = np.asarray(dv, float)
    v = v[np.isfinite(v)]
    if 'log' in vmode:
        y = np.log10(abs(v))
    elif 'linear' in vmode:
        y = v
    kde = gaussian_kde(y, bw_method=bw_method)
    bw  = kde.factor * y.std()  # σ_KDE in linear dv space
    vbin = bw / points_per_sigma
    return vbin, bw

def get_kcdf(dv, vmode='log_abs', vbin=None,  bw='scott', extend_sigma=0.5, nmax=None):
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

def save_CDF(tracer, zmin, zmax, vmode, kind='both', vbin_fine=0.005):
    """
    Save KCDF and/or observed CDF (obsCDF) for Δv in a single function.

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
                We split the data into Δv_pos = {Δv > 0} and Δv_neg = {Δv < 0} and fit two independent CDFs:
            - "log-abs":
                Model only the absolute Δv distribution using y = log10(|Δv|)
            - "linear-...":
                Linear-space CDF (obsCDF not implemented for linear in original code).
    kind : {"kcdf", "obs", "both"}, optional
        Which CDF(s) to save:
            - "kcdf" : only KCDF
            - "obs"  : only observed CDF (histogram-based)
            - "both" : save both KCDF and obsCDF
    vbin_fine : float, optional
        Bin width for obsCDF in log-space (same role as in save_obsCDF).
    """

    # Load and clean dv once
    dv_raw = GET_REPEATS_DV(tracer[:3], zmin, zmax)

    # For obsCDF log-binning
    catasmin, catasmax = -3, 6  # same as before

    # Convenience flags
    do_kcdf = kind in ('kcdf', 'both')
    do_obs  = kind in ('obs', 'both')

    # ---------------- LOG MODES ----------------
    if 'log' in vmode:
        dv_clean = dv_raw[dv_raw != 0] # Exclude exact zeros (cannot take log10)
        # ---------------- KCDF PART ----------------
        if do_kcdf:
            if 'signed' in vmode:  # positive and negative separately
                cdf_p_fn = REPEAT_DIR + f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_+.npz'
                cdf_n_fn = REPEAT_DIR + f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_-.npz'
                # Positive tail
                if not os.path.exists(cdf_p_fn) or overwrite:
                    dv_pos = dv_clean[dv_clean > 0]
                    grid_p, pdf_p, cdf_p, log_vbin, F_p = get_kcdf(dv_pos, vmode)
                    np.savez(cdf_p_fn, grid=grid_p, pdf=pdf_p, cdf=cdf_p, log_vbin=log_vbin)
                    logger.info('Save KCDF', cdf_p_fn)
                # Negative tail (use |Δv| for kernel input)
                if not os.path.exists(cdf_n_fn) or overwrite:
                    dv_neg = dv_clean[dv_clean < 0]
                    grid_n, pdf_n, cdf_n, log_vbin, F_n = get_kcdf(-dv_neg, vmode)
                    np.savez(cdf_n_fn, grid=grid_n, pdf=pdf_n, cdf=cdf_n, log_vbin=log_vbin)
                    logger.info('Save KCDF', cdf_n_fn)
            if 'abs' in vmode:  # absolute Δv
                cdf_fn = REPEAT_DIR + f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
                if not os.path.exists(cdf_fn) or overwrite:
                    dv_abs = abs(dv_clean)
                    grid, pdf, cdf, log_vbin, F = get_kcdf(dv_abs, vmode)
                    np.savez(cdf_fn, grid=grid, pdf=pdf, cdf=cdf, log_vbin=log_vbin)
                    logger.info('Save KCDF', cdf_fn)

        # ---------------- obsCDF PART ----------------
        if do_obs:
            if 'signed' in vmode:
                cdf_p_fn = REPEAT_DIR + f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_+.npz'
                cdf_n_fn = REPEAT_DIR + f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_-.npz'
                # Positive tail
                if not os.path.exists(cdf_p_fn) or overwrite:
                    dv_pos = dv_clean[dv_clean > 0]
                    pdf_p, bins_p = np.histogram(np.log10(dv_pos), bins=np.arange(catasmin, catasmax, vbin_fine), density=True )
                    cdf_p = np.cumsum(pdf_p) * vbin_fine
                    grid_p = 0.5 * (bins_p[1:] + bins_p[:-1])
                    np.savez(cdf_p_fn, grid=grid_p, pdf=pdf_p, cdf=cdf_p, vbin=vbin_fine)
                    logger.info('Save obsCDF', cdf_p_fn)
                # Negative tail
                if not os.path.exists(cdf_n_fn) or overwrite:
                    dv_neg = dv_clean[dv_clean < 0]
                    pdf_n, bins_n = np.histogram(np.log10(-dv_neg), bins=np.arange(catasmin, catasmax, vbin_fine), density=True)
                    cdf_n = np.cumsum(pdf_n) * vbin_fine
                    grid_n = 0.5 * (bins_n[1:] + bins_n[:-1])
                    np.savez(cdf_n_fn, grid=grid_n, pdf=pdf_n, cdf=cdf_n, vbin=vbin_fine)
                    logger.info('Save obsCDF', cdf_n_fn)
            if 'abs' in vmode:
                cdf_fn = REPEAT_DIR + f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
                if not os.path.exists(cdf_fn) or overwrite:
                    dv_abs = abs(dv_clean)
                    pdf_fine, bins_fine = np.histogram(np.log10(dv_abs), bins=np.arange(catasmin, catasmax, vbin_fine), density=True)
                    cdf_data = np.cumsum(pdf_fine) * vbin_fine
                    grid = 0.5 * (bins_fine[1:] + bins_fine[:-1])
                    np.savez(cdf_fn, grid=grid, pdf=pdf_fine, cdf=cdf_data, vbin=vbin_fine)
                    logger.info('Save obsCDF', cdf_fn)
    # --- LINEAR MODES (KCDF ONLY, as in original code) ----------------------
    elif 'linear' in vmode:
        if do_kcdf:
            cdf_fn = REPEAT_DIR + f'/vmode/KCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
            if not os.path.exists(cdf_fn) or overwrite:
                grid, pdf, cdf, vbin, F = get_kcdf(dv_raw, vmode)
                np.savez(cdf_fn, grid=grid, pdf=pdf, cdf=cdf, vbin=vbin)
                logger.info('Save obsCDF', cdf_fn)
        # obsCDF for linear vmode was not defined
    return 0


###################################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--ztype", help="z bins type", type = str, choices=['all','bin','LSS'], default='LSS')
    parser.add_argument("--vmode", help="dv bin mode", choices=['log_signed', 'log_abs', 'linear'], default='log_abs')
    parser.add_argument("--cdfmode", help="CDF modeling mode", choices=['obs', 'kcdf', 'both'], default='both')
    parser.add_argument("--outputdir", help="output directory for results", default= '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1' )
    args = parser.parse_args()
    for tracer in args.tracers:
        zmin, zmax = REDSHIFT_OVERALL[tracer[:3]]
        if args.ztype == 'all':
            logger.info(f'Calculate CDF for repeats {tracer} z{zmin}-{zmax}')
            save_CDF(tracer, zmin, zmax, args.vmode, args.cdfmode)
        elif args.ztype == 'LSS':
            for indz, (z1, z2) in enumerate(REDSHIFT_LSS[tracer[:3]]): 
                logger.info(f'Calculate CDF for repeats {tracer} z{z1}-{z2}')
                save_CDF(tracer, z1, z2, args.vmode, kind=args.cdfmode)
        elif args.ztype == 'bin':
            step = 0.1
            zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
            zbins = list(zip(zrange[:-1], zrange[1:]))
            for indz, (z1, z2) in enumerate(zbins):
                logger.info(f'Calculate CDF for repeats {tracer} z{z1}-{z2}')
                save_CDF(tracer, z1, z2, args.vmode, kind=args.cdfmode)

""""
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

def save_obsCDF(tracer, zmin, zmax, vmode, vbin_fine=0.005):
    '''
    Save the observed CDF (obsCDF) for dv.

    Parameters
    ----------
    tracer : str
        Tracer name (e.g. 'QSO', 'LRG', 'ELG'). The first 3 letters select the tracer type.
    zmin, zmax : float
        Redshift interval of the Δv sample.
    vmode : str
        Controls how the Δv distribution is modeled. 
    '''
    dv = generate_dv(REPEAT_DIR, tracer[:3], zmin, zmax)
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    if 'log' in vmode:
        catasmin, catasmax, catasbin = -3, 6, 0.2
        if 'signed' in vmode: # model the negative and positive part separately
            cdf_p_fn = REPEAT_DIR+f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_+.npz'
            cdf_n_fn = REPEAT_DIR+f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}_-.npz'
            if not os.path.exists(cdf_p_fn) or overwrite == True:
                dv_pos = dv[dv > 0]
                x_pos = np.log10(dv_pos)
                pdf_p, bins_p=np.histogram(x_pos, bins=np.arange(catasmin,catasmax,vbin_fine),density=True)
                cdf_p  = np.cumsum(pdf_p) * vbin_fine 
                np.savez(cdf_p_fn, grid=(bins_p[1:]+bins_p[:-1])/2, pdf=pdf_p, cdf=cdf_p, vbin=vbin_fine)
                print('[SAVE obsCDF]', cdf_p_fn, flush=True)
            if not os.path.exists(cdf_n_fn) or overwrite == True:
                dv_neg = dv[dv < 0]
                x_pos = np.log10(-dv_neg)
                pdf_n, bins_n=np.histogram(x_pos, bins=np.arange(catasmin,catasmax,vbin_fine),density=True)
                cdf_n  = np.cumsum(pdf_n) * vbin_fine 
                np.savez(cdf_n_fn, grid=(bins_n[1:]+bins_n[:-1])/2, pdf=pdf_n, cdf=cdf_n, vbin=vbin_fine)
                print('[SAVE obsCDF]', cdf_n_fn, flush=True)
        if 'abs' in vmode: # model the absolute dv
            cdf_fn = REPEAT_DIR+f'/vmode/obsCDF_{tracer}_z{zmin:.1f}-{zmax:.1f}_{vmode}.npz'
            if not os.path.exists(cdf_fn) or overwrite == True:
                dv = abs(dv)
                pdf_fine,bins_fine=np.histogram(dv, bins=np.arange(catasmin,catasmax,vbin_fine),density=True)
                cdf_data     = np.cumsum(pdf_fine) * vbin_fine 
                np.savez(cdf_fn, grid=(bins_fine[1:]+bins_fine[:-1])/2, pdf=pdf_fine, cdf=cdf_data, vbin=vbin_fine)
                print('[SAVE obsCDF]', cdf_fn, flush=True)
    return 0


"""