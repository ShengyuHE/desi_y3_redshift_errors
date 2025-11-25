
#!/usr/bin/env python
### compute and save the variance of the repeat quantities: median, RMS and fc ###

import os, sys
import random
import argparse
import numpy as np
from astropy.table import Table,join,unique,vstack
import random

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_OVERALL
from helper import GET_REPEATS_DV, GET_CTHR

c = 299792.458

def bootstrap_metrics(dv, cthr, B=10000, seed=1234):
    """
    Estimate median(|dv|), RMS(core), and fc using bootstrap resampling.

    Parameters
    ----------
    dv : array Δv values (km/s) for one tracer or z-bin
    cthr : float Catastrophic threshold (e.g., 1000 or 10000 km/s)
    B : int Number of bootstrap resamples

    Returns
    -------
    dict with medians and 16-84% confidence intervals.
    """
    rng = np.random.default_rng(seed)
    dv_abs = np.abs(dv)
    N = len(dv_abs)

    med, rms, fc = np.empty(B), np.empty(B), np.empty(B)

    for b in range(B):
        idx = rng.integers(0, N, N)
        x = dv_abs[idx]
        core = x[x < cthr]
        med[b] = np.median(x)
        rms[b] = np.sqrt(np.mean(core**2)) if core.size else np.nan
        fc[b]  = np.mean(x >= cthr)*100
    out = {
        'median': {'mean': np.nanmean(med), 'std': np.nanstd(med, ddof=1)},
        'rms':    {'mean': np.nanmean(rms), 'std': np.nanstd(rms, ddof=1)},
        'fc':     {'mean': np.nanmedian(fc),  'std': np.nanstd(fc, ddof=1)}
    }
    return out

def loo_medians_fast(a):
    """
    Return the N delete-1 medians of array a in O(N log N).
    """
    a = np.asarray(a)
    N  = a.size
    # Sort once and map each element to its rank
    order = np.argsort(a, kind='mergesort')   # stable; any kind works here
    y     = a[order]                          # sorted values
    rank  = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    if N % 2 == 0:
        # N = 2k → N-1 odd → median is element at index k-1 of y' (y with one removed)
        k = N // 2
        # If you remove something left of k, the (k-1)th of y' is y[k]; else it's y[k-1]
        med_loo = np.where(rank < k, y[k], y[k-1]).astype(float)
    else:
        # N = 2k+1 → N-1 even → median is average of indices (k-1, k) in y'
        k = N // 2
        left  = (y[k-1] + y[k])   * 0.5      # remove right side (r ≥ k+1)
        mid   = (y[k-1] + y[k+1]) * 0.5      # remove one of the middle pair (r == k)
        right = (y[k]   + y[k+1]) * 0.5      # remove left side (r ≤ k-1)
        med_loo = np.empty(N, dtype=float)
        med_loo[rank <= k-1] = right
        med_loo[rank == k]   = mid
        med_loo[rank >= k+1] = left
    return med_loo


def jackknife_metrics(dv, cthr):
    """
    Delete-1 jackknife variance for median(|dv|), RMS(core), and fc.

    Parameters
    ----------
    dv : array Δv values (km/s)
    cthr : float Catastrophic threshold (km/s)

    Returns
    -------
    dict with mean and ±1σ jackknife errors.
    """
    dv_abs = np.abs(dv)
    N = len(dv_abs)
    if N <= 1:
        return {k:(np.nan,(np.nan,np.nan)) for k in ['median','rms','fc']}
    med, rms, fc = np.empty(N), np.empty(N), np.empty(N)
    med_loo = loo_medians_fast(dv_abs) # sort once and no need for loop
    for i in range(N):
        x = np.delete(dv_abs, i)
        core = x[x < cthr]
        # med[i] = np.median(x) # use the function loo_medians_fast without loop
        rms[i] = np.sqrt(np.mean(core**2)) if core.size else np.nan
        fc[i]  = np.mean(x >= cthr)*100
    def jk_std(arr):
        G = len(arr)
        mean = np.nanmean(arr)
        var = (G - 1) / G * np.nansum((arr - mean)**2)
        return np.sqrt(var)
    med_mean, rms_mean, fc_mean = map(np.nanmean, [med_loo, rms, fc])
    med_err,  rms_err,  fc_err  = map(jk_std, [med_loo, rms, fc])
    out = {
        'median': {'mean': med_mean, 'std': med_err},
        'rms':    {'mean': rms_mean, 'std': rms_err},
        'fc':     {'mean': fc_mean,  'std': fc_err}
    }
    return out


#################################################################################################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    parser.add_argument("--method", help="method to compute the variance", choices = ['bootstrap', 'jackknife'], default='bootstrap')
    parser.add_argument("--repeatdir", help="base directory for repeat catalogs", default='/pscratch/sd/s/shengyu/repeats/DA2/loa-v1')
    parser.add_argument("--outputdir", help="output directory for results", default= '/global/homes/s/shengyu/desi_y3_redshift_errors/main/repeat_obs/results' )
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO','QSO_3cut'], default=['BGS','LRG','ELG','QSO','QSO_3cut'], nargs = '+')
    args = parser.parse_args()

    tracers = args.tracers
    method = args.method

    os.makedirs(args.outputdir, exist_ok=True)

    rows = []
    for tracer in args.tracers:
        # compute the overall matric
        zmin, zmax = REDSHIFT_OVERALL[tracer[:3]]
        dv = GET_REPEATS_DV(args.repeatdir, tracer[:3], zmin, zmax)
        cthr = GET_CTHR(tracer)
        tag = f"{tracer}_z{zmin}_{zmax}"
        print(tag, method, flush=True)
        if args.method == 'bootstrap':
            res = bootstrap_metrics(dv, cthr)
        elif args.method == 'jackknife':
            res = jackknife_metrics(dv, cthr)
        N = dv.size
        rows.append({'tracer': tag, 'N': N,  'method': method, 
                        'med_mean': res['median']['mean'], 'med_std': res['median']['std'],
                        'rms_mean': res['rms']['mean'], 'rms_std': res['rms']['std'],
                        'fc_mean': res['fc']['mean'], 'fc_std': res['fc']['std'], })    
        print(f"  Median={res['median']['mean']:.2f}±{res['median']['std']:.2f}, "
                f"RMS={res['rms']['mean']:.2f}±{res['rms']['std']:.2f}, "
                f"fc={res['fc']['mean']:.3f}±{res['fc']['std']:.3f}", flush=True) 
        # compute the matric in zbin 
        tag = 0
        step = 0.1
        zrange = np.round(np.arange(zmin, zmax+ step/2, step), 1)
        zbins = list(zip(zrange[:-1], zrange[1:]))
        for indz, (z1, z2) in enumerate(zbins):
            tag = f"{tracer}_z{z1}_{z2}"
            print(tag, method, flush=True)
            dv = GET_REPEATS_DV(args.repeatdir, tracer[:3], z1, z2)
            cthr = GET_CTHR(tracer)
            if args.method == 'bootstrap':
                res = bootstrap_metrics(dv, cthr)
            elif args.method == 'jackknife':
                res = jackknife_metrics(dv, cthr)
            N = dv.size
            rows.append({'tracer': tag, 'N': N,  'method': method, 
                         'med_mean': res['median']['mean'], 'med_std': res['median']['std'],
                         'rms_mean': res['rms']['mean'], 'rms_std': res['rms']['std'],
                         'fc_mean': res['fc']['mean'], 'fc_std': res['fc']['std'], })    
            print(f"  Median={res['median']['mean']:.2f}±{res['median']['std']:.2f}, "
                    f"RMS={res['rms']['mean']:.2f}±{res['rms']['std']:.2f}, "
                    f"fc={res['fc']['mean']:.3f}±{res['fc']['std']:.3f}", flush=True)
    # Save CSV
    outtab = Table(rows=rows, names=('tracer','N','method','med_mean','med_std','rms_mean','rms_std','fc_mean','fc_std'))
    outfile = os.path.join(args.outputdir, f"repeat_metrics_{method}.csv")
    outtab.write(outfile, overwrite=True, format='ascii.csv')
    print(f"[OK] Saved: {outfile}")
