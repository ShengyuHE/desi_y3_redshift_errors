from __future__ import annotations
import os
import logging
import numpy as np
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from utils import setup_logging
from pathlib import Path
try:
    from mockfactory import Catalog
except ImportError:
    Catalog = None

setup_logging()
logger = logging.getLogger('dv_tools') 

##### constant #####
CSPEED = 299792.458 # in km/s
REPEAT_DIR = Path('/global/cfs/cdirs/desi/users/shengyu/repeats/')
LSS_CAT_DIR = Path('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/nonKP')

def get_cthr(tracer):
    if tracer in ['BGS', 'BGS_BRIGHT-21.35', 'BGS_cdf']:
        cthr = 600
    if tracer in ['LRG', 'LRG_cdf']:
        cthr = 2000
    elif tracer in ['ELG', 'ELGnotqso', 'ELG_cdf']:
        cthr = 600
    elif tracer in ['QSO', 'QSO_cdf']:
        cthr = 10000
    elif tracer in ['QSO_3cut', 'QSO_3cut_cdf']:
        cthr = 3000
    return cthr

def _get_repeats_numbers(tracer, z1, z2, table_path='/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/repeat_obs/results/repeat_numbers.csv'):
    """
    Read repeat_numbers.csv and return (N, N_p, N_n)
    for a given tracer and z-range (z1, z2).
    """
    # Load table
    tab = Table.read(table_path, format='ascii.csv')
    # Construct tag
    tag = f"{tracer}_{z1}_{z2}"
    # Find matching row
    mask = tab['tag'] == tag
    if not np.any(mask):
        raise ValueError(f"Tag {tag} not found in table {table_path}")
    row = tab[mask][0]
    return row['N'], row['N_p'], row['N_n']

def _save_target_ids(tracer):
    if Catalog is None:
        raise ImportError("mockfactory is required to save repeat target ids.")
    if 'ELG' in tracer: tracer = 'ELGnotqso'
    if 'BGS' in tracer: tracer = 'BGS_BRIGHT-21.35'
    cat = Catalog.read(f'{LSS_CAT_DIR}/{tracer}_clustering.dat.fits')
    target_ids = np.asarray(cat['TARGETID'])
    np.save(REPEAT_DIR/f'{tracer[:3]}_target_ids.npy', target_ids)
    
def get_dv_qu(dv, cthr):
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    dv_smear = dv[abs(dv) < cthr]
    MAD = np.median(abs(dv))*1.4828 # median absolute deviation 
    RMS = np.sqrt(np.mean(dv_smear**2)) # residual mean square of vsmear part
    fc= np.mean(abs(dv) >= cthr)*100 # fc = (np.sum(abs(ds) > cthr)) /len(ds)*100 
    return {'cthr':cthr, 'mad':MAD, 'rms':RMS, 'fc':fc}

def _get_repeat_zcols(d):
    """Return the redshift column pair for old and new repeat catalogs."""
    for z1, z2 in [('Z1', 'Z2'), ('Z_1', 'Z_2'), ('Z_QF_1', 'Z_QF_2')]:
        if z1 in d.colnames and z2 in d.colnames:
            return z1, z2
    raise KeyError(f"Could not find repeat redshift columns in {d.colnames}")

def get_repeats_dv(tracer='LRG', zmin=0.4, zmax=0.6, kind='Z1/Z2', use_lss = True, survey='DA2', verspec='loa-v1', repeat_dir = REPEAT_DIR):
    repeat_dir = repeat_dir / survey / verspec
    d = Table.read(repeat_dir / f'{tracer[:3]}repeats.fits', hdu=1)
    z1_col, z2_col = _get_repeat_zcols(d)
    # sel = np.full(len(d),True)
    sel = np.isfinite(d[z1_col]) & np.isfinite(d[z2_col])
    if use_lss == True:
        target_ids = np.load(repeat_dir/f'{tracer[:3]}_target_ids.npy')
        sel_lss = np.isin(d['TARGETID'], target_ids)
    else:
        sel_lss = np.full(len(d),True)
    if kind in ['Z1', 'Z2']:
        ztrue = d[z1_col] if kind == 'Z1' else d[z2_col]
        selz = (zmin<ztrue)&(ztrue<zmax)
    elif kind == 'mean':
        ztrue = (d[z1_col]+d[z2_col])/2
        selz = (zmin<ztrue)&(ztrue<zmax)
    elif kind == 'Z1/Z2' or kind == 'Z1/Z2(mean)':
        ztrue = (d[z1_col]+d[z2_col])/2
        selz = ((zmin<d[z1_col])&(d[z1_col]<zmax))|((zmin<d[z2_col])&(d[z2_col]<zmax))
    elif kind == 'Z1/Z2(Z1)':
        ztrue = (d[z1_col])
        selz = ((zmin<d[z1_col])&(d[z1_col]<zmax))|((zmin<d[z2_col])&(d[z2_col]<zmax))
    elif kind == 'Z1/Z2(Z2)':
        ztrue = (d[z2_col])
        selz = ((zmin<d[z1_col])&(d[z1_col]<zmax))|((zmin<d[z2_col])&(d[z2_col]<zmax))
    else: 
        ztrue = (d[z1_col]+d[z2_col])/2
        selz = np.full(len(d),True)
    mask = sel & selz & sel_lss
    ztrue = ztrue[mask]
    d_zbin = d[mask]
    dv = (d_zbin[z1_col]-d_zbin[z2_col])/(1+ztrue)*CSPEED
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    cthr = get_cthr(tracer)
    return dv, get_dv_qu(dv, cthr)

def set_edges(type= 'log2', lim = 1000., num=60):
    if type == 'logbin':
        catasmin, catasmax, catasbin = -3.5, 6.1, 0.1
        edges=np.arange(catasmin, catasmax, catasbin)
    elif type == 'linear':
        edges = np.linspace(-lim, +lim, num)
    elif type == 'log2':
        n_side = num // 2
        dmin = lim * 1e-3
        dpos = 2.0 ** np.linspace(np.log2(dmin), np.log2(lim), n_side + 1)
        edges = np.concatenate([-dpos[::-1], dpos[1:]])
    else:
        ValueError(f"not validated {type}")
    return edges
    
def suggest_vbin(dv, bin_mode='log_abs', bw_method='scott', points_per_sigma=5):
    """
    Suggest an optimal linear dv bin width for evaluating the KDE model.

    The recommended bin width is computed as:
        vbin ≈ σ_KDE / points_per_sigma,
    where σ_KDE = kde.factor * std(v) (or |v| if use_abs=True).
    """
    v = np.asarray(dv, float)
    v = v[np.isfinite(v)]
    if 'log' in bin_mode:
        y = np.log10(abs(v))
    elif 'linear' in bin_mode:
        y = v
    kde = gaussian_kde(y, bw_method=bw_method)
    bw  = kde.factor * y.std()  # σ_KDE in linear dv space
    vbin = bw / points_per_sigma
    return vbin, bw

def sample_from_cdf_v2(tracer, z1, z2, N, survey='DA2', verspec='loa-v1', return_data = False, cdf_dir= None, seed=None):
    """
    Read one `cdf/*.npz` file and sample values using inverse-CDF sampling.

    Expected npz keys (as in this repo):
      - grid: 1D sample grid
      - cdf:  1D cumulative distribution on `grid`
      - pdf:  1D pdf on `grid` (optional for sampling, returned if requested)

    Parameters
    ----------
    tracer : str
        Tracer name, e.g. "BGS, ""LRG", "ELG", "QSO".
    z1 : float
        Lower edge of z bin, e.g. 0.8.
    z2 : float
        Upper edge of z bin, e.g. 0.9.
    N : int
        Number of random samples to draw.
    seed : int | None, default None
        RNG seed for reproducibility.
    return_data : bool, default False
        If True, also return a dict with loaded arrays and metadata.
    cdf_dir : str, default "cdf"
        Directory that stores CDF files.

    Returns
    -------
    samples : ndarray, shape (N,)
        Random samples distributed according to the input CDF.
    data : dict, optional
        Returned only when `return_data=True`.
    """
    if N <= 0:
        raise ValueError("`N` must be > 0")
    if cdf_dir is None:
        cdf_dir = REPEAT_DIR / survey / verspec / 'verr_mode'
    tracer = tracer.upper().strip()
    supported = {"BGS","LRG", "ELG", "QSO"}
    if tracer not in supported:
        raise ValueError(f"`tracer` must be one of {sorted(supported)}")

    npz_name = f"CDF_verr_nonparam_{tracer}_z{z1:.1f}-{z2:.1f}.npz"
    npz_path = Path(cdf_dir) / npz_name
    if not npz_path.exists():
        raise FileNotFoundError(f"CDF file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as d:
        logger.info(f"use {npz_path} to generate redshift errors")
        required = {"grid", "cdf"}
        missing = required.difference(d.files)
        if missing:
            raise KeyError(f"Missing required keys in {npz_path}: {sorted(missing)}")
        grid = np.asarray(d["grid"], dtype=np.float64).reshape(-1)
        cdf = np.asarray(d["cdf"], dtype=np.float64).reshape(-1)
        pdf = np.asarray(d["pdf"], dtype=np.float64).reshape(-1) if "pdf" in d.files else None
    if grid.size != cdf.size:
        raise ValueError("`grid` and `cdf` must have the same length")
    if grid.size < 2:
        raise ValueError("`grid`/`cdf` must contain at least 2 points")
    # Ensure interpolation axes are strictly ordered and CDF is valid.
    order = np.argsort(grid)
    grid = grid[order]
    cdf = cdf[order]
    if pdf is not None and pdf.size == order.size:
        pdf = pdf[order]

    # If pdf exists, rebuild CDF from integral on the actual grid.
    # This is robust for non-uniform grids where cumsum(pdf) is incorrect.
    if pdf is not None and pdf.size == grid.size:
        cdf = np.zeros_like(grid, dtype=np.float64)
        dx = np.diff(grid)
        cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)
    
    cdf = np.maximum.accumulate(cdf)
    cdf = np.clip(cdf, 0.0, None)
    cdf0, cdf1 = float(cdf[0]), float(cdf[-1])
    if not np.isfinite(cdf0) or not np.isfinite(cdf1) or cdf1 <= cdf0:
        raise ValueError("Invalid CDF values: cannot normalize for sampling")
    cdf = (cdf - cdf0) / (cdf1 - cdf0)

    # Deduplicate repeated CDF points for stable inverse interpolation.
    keep = np.r_[True, np.diff(cdf) > 0.0]
    cdf_u = cdf[keep]
    grid_u = grid[keep]
    if cdf_u.size < 2:
        raise ValueError("CDF has insufficient dynamic range for sampling")

    rng = np.random.default_rng(seed)
    u = rng.random(N)
    samples = np.interp(u, cdf_u, grid_u)
    if return_data:
        return samples, {"grid": grid, "cdf": cdf, "pdf": pdf, "npz_path": str(npz_path),}
    return samples

def _sample_from_cdf(cdf_fn, Ngal, bin_mode, seed=1234):
    """
    Returns
    -------
    dv : ndarray
        Sampled Δv array of length Ngal.
    inv_cdf : function
        Inverse-CDF interpolator used for sampling.
    """
    logger.info(f"use {cdf_fn} to generate redshift errors")
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
    if 'log' in bin_mode:
        if 'abs' in bin_mode:
            u = np.random.uniform(0, 1, int(Ngal / 2))
            y = inv_cdf(u)
            dv = np.append(10**y, -10**y)
            if Ngal % 2 == 1:
                dv = np.append([0.0], dv)
            np.random.shuffle(dv)
        elif 'signed' in bin_mode:
            u = np.random.uniform(0, 1, int(Ngal))
            dv = 10**inv_cdf(u)
    elif 'linear' in bin_mode:
        u = np.random.uniform(0, 1, int(Ngal))
        dv = inv_cdf(u)
    return dv, inv_cdf

def sample_from_cdf_v1(tracer, z1, z2, N, survey='DA2', verspec='loa-v1', dv_mode = 'verr_empirical', cdf_mode = 'CDF', bin_mode = 'log_abs', cdf_dir=None,  seed=1234):
    """
    Generate model Δv samples for a given tracer and redshift bin.

    Parameters
    ----------
    N : int Number of Δv values to sample.
    dv_mode
        - repeats
        - verr_empirical
        - 
    # cdf_mode : {"KCDF", "HCDF", "CDF"} Type of CDF used for sampling.
    vmode : {"log_signed", "log_abs", "linear"}
        Modeling mode:
        - "log_abs"    : sample |Δv| from log-CDF.
        - "log_signed" : sample positive/negative Δv separately using observed N_p/N_n fractions.
        - "linear"     : sample Δv directly.
    """
    # logger.info(f"use {dv_mode} mode, {tracer} in z{z1}-{z2}, to generate redshift errors")

    if cdf_dir is None:
        cdf_dir = REPEAT_DIR / survey / verspec
    mode = 'verr' if 'verr' in dv_mode else dv_mode 
    if 'verr' in dv_mode:
        cdf_mode = 'CDF'
        cdf_dir = cdf_dir / 'verr_mode'
    elif 'repeat' in dv_mode:
        cdf_mode = 'HCDF' 
        cdf_dir = cdf_dir / 'repeat_mode'
    if bin_mode == "log_abs":
        cdf_fn = cdf_dir / f"{cdf_mode}_{dv_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}.npz"
        dv, _ = _sample_from_cdf(cdf_fn, N, bin_mode, seed)
        return np.asarray(dv, float)
    elif bin_mode == "log_signed":
        (_N, _p, _n) = _get_repeats_numbers(tracer, z1, z2)
        N_p = int(N*float(_p/_N))
        N_n = N-N_p
        dv_list = []
        for sign, Num in [('+', N_p), ('-', N_n)]:
            cdf_fn = cdf_dir / f"{cdf_mode}_{dv_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}_{sign}.npz"
            sample, _ = _sample_from_cdf(cdf_fn, Num, bin_mode, seed)
            sample = np.asarray(sample, float)
            dv_list.append(sample if sign=='+' else -sample)
        dv = np.concatenate(dv_list)
        np.random.shuffle(dv)
        return np.asarray(dv, float)
    elif bin_mode == "linear":
        # fn = f"{dir}/{mode}_mode/{cdf_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}.npz"
        fn = cdf_dir / f"{cdf_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_logabs.npz"
        dv_model, _ = _sample_from_cdf(fn, N, bin_mode, seed)
        return np.asarray(dv_model, float)
    else:
        raise ValueError(f"Unknown mode: {bin_mode}")

def F_pdf(x, pars, dist="g", loc=0.0, cthr=None):
    """
    F(x): assumed redshift error profile, numerically normalized on x-grid.
    pars must contain required parameters depending on dist.
    """
    x = np.asarray(x, float)
    loc = float(pars.get("loc", loc))
    if dist in ("g"):
        sigma = float(pars["sigma"])
        z = (x - loc) / sigma
        f = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
    elif dist in ("l"):
        gamma = float(pars["gamma"])
        u = (x - loc) / gamma
        f = 1.0 / (np.pi * gamma * (1.0 + u**2))
    elif dist in ("g+l", "l+g"):
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        eta = float(pars.get("eta", 0.5))
        eta = np.clip(eta, 0.0, 1.0)
        z = (x - loc) / sigma
        u = (x - loc) / gamma
        f_g = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
        f_l = 1.0 / (np.pi * gamma * (1.0 + u**2))
        f = (1.0 - eta) * f_g + eta * f_l
    elif dist in ("v"):
        from scipy.special import wofz
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        z = ((x - loc) + 1j * gamma) / (sigma * np.sqrt(2.0))
        f = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError(f"Unknown dist: {dist}")
    # optional truncation
    if cthr is not None:
        f = np.where(np.abs(x) < cthr, f, 0.0)
    # numerical renormalization
    area = np.trapz(f, x)
    return (f / area) if area > 0 else None

def G_from_F_fft(x, f_x, cthr=None):
    """
    G(d) = ∫ F(t)F(t-d) dt via FFT autocorrelation.
    Returns (d, g).
    """
    from numpy.fft import rfft, irfft, fftshift
    x = np.asarray(x, float)
    f_x = np.asarray(f_x, float)
    dx = x[1] - x[0]
    n = x.size
    g = irfft(rfft(f_x) * np.conj(rfft(f_x)), n=n) * dx
    g = fftshift(g)
    d = (np.arange(n) - n // 2) * dx
    g = np.maximum(g, 0.0)
    if cthr is not None:
        g = np.where(np.abs(d) < cthr, g, 0.0)
    area = np.trapz(g, d)
    g = g / area if area > 0 else g
    return d, g

def F_from_G_ifft(d, g_d, cthr=None):
    """
    Reconstruct F from G assuming (roughly) F_k = sqrt(G_k) with no phase.
    """
    from numpy.fft import rfft, irfft, fftshift, ifftshift
    d = np.asarray(d, float)
    g_d = np.asarray(g_d, float)
    dx = d[1] - d[0]
    n = d.size
    g_k = rfft(ifftshift(g_d))
    f_k = np.sqrt(np.maximum(g_k.real, 0.0))
    f = irfft(f_k, n=n) / np.sqrt(dx)
    f = fftshift(f)
    f = np.maximum(f, 0.0)
    if cthr is not None:
        f = np.where(np.abs(d) < cthr, f, 0.0)
    area = np.trapz(f, d)
    f = f / area if area > 0 else f
    return d, f

# ---------- Fitting ----------

def _spec_from_dist(dv, dist):
    """Initial guesses + bounds depending on dist."""
    sigma0 = max(np.std(dv) / np.sqrt(2), 1e-3)
    q75, q25 = np.percentile(dv, [75, 25])
    gamma0 = max((q75 - q25) / 4.0, 1e-3)
    spec = {
        "g":          (["sigma"],                 [sigma0],              [(1e-6, None)]),
        "l":          (["gamma"],                 [gamma0],              [(1e-6, None)]),
        "g+l":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "l+g":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "v":          (["sigma", "gamma"],        [sigma0, gamma0],      [(1e-6, None), (1e-6, None)]),
    }
    if dist not in spec:
        raise ValueError(f"Unknown dist: {dist}")
    return spec[dist]

def _spec_from_dist_direct(dv, dist):
    """Initial guesses + bounds for fitting G(d) directly."""
    sigma0 = max(np.std(dv), 1e-3)
    q75, q25 = np.percentile(dv, [75, 25])
    gamma0 = max((q75 - q25) / 2.0, 1e-3)
    spec = {
        "g":          (["sigma"],                 [sigma0],              [(1e-6, None)]),
        "l":          (["gamma"],                 [gamma0],              [(1e-6, None)]),
        "g+l":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "l+g":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "v":          (["sigma", "gamma"],        [sigma0, gamma0],      [(1e-6, None), (1e-6, None)]),
    }
    if dist not in spec:
        raise ValueError(f"Unknown dist: {dist}")
    return spec[dist]

def G_pdf_direct(d, pars, dist="g", loc=0.0, cthr=None):
    """
    Parametric model for G(d) evaluated directly on the dv axis.

    This uses the same distribution names as F_pdf, but the fitted widths are
    widths of the repeat-difference distribution G rather than widths of F.
    If cthr is provided, the PDF is truncated to |d| < cthr and renormalized.
    """
    d = np.asarray(d, float)
    loc = float(pars.get("loc", loc))
    if dist in ("g"):
        sigma = float(pars["sigma"])
        z = (d - loc) / sigma
        g = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
    elif dist in ("l"):
        gamma = float(pars["gamma"])
        u = (d - loc) / gamma
        g = 1.0 / (np.pi * gamma * (1.0 + u**2))
    elif dist in ("g+l", "l+g"):
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        eta = float(pars.get("eta", 0.5))
        eta = np.clip(eta, 0.0, 1.0)
        z = (d - loc) / sigma
        u = (d - loc) / gamma
        g_g = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
        g_l = 1.0 / (np.pi * gamma * (1.0 + u**2))
        g = (1.0 - eta) * g_g + eta * g_l
    elif dist in ("v"):
        from scipy.special import wofz
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        z = ((d - loc) + 1j * gamma) / (sigma * np.sqrt(2.0))
        g = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError(f"Unknown dist: {dist}")
    if cthr is not None:
        g = np.where(np.abs(d) < cthr, g, 0.0)
    return g

def _direct_norm(pars, dist="g", loc=0.0, cthr=None, margin=0.5):
    """Numerical normalization for the direct G model."""
    n = 4097
    if cthr is not None:
        d_grid = np.linspace(-cthr, cthr, n)
    else:
        scale = max(float(v) for k, v in pars.items() if k != "loc")
        L = max(10.0 * scale, 1.0) * (1.0 + margin)
        d_grid = np.linspace(loc - L, loc + L, n)
    g_grid = G_pdf_direct(d_grid, pars, dist=dist, loc=loc, cthr=cthr)
    area = np.trapz(g_grid, d_grid)
    return area if area > 0 and np.isfinite(area) else np.nan

def _make_x_grid(dv, theta0, cthr=None, loc=0.0, margin=0.5):
    """Build x-grid used for numerical PDF + FFTs."""
    x_n = 2 ** int(np.ceil(np.log2(4 * len(dv))))
    if cthr is not None:
        L = cthr * (1.0 + margin)
    else:
        L = max(2 * np.max(np.abs(dv)), 10 * np.max(theta0))
    return np.linspace(loc - L, loc + L, int(x_n))

def fit_repeats(dv, dist="g", fit_mode="hist", bins=None, cthr=None, loc=0.0, margin=0.5):
    """
    Fit dv samples (assumed drawn from G(d)) by optimizing loss.
    Returns: best_params_dict, best_loss, scipy OptimizeResult
    """
    from scipy.optimize import minimize

    eps = 1e-12
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    if cthr is not None:
        dv = dv[np.abs(dv) < cthr]
    if dv.size == 0:
        raise ValueError("No finite dv values remain after applying cthr.")
    names, theta0, bounds = _spec_from_dist(dv, dist)
    theta0 = np.asarray(theta0, float)
    x = _make_x_grid(dv, theta0, cthr=cthr, loc=loc, margin=margin)
    if fit_mode == "hist" and bins is None:
        bins = set_edges(lim=cthr or np.max(np.abs(dv)), num=60)
    def loss(theta):
        pars = {"loc": loc, **{k: float(v) for k, v in zip(names, theta)}}
        f = F_pdf(x, pars, dist=dist, loc=loc, cthr=cthr)
        if f is None:
            return np.inf
        d_model, g_model = G_from_F_fft(x, f, cthr=cthr)
        if fit_mode == "hist":
            g_obs, edges = np.histogram(dv, bins=bins, density=True)
            d_centers = 0.5 * (edges[1:] + edges[:-1])
            g_pred = np.interp(d_centers, d_model, g_model, left=0.0, right=0.0)
            mask = g_obs > 0
            return np.sum((np.log(g_obs[mask] + eps) - np.log(g_pred[mask] + eps)) ** 2)
        # "direct": evaluate likelihood at dv points
        g_eval = np.interp(dv, d_model, g_model, left=0.0, right=0.0)
        return -np.sum(np.log(g_eval + eps))
    res = minimize(loss, x0=theta0, bounds=bounds, method="L-BFGS-B")
    best = {k: float(v) for k, v in zip(names, res.x)}
    print(f"Best-fit: {best}, loss={res.fun:.2f}")
    return best, float(res.fun), res

def fit_direct(dv, dist="g", fit_mode="hist", bins=None, cthr=None, loc=0.0, margin=0.5):
    """
    Fit dv samples by optimizing a parametric model for G(d) directly.

    This is analogous to fit_repeats, but it does not build F or use the FFT
    autocorrelation. The returned parameters describe G(dv) itself.
    Returns: best_params_dict, best_loss, scipy OptimizeResult
    """
    from scipy.optimize import minimize

    eps = 1e-12
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    if cthr is not None:
        dv = dv[np.abs(dv) < cthr]
    if dv.size == 0:
        raise ValueError("No finite dv values remain after applying cthr.")
    names, theta0, bounds = _spec_from_dist_direct(dv, dist)
    theta0 = np.asarray(theta0, float)
    if fit_mode == "hist" and bins is None:
        bins = set_edges(lim=cthr or np.max(np.abs(dv)), num=60)
    if fit_mode == "hist":
        g_obs, edges = np.histogram(dv, bins=bins, density=True)
        d_centers = 0.5 * (edges[1:] + edges[:-1])
        mask = g_obs > 0
    elif fit_mode != "direct":
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    def loss(theta):
        pars = {"loc": loc, **{k: float(v) for k, v in zip(names, theta)}}
        norm = _direct_norm(pars, dist=dist, loc=loc, cthr=cthr, margin=margin)
        if not np.isfinite(norm) or norm <= 0:
            return np.inf
        if fit_mode == "hist":
            g_pred = G_pdf_direct(d_centers, pars, dist=dist, loc=loc, cthr=cthr) / norm
            return np.sum((np.log(g_obs[mask] + eps) - np.log(g_pred[mask] + eps)) ** 2)
        g_eval = G_pdf_direct(dv, pars, dist=dist, loc=loc, cthr=cthr) / norm
        return -np.sum(np.log(g_eval + eps))

    res = minimize(loss, x0=theta0, bounds=bounds, method="L-BFGS-B")
    best = {k: float(v) for k, v in zip(names, res.x)}
    print(f"Best-fit direct G: {best}, loss={res.fun:.2f}")
    return best, float(res.fun), res
