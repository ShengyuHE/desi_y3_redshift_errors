import numpy as np

##### constant #####
CSPEED = 299792.458 # in km/s

def get_repeats_numbers(tracer, z1, z2, table_path='/global/homes/s/shengyu/desi_y3_redshift_errors/main/repeat_obs/results/repeat_numbers.csv'):
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

def get_cthr(tracer):
    if tracer in ['BGS']:
        cthr = 600
    if tracer in ['LRG']:
        cthr = 1000
    elif tracer in ['ELG']:
        cthr = 600
    elif tracer in ['QSO']:
        cthr = 10000
    elif tracer in ['QSO_3cut']:
        cthr = 3000
    return cthr

def get_repeats_ds(tracer, zmin, zmax, zture = None, repeat_dir = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'):
    from astropy.table import Table
    d = Table.read(f'{repeat_dir}/{tracer}repeats.fits', hdu=1)
    sel      = np.full(len(d),True)
    sel = np.isfinite(d['Z1']) & np.isfinite(d['Z2'])
    selz = ((zmin<d["Z2"])&(d["Z2"]<zmax))
    d_zbin = d[sel & selz]
    if zture is None:
        ztrue = (d_zbin['Z1']+d_zbin['Z2'])/2
    ds_zbin = (d_zbin['Z1']-d_zbin['Z2'])/(1+ztrue)*CSPEED
    ds = np.asarray(ds_zbin, float)
    ds = ds[np.isfinite(ds)]
    cthr = get_cthr(tracer)
    ds_smear = ds[abs(ds) < cthr]
    MED = np.median(abs(ds))*1.4828/np.sqrt(2) # median absolute deviation 
    RMS = np.sqrt(np.mean(ds_smear**2)) # residual mean square of vsmear part
    fc= np.mean(abs(ds) >= cthr)*100 # fc = (np.sum(abs(ds) > cthr)) /len(ds)*100 
    qu = {'cthr':cthr, 'med':MED, 'rms':RMS, 'fc':fc}
    return ds, qu

def set_edges(type= 'log2', lim = 1000., num=60):
    if type == 'logbin':
        catasmin, catasmax, catasbin = -3.5, 6.1, 0.1
        edges=np.arange(catasmin, catasmax, catasbin)
    if type == 'linear':
        edges = np.linspace(-lim, +lim, num)
    if type == 'log2':
        n_side = num // 2
        dmin = lim * 1e-3
        dpos = 2.0 ** np.linspace(np.log2(dmin), np.log2(lim), n_side + 1)
        edges = np.concatenate([-dpos[::-1], dpos[1:]])
    return edges
    
def F_pdf(x, pars, dist="gaussian", cthr=None):
    """
    F(x) --> the assumed profile of redshift errors 
    """
    dist = dist.lower()
    loc  = pars.get("loc", 0.0)
    if dist in ("g"):
        # Gaussian profile
        sigma = float(pars["sigma"])
        z = (x - loc) / sigma
        f = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
    elif dist in ("l"):
        # Lorentzian profile
        gamma = float(pars["gamma"])
        u = (x - loc) / gamma
        f = 1.0 / (np.pi * gamma * (1.0 + u**2))
    elif dist in ("g+l", "l+g"):
        # Gaussian+Lorentzian combined profile
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        eta = float(pars.get("eta", 0.5))  # eta in [0,1]
        eta = np.clip(eta, 0.0, 1.0)
        z = (x - loc) / sigma
        u = (x - loc) / gamma
        f_g = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * sigma)
        f_l = 1.0 / (np.pi * gamma * (1.0 + u**2))
        f = (1.0 - eta) * f_g + eta * f_l
    elif dist in ("v"):
        # Voigt profile: convolution of Gaussian(sigma) and Lorentzian(gamma)
        from scipy.special import wofz
        sigma = float(pars["sigma"])
        gamma = float(pars["gamma"])
        z = ((x - loc) + 1j * gamma) / (sigma * np.sqrt(2.0))
        f = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError(f"Unknown dist: {dist}")
    # renormalize numerically
    if cthr is not None:
        f = np.where(np.abs(x) < cthr, f, 0.0)
    area = np.trapz(f, x)
    return f / area if area > 0 else None

def G_from_F_fft(x, f_x, cthr=None):
    from numpy.fft import rfft, irfft, fftshift, ifftshift
    """
    G(d) --> repeats distribution
    Given redshift errors F(t), compute G(d)=∫F(t) F(t-d) dt -> Gk=Fk^2
    using FFT autocorrelation. Returns d_grid and g(d) on that grid.
    """
    x = np.asarray(x)
    f_x = np.asarray(f_x)
    dx = x[1] - x[0]
    n = x.size
    g = irfft(rfft(f_x) * np.conj(rfft(f_x)), n=n) * dx  # autocorr * dx
    g = fftshift(g)
    d = (np.arange(n) - n//2) * dx
    # numerical cleanup + renormalize (should be ~1 already)
    g = np.maximum(g, 0.0)
    if cthr is not None:
        g = np.where(np.abs(d) < cthr, g, 0.0)
    g /= np.trapz(g, d)
    return d, g

def F_from_G_ifft(d, g_d, cthr=None):
    from numpy.fft import rfft, irfft, fftshift, ifftshift
    """
    Reconstruct F(x) from G(y) assuming in Fourrier space Fk=sqrt(Gk):
    - F is real and even
    - Fourier transform of F is non-negative
    - F has no phase
    """
    d = np.asarray(d)
    g_d = np.asarray(g_d)
    dx = d[1] - d[0]
    n = d.size
    # shift back to FFT ordering
    g_k = rfft(ifftshift(g_d))
    f_k = np.sqrt(np.maximum(g_k.real, 0.0))
    f = irfft(f_k, n=n) / np.sqrt(dx)
    f = fftshift(f)
    f = np.maximum(f, 0.0)
    if cthr is not None:
        f = np.where(np.abs(d) < cthr, f, 0.0)
    f /= np.trapz(f, d)
    return d, f

def fit_dv_to_F(dv, dist = 'gaussian', fit_mode = 'direct',
                bins = 100, cthr= 1000,  loc = 0.0, margin = 0.5):
    from scipy.optimize import minimize
    """
    Fit the observed dv (G(d)) to the F(x)
    dist: 'g':gaussian', 'l':lorentzian, 'g+l'/'l+g':mix with eta, 'v':voigt
    """
    dv = np.asarray(dv)
    dist = dist.lower()
    if cthr is not None:
        dv = dv[np.abs(dv) < cthr]
    # initial guesses from dv
    sigma0 = max(np.std(dv) / np.sqrt(2), 1e-3)
    q75, q25 = np.percentile(dv, [75, 25])
    gamma0 = max((q75 - q25) / 4.0, 1e-3)
    # parameter space
    spec = {
        "g":          (["sigma"],              [sigma0],            [(1e-6, None)]),
        "l":          (["gamma"],              [gamma0],            [(1e-6, None)]),
        "g+l":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "l+g":        (["sigma", "gamma", "eta"], [sigma0, gamma0, 0.5], [(1e-6, None), (1e-6, None), (0.0, 1.0)]),
        "v":          (["sigma", "gamma"],     [sigma0, gamma0],    [(1e-6, None), (1e-6, None)]),
    }
    if dist not in spec:
        raise ValueError(f"Unknown dist: {dist}")
    names, theta0, bounds = spec[dist]
    theta0 = np.asarray(theta0, float)

    # extend bin range
    eps = 1e-12
    x_n = 2 ** int(np.ceil(np.log2(4 * len(dv))))
    if cthr is not None:
        L = cthr * (1.0 + margin)
    else:
        L = max(2*np.max(np.abs(dv)), 10 * theta0)
    x = np.linspace(loc - L, loc + L, int(x_n))

    # loss function
    def loss(theta):
        pars = {"loc": loc}
        pars.update({k: float(v) for k, v in zip(names, theta)})
        if fit_mode == 'hist':
            # empirical G(d) from dv
            g_obs, edges = np.histogram(dv, bins=bins, density=True)
            d_centers = 0.5 * (edges[1:] + edges[:-1])
            # grid size
            mask = g_obs > 0
            f = F_pdf(x, pars, dist=dist)
            if f is None: return np.inf
            d_model, g_model = G_from_F_fft(x, f, cthr = cthr)
            g_pred = np.interp(d_centers, d_model, g_model, left=0.0, right=0.0)
            return np.sum((np.log(g_obs[mask] + eps) - np.log(g_pred[mask] + eps))**2)
            return np.sum((g_obs - g_pred) ** 2)
        elif fit_mode == 'direct':
            f = F_pdf(x, pars, dist=dist)
            if f is None: return np.inf
            d, g = G_from_F_fft(x, f, cthr = cthr)
            g_eval = np.interp(dv, d, g, left=0.0, right=0.0)
            return -np.sum(np.log(g_eval + eps))

    res = minimize(loss, x0=theta0, bounds=bounds, method="L-BFGS-B")
    print(f"Best-fit: {dict(zip(names, res.x))}, loss= {res.fun:.2f}",)
    return res.x, res.fun, res


'''
old script we model the repeats observations

def sample_from_cdf(cdf_fn, Ngal, vmode, seed=1234):
    """
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
    N : int Number of Δv values to sample.
    cdf_kind : {"KCDF", "obsCDF"} Type of CDF used for sampling.
    vmode : {"log_signed", "log_abs", "linear"}
        Modeling mode:
        - "log_abs"    : sample |Δv| from log-CDF.
        - "log_signed" : sample positive/negative Δv separately using observed N_p/N_n fractions.
        - "linear"     : sample Δv directly.
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
'''