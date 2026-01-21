import os
import numpy as np
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

##### constant #####
CSPEED = 299792.458 # in km/s
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'

def get_repeats_numbers(tracer, z1, z2, table_path='/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/repeat_obs/results/repeat_numbers.csv'):
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
        cthr = 2000
    elif tracer in ['ELG']:
        cthr = 600
    elif tracer in ['QSO']:
        cthr = 10000
    elif tracer in ['QSO_3cut']:
        cthr = 3000
    return cthr

def get_repeats_dv(tracer, zmin, zmax, kind='Z1', repeat_dir = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'):
    d = Table.read(f'{repeat_dir}/{tracer[:3]}repeats.fits', hdu=1)
    # sel = np.full(len(d),True)
    sel = np.isfinite(d['Z1']) & np.isfinite(d['Z2'])
    if kind in ['Z1', 'Z2']:
        ztrue = d[kind]
        selz = (zmin<ztrue)&(ztrue<zmax)
    elif kind == 'mean':
        ztrue = (d['Z1']+d['Z2'])/2
        selz = (zmin<ztrue)&(ztrue<zmax)
    elif kind == 'Z1/Z2':
        ztrue = (d['Z1']+d['Z2'])/2
        selz = ((zmin<d['Z1'])&(d['Z1']<zmax))|((zmin<d['Z2'])&(d['Z2']<zmax))
    elif kind == 'Z1/Z2(Z1)':
        ztrue = (d['Z1'])
        selz = ((zmin<d['Z1'])&(d['Z1']<zmax))|((zmin<d['Z2'])&(d['Z2']<zmax))
    elif kind == 'Z1/Z2(Z2)':
        ztrue = (d['Z2'])
        selz = ((zmin<d['Z1'])&(d['Z1']<zmax))|((zmin<d['Z2'])&(d['Z2']<zmax))
    ztrue = ztrue[sel & selz]
    d_zbin = d[sel & selz]
    dv = (d_zbin['Z1']-d_zbin['Z2'])/(1+ztrue)*CSPEED
    dv = np.asarray(dv, float)
    dv = dv[np.isfinite(dv)]
    cthr = get_cthr(tracer)
    dv_smear = dv[abs(dv) < cthr]
    MED = np.median(abs(dv))*1.4828/np.sqrt(2) # median absolute deviation 
    RMS = np.sqrt(np.mean(dv_smear**2)) # residual mean square of vsmear part
    fc= np.mean(abs(dv) >= cthr)*100 # fc = (np.sum(abs(ds) > cthr)) /len(ds)*100 
    qu = {'cthr':cthr, 'med':MED, 'rms':RMS, 'fc':fc}
    return dv, qu

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

def model_dv_from_cdf(tracer, z1, z2, N, dv_mode = 'verr', cdf_mode = 'HCDF', bin_mode = 'log_abs', seed=1234):
    """
    Generate model Δv samples for a given tracer and redshift bin.

    Parameters
    ----------
    N : int Number of Δv values to sample.
    dv_mode
        - repeats
        - verr_empirical
        - 
    cdf_mode : {"KCDF", "HCDF"} Type of CDF used for sampling.
    vmode : {"log_signed", "log_abs", "linear"}
        Modeling mode:
        - "log_abs"    : sample |Δv| from log-CDF.
        - "log_signed" : sample positive/negative Δv separately using observed N_p/N_n fractions.
        - "linear"     : sample Δv directly.
    """
    # if dv_mode == 'repeat': 
    #     cdf_mode = 'HCDF'
    #     bin_mode = 'log_signed'
    mode = 'verr' if 'verr' in dv_mode else dv_mode 

    if bin_mode == "log_abs":
        cdf_fn = f"{REPEAT_DIR}/{mode}_mode/{cdf_mode}_{dv_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}.npz"
        dv, _ = sample_from_cdf(cdf_fn, N, bin_mode, seed)
        return np.asarray(dv, float)
    elif bin_mode == "log_signed":
        (_N, _p, _n) = get_repeats_numbers(tracer, z1, z2)
        N_p = int(N*float(_p/_N))
        N_n = N-N_p
        dv_list = []
        for sign, Num in [('+', N_p), ('-', N_n)]:
            cdf_fn = f"{REPEAT_DIR}/{mode}_mode/{cdf_mode}_{dv_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}_{sign}.npz"
            sample, _ = sample_from_cdf(cdf_fn, Num, bin_mode, seed)
            sample = np.asarray(sample, float)
            dv_list.append(sample if sign=='+' else -sample)
        dv = np.concatenate(dv_list)
        np.random.shuffle(dv)
        return np.asarray(dv, float)
    elif bin_mode == "linear":
        fn = f"{REPEAT_DIR}/{mode}_mode/{cdf_mode}_{tracer}_z{z1:.1f}-{z2:.1f}_{bin_mode}.npz"
        dv_model, _ = sample_from_cdf(fn, N, bin_mode, seed)
        return np.asarray(dv_model, float)
    else:
        raise ValueError(f"Unknown mode: {bin_mode}")



class ParamRedshiftErrorModel:
    """
    Encapsulates:
    - F(x): assumed redshift error PDF (Gaussian / Lorentzian / mix / Voigt)
    - G(d): repeats distribution via autocorrelation of F
    - fitting dv samples by maximizing likelihood under G(d)
    """

    def __init__(
        self,
        dist="g",
        loc=0.0,
        cthr=None,
        bins=100,
        margin=0.5,
        fit_mode="direct",
    ):
        self.dist = dist.lower()
        self.loc = float(loc)
        self.cthr = cthr
        self.bins = int(bins)
        self.margin = float(margin)
        self.fit_mode = fit_mode

    # ---------- Core PDFs ----------

    def F_pdf(self, x, pars):
        """
        F(x): assumed redshift error profile, numerically normalized on x-grid.
        pars must contain required parameters depending on dist.
        """
        x = np.asarray(x, float)
        dist = self.dist
        loc = float(pars.get("loc", self.loc))
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
        if self.cthr is not None:
            f = np.where(np.abs(x) < self.cthr, f, 0.0)
        # numerical renormalization
        area = np.trapz(f, x)
        return (f / area) if area > 0 else None

    def G_from_F_fft(self, x, f_x, cthr=None):
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
        cthr_use = self.cthr if cthr is None else cthr
        if cthr_use is not None:
            g = np.where(np.abs(d) < cthr_use, g, 0.0)
        area = np.trapz(g, d)
        g = g / area if area > 0 else g
        return d, g

    def F_from_G_ifft(self, d, g_d, cthr=None):
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
        cthr_use = self.cthr if cthr is None else cthr
        if cthr_use is not None:
            f = np.where(np.abs(d) < cthr_use, f, 0.0)
        area = np.trapz(f, d)
        f = f / area if area > 0 else f
        return d, f

    # ---------- Fitting ----------

    def _spec_from_dist(self, dv):
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
        if self.dist not in spec:
            raise ValueError(f"Unknown dist: {self.dist}")
        return spec[self.dist]

    def _make_x_grid(self, dv, theta0):
        """Build x-grid used for numerical PDF + FFTs."""
        x_n = 2 ** int(np.ceil(np.log2(4 * len(dv))))
        if self.cthr is not None:
            L = self.cthr * (1.0 + self.margin)
        else:
            L = max(2 * np.max(np.abs(dv)), 10 * np.max(theta0))
        return np.linspace(self.loc - L, self.loc + L, int(x_n))

    def fit_dv(self, dv):
        """
        Fit dv samples (assumed drawn from G(d)) by optimizing loss.
        Returns: best_params_dict, best_loss, scipy OptimizeResult
        """
        from scipy.optimize import minimize

        eps = 1e-12
        dv = np.asarray(dv, float)
        if self.cthr is not None:
            dv = dv[np.abs(dv) < self.cthr]
        names, theta0, bounds = self._spec_from_dist(dv)
        theta0 = np.asarray(theta0, float)
        x = self._make_x_grid(dv, theta0)

        def loss(theta):
            pars = {"loc": self.loc, **{k: float(v) for k, v in zip(names, theta)}}
            f = self.F_pdf(x, pars)
            if f is None:
                return np.inf

            d_model, g_model = self.G_from_F_fft(x, f, cthr=self.cthr)
            if self.fit_mode == "hist":
                g_obs, edges = np.histogram(dv, bins=self.bins, density=True)
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

