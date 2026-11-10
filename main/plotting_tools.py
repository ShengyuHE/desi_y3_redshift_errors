
import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import re
import healpy as hp
import pandas as pd
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

sys.path.append('../')

##### Basic settings #####
def get_namespace(tracer, zrange):
    return {
        ('BGS_BRIGHT-21.35', (0.1, 0.4)): 'BGS1',
        ('BGS', (0.1, 0.4)): 'BGS1',
        ('LRG', (0.4, 0.6)): 'LRG1',
        ('LRG', (0.6, 0.8)): 'LRG2',
        ('LRG', (0.8, 1.1)): 'LRG3',
        ('ELG_LOPnotqso', (0.8, 1.1)): 'ELG1',
        ('ELG_LOPnotqso', (1.1, 1.6)): 'ELG2',
        ('ELG', (0.8, 1.1)): 'ELG1',
        ('ELG', (1.1, 1.6)): 'ELG2',
        ('QSO', (0.8, 2.1)): 'QSO1',
    }[(tracer, zrange)]


CONF = {
    "BGS": {
        r"[S$\,\mathrm{II}$]": 6717,
        r"[N$\,\mathrm{II}$]": 6548,
        r"H$\,\alpha$": 6563,
        r"H$\,\beta$": 4861,
        r"[O$\,\mathrm{III}$] 4959": 4959,
        r"[O$\,\mathrm{III}$] 5007": 5007,
        r"[O$\,\mathrm{III}$]": 5003,
        r"[O$\,\mathrm{II}$]": 3727,
        r"Mg$\,\mathrm{II}$": 2800,
    },
    "QSO": {
        r"Mg$\,\mathrm{II}$": 2800,
        r"H$\,\alpha$": 6563,
        r"H$\,\beta$": 4861,
        r"H$\,\gamma$": 4340,
        r"C$\,\mathrm{III}$]": 1908,
        r"C$\,\mathrm{IV}$": 1549,
        r"Ly$\,\alpha$": 1215.67,
        r"[O$\,\mathrm{III}$]": 5003,
        r"[O$\,\mathrm{II}$]": 3727,
    },
    "LRG": {},
    "ELG": {},
}

def get_spec_lines(tracer):
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
    if tracer not in CONF:
        raise ValueError(f"Unknown tracer '{tracer}'")
    return dict(CONF[tracer])

CONF_TOL_TRACER = {'BGS': 0.02, 'LRG': 0.01, 'ELG': 0.01, 'QSO': 0.05,}

##### Color settings #####
COLOR_OVERALL = dict(BGS = 'green',
                    LRG = 'crimson',
                    ELG = 'blue',
                    QSO = 'purple')

COLOR_TRACERS = dict(BGS1='green', 
                    LRG1='orange', LRG2='orangered', LRG3='red',
                    ELG1='skyblue', ELG2= 'steelblue',
                    QSO1='purple')

COLOR_TRACER_GRADIENT = dict(BGS='Greens', LRG='Reds',ELG='Blues', QSO='Purples',
                             LRG1='Wistia',LRG2='Oranges',LRG3='Reds', QSO1='Purples',)

TPS_LABELS = dict(xi ={'x':r"$s\ [h^{-1}\mathrm{Mpc}]$",'y':r"$s^2\xi_\ell$", 'dy0':r"$\Delta\xi_0/\sigma$", 'dy2':r"$\Delta\xi_2/\sigma$"},
                  pk ={'x':r"$k\ [h\,\mathrm{Mpc}^{-1}]$",'y':r"$kP_\ell(k) \ [h^{-1}\mathrm{Mpc}]^2$", 'dy0':r"$\Delta P_0/\sigma$", 'dy2':r"$\Delta P_2/\sigma$"},
                  mpslog ={'x':r"$s \ [h^{-1}\mathrm{Mpc}]$",'y':r"$s\,\xi_\ell\ [h^{-1}\mathrm{Mpc}]$", 'dy0':r"$\Delta\xi_0/\sigma$", 'dy2':r"$\Delta\xi_2/\sigma$"},
                  mpslog0 ={'x':r"$s \ [h^{-1}\mathrm{Mpc}]$",'y':r"$s\,\xi_0 \ [h^{-1}\mathrm{Mpc}]$", 'dy':r"$\Delta\xi_0/\sigma$"},
                  mpslog2 ={'x':r"$s \ [h^{-1}\mathrm{Mpc}]$",'y':r"$s\,\xi_2 \ [h^{-1}\mathrm{Mpc}]$", 'dy':r"$\Delta\xi_2/\sigma$"},
                  wplog ={'x':r"$r_p \ [h^{-1}\mathrm{Mpc}]$",'y':r"$r_p \, w_P \ [h^{-1}\mathrm{Mpc}]^2$", 'dy0':r"$\Delta w_p/\sigma$"},
                  mesh2 = {'x':r"$k\ [h\,\mathrm{Mpc}^{-1}]$",'y':r"$kP_\ell \ [h^{-1}\mathrm{Mpc}]^2$", 'dy0':r"$\Delta P_0/\sigma$", 'dy2':r"$\Delta P_2/\sigma$"},
                  mesh3_sugiyama= {'x':r"$k\ [h\,\mathrm{Mpc}^{-1}]$",'y':r"$k^2B_{\ell_1 \ell_2 L} \, [h^{-1}\mathrm{Mpc}]^4 $", 'dy0':r"$\Delta B_{000}/\sigma$", 'dy2':r"$\Delta B_{202}/\sigma$"}
                  )

#### Fitting settings ####

PARAM_LABELS = {
    'h': r'$h$',
    'omega_cdm': r'$\omega_{\rm cdm}$',
    'omega_b': r'$\omega_b$',
    'logA': r'$\ln(10^{10} A_s)$',
    'n_s': r'$n_s$',
    'H0': r'$H_0$',
    'Omega_m': r'$\Omega_m$',
    'sigma8_m': r'$\sigma_8$',
}


PLANCK_COSMOLOGY = {
    "Omega_m": 0.315191868,
    "H_0": 67.36,
    "H0": 67.36,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "sigma8_m": 0.80758,
    "h": 0.6736,
    "A_s": 2.083e-9,
    "logA": 3.034,
    "n_s": 0.9649,
    "N_ur": 2.0328,
    "N_ncdm": 1.0,
    "omega_ncdm": 0.0006442,
    "w_0": -1,
    "w_a": 0.0
}

##### Functions for CATAS #####
def identify_line_confusions(d, line_set, name_set, focus, remove=(), tol=1e-1, cols = ['Z1', 'Z2']):
    """
    Identify potential line confusions given two redshifts Z1 (trusted) and Z2 (alt)
    and a set of rest-frame line wavelengths.
    """
    if remove is None: remove = ()
    # Filter out removed names (preserves order)
    kept = [(l, n) for l, n in zip(line_set, name_set) if n not in set(remove)]
    if len(kept) == 0:
        return {}
    lines, names = map(list, zip(*kept))
    if focus not in names:
        raise ValueError(f"Focus line '{focus}' not found after applying remove={remove}.")
    focus_idx = names.index(focus)
    lam_focus = float(lines[focus_idx])

    # Pre-extract columns (works for numpy structured arrays and astropy Table)
    targetid = np.asarray(d["TARGETID"])
    z1 = np.asarray(d[cols[0]], dtype=float)
    z2 = np.asarray(d[cols[1]], dtype=float)
    confusion_dict = {}
    for lam, name in zip(lines, names):
        if name == focus:
            continue
        lam = float(lam)
        key = f"{focus}→{name}"
        # Predicted catastrophic redshift if the line is mis-identified
        z_cata1 = lam_focus / lam * (1.0 + z1) - 1.0  # focus mistaken as this line
        z_cata2 = lam / lam_focus * (1.0 + z1) - 1.0  # this line mistaken as focus
        m = (np.abs(z2 - z_cata1) < tol) | (np.abs(z2 - z_cata2) < tol)
        if np.any(m):
            confusion_dict[key] = [
                {"TARGETID": int(t), "Z1": float(a), "Z2": float(b)}
                for t, a, b in zip(targetid[m], z1[m], z2[m])
            ]
        else:
            confusion_dict[key] = []  # keep empty lists if you want consistent keys
    return confusion_dict

def identify_sky_residuals(d, skyZ_list, tol=1e-3, cols = ['Z1', 'Z2']):
    """
    Identify potential sky-residual-driven failures by matching Z1 and/or Z2
    to a set of known sky redshift spikes.
    """
    skyZ = np.asarray(list(skyZ_list), dtype=float)
    if skyZ.size == 0: return {}
    targetid = np.asarray(d["TARGETID"])
    # Load requested z columns
    zcols = {col: np.asarray(d[col], dtype=float) for col in cols}
    sky_dict = {}
    for zsky in skyZ:
        key = f"sky z≈{zsky:g}"
        # Match if ANY requested column is close
        m = np.zeros(len(targetid), dtype=bool)
        for col, z in zcols.items():
            m |= (np.abs(z - zsky) < tol)
        if np.any(m):
            sky_dict[key] = [
                {"TARGETID": int(t), **{col: float(zcols[col][i]) for col in cols}}
                for i, t in enumerate(targetid) if m[i]]
        else:
            sky_dict[key] = []
    return sky_dict

def plot_confusion_lines(ax, line_set, name_set, focus = None, remove = None, **args):
    alpha = args.get('alpha', 1.0)
    lw = args.get('lw', 1.0)
    if focus not in name_set and focus is not None:
        raise ValueError(f"Focus line '{focus}' not found in names list.")
    if remove != None:
        lines, names = zip(*[(l, n) for l, n in zip(line_set, name_set) if n not in remove])
    else:
        lines, names = line_set, name_set
    # Loop over all possible pairs
    x = np.linspace(-0.01, 4.0, 2)
    if focus != None:
        focus_idx = names.index(focus)
        # colormap = get_cmap('a')
        # colors = [tuple(c) for c in colormap(np.linspace(0, 1, len(lines)+1))]
        colors = plt.cm.tab10(np.linspace(0, 1, len(lines)+1))
        if focus in [r"C$\,\mathrm{IV}$"]:
            colors = list(reversed(colors))
        for j, (lam, name) in enumerate(zip(lines, names)):
            if name == focus:
                continue
            # case 1: focus line mistaken for others
            y1 = lines[focus_idx]/lam * (1+x) - 1
            ax.plot(x, y1, '--', color=colors[j], lw=lw, label=f'{focus}'+r'$\longleftrightarrow$'+f'{name}', alpha=alpha)
            # case 2: others mistaken for focus
            y2 = lam/lines[focus_idx] * (1+x) - 1
            ax.plot(x, y2, '--', color=colors[j], lw=lw, alpha=alpha)
    else:
        print(lines)
        colors = plt.cm.tab20(np.linspace(0, 1, len(lines) * (len(lines)-1)))
        for k, (i, j) in enumerate([(i, j) for i in range(len(lines)) for j in range(len(lines)) if i != j]):
            true, false = lines[i], lines[j]
            y = true/false * (1 + np.linspace(-0.01, 3.0, 2)) - 1
            ax.plot(x, y, ':', color=colors[k], lw=0.5, alpha=alpha,
                    label=f'{names[i]}'+r'$\leftrightarrow$'+f'{names[j]}')
            
def plot_sky_residuals(ax, residuals, label=None):
    colors = ["#3B8B7A" , "#9540e5", "#9A3412"]
    # colors = ["#D55E00", "#0072B2", "#CC79A7"]
    for i,z in enumerate(residuals):
        if label == None:
            this_label = f'skyres z'+r'$\approx$'+f'{z:.2f}'
        else:
            this_label = label 
        ax.axhline(z, color=colors[i], lw=1.5, ls='-.', label=this_label)
        ax.axvline(z, color=colors[i], lw=1.5, ls='-.')

def plot_conf_dots(ax, conf, cols = ['Z1','Z2']):
    for key, rows in conf.items(): 
        if isinstance(rows, dict):
            rows = [rows]
        z1 = np.array([r[cols[0]] for r in rows], dtype=float)
        z2 = np.array([r[cols[1]] for r in rows], dtype=float)
        ax.plot(z1, z2, "o", alpha=0.6, markersize=3, markerfacecolor="none", markeredgecolor="k",)


def plot_moll(hmap, whmap=None, min=None, max=None, nest=False, title='', label=r'number density [$\#$ deg$^{-2}$]', rot=115, projection='mollweide'):
     # transform healpix map to 2d array
    colors=['#348ABD', '#A60628', 'tab:blue','#467821','#D55E00','#CC79A7','#56B4E9','#009E73','gold','#0072B2']

    handles = None
    pixarea = hp.nside2pixarea(hp.get_nside(hmap), degrees=True)
    m = hp.ma(hmap / pixarea) if whmap is None else hp.ma(whmap / hmap)
    mask_2 = np.zeros(len(m))
    mask_2[m <= 0] = 1
    m.mask=mask_2
    map_to_plot = hp.cartview(m, nest=nest, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()
    # build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1] + 1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0] + 1)

    ra_edge[ra_edge > 180] -= 360    # scale conversion to [-180, 180]
    ra_edge = -ra_edge               # reverse the scale: East to the left
    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96, top=0.90)
    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap='viridis', edgecolor='none', lw=0)
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array([f'{lab}°' for lab in tick_labels])
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_xlabel('R.A. [deg]', labelpad=12, fontsize=12)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Dec. [deg]', fontsize=12, labelpad=12)
    if label is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, -0.1, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.9, aspect=40, ticks=None)
        cb.outline.set_visible(False)
        cb.set_label(label, x=0.5, labelpad=12, fontsize=12)
        cb.ax.tick_params(size=0)
    ax.grid(True)
    plt.tight_layout()
    return ax, mesh

##### Functions for full-shape #####

def plot_observable(self, ax_top=None, ax_bottom=None, **plot_kwargs):
    """
    Plot the observable into provided axes
    """
    show_legend = True
    corr_type = plot_kwargs.get('corr_type', 'pk')
    color = plot_kwargs.get('color', f'C0')
    linestyle = plot_kwargs.get('linestyle', '-')
    fmt = plot_kwargs.get('fmt', 'o')
    (tracer, i, plot_sysmodel) = (plot_kwargs[key] for key in ["tracer", "index", "sys_model"])
    data, theory, std = self.data, self.theory, self.std
    if corr_type == 'xi':
        # Plot the observable (top panel)
        for ill, ell in enumerate(self.ells):
            ax_top.errorbar(self.s[ill], self.s[ill]**2 * data[ill], yerr=self.s[ill]**2 * std[ill],
                            color=color, linestyle='none', marker='o')
            ax_top.plot(self.s[ill], self.s[ill]**2 * theory[ill], color = color, ls = linestyle, label = label)
        ax_top.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if show_legend:
            ax_top.legend()
        ax_top.grid(True)
        # Plot the residuals (bottom panel)
        for ill, ell in enumerate(self.ells):
            ax_bottom.plot(self.s[ill], (data[ill] - theory[ill]) / std[ill], color = color, ls = linestyle)
            ax_bottom.set_ylim(-4, 4)
            for offset in [-2., 2.]:
                ax_bottom.axhline(offset, color='k', linestyle='--')
            ax_bottom.set_ylabel(r'$\Delta \xi_{{{0:d}}} / \sigma_{{ \xi_{{{0:d}}} }}$'.format(ell))
        ax_bottom.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
        ax_bottom.grid(True)
    if corr_type == 'pk':
        # Plot the observable (top panel)
        for ill, ell in enumerate(self.ells):
            if plot_sysmodel == 'standard':
                label = f'{tracer}{i+1} std'
                if ell == 2: label = None
                ax_top.errorbar(self.k[ill], self.k[ill] * data[ill], yerr=self.k[ill] * std[ill],
                                color=color, fmt = fmt, label = label, markersize= 4)
                ax_top.plot(self.k[ill], self.k[ill] * theory[ill], color = color, ls = linestyle)
            if plot_sysmodel == 'dv-obs':
                label = f'{tracer}{i+1} dv-obs'
                if ell == 2: label = None
                ax_top.errorbar(self.k[ill], self.k[ill] * data[ill], yerr=self.k[ill] * std[ill],
                                color=color, fmt = fmt, label = label, markerfacecolor='none', markersize= 4)
                ax_top.plot(self.k[ill], self.k[ill] * theory[ill], color = color, ls = linestyle)
        if show_legend:
            ax_top.legend(loc=1)
        ax_top.set_ylabel(r'$k P_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{-1}$]')
        # Plot the residuals (bottom panel)
        for ill, ell in enumerate(self.ells):
            ax_bottom[ill].plot(self.k[ill], (data[ill] - theory[ill]) / std[ill], color = color, ls = linestyle)
            ax_bottom[ill].set_ylim(-4, 4)
            for offset in [-2., 2.]:
                ax_bottom[ill].axhline(offset, color='k', linestyle='--')
            ax_bottom[ill].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
        ax_bottom[1].set_xlabel(r'$k$ [h/$\mathrm{Mpc}$]')

def plot_observable_bao(self, ax_top=None, ax_bottom=None, **plot_kwargs):
    """
    Plot data and theory BAO correlation function peak.
    """
    lax = [ax_top, ax_bottom]
    if ax_bottom == None:
        lax = [ax_top]
    show_legend = False
    color = plot_kwargs.get('color', f'C0')
    linestyle = plot_kwargs.get('linestyle', '-')
    fmt = plot_kwargs.get('fmt', 'o')
    plot_sysmodel =  plot_kwargs.get('sys_model', 'standard')
    data, theory, std = self.data, self.theory, self.std
    nobao = self.theory_nobao
    for ill, ell in enumerate(self.ells):
        if plot_sysmodel == 'standard':
            # lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill] - nobao[ill]), yerr=self.s[ill]**2 * std[ill], 
            #                 color=color, fmt = fmt, markersize= 6, label = 'std')
            lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill]), yerr=self.s[ill]**2 * std[ill], 
                            color=color, fmt = fmt, markersize= 6, label = 'std')
        elif plot_sysmodel == 'dv-obs':
            lax[ill].errorbar(self.s[ill], self.s[ill]**2 * (data[ill]), yerr=self.s[ill]**2 * std[ill], 
                            color=color, fmt = fmt, markersize= 6, markerfacecolor='none', label = 'dv')
        lax[ill].plot(self.s[ill], self.s[ill]**2 * (theory[ill]), 
                      color=color, linestyle = linestyle)
        lax[ill].set_ylabel(r'$s^{{2}} \Delta \xi_{{{:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(ell))
        if ill == 0:
            lax[ill].legend(loc =3, ncol=2)
        # if 2 not in self.ells:
            # lax[ill].legend(loc =3, ncol=2)
    for ax in lax: ax.grid(True)
    lax[-1].set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')

def make_key(tracer, region, zerr, stats, theory=None, pk_kmax=None, bk_kmax=None, **kwargs):
    if pk_kmax is None or bk_kmax is None:
        stat_label = 'S2+S3' if 'mesh3' in stats else 'S2'
        return f'{tracer}_{stat_label}_{zerr}'    
    if 'mesh3' in stats:
        return f'{tracer}_{theory[:4]}_S2_{pk_kmax:.2f}_S3_{bk_kmax:.2f}'
    return f'{tracer}_{theory[:4]}_S2_{pk_kmax:.2f}'

def _read_legacy_npy_chain(fn):
    """Read old desilike ``chain_*.npy`` files into current MCSamples."""
    from desilike.parameter import Variable
    from desilike.samples import MCSamples as Chain

    state = np.load(fn, allow_pickle=True).item()
    if not isinstance(state, dict) or 'data' not in state:
        raise ValueError(f'Unsupported legacy chain format in {fn!r}')

    variables = []
    first = state['data'][0]
    first_shape = tuple(first.get('param', {}).get('shape', ()))
    sample_shape = np.shape(first['value'])[:None if not first_shape else -len(first_shape)]
    for item in state['data']:
        param = dict(item.get('param', {}))
        name = '.'.join(part for part in [param.get('namespace', ''), param.get('basename', '')] if part)
        if not name:
            continue
        values = np.asarray(item['value'])
        shape = values.shape[len(sample_shape):]
        variable = Variable(name=name, value=param.get('value', None),
                            derived=bool(param.get('derived', False)),
                            latex=param.get('latex', None),
                            shape=shape)
        variable._value = values
        variables.append(variable)
    return Chain(data=variables, attrs=state.get('attrs', {}))


def read_chain(fn, remove_burnin=0.6, step=2, ravel=True):
    """Read sampler chain files with the fiducial desilike MCSamples reader."""
    from desilike.samples import MCSamples as Chain

    if isinstance(fn, (list, tuple)):
        chains = [read_chain(one_fn, remove_burnin=remove_burnin, step=step, ravel=ravel) for one_fn in fn]
        if not chains:
            raise ValueError('No chain files provided.')
        if len(chains) == 1:
            return chains[0]
        return Chain.concatenate(*chains)
    fn = str(fn)
    chain = _read_legacy_npy_chain(fn) if fn.endswith('.npy') else Chain.read(fn)
    chain = chain.remove_burnin(remove_burnin)[::step]
    if ravel:
        chain = chain.ravel()
    return chain

def chain_fit_results(chain, params=None, q=(0.1587, 0.8413)):
    """Return mean/median/central-interval summaries for chain parameters."""
    fit_results = {}
    if hasattr(chain, 'params'):
        available = {str(param) for param in chain.params()}
        default_params = chain.params(varied=True)
    else:
        available = {str(param) for param in chain.names()}
        try:
            default_params = chain.names(derived=False)
        except TypeError:
            default_params = chain.names()
    if params is None:
        params = default_params
    for param in params:
        param = str(param)
        if param not in available:
            continue
        qlo, qhi = chain.quantile(params=param, q=q)
        fit_results[param] = {
            'mean': float(chain.mean(params=param)),
            'median': float(chain.median(params=param)),
            'q16': float(qlo),
            'q84': float(qhi),
        }
    return fit_results

def plot_mcmc_walkers(chain, params, nwalkers, true_values = None):
    from desilike.samples import plotting, Chain
    from getdist import plots
    ndim            = len(params)
    chain_samples   = dict(zip(chain.basenames(), chain.data))
    samples         = np.array([chain_samples[p] for p in params])
    medians         = np.array(chain.median(params=params))
    # true_values     = set_true_values(params)
    fig, ax = plt.subplots(ndim, sharex=True, figsize=(16, 2 * ndim))
    for i in range(nwalkers):
        for j in range(ndim):
            ax[j].plot(samples[j, :, i], c = 'green', lw=0.3)
            ax[j].set_ylabel(params[j], fontsize=15)
            ax[j].grid(True)
            ax[j].axhline(medians[j], c='blue', lw=1.2)
            if true_values != None:
                ax[j].axhline(true_values[j], c='red', lw=1.2)


def _fit_label_from_args(data_args):
    use_dv = data_args.get('use_dv', 'None')
    if use_dv in [None, False, 'None', 'False']:
        label = 'standard'
    else:
        label = str(use_dv)
    if data_args.get('z_evol', False):
        label = f'{label} + zevol'
    return label

def _stat_label_from_result(result):
    if 'stat_label' in result:
        return str(result['stat_label'])
    if 'stats' in result:
        return '+'.join(result['stats'])
    return 'unknown'

def _stat_cmap(stat_label, stat_cmaps=None):
    stat_cmaps = stat_cmaps
    return plt.colormaps[stat_cmaps.get(str(stat_label), 'viridis')]

def build_fit_summary_table(chains_dict, params):
    rows = []
    for key, result in chains_dict.items():
        data_args = result.get('args', [{}])[0]
        fit_results = result.get('fit_results', {})
        fit_label = _fit_label_from_args(data_args)
        stat_label = _stat_label_from_result(result)
        row = {
            'key': str(key),
            'tracer': str(data_args.get('tracer', str(key).split('_')[0])),
            'fit_label': fit_label,
            'stat_label': stat_label,
            'comparison_label': f'{fit_label}, {stat_label}',
            'bestfit': fit_results,
        }
        for param in params:
            summary = fit_results.get(param, {})
            row[param] = summary.get('mean', np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _value_from_mapping(values, param):
    if values is None:
        return None
    try:
        value = values[param]
    except Exception:
        try:
            value = values.get(param, None)
        except AttributeError:
            return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        array = np.asarray(value)
        if array.size == 0:
            return None
        return float(array.reshape(-1)[0])


def profile_fit_results(profiles, params=None):
    """Return best-fit/error summaries from a desilike Profiles object."""
    try:
        profile = profiles.choice(index='argmax', squeeze=True)
    except Exception:
        profile = profiles
    best = getattr(profile, 'best', None)
    if best is None:
        best = getattr(profile, 'bestfit', None)
    error = getattr(profile, 'error', None)

    if params is None:
        params = []
        try:
            params = [str(param) for param in best.params()]
        except Exception:
            try:
                params = [str(param) for param in best.keys()]
            except Exception:
                params = []

    fit_results = {}
    for param in params:
        param = str(param)
        value = _value_from_mapping(best, param)
        if value is None:
            continue
        summary = {'best': value}
        err = _value_from_mapping(error, param)
        if err is not None:
            summary['error'] = err
        fit_results[param] = summary
    return fit_results


def build_profile_summary_table(profiles_dict, params):
    rows = []
    for key, result in profiles_dict.items():
        data_args = result.get('args', [{}])[0]
        profiles = result.get('profiles', result)
        fit_results = result.get('fit_results')
        if fit_results is None:
            fit_results = profile_fit_results(profiles, params=params)
        fit_label = _fit_label_from_args(data_args)
        stat_label = _stat_label_from_result(result)
        row = {
            'key': str(key),
            'tracer': str(data_args.get('tracer', str(key).split('_')[0])),
            'fit_label': fit_label,
            'stat_label': stat_label,
            'comparison_label': f'{fit_label}, {stat_label}',
            'bestfit': fit_results,
        }
        for param in params:
            summary = fit_results.get(param, {})
            row[param] = summary.get('best', np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_yerr(summary, center='mean', error='std', low='q16', high='q84'):
    if not summary or center not in summary:
        return None
    y = float(summary[center])
    if error in summary:
        return float(summary[error])
    if low in summary and high in summary:
        return np.array([[max(0.0, y - float(summary[low]))], [max(0.0, float(summary[high]) - y)]])
    return None


def _display_comparison_label(fit_label, stat_label, fit_label_display=None, stat_label_display=None):
    fit_label_display = fit_label_display or {}
    stat_label_display = stat_label_display or {}
    return f'{fit_label_display.get(fit_label, fit_label)}, {stat_label_display.get(stat_label, stat_label)}'


def _combo_style(fit_index, nfit, cmap):
    # Stat selects the color family; zerr selects shade within that family.
    fit_norm = 0.0 if nfit <= 1 else fit_index / (nfit - 1)
    shade = 0.7 - 0.3 * fit_norm
    color = cmap(shade)
    return color

def plot_chain_comparison(chains_dict, params=('H0', 'Omega_m', 'sigma8_m'), tracers=None,
                          fit_labels=None, stat_labels=None, center='mean', error='std',
                          low='q16', high='q84', markers=('o', 's', 'D', '^', 'v'),
                          offset_width=0.12, fit_label_display=None, stat_label_display=None,
                          stat_cmaps=None, profiles_dict=None, profile_marker='o',
                          profile_color='black', profile_offset=0.01,
                          profile_label='', true_values=None, reference_label='Planck'):
    table = build_fit_summary_table(chains_dict, params=params)
    if table.empty:
        raise ValueError('No chain results available to plot')
    profile_table = None
    if profiles_dict:
        profile_table = build_profile_summary_table(profiles_dict, params=params)

    if tracers is None:
        tracers = list(table['tracer'].dropna().unique())
    else:
        tracers = [str(tracer) for tracer in tracers]
        table = table.loc[table['tracer'].isin(tracers)]

    if fit_labels is None:
        fit_labels = list(table['fit_label'].dropna().unique())
    else:
        fit_labels = [str(label) for label in fit_labels]
        table = table.loc[table['fit_label'].isin(fit_labels)]

    if stat_labels is None:
        stat_labels = list(table['stat_label'].dropna().unique())
    else:
        stat_labels = [str(label) for label in stat_labels]
        table = table.loc[table['stat_label'].isin(stat_labels)]

    comparison_labels = [
        f'{fit_label}, {stat_label}'
        for stat_label in stat_labels
        for fit_label in fit_labels
    ]
    comparison_labels = [label for label in comparison_labels if label in set(table['comparison_label'])]

    params = [str(param) for param in params if str(param) in table.columns]
    if not params:
        raise ValueError('None of the requested parameters are available')
    if not tracers:
        raise ValueError('No tracers found')
    if not comparison_labels:
        raise ValueError('No fit/stat combinations found')

    fig, axes = plt.subplots(len(params), 1, figsize=(8.0, 1.5 * len(params) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    x_positions = np.arange(len(tracers))
    offsets = np.linspace(-offset_width, offset_width, len(comparison_labels))
    points_plotted = 0
    profile_points_plotted = 0

    if true_values is None:
        true_values = PLANCK_COSMOLOGY
    for ax, param in zip(axes, params):
        labeled_combinations = set()
        for icombo, comparison_label in enumerate(comparison_labels):
            fit_label, stat_label = comparison_label.split(', ', 1)
            ifit = fit_labels.index(fit_label)
            istat = stat_labels.index(stat_label)
            display_label = _display_comparison_label(
                fit_label, stat_label,
                fit_label_display=fit_label_display,
                stat_label_display=stat_label_display,
            )

            for itracer, tracer in enumerate(tracers):
                rows = table.loc[
                    (table['tracer'] == tracer)
                    & (table['fit_label'] == fit_label)
                    & (table['stat_label'] == stat_label)
                ]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                summary = row['bestfit'].get(param, {})
                if center not in summary:
                    continue

                x = x_positions[itracer] + offsets[icombo]
                cmap = _stat_cmap(stat_label, stat_cmaps=stat_cmaps)
                color = _combo_style(ifit, len(fit_labels), cmap)
                marker = markers[ifit % len(markers)]
                y = float(summary[center])
                yerr = _summary_yerr(summary, center=center, error=error, low=low, high=high)
                ax.errorbar(x, y, yerr=yerr, color=color,
                            marker=marker, markerfacecolor=color, 
                            markeredgecolor=color, markeredgewidth=1.1,
                            ls='', capsize=2.5, lw=1.0, ms=5.5,
                            label=display_label if comparison_label not in labeled_combinations else None,)
                labeled_combinations.add(comparison_label)
                points_plotted += 1

                if profile_table is not None:
                    profile_rows = profile_table.loc[
                        (profile_table['tracer'] == tracer)
                        & (profile_table['fit_label'] == fit_label)
                        & (profile_table['stat_label'] == stat_label)
                    ]
                    if not profile_rows.empty:
                        profile_summary = profile_rows.iloc[0]['bestfit'].get(param, {})
                        if 'best' in profile_summary:
                            ax.errorbar(
                                x + profile_offset, float(profile_summary['best']),
                                yerr=None, color=profile_color,
                                marker=profile_marker, markerfacecolor='none',
                                markeredgecolor=profile_color, markeredgewidth=1.0,
                                ls='', lw=0.9, ms=4.5,
                                label=profile_label if profile_points_plotted == 0 else None,
                            )
                            profile_points_plotted += 1
        if param in true_values and np.isfinite(true_values[param]):
            ax.axhline(true_values[param], color='gray', ls='--', lw=1.0, label=reference_label)
        
        ax.set_ylabel(PARAM_LABELS.get(param, param), fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tracers, fontsize=12)
        ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
        ax.grid(axis='y', alpha=0.2)

    if points_plotted == 0:
        raise ValueError('No matching fit-summary points found to plot')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), fontsize=10,
                   ncol=min(len(comparison_labels), 4), frameon=False)
    fig.tight_layout()
    return fig
