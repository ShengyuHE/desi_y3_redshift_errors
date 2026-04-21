
import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

sys.path.append('../')
from helper import PLANCK_COSMOLOGY

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

def get_spec_lines(tracer):
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
    CONF = {
        "BGS": {
            r"[S$\,\mathrm{II}$]": 6717,
            r"[N$\,\mathrm{II}$]": 6548,
            r"[H$\,\alpha$]": 6563,
            r"[H$\,\beta$]": 4861,
            r"[O$\,\mathrm{III1}$]": 4959,
            r"[O$\,\mathrm{III2}$]": 5007,
            r"[O$\,\mathrm{III}$]": 5003,
            r"[O$\,\mathrm{II}$]": 3727,
            r"[Mg$\,\mathrm{II}$]": 2800,
        },
        "QSO": {
            r"[Mg$\,\mathrm{II}$]": 2800,
            r"[H$\,\alpha$]": 6563,
            r"[H$\,\beta$]": 4861,
            r"[H$\,\gamma$]": 4340,
            r"[C$\,\mathrm{III}$]": 1908,
            r"[C$\,\mathrm{IV}$]": 1549,
            r"[Ly$\,\alpha$]": 1215.67,
            r"[O$\,\mathrm{III}$]": 5003,
            r"[O$\,\mathrm{II}$]": 3727,
        },
        "LRG": {},
        "ELG": {},
    }
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

COLOR_TRACER_GRADIENT = dict(BGS='Greens', LRG='Reds',ELG='Blues', QSO='Purples')

TPS_LABELS = dict(xi ={'x':r"$s\,[h^{-1}\mathrm{Mpc}]$",'y':r"$s^2\xi_\ell(s)$", 'dy0':r"$\Delta\xi_0/\sigma$", 'dy2':r"$\Delta\xi_2/\sigma$"},
                  pk ={'x':r"$k\,[\mathrm{Mpc}^{-1}h]$",'y':r"$kP_\ell(k)$", 'dy0':r"$\Delta P_0/\sigma$", 'dy2':r"$\Delta P_2/\sigma$"},
                  mpslog ={'x':r"$s\,[h^{-1}\mathrm{Mpc}]$",'y':r"$s^2\xi_\ell(s)$", 'dy0':r"$\Delta\xi_0/\sigma$", 'dy2':r"$\Delta\xi_2/\sigma$"},
                  wplog ={'x':r"$r_p$",'y':r"$r_p w_P$", 'dy0':r"$\Delta w_p/\sigma$"},
                  mesh2 = {'x':r"$k\,[\mathrm{Mpc}^{-1}h]$",'y':r"$kP_\ell(k)$", 'dy0':r"$\Delta P_0/\sigma$", 'dy2':r"$\Delta P_2/\sigma$"},
                  mesh3_sugiyama= {'x':r"$k\,[\mathrm{Mpc}^{-1}h]$",'y':r"$k^2B_\ell(k)$", 'dy0':r"$\Delta B_{000}/\sigma$", 'dy2':r"$\Delta B_{202}/\sigma$"}
                  )

##### Functions #####
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
        if focus in [r"[C$\,\mathrm{IV}$]"]:
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
    colors = ["#3B8B7A" ,"#5555d2", "#9540e5",]
    for i,z in enumerate(residuals):
        if label == None:
            this_label = f'skyres z'+r'$\approx$'+f'{z:.2f}'
        else:
            this_label = label 
        ax.axhline(z, color=colors[i], lw=1.5, ls=':', label=this_label)
        ax.axvline(z, color=colors[i], lw=1.5, ls=':')

def plot_conf_dots(ax, conf, cols = ['Z1','Z2']):
    for key, rows in conf.items(): 
        if isinstance(rows, dict):
            rows = [rows]
        z1 = np.array([r[cols[0]] for r in rows], dtype=float)
        z2 = np.array([r[cols[1]] for r in rows], dtype=float)
        ax.plot(z1, z2, "o", alpha=0.6, markersize=3, markerfacecolor="none", markeredgecolor="k",)

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

def convert_chain(chain):
    from desilike.samples import plotting, Chain
    chain.set(chain['Omega_m'].clone(value=(chain['omega_cdm'] + chain['omega_b'] + PLANCK_COSMOLOGY['omega_ncdm'])/chain['h']**2, param={'basename': 'Omega_m', 'derived': True, 'latex': r'\Omega_m'}))
    chain.set(chain['H0'].clone(value=(chain['h']*100), param={'basename': 'H0', 'derived': True, 'latex': r'H_0'}))
    return 0

def read_bao_chain(filename, burnin=0.5, slice_step=1, apmode='qisoqap'):
    from desilike.samples import plotting, Chain
    if isinstance(filename, list):
        chains = []
        for fn in filename:
            chains.append(Chain.load(fn))
        chain = chains[0].concatenate([chain.remove_burnin(burnin)[::slice_step] for chain in chains])
    else:
        chain = Chain.load(filename)
        chain = chain.remove_burnin(burnin)[::slice_step]
    if apmode == 'qparqper':
        qiso = (chain['qpar']**(1./3.) * chain['qper']**(2./3.)).clone(param=dict(basename='qiso', derived=True, latex=r'q_{\rm iso}'))
        qap = (chain['qpar'] / chain['qper']).clone(param=dict(basename='qap', derived=True, latex=r'q_{\rm AP}'))
        chain.set(qiso)
        chain.set(qap)
    if apmode == 'qisoqap':
        qpar = (chain['qiso'] * chain['qap']**(2/3)).clone(param=dict(basename='qpar', derived=True, latex=r'q_{\parallel}'))
        qper = (chain['qiso'] * chain['qap']**(-1/3)).clone(param=dict(basename='qper', derived=True, latex=r'q_{\perp}'))
        chain.set(qpar)
        chain.set(qper)
    alpha_iso = chain['qiso'].clone(param=dict(basename='alpha_iso', derived=True, latex=r'(D_{\mathrm{V}}/r_{d})/(D_{\mathrm{V}}/r_{d})^{\rm fid}'))
    chain.set(alpha_iso)
    if apmode in ['qisoqap', 'qparqper']:
        alpha_ap = chain['qap'].clone(param=dict(basename='alpha_ap', derived=True, latex=r'(D_{\mathrm{H}}/D_{\mathrm{M}})/(D_{\mathrm{H}}/D_{\mathrm{M}})^{\rm fid}'))
        alpha_par = chain['qpar'].clone(param=dict(basename='alpha_par', derived=True, latex=r'(D_{\mathrm{H}}/r_{d})/(D_{\mathrm{H}}/r_{d})^{\rm fid}'))
        alpha_per = chain['qper'].clone(param=dict(basename='alpha_per', derived=True, latex=r'(D_{\mathrm{M}}/r_{d})/(D_{\mathrm{M}}/r_{d})^{\rm fid}'))
        chain.set(alpha_ap)
        chain.set(alpha_par)
        chain.set(alpha_per)
    return chain

def plot_DM():
    return 0

def plot_mcmc_contour(chain, params, plot_args=None):
    from desilike.samples import plotting, Chain
    from getdist import plots
    g = plots.get_subplot_plotter()
    g.settings.fig_width_inch= 8
    g.settings.legend_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.axes_fontsize = 16
    g.settings.figure_legend_frame = False
    plotting.plot_triangle(chain, title_limit=1, filled = True, params = params,
                            #    legend_labels = labels, legend_loc= 'upper right',
                                contour_lws = 1.5,
                                # contour_ls = lss, contour_lws = lws, contour_colors = colors, 
                                # param_limits=param_limits, 
                                smoothed=True, show=False, g=g)
    # true_values     = set_true_values(params)
    # for i in range(len(true_values)):
    #     for j in range(i+1):
    #         g.subplots[i,j].axvline(true_values[j], c = 'k', ls = ':', lw = 1.2)
    #         if i != j:
    #             g.subplots[i,j].axhline(true_values[i], c = 'k', ls = ':', lw = 1.2)

