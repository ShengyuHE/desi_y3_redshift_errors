
import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import numpy as np
from getdist import plots
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from desilike.samples import plotting, Chain

sys.path.append('../')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX, PLANCK_COSMOLOGY

line_confusion_limit = dict(BGS = [0, 2.0], LRG = [0, 1.7], ELG = [0, 1.9], QSO = [0, 4.2],)

def plot_confusion_lines(ax, line_set, name_set, focus = 'Mg[II]', remove = None):
    alpha = 1.0
    if focus not in name_set:
        raise ValueError(f"Focus line '{focus}' not found in names list.")
    if remove != None:
        lines, names = zip(*[(l, n) for l, n in zip(line_set, name_set) if n not in remove])
    else:
        lines, names = line_set, name_set
    # Loop over all possible pairs
    x = np.linspace(-0.01, 4.0, 2)
    if focus != None:
        focus_idx = names.index(focus)
        colormap = get_cmap('plasma')
        colors = [tuple(c) for c in colormap(np.linspace(0, 1, 7))] 
        if focus != 'C[IV]':
             colors = colors[1:]
        for j, (lam, name) in enumerate(zip(lines, names)):
            if name == focus:
                continue
            # case 1: focus line mistaken for others
            y1 = lines[focus_idx]/lam * (1+x) - 1
            ax.plot(x, y1, '--', color=colors[j], lw=1.0, label=f'{focus}'+r'$\longleftrightarrow$'+f'{name}', alpha=alpha)
            # case 2: others mistaken for focus
            y2 = lam/lines[focus_idx] * (1+x) - 1
            ax.plot(x, y2, '--', color=colors[j], lw=1.0, alpha=alpha)
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(lines) * (len(lines)-1)))
        for k, (i, j) in enumerate([(i, j) for i in range(len(lines)) for j in range(len(lines)) if i != j]):
            true, false = lines[i], lines[j]
            y = true/false * (1 + np.linspace(-0.01, 3.0, 2)) - 1
            ax.plot(x, y, ':', color=colors[k], lw=0.3, alpha=alpha,
                    label=f'{names[i]}'+r'$\leftrightarrow$'+f'{names[j]}')

def plot_sky_residuals(ax, residuals):
    """
    Plot vertical and horizontal lines at given redshift values on an Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the lines.
    residuals : sequence of float
        List/array of redshift values where sky residuals are suspected.
    """
    colors = ['black','purple']
    for i,z in enumerate(residuals):
        this_label = f'skyres z'+r'$\approx$'+f'{z:.2f}'
        ax.axhline(z, color=colors[i], lw=1.2, ls=':', label=this_label)
        ax.axvline(z, color=colors[i], lw=1.2, ls=':')

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
    chain.set(chain['Omega_m'].clone(value=(chain['omega_cdm'] + chain['omega_b'] + PLANCK_COSMOLOGY['omega_ncdm'])/chain['h']**2, param={'basename': 'Omega_m', 'derived': True, 'latex': r'\Omega_m'}))
    chain.set(chain['H0'].clone(value=(chain['h']*100), param={'basename': 'H0', 'derived': True, 'latex': r'H_0'}))
    return 0

def read_bao_chain(filename, burnin=0.5, slice_step=1, apmode='qisoqap'):
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

