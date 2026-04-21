import os
# os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import logging
import numpy as np
import lsstypes as types

from cosmoprimo.fiducial import DESI, AbacusSummit
sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_GLOBAL, REDSHIFT_BIN_LSS, REDSHIFT_ABACUSHF_V1
from cat_tools import get_measurement_fn

def load_bins(corr_type, bins_type = 'test'):
    if corr_type == 'xi':
        if bins_type in ['test', 'y3_sys']:
            rmin, rmax, rbin, lenr = 20, 200, 4, 45
        elif bins_type in ['y3_bao']:
            rmin, rmax, rbin, lenr = 60, 150, 4, 23
        return (rmin, rmax, rbin, lenr)
    elif corr_type in ['pk', 'mesh2']:
        if bins_type in ['y3_bao', 'test', 'y3_sys']:
            kmin, kmax, kbin, lenk = 0.02, 0.3, 0.005, 56
        elif bins_type in ['y3_fs']: 
            kmin, kmax, kbin, lenk = 0.02, 0.2, 0.005, 36
        elif bins_type in ['test_covbox']:
            kmin, kmax, kbin, lenk = 0.03, 0.2, 0.005, 34     
        return (kmin, kmax, kbin, lenk)
    elif corr_type == 'mpslog':
        if bins_type in ['test', 'y3_sys']:
            smin, smax = 0.10, 30
        return (smin, smax, None, None)
    elif corr_type == 'wplog':
        if bins_type in ['test', 'y3_sys']:
            rpmin, rpmax = 0.10, 30
        return (rpmin, rpmax, None, None)
    elif corr_type == 'mesh3_sugiyama':
        if bins_type in ['test', 'y3_sys']:
            kmin, kmax, kbin, lenk = 0, 0.2, 0.01, 20 #Sigiyama space
        return (kmin, kmax, kbin, lenk)
    elif corr_type == 'mesh3_scoccimarro':
        return (None, None, None, None)
    else:
        raise ValueError(f"Invalid corr_type '{corr_type}'. Expected one of ['xi', 'pk', 'mpslog', 'wp', 'mesh2', 'mesh3_sugiyama', 'mesh3_scoccimarro'].")

def read_data_from_fn(fn, corr_type, bin_type = 'test', ells = (0,2)):
    _min, _max, _bin, _len = load_bins(corr_type, bin_type)
    bin_set = (_min, _max, _bin, _len)
    if corr_type in ['xi', 'mpslog','wplog']:
        if corr_type == 'xi': corr_type = 'xipoles'
        result = TwoPointCorrelationFunction.load(fn.format(corr_type))
        result = result[::_bin,::]
        result.select((_min, _max))
        if corr_type in ['xipoles', 'mpslog']:
            s, xi  = project_to_multipoles(result, ells=[0,2])
        elif corr_type in ['wplog']:
            s, xi = project_to_wp(result)
        return (s, xi), bin_set
    elif corr_type in ['pk']:
        if corr_type == 'pk': corr_type = 'pkpoles'
        result = PowerSpectrumMultipoles.load(fn.format(corr_type))
        result = result.select((_min,_max,_bin))
        pk = np.real(result.get_power())
        k = result.kavg
        return (k, pk), bin_set
    elif corr_type in ['mesh2']:
        result = types.read(fn.format('mesh2_spectrum_poles'))
        sl = slice(0, None, 5)  # rebin to dk = 0.005 h/Mpc
        oklim = (_min, _max)  # fitted k-range, no need to go to higher k
        result = result.select(k=sl).select(k=oklim)
        k = result.get(ells=0).coords('k')
        pk = [result.get(ells=ell).values()['value'] for ell in result.ells]
        return (k, pk), bin_set
    elif corr_type in ['mesh3_scoccimarro']:
        result = types.read(fn.format('mesh3_spectrum_poles_scoccimarro'))
        k = result.get(ells=0).coords('k')
        k1, k2, k3 = np.asarray(k).T
        bk = [result.get(ells=ell).values()['value'] for ell in result.ells]
        return ((k1, k2, k3), bk), bin_set
    elif corr_type in ['mesh3_sugiyama']:
        result = types.read(fn.format('mesh3_spectrum_poles_sugiyama'))
        sl = slice(0, None, 1)
        oklim = (_min, _max)
        result = result.select(k=sl).select(k=oklim)
        k = result.get(ells=(0, 0, 0)).coords('k')[:,1]
        bk = {ell: result.get(ells=ell).values()['value'] for ell in result.ells}
        return (k, bk), bin_set
    else:
        ValueError(f"{corr_type} not available")

def load_data_fns(args, corr_type = 'xi', data_type='cubic_sys', bins_type = None):
    return 0
    
def load_cov(args, corr_type= 'xi', cov_type = 'EZmocks_fn', bins_type = 'test', ells=(0,2)):
    return 0


def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial


def _get_default_ref_from_prior(prior, value=None):
    """Build a compact reference distribution from a prior for sampler initialization."""
    if not prior: return None
    dist = prior.get('dist', None)
    limits = prior.get('limits', [-np.inf, np.inf])
    if limits is None: limits = [-np.inf, np.inf]
    limits = list(limits)
    if dist == 'norm':
        scale = prior.get('scale', None)
        if scale is None or scale <= 0:
            return None
        return {'dist': 'norm',
                'loc': prior.get('loc', value if value is not None else 0.),
                'scale': scale / 5.,
                'limits': limits,}
    
    if dist == 'uniform':
        if len(limits) != 2 or not np.all(np.isfinite(limits)):
            return None
        lo, hi = limits
        return {'dist': 'norm',
                'loc': value if value is not None else 0.5 * (lo + hi),
                'scale': (hi - lo) / 20.,
                'limits': [lo, hi],
        }
    return None

def _get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=True, tracer=None, sigma8_fid=1.):
    """
    Build a dictionary of parameter priors.

    Parameters
    ----------
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2_spectrum', 'mesh2_spectrum'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float, optional
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        :meth:`Parameter.update` (e.g. ``{'fixed': True}`` or ``{'prior': {...}}``).
    """
    params = {}
    scale_eft = 12.5
    scale_sn0 = 2.0
    scale_sn2 = 5.0

    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7. * sigma8_fid**2, 'scale': 5}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3p'] = {'fixed': True}
            else:
                params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23. / 42. * sigma8_fid**4, 'scale': sigma8_fid**4},
                                 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_pp'] = {'fixed': True}
            else:
                params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3'] = {'fixed': True}
            else:
                params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}, 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_p'] = {'fixed': True}
            else:
                params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            shotnoise = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    for config in params.values():
        if config.get('fixed', False):
            continue
        ref = _get_default_ref_from_prior(config.get('prior', None), value=config.get('value', None))
        if ref is not None and 'ref' not in config:
            config['ref'] = ref
    return params

def get_theory(stat: str, theory_options: dict, cosmology: object=None, data_attrs: dict=None, data=None):
    """
    Return a configured theory desilike calculator for the requested statistic.

    Parameters
    ----------
    stat : str
        Statistic name, e.g. 'mesh2_spectrum' or 'mesh3_spectrum'.
    theory_options : dict
        Theory options dict containing at least 'model' and possibly other keys.
    cosmology : Cosmoprimo
        Cosmology calculator.
    data_attrs : dict
        Data attributes ('z', 'recon_mode', 'recon_smoothing_radius', 'tracers', ...).

    Returns
    -------
    theory : BaseCalculator
        Initialized theory object from desilike for the requested statistic.
    """
    from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, BAOPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)
    theory_options = dict(theory_options)
    fiducial = get_fiducial()
    template = None
    theory_options.setdefault('cosmology', {'template': 'direct'})
    cosmology_options = theory_options['cosmology']
    z = data_attrs['z']
    if cosmology_options['template'] == 'direct':
        template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmology, z=z)
    elif cosmology_options['template'] == 'shapefit':
        template = ShapeFitPowerSpectrumTemplate(fiducial=fiducial, z=z)
    elif cosmology_options['template'] == 'bao':
        kw = {name: cosmology_options[name] for name in ['apmode', 'now'] if name in cosmology_options}
        template = BAOPowerSpectrumTemplate(fiducial=fiducial, z=z, **kw)
    if template is None:
        raise ValueError(f'template not found for {stat} and {repr(cosmology_options["template"])}')
    theory = None
    if 'mesh2_spectrum' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis', 'b3_coev', 'A_full']}
            theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
            if theory_options['marg']:
                for param in theory.init.params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
    elif 'mesh3_spectrum' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis']}
            theory = FOLPSv2TracerBispectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory


