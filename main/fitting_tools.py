import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import logging
import numpy as np

from cosmoprimo.fiducial import DESI, AbacusSummit
from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction, utils
from pypower import CatalogFFTPower, PowerSpectrumMultipoles
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, StandardPowerSpectrumTemplate
from desilike.theories.galaxy_clustering import FOLPSTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, TNSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, BAOCompressionObservable
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX, RSF_COV_ERROR, RSF_EZMOCKS_ERROR, EDGES, GET_RECON_BIAS
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX
Z_VSMEAR = REDSHIFT_LSS_VSMEAR
Z_CUBIC = REDSHIFT_LSS_CUBICBOX

def load_bins(corr_type, bins_type = 'test'):
    if corr_type == 'xi':
        if bins_type in ['test']:
            rmin, rmax, rbin, lenr = 20, 200, 4, 45
        elif bins_type in ['y3_bao', 'y3_sys']:
            rmin, rmax, rbin, lenr = 60, 150, 4, 23
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'xi'.")
        return (rmin, rmax, rbin, lenr)
    elif corr_type == 'pk':
        if bins_type in ['y3_bao']:
            kmin, kmax, kbin, lenk = 0.02, 0.3, 0.005, 56
        elif bins_type in ['y3_fs', 'y3_sys', 'test']: 
            kmin, kmax, kbin, lenk = 0.02, 0.2, 0.005, 36
        elif bins_type in ['test_covbox']:
            kmin, kmax, kbin, lenk = 0.03, 0.2, 0.005, 34     
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'pk'.")
        return (kmin, kmax, kbin, lenk)
    elif corr_type == 'mpslog':
        if bins_type in ['test']:
            smin, smax = 0.10, 30
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'mpslog'.")
        return (smin, smax, None, None)
    elif corr_type == 'wp':
        if bins_type in ['test']:
            rpmin, rpmax = 0.10, 30
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'wp'.")
        return (rpmin, rpmax, None, None)
    elif corr_type == 'bk':
        if bins_type in ['test']:
            kmin, kmax, kbin, lenk = 0, 0.2, 0.01, 20 #Sigiyama space
        else:
            raise ValueError(f"Unknown bins_type '{bins_type}' for correlation type 'bk'.")
        return (kmin, kmax, kbin, lenk)
    else:
        raise ValueError(f"Invalid corr_type '{corr_type}'. Expected one of ['xi', 'pk', 'mpslog', 'wp', 'bk'].")

def load_data_fns(args, corr_type = 'xi', data_type='cubic_sys', bins_type = None):
    """
    Dispatch to the correct file-listing helper based on `args`.

    Parameters
    ----------
    args : dict
    corr_type : {'xi', 'pk'}
    data_type:{'cubic_cosmo', 'cubic_sys'}
    bins_type : any, optional --> load_bins function

    Returns
    -------
    list[str]: List of matching file paths.
    """
    data_type = args.get("data_type", 'cubic_sys')
    try:
        tracer, zeff, zmin, zmax, grid_cosmo, sys_model = (args[k] for k in ["tracer", "z_eff", "zmin", "zmax", "grid_cosmo", "sys_model"])
    except KeyError as e:
        raise KeyError(f"[DEBUG] Missing required key {e} for data_type='cubic_sys'.")
    # -------------------------------------------------
    # 1) cubic_cosmo: load cubic cosmo from Abacus HOD cubic mocks
    # -------------------------------------------------
    if data_type == 'cubic_cosmo':
        base_dir = f'/global/cfs/cdirs/desi/users/arocher/Y3/AbacusBox/{tracer}/cosmo_2/z{zeff:.3f}/'
        # load the mocks from Abacus_HOD cubicbox data
        if corr_type == 'xi':
            recon = args.get("recon", False)
            if recon:
                files=glob.glob(base_dir+f'/AbacusSummit_base_c002_ph*/allcounts_rec_c{grid_cosmo}_{tracer}_z{zeff:.3f}_SHOD_kibo-v1_v0.1_PIP.npy')
            else:
                files=glob.glob(base_dir+f'/AbacusSummit_base_c002_ph*/allcounts_c{grid_cosmo}_{tracer}_z{zeff:.3f}_SHOD_kibo-v1_v0.1_PIP.npy')
        else:
            # pk not implemented for cubic_cosmo in the code
            raise ValueError("[DEBUG] 'pk' not implemented for data_type='cubic_cosmo'.")
    # -------------------------------------------------
    # 2) cubic_sys: load contaminated mocks from Abacus HOD cubic
    # -------------------------------------------------
    elif data_type == 'cubic_sys':
        # load the contaminated mocks built from Abacus_HOD cubicbox data
        option = args.get("option", 'None')
        if sys_model not in ['standard', 'dv-obs']:
            raise ValueError(f"[DEBUG] Invalid sys_model: '{sys_model}'. Expected one of ['standard', 'dv-obs']")
        cat_dir = '/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox'
        base_dir = cat_dir+f'/{tracer}/obs_z{zmin:.1f}-{zmax:.1f}/AbacusSummit_base_c{grid_cosmo}_ph*/mpspk'
        print(base_dir)
        if option == 'QSO_test':
            zmin = 0.8
            zmax = 2.1
            base_dir = cat_dir+f'/{tracer}/obs_QSO/AbacusSummit_base_c{grid_cosmo}_ph*/mpspk'
        if corr_type == 'xi':
            recon = args.get("recon", False)
            if recon:
                files=glob.glob(base_dir+f'/xipoles_recon_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}.npy')
            else:
                files=glob.glob(base_dir+f'/xipoles_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}.npy')
            return files
        elif corr_type == 'pk':
            files = glob.glob(base_dir+f'/pkpoles_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}.npy')
    else:
        raise ValueError(f"[DEBUG] Invalid data_type: '{data_type}'.")
    files.sort()
    if len(files) == 0:
        raise ValueError(f"No files found matching data_args")
    return files
    
def load_cov(args, corr_type= 'xi', cov_type = 'EZmocks_fn', bins_type = 'test', ells=(0,2)):
    """
    Unified covariance loader.

    Parameters
    ----------
    args : dict
    corr_type : {'xi','pk'}
    bins_type : str or None
    cov_type : str
        - 'rescalc'     -> RascalC analytical covariance (recon-aware)
        - 'cov_fn'      -> load precomputed EZmocks covariance text file (by bins)
        - 'cov_mock'    -> build from EZmocks or covbox
    ells : tuple

    Returns
    -------
    np.ndarray: Covariance matrix.
    """
    if corr_type not in {'xi', 'pk'}:
        raise ValueError(f"[DEBUG] Invalid corr_type='{corr_type}'. Use 'xi' or 'pk'.")
    try:
        tracer, zeff, zmin, zmax = (args[k] for k in ["tracer", "z_eff", "zmin", "zmax"])
    except KeyError as e:
        raise KeyError(f"[DEBUG] Missing required key {e} for data_type='cubic_sys'.")
    # -------------------------------------------------
    # 1) RascalC: file on disk, only for corr_type='xi'
    # -------------------------------------------------
    if cov_type.lower() in {'rascalc', 'rascalc_fn'}:
        if corr_type == 'xi':
            recon = args.get("recon", False)
            if recon:
                basedir = f'/global/homes/s/shengyu/project_rc/main/Y3/data/covariance/recon_sm15_IFFT_recsym'
                if tracer == 'QSO':
                    basedir = f'/global/homes/s/shengyu/project_rc/main/Y3/data/covariance/recon_sm30_IFFT_recsym_z0.8-2.1'
            else:
                basedir = f'/global/homes/s/shengyu/project_rc/main/Y3/data/covariance'
            if tracer == 'LRG':
                cov_matrix = np.loadtxt(basedir+f'/xi024_LRG_GCcomb_z{zmin}-{zmax}_default_FKP_lin4_s20-200_cov_RascalC.txt')
            if tracer == 'ELG':
                if zmin == 0.8:
                    cov_matrix = np.loadtxt(basedir+f'/xi024_ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_lin4_s20-200_cov_RascalC.txt')
                elif zmin in [1.1,1.3]:
                    cov_matrix = np.loadtxt(basedir+f'/xi024_ELG_LOPnotqso_GCcomb_z1.1-1.6_default_FKP_lin4_s20-200_cov_RascalC.txt')
            if tracer == 'QSO':
                cov_matrix = np.loadtxt(basedir+f'/xi024_QSO_GCcomb_z0.8-2.1_default_FKP_lin4_s20-200_cov_RascalC.txt')
            # Optionally cut to BAO bins
            if bins_type in (None, 'test'):
                return cov_matrix # s: [20,200] with bin=4 in original binning 
            elif bins_type in ['y3_bao', 'y3_sys']:
                mono_indices = np.arange(10, 33)  # s: [60, 150] in BAO fitting
                quad_indices = np.arange(55, 78)
            selected_indices = np.concatenate([mono_indices, quad_indices])
            if tracer in ['BGS', 'QSO']:
                cov_cut = cov_matrix[np.ix_(mono_indices, mono_indices)]
            else:
                cov_cut = cov_matrix[np.ix_(selected_indices, selected_indices)]
            return cov_cut
        else: 
            raise ValueError(f"[DEBUG] pk don't use RascalC analytical covariance")
    # -------------------------------------------------
    # 2) Precomputed EZmocks covariance (single file)
    # -------------------------------------------------
    elif 'fn' in cov_type:
        ez_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/EZmocks/covariance/{corr_type}'
        if corr_type == 'xi':
            raise ValueError(f"[DEBUG] Haven't caluclated the EZmocks covariance for xi")
            recon = args.get("recon", False)
            if recon: ez_dir = ez_dir + '/recon'
            rmin, rmax, rbin, lenr = load_bins(corr_type, bins_type)
            cov_fn = f'{ez_dir}/cov_EZmocks_1.5Gpc_{corr_type}02_{tracer}_z{zmin}-{zmax}_s{rmin}-{rmax}.txt'
        else:  # pk
            kmin, kmax, kbin, lenk = load_bins(corr_type, bins_type)
            cov_fn = f'{ez_dir}/cov_EZmocks_1.5Gpc_{corr_type}02_{tracer}_z{zmin}-{zmax}_k{kmin}-{kmax}.txt'
        cov = np.loadtxt(cov_fn)
        return cov
    # -------------------------------------------------
    # 3) Build covariance from mocks (CovBox / EZmocks)
    # -------------------------------------------------
    elif 'mock' in cov_type:
        logging.getLogger("TwoPointEstimator").setLevel(logging.WARNING)
        logging.getLogger("PowerSpectrumMultipoles").setLevel(logging.WARNING)
        recon = args.get("recon", False)
        if 'covbox' in cov_type:
            mock_range = range(3000, 4500)
            base_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CovBox/{tracer}/z{zeff:.3f}/{corr_type}'
            if recon: base_dir = base_dir+'/recon'
            data_fns = base_dir + f'/{corr_type}poles_{tracer}_z{zeff:.1f}_ph{{}}.npy'
        elif 'EZ' in cov_type:
            mock_range = range(1, 2001)
            base_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/EZmocks/{tracer}/z{zeff:.3f}/{corr_type}'
            if recon: base_dir = base_dir+'/recon'
            data_fns = base_dir + f'/{corr_type}poles_{tracer}_z{zmin:.1f}-{zmax:.1f}_{{}}.npy'
        if len(data_fns) == 0:
            raise ValueError(f"No files found matching mocks")
        if corr_type == 'xi':
            xi_ = []
            (rmin, rmax, rbin, lenr) = load_bins(corr_type, bins_type)
            for mock_id in mock_range:
                mock_id04 =  f"{mock_id:04}"
                data_fn = data_fns.format(mock_id04)
                if not os.path.isfile(data_fn):
                    # print(f"File not found: {data_fn}. Skipping mock_id={mock_id04}")
                    continue
                result = TwoPointCorrelationFunction.load(data_fn)
                result = result[::rbin,::]
                result.select((rmin, rmax+1))
                s, xi = project_to_multipoles(result, ells=[0,2])
                xi_.append(xi)
            # print(s)
            xi_ = np.array(xi_)  # Shape (N_mocks, N_bins)
            xi_ = xi_.reshape(len(xi_), -1)
            cov = np.cov(xi_, rowvar=False, ddof=1)
            if 2 not in ells:
                cov = cov[:lenr,:lenr]
            # cov_fn = glob.glob(basedir+f'/{tracer}/z{z_eff:.3f}/{corr_type}/xipoles*.npy') # not work because we save the DD, RR
            return cov
        if corr_type == 'pk':
            pk_ = []
            recon = args.get("recon", False)
            (kmin, kmax, kbin, lenk) = load_bins(corr_type, bins_type)
            for mock_id in mock_range:
                mock_id04 =  f"{mock_id:04}"
                data_fn = data_fns.format(mock_id04)
                if not os.path.isfile(data_fn):
                    # print(f"File not found: {data_fn}. Skipping mock_id={mock_id04}")
                    continue
                result = PowerSpectrumMultipoles.load(data_fn)
                result = result.select((kmin,kmax,kbin))
                pk = np.real(result.get_power())
                k = result.kavg
                pk_.append(pk)
            # print(k)
            pk_ = np.array(pk_)  # Shape (N_mocks, N_bins)
            pk_ = pk_.reshape(len(pk_), -1)
            cov = np.cov(pk_, rowvar=False, ddof=1)
            # cov = np.cov(pk_, rowvar=False, ddof=1)
            if 2 not in ells:
                cov = cov[:lenk,:lenk]
            # cov = glob.glob(base_dir+f'/{tracer}/z{z_eff:.3f}/{corr_type}/*')
            return cov
    else:
        raise ValueError(f"[DEBUG] Invalid cov_type: '{cov_type}'.")

def get_template(task, z_eff = 1.0, ells = (0,2), cosmo=DESI()):
    '''
    task: {FM/SF}_{fit_cosmology}_{fiducial_cosmology}_{theory}
    return: initialized template
    '''
    # for param in ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s', 'm_ncdm', 'Omega_m']:
    # print(f'\'{param}\': {cosmo[param]},')
    if 'FM' in task:
        template = DirectPowerSpectrumTemplate(z=z_eff, fiducial=cosmo)
        template.init.params['h'].update(prior={'dist': 'uniform', 'limits': [0.2, 1.0]})
        template.init.params['omega_cdm'].update(delta=0.01)
        template.init.params['logA'].update(delta=0.07)
        template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}, delta=0.01)
        template.init.params['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}
        # template.init.params['Omega_m'] = {'derived': True, 'latex': r'\Omega_m'}
        # template.init.params['rs_drag'] = {'derived': True, 'latex': r'r_s'}
        if 'nuCDM' in task:
            # template.init.params['m_ncdm'].update(fixed=False)
            template.init.params['m_ncdm'].update(fixed=False, prior=dict(dist='uniform', limits=[0.0, 4.0]))
        if 'wCDM' in task:
            template.init.params['w_fld'].update(fixed=False)
        if 'w0waCDM' in task:
            template.init.params['w0_fld'].update(fixed=False)
            template.init.params['wa_fld'].update(fixed=False)
    elif 'SF' in task:
        template = ShapeFitPowerSpectrumTemplate(z=z_eff, fiducial=cosmo)
        template.init.update(apmode='qisoqap')
        template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
        template.init.params['qap'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
        template.init.params['df'].update(delta=0.05)
        template.init.params['dm'].update(prior={'limits': [-0.8, 0.8]})
    elif 'BAO' in task:
        apmode = 'qisoqap'
        if 2 not in ells: apmode = 'qiso'
        template = BAOPowerSpectrumTemplate(z=z_eff, cosmo=cosmo, apmode=apmode)
    else:
        raise ValueError(f"Unknown template type (BAO, SF, FM): {task}")
    return template

def get_theory(task, template=None, ells = (0,2), emulator_fn=None, smoothing_radius=15):
    if 'FM' in task or 'SF' in task:
        theory_keys = ['folpsrc', 'folps', 'vel', 'tns'] 
        theory_map = {
            'vel': LPTVelocileptorsTracerPowerSpectrumMultipoles,
            # 'folpsrc': FOLPSRCTracerPowerSpectrumMultipoles,
            'folps': FOLPSTracerPowerSpectrumMultipoles,
            'tns': TNSTracerPowerSpectrumMultipoles,
        }
        matched_key = next((key for key in theory_keys if key in task), None)
        if matched_key is None:
            raise ValueError(f"Unsupported theory type in task: {task}")
        theory = theory_map[matched_key]
        if emulator_fn != None:
            emulator = EmulatedCalculator.load(emulator_fn)
            return theory(pt=emulator)
        else:
            if template is None:
                template = DirectPowerSpectrumTemplate(fiducial = DESI())
            return theory(template=template)
    elif 'BAO' in task:
        # theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=ells, broadband='pcs2')
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=ells, smoothing_radius = 15, 
                                                                    broadband='pcs2', mode = 'recsym',)
        return theory

def get_observable_likelihood(task, data_args, fit_args):
    """
    Defines the modeling task and relevant configuration for fitting.

    Task format:
        {FM|SF}_{fit_cosmology}_{fiducial_cosmology}_{theory}
        where:
            - FM: Full Modeling
            - SF: ShapeFit
            - BAO: BAO fitting

    Parameters:
        data_args (dict): Contains the following keys:
            - 'tracer': Tracer
            - 'zmin' and 'zmax': Redshift bins
            - 'z_eff': Effective redshift
            - 'grid_cosmo': cosmology
            - 'sys_model': Systematics model applied ['standard','dv-obs']
            - 'recon': [True,False], if using reconstruction data

        fit_args (dict): Contains the following keys:
            - 'corr_type': Type of statistic ('pk' or 'xi')
            - 'bins_type': Type of binning used
            - 'data_type': Data used ('sys')
            - 'cov_type': Covariance mode ('fn' or 'cov')
                - 'fn': Load a precomputed covariance matrix from file
                - 'cov': Compute/load covariance from mock realizations
            - 'emulator_fn': Path to the emulator function (if applicable)

    Returns:
        tuple: (likelihood, observable, theory)
    """
    (tracer, zeff, grid_cosmo) = (data_args[key] for key in ["tracer", "z_eff", 'grid_cosmo'])
    (corr_type, bins_type, data_type, cov_type) = (fit_args[key] for key in ["corr_type", "bins_type", "data_type", "cov_type"])
    # seed = data_args.get("seed", None)
    rsf = data_args.get("rsf", 1)
    if grid_cosmo == '000':
        cosmo = DESI()
    elif grid_cosmo == '002':
        cosmo = AbacusSummit(2)
    if 'BAO' in task:
        if corr_type != 'xi':
            raise ValueError(f"BAO task requires corr_type='xi', but got '{corr_type}'")
        option = fit_args.get("option", 'No')
        (rmin, rmax, rbin, lenr) = load_bins(corr_type, bins_type)
        slim={0: (rmin, rmax, rbin), 2: (rmin, rmax, rbin)}
        ells = (0,2); apmode = 'qisoqap'
        broadband = None
        if 'recon' in task:
            data_args["recon"] = True
        if '1d' in option: ells = (0,)
        if tracer == 'QSO':
            smoothing_radius = 30
        else:
            smoothing_radius = 15
        data_fns = load_data_fns(data_args, corr_type = corr_type, data_type=data_type)
        cov = load_cov(data_args, corr_type=corr_type, cov_type=cov_type, bins_type=bins_type, ells=ells)
        template = get_template(task, z_eff=zeff, ells=ells, cosmo=cosmo)
        theory = get_theory(task, template=template, ells=ells, smoothing_radius=smoothing_radius)
        if broadband == 'fixed':
            for param in theory.init.params.select(basename=['al*_*', 'bl*_*']):
                param.update(fixed=True)
        if 2 not in ells:
            slim={0: (rmin, rmax, rbin)}
            for param in theory.init.params.select(basename='*l2_*'):
                param.update(fixed=True)
            for param in theory.init.params.select(basename='qap'):
                param.update(fixed=True)
        observable = TracerCorrelationFunctionMultipolesObservable(data = data_fns, covariance = cov, 
                                                                slim=slim, theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory, scale_covariance = 1/rsf)
        likelihood.all_params[f'sigmaper'].update(fixed = False, prior=dict(dist='norm', loc=3.0, scale=1.0))
        likelihood.all_params[f'sigmapar'].update(fixed = False, prior=dict(dist='norm', loc=6.0, scale=1.0))
        likelihood.all_params[f'sigmas'].update(fixed = False, prior=dict(dist='norm', loc=2.0, scale=2.0))
    elif 'FM' in task or 'SF' in task:
        if corr_type != 'pk':
            raise ValueError(f"{task} task requires corr_type='pk', but got '{corr_type}'")
        option = fit_args.get("option", 'No')
        (kmin, kmax, kbin, lenk) = load_bins(corr_type, bins_type)
        klim = {ell*2: (kmin,kmax,kbin) for ell in range(2)}
        ells = (0,2)
        if '1d' in option:
            ells = (0)
            klim = {ell*2: (kmin,kmax,kbin) for ell in range(1)}
        data_fns = load_data_fns(data_args, corr_type = corr_type, data_type=data_type)
        cov = load_cov(data_args, corr_type=corr_type, cov_type=cov_type, ells=ells) # cov_type = 'fn' or 'mock'
        if 'emulator_fn' in fit_args:
            # print('[LOADING EMULATOR]:', fit_args['emulator_fn'])
            theory = get_theory(task, emulator_fn = fit_args['emulator_fn'])
        else:
            template = get_template(task, z_eff = zeff, cosmo=cosmo)
            theory = get_theory(task, template=template, ells=ells)
        observable = TracerPowerSpectrumMultipolesObservable(data = data_fns, covariance = cov, 
                                                                klim=klim, theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory, scale_covariance = 1/rsf)
        if 'fixfc' in option:
            print('[LIKELIHOOD] fix redshift catastrophic rate fc', flush=True)
            likelihood.varied_params['fcp'].update(fixed = True, value = 0.05)
    else:
        raise ValueError(f"Unknown fit type (BAO, SF, FM): {task}")
    likelihood()
    return (likelihood, observable, theory)

"""
def run_reconstruction(Reconstruction, positions, tracer, boxsize=None, nmesh=None, cellsize=20., smoothing_radius=None, convention='recsym', dtype='f8'):
    # RecSym = remove large scale RSD from randoms
    f, bias, s = GET_RECON_BIAS(tracer = tracer, grid_cosmo = '000')
    if smoothing_radius == None: smoothing_radius = s
    print(f"RECONTRUCTION: recon_sm{smoothing_radius:.0f}_IFFT_{convention}" )
    recon = Reconstruction(f=f, bias=bias, boxsize=boxsize, nmesh=nmesh, cellsize=cellsize, los='z', positions=positions, boxpad=1.2, position_type = 'xyz', dtype=dtype)
    recon.assign_data(positions)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()
    field = 'rsd' if convention == 'rsd' else 'disp+rsd'
    field = 'disp+rsd' if convention == 'recsym' else 'disp'
    if Reconstruction is IterativeFFTParticleReconstruction:
        positions_rec = recon.read_shifted_positions('data', field=field, dtype=dtype)
    else:
        positions_rec = recon.read_shifted_positions(positions, field=field, dtype=dtype)
    positions_rec = positions_rec%boxsize-boxsize/2.0
    return positions_rec


def get_observable_likelihood(task, data_args, fit_args):
    '''
    task: FMfit_cubic_sys, BAOfit_recon_cubic_sys, SFfit_cubic_sys
    data_args: dict{"tracer", "z_eff", "zmin", "zmax", "sys_model", "grid_cosmo"}
    fit_args = dict{"corr_type", "bins_type", "cov_type", "theory_model", "emulator_fn"}
    return: (likelihood, observable, theory)
    '''
    (tracer, z_eff, zmin, zmax, sys_model) = (data_args[key] for key in ["tracer", "z_eff", "zmin", "zmax", "sys_model"])
    if 'BAOfit' in task:
        corr_type = fit_args.get("corr_type", "xi")
        bins_type = fit_args.get("bins_type", "y3_bao")
        cov_type = fit_args.get("cov_type", "EZmocks")
        grid_cosmo = fit_args.get("grid_cosmo", "000")
        (rmin, rmax, rbin, lenr) = load_bins(corr_type, bins_type)
        slim={0: (rmin, rmax, rbin), 2: (rmin, rmax, rbin)}
        broadband = None
        if 'recon' in task:
            data_args["recon"] = True
        if tracer in ['LRG','ELG']:
            ells = (0,2)
            apmode = 'qisoqap'
            smoothing_radius = 15
        elif tracer in ['BGS','QSO']:
            ells = (0,)
            apmode = 'qiso'
            smoothing_radius = 15
            if tracer == 'QSO':
                smoothing_radius = 30
        if 'sys' in task:
            data_fns = load_cubic_sys_fns(data_args, corr_type)
        rsf = True if 'rsf' in 'cov_type' else False
        if 'mock' in cov_type:
            print("Loading covariance from mocks", flush = True)
            cov = load_mock_cov(data_args, cov_type, corr_type, bins_type, rsf = rsf, ells=ells)
        elif 'fn' in cov_type:
            print("Loading exist EZmocks covariance", flush = True)
            cov = load_EZcov_fn(data_args,  cov_type, corr_type, bins_type, rsf = rsf, ells=ells)
        elif 'RescalC' in cov_type:
            print("Loading RescalC analytical covariance", flush = True)
            cov = load_rescalc_cov(data_args, corr_type, bins_type, ells=ells)
        if grid_cosmo == '000':
            cosmo = DESI()
        elif grid_cosmo == '002':
            cosmo = AbacusSummit(2)
        template = BAOPowerSpectrumTemplate(z=z_eff, cosmo = cosmo, apmode = apmode)
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, ells=ells, smoothing_radius = smoothing_radius, 
                                                                    broadband='pcs2', mode = 'recsym',)
        if broadband == 'fixed':
            for param in theory.init.params.select(basename=['al*_*', 'bl*_*']):
                param.update(fixed=True)
        if 2 not in ells:
            slim={0: (rmin, rmax, rbin)}
            for param in theory.init.params.select(basename='*l2_*'):
                param.update(fixed=True)
            for param in theory.init.params.select(basename='qap'):
                param.update(fixed=True)
        observable = TracerCorrelationFunctionMultipolesObservable(data = data_fns, covariance = cov, 
                                                                slim=slim, theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory)
        likelihood.all_params[f'sigmaper'].update(fixed = False, prior=dict(dist='norm', loc=3.0, scale=1.0))
        likelihood.all_params[f'sigmapar'].update(fixed = False, prior=dict(dist='norm', loc=6.0, scale=1.0))
        likelihood.all_params[f'sigmas'].update(fixed = False, prior=dict(dist='norm', loc=2.0, scale=2.0))
        # likelihood.all_params[f'sigmaper'].update(fixed = False, prior=dict(dist='norm', loc=3.0, scale=10.0))
        # likelihood.all_params[f'sigmapar'].update(fixed = False, prior=dict(dist='norm', loc=6.0, scale=10.0))
        # likelihood.all_params[f'sigmas'].update(fixed = False, prior=dict(dist='norm', loc=2.0, scale=20.0))
        
    elif 'FMfit' in task or 'SFfit' in task:
        corr_type = fit_args.get("corr_type", "pk")
        bins_type = fit_args.get("bins_type", "y3_sys")
        cov_type = fit_args.get("cov_type", "EZmocks")
        fit_cosmo = fit_args.get("fit_cosmo", "LCDM")
        grid_cosmo = fit_args.get("grid_cosmo", "000")
        theory_model = fit_args.get("theory_model", "folps")
        option = fit_args.get("option", 'None')
        (kmin, kmax, kbin, lenk) = load_bins(corr_type, bins_type)
        if tracer in ['LRG', 'ELG', 'QSO']:
            ells = (0,2)
            klim = {ell*2: (kmin,kmax,kbin) for ell in range(2)}
            if '1d' in option:
                ells = (0)
                klim = {ell*2: (kmin,kmax,kbin) for ell in range(1)}
        if 'sys' in task:
            data_fns = load_cubic_sys_fns(data_args, corr_type)
        rsf = True if 'rsf' in cov_type else False
        if 'mock' in cov_type:
            print("Loading covariance from mocks", flush = True)
            cov = load_mock_cov(data_args, cov_type, corr_type, bins_type, rsf = rsf, ells=ells)
        elif 'fn' in cov_type:
            print("Loading exist EZmocks covariance", flush = True)
            cov = load_EZcov_fn(data_args,  cov_type, corr_type, bins_type, rsf = rsf, ells=ells)
            if '1d' in option:
                cov = cov[:lenk, :lenk]
        if 'emulator_fn' in fit_args:
            if 'vel' in theory_model:
                theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(fit_args['emulator_fn']))
            if 'folps' in theory_model:
                theory = FOLPSTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(fit_args['emulator_fn']))
        else:
            fiducial_cosmo = DESI()
            if 'FMfit' in task:
                template = DirectPowerSpectrumTemplate(z=z_eff, fiducial = fiducial_cosmo)
                template.init.params['h'].update(prior={'dist': 'uniform', 'limits': [0.2, 1.0]})
                template.init.params['omega_cdm'].update(delta=0.01)
                template.init.params['logA'].update(delta=0.07)
                if 'logA' in option:
                    if '5.0' in option:
                        template.init.params['logA'].update(prior={'dist': 'uniform', 'limits': [2.0, 5.0]}, delta=0.07)
                    elif '4.5' in option:
                        template.init.params['logA'].update(prior={'dist': 'uniform', 'limits': [2.0, 4.5]}, delta=0.07)
                    print('logA prior:', template.params['logA'].prior, flush=True)
                template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
                template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}, delta=0.01)
                if 'fix_ns' in option:
                    template.init.params['n_s'].update(fixed=True)
                    print('fix ns', flush=True)
                template.init.params['sigma8'] = {'derived': True, 'latex': r'\sigma_8'}
                # template.init.params['Omega_m'] = {'derived': True, 'latex': r'\Omega_m'}
                # template.init.params['rs_drag'] = {'derived': True, 'latex': r'r_s'}
                if 'wCDM' in fit_cosmo:
                    template.init.params['w0_fld'].update(fixed=False)
                if 'w0waCDM' in fit_cosmo:
                    template.init.params['wa_fld'].update(fixed=False)
                if 'vel' in theory_model:
                    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template,)
                if 'folps' in theory_model:
                    theory = FOLPSTracerPowerSpectrumMultipoles(template=template,)
            elif 'SFfit' in task:
                template = ShapeFitPowerSpectrumTemplate(z = z_eff, fiducial = fiducial_cosmo)
                template.init.update(apmode='qisoqap')
                template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['qap'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['df'].update(delta=0.05)
                # limit the range of dm (otherwise it's easy to have bimodial posterior)
                template.init.params['dm'].update(prior={'limits': [-0.8, 0.8]})
                if 'vel' in theory_model:
                    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template,)
                if 'folps' in theory_model:
                    theory = FOLPSTracerPowerSpectrumMultipoles(template=template,)
        observable = TracerPowerSpectrumMultipolesObservable(data = data_fns, covariance = cov, 
                                                            klim=klim, theory=theory)
        likelihood = ObservablesGaussianLikelihood(observables=observable, theory=theory)
    else:
        raise ValueError(f"Unknown task type (BAOfit, SFfit, FMfit): {task}")
    likelihood()
    return (likelihood, observable, theory)



    
"""