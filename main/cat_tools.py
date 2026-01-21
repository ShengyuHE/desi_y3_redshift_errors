import os
import sys
import logging
import numpy as np
from mockfactory import Catalog
from astropy.table import Table,join,Column
from scipy.interpolate import interp1d

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import NRAN, NRAN_TEST, TRACER_CUTSKY_INFO
from utils import setup_logging
from dv_tools import get_repeats_dv, get_cthr, get_repeats_numbers

setup_logging()
logger = logging.getLogger('cat_tools') 

GLOBAL_SEED = 123
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'
BASE_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3'
RANDOM_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1/Cutsky/random'

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

def comoving_radial_distance(z):
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    # Abacussummit cosmology -- Planck 2018
    h          = 0.6736
    omega_b    = 0.02237
    omega_cdm  = 0.1200
    omega_ncdm = 0.00064420
    Om0 = (omega_b + omega_cdm + omega_ncdm) / h**2
    Ob0 = omega_b / h**2
    _cosmo = FlatLambdaCDM(H0=67.36, Om0=Om0, Ob0=Ob0, Tcmb0=2.7255 * u.K,  Neff=3.044)
    return _cosmo.comoving_distance(z).to(u.Mpc).value * _cosmo.h

def get_proposed_mattrs(tracer):
    if 'BGS' in tracer:
        mattrs = dict(boxsize=4000., cellsize=7)
    elif 'LRG+ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'LRG' in tracer:
        mattrs = dict(boxsize=7000., cellsize=7)
    elif 'ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'QSO' in tracer:
        mattrs = dict(boxsize=10000., cellsize=10)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    # mattrs.update(cellsize=10)
    return mattrs

def get_catalog_fn(version='AbacusHF-v1', domain = 'cubic', tracer='LRG', zrange=(0.4, 0.6), zsnap = 0.5, mock_id=0, random = False, nran=None, **kwargs):
    mock_id03 =  f"{mock_id:03}"
    if version == 'AbacusHF-v1':
        if domain == 'cubic':
            # load the data
            if random == True: raise ValueError(f"No random needs for cubic mocks")
            cubic_name = f'/abacus_HF_{tracer}_{zfmt(zsnap)}_DR2_v1.0_AbacusSummit_base_c000_ph{mock_id03}_clustering.dat.fits'
            cubic_fn = BASE_DIR+ f'/{version}' +f'/Boxes/{tracer}/sn{zfmt(zsnap)}/AbacusSummit_base_c000_ph{mock_id03}'+cubic_name
            return cubic_fn
        '''
        elif domain == 'cutsky':
            if random == True:
                if nran == None:
                    nran = NRAN_TEST[tracer]
                return [RANDOM_DIR+f'/rands_intiles_DARK_{i}_NO_imagingmask_withz.ran.fits'.format(i) for i in range(8, 8+nran-1)]
            else:
                tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
                fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
                cutsky_name = f'cutsky_abacusHF_DR2_{tracer_type}_z{zfmt(zsnap)}_zcut_{fit_range}_clustering.dat.fits'
                cat_fn = BASE_DIR+ f'/{version}'+ f'/Cutsky/{tracer_type[:3]}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering/'+cutsky_name
                return cat_fn
        '''
    if version == 'AbacusHF-v2':
        if domain == 'cubic':
            # load the data
            if random == True: raise ValueError(f"No random needs for cubic mocks")
            cubic_name = f'/abacus_HF_{tracer}_{zfmt(zsnap)}_DR2_v2.0_AbacusSummit_base_c000_ph{mock_id03}_base_clustering.dat.h5'
            if tracer[:3] == 'ELG':
                cubic_name= f"abacus_HF_{tracer}_{zfmt(zsnap)}_DR2_v2.0_AbacusSummit_base_c000_ph0${mock_id03}_base_conf_nfwexp_clustering.dat.h5"
            cubic_fn = BASE_DIR+ f'/{version}' +f'/Boxes/{tracer}/sn{zfmt(zsnap)}/AbacusSummit_base_c000_ph{mock_id03}'+cubic_name
            return cubic_fn
        
def get_measurement_fn(version='AbacusHF-v1', domain = 'cubic', tracer='LRG', zrange=(0.4, 0.6), zsnap = 0.5, mock_id=0, use_dv = False,  weight_type='default', use_jax = False, **kwargs):
    mock_id03 =  f"{mock_id:03}"
    base_dir = BASE_DIR+f'/{version}'   # now base_dir is a Path
    if domain == 'cubic' in domain:
        mock_dir = base_dir+ f'/Boxes/{tracer}/sn{zfmt(zsnap)}/AbacusSummit_base_c000_ph{mock_id03}'
    # elif 'cutsky' in domain:
        # mock_dir = base_dir+ f'/Cutsky/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering'
    fn_path = mock_dir+ '/mpspk'
    os.makedirs(fn_path, exist_ok=True)
    if use_dv in ['repeat', 'verr']:
        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0+dv_{use_dv}.npy'
    else:
        fn_2pt = fn_path + f'/{{}}_{tracer}_zp{zsnap:.3f}_DR2_v1.0.npy'
    if use_jax: fn_2pt = os.path.splitext(fn_2pt)[0] + '.h5'
    return fn_2pt

def read_positions_weights(version='AbacusHF-v1', domain = 'cubic', tracer='LRG', zrange=(0.4, 0.6), zsnap = 0.5, mock_id=0, weight_type='default', use_dv = False, use_random=False, nran=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.rank
    """
    Return the positions of tracer galaxies for either cubic-box or light-cone mocks, formatted for pycorr / Corrfunc two-point estimators.

    Parameters
    ----------
    use_zerr : bool, optional
        If default, use Z_RSD (true RSD displacement) or Z Redshift.
        If LSS, use Z_OBS_LSS, dv-obs fitted to LSS bins
        If global, use Z_OBS_GLOBAL, dv-obs fitted to global bins
        If bin, use Z_OBS_BIN, dv-obs fitted to 0.1 zbins
    use_random: bool, optional
        If True, load the random catalogs for cutsky
    Returns
    -------
    positions : ndarray of shape (3, N)
    weights : ndarray of shape (3, N)
    """    
    if domain == 'cubic':
        # basic settings
        los = 'z'
        boxsize = 2000
        cubic_fn = get_catalog_fn(version, domain , tracer, zrange, zsnap, mock_id) # load the data
        if rank == 0: logger.info(f'Load {cubic_fn}')
        cat = Catalog.read(cubic_fn, mpicomm=MPI.COMM_SELF)
        if los == 'z':
            if use_dv is not False:
                if use_dv == 'repeat': dv_label = 'DV_REP'
                if use_dv == 'verr': dv_label = 'DV_ERR'
                if rank == 0: logger.info(f'use Z positions shifted in {use_dv} mode')
                positions = np.array([cat['X'], cat['Y'], cat[f'Z_{dv_label}']])%boxsize
            else:
                positions = np.array([cat['X'], cat['Y'], cat['Z_RSD']])%boxsize
        positions = positions
        weights = None
        return np.array(positions), np.array(weights)
    '''
    elif domain == 'cutsky':
        # load the data
        (zmin, zmax) = (zrange[0], zrange[1])
        tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
        fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
        if use_random == True:
            ran_fns = get_catalog_fn(version, domain , tracer, zrange, zsnap, mock_id, random=True, nran=nran)
            chunks = np.array_split(ran_fns, size)
            _chunk = chunks[rank]
            pos_list = []
            wei_list = []
            for ran_fn in _chunk:
                logger.info(f'Load with rank{rank} {ran_fn}')
                cat = Table.read(ran_fn)
                Zcol = f'Z_{tracer_type}'
                sel = np.isfinite(cat[Zcol])
                selz = (cat[Zcol] >= zmin) & (cat[Zcol] < zmax)
                if tracer_type == 'ELG_LOP':
                    sel &= cat['ELG_LOP_MASK']
                cat_sel = cat[sel & selz]
                pos = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, comoving_radial_distance(cat_sel[Zcol])])
                # Remove any non-finite coords
                mask_good = np.all(np.isfinite(pos), axis=0)
                if (~mask_good).sum() > 0:
                    logger.info(f"Warning randoms: dropping {(~mask_good).sum()} non-finite points in {ran_fn}")
                    pos = pos[:, mask_good]
                w = np.ones(pos.shape[1], dtype=float)
                pos_list.append(pos)
                wei_list.append(w)
            if pos_list:
                local_positions = np.hstack(pos_list)        # shape (3, N_local)
                local_weights = np.hstack(wei_list)          # shape (N_local,)
            else:
                local_positions = np.empty((3, 0), dtype=float)
                local_weights = np.empty((0,), dtype=float)
            mpicomm.Barrier()
            _positions = mpicomm.allgather(local_positions)
            _weights = mpicomm.allgather(local_weights)
            positions = np.hstack(_positions)
            weights = np.hstack(_weights)
        else:
            cat_fn = get_catalog_fn(version, domain , tracer, zrange, zsnap, mock_id)
            cat = Table.read(cat_fn)
            if rank == 0: logger.info(f'Load {cat_fn}')
            sel = np.isfinite(cat['Z'])
            selz = (cat['Z'] >= zmin) & (cat['Z'] < zmax) 
            # selr  = select_region(catalog['RA'], catalog['DEC'], region=region)
            if use_zerr in ['global', 'bin', 'LSS']:
                Z_err = f'Z_OBS_{use_zerr.upper()}'
                if rank == 0: logger.info(f'use Z with redshift error shifted in {Z_err}')
                selz_obs = (cat[Z_err] >= zmin) & (cat[Z_err] < zmax) 
                cat_sel = cat[sel&selz&selz_obs]
                positions = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, comoving_radial_distance(cat_sel[Z_err])])
            else:
                cat_sel = cat[sel&selz]
                positions = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, comoving_radial_distance(cat_sel['Z'])])
            mask_good = np.all(np.isfinite(positions), axis=0)
            if (~mask_good).sum() > 0:
                if rank == 0: logger.info(f"Data warning: dropping {(~mask_good).sum()} non-finite points")
                positions = positions[:, mask_good]
            weights = np.ones(len(cat_sel), dtype=float)
            # if 'default' in weight_type:
                # weights = cat['WEIGHT'].data
    '''
    

