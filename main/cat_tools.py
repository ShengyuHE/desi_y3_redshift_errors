
import os
import sys
import numpy as np
from astropy.table import Table,join,Column
from scipy.interpolate import interp1d
from cosmoprimo.fiducial import DESI, AbacusSummit

sys.path.append('/global/homes/s/shengyu/desi_y3_redshift_errors/main/')
from helper import REDSHIFT_BIN_OVERALL, REDSHIFT_BIN_LSS, REDSHIFT_ABACUSHF_v1, NRAN, TRACER_CUTSKY_INFO
from helper import GET_REPEATS_DV, GET_CTHR, GET_REPEATS_NUMBER

GLOBAL_SEED = 123
REPEAT_DIR = '/pscratch/sd/s/shengyu/repeats/DA2/loa-v1'
ABACUS_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1'
RANDOM_DIR = '/pscratch/sd/s/shengyu/galaxies/catalogs/Y3/AbacusHF-v1/Cutsky/random'

def zfmt(x):
    return f"{x:.3f}".replace(".", "p")

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

def sample_from_cdf(cdf_fn, Ngal, vmode, seed=1234):
    """
    Sample Δv from a stored |Δv| CDF.

    Parameters
    ----------
    cdf_fn : str
        Path to the CDF file (npz) containing arrays {grid, cdf}.
    Ngal : int
        Number of Δv samples to generate.
    vmode : {"log_abs", "log_signed", "linear"}
        Sampling mode
    seed : int, optional
        Random seed.

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
    tracer : str
        Tracer name (e.g., "LRG", "ELG", "QSO").
    z1, z2 : float
        Lower and upper redshift bounds of the bin.
    N : int
        Number of Δv values to sample.
    cdf_kind : {"KCDF", "obsCDF"}
        Type of CDF used for sampling.
    vmode : {"log_signed", "log_abs", "linear"}
        Modeling mode:
        - "log_abs"    : sample |Δv| from log-CDF.
        - "log_signed" : sample positive/negative Δv separately using observed N_p/N_n fractions.
        - "linear"     : sample Δv directly.
    seed : int
        Random seed.

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
    

def read_positions_weights(args, use_zerr = None, use_random = False):
    """
    Return the positions of tracer galaxies for either cubic-box or light-cone mocks, formatted for pycorr / Corrfunc two-point estimators.

    Parameters
    ----------
    args : dict
        Dictionary containing the following keys:
            - 'domain' : str
                Either 'cubic' or 'cutsky'. Determines how coordinates are read.
            - 'tracer' : str
                Tracer name, e.g. 'LRG', 'ELG', 'QSO'.
            - 'zind' : int
                index of redshift bin.
            - 'mock_id03' : str or int
                Mock identifier formatted as a 3-digit string, e.g. '000'.

    use_zerr : bool, optional
        If default, use Z_RSD (true RSD displacement) or Z Redshift.
        If True, use Z_OBS (includes modeled redshift errors).
        If global, use Z_OBS_GLOBAL, 
        If bin, use Z_OBS_BIN, 

    use_random: bool, optional
        If True, load the random catalogs for cutsky

    Returns
    -------
    positions : ndarray of shape (3, N)
    weights : ndarray of shape (3, N)
    """
        
    domain = args.get('domain', 'cubic')
    (tracer, indz, mock_id03) = (args[key] for key in ["tracer", "indz", "mock_id03"])
    zsnap = REDSHIFT_ABACUSHF_v1[tracer][indz]
    if domain == 'cubic':
        # basic settings
        los = args.get('los', 'z')
        boxsize = args.get('boxsize', 2000)
        # load the data
        cubic_name = f'/abacus_HF_{tracer}_{zfmt(zsnap)}_DR2_v1.0_AbacusSummit_base_c000_ph{mock_id03}_clustering.dat.fits'
        cubic_fn = ABACUS_DIR+ f'/Boxes/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}' +cubic_name
        print(f'[LOAD]: {cubic_fn}', flush=True)
        cat = Table.read(cubic_fn)
        if los == 'z':
            if use_zerr == 'LSS':
                positions = np.array([cat['X'], cat['Y'], cat['Z_OBS']])%boxsize
            else:
                positions = np.array([cat['X'], cat['Y'], cat['Z_RSD']])%boxsize
        positions = positions
        weights = None
    elif domain == 'cutsky':
        # basic settings
        cosmo = DESI()
        z2d = cosmo.comoving_radial_distance
        # load the data
        if args.zmin is None or args.zmax is None:
            (zmin, zmax) = REDSHIFT_BIN_LSS[tracer][indz]
        tracer_type = TRACER_CUTSKY_INFO[tracer]['tracer_type']
        fit_range = TRACER_CUTSKY_INFO[tracer]['fit_range']
        if use_random == True:
            ran_fns = [RANDOM_DIR+f'/rands_intiles_DARK_{i}_NO_imagingmask_withz.ran.fits'.format(i) for i in range(8, 8+NRAN[tracer])]
            pos_list = []
            wei_list = []
            print(f'[LOAD]: {ran_fns[0]}', flush=True)
            for ran_fn in ran_fns:
                cat = Table.read(ran_fn)
                Zcol = f'Z_{tracer_type}'
                sel = np.isfinite(cat[Zcol])
                selz = (cat[Zcol] >= zmin) & (cat[Zcol] < zmax)
                if tracer_type == 'ELG_LOP':
                    sel &= cat['ELG_LOP_MASK']
                cat_sel = cat[sel & selz]
                # print(tracer, zmin, zmax, len(cat), len(cat_sel))
                pos = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, z2d(cat_sel[Zcol])])
                # Remove any non-finite coords
                mask_good = np.all(np.isfinite(pos), axis=0)
                if (~mask_good).sum() > 0:
                    print(f"[WARN] Randoms: dropping {(~mask_good).sum()} non-finite points in {ran_fn}", flush=True)
                    pos = pos[:, mask_good]
                w = np.ones(pos.shape[1], dtype=bool)
                pos_list.append(pos)
                wei_list.append(w)
            positions = np.hstack(pos_list)
            weights = np.hstack(wei_list)
        else:
            cutsky_name = f'cutsky_abacusHF_DR2_{tracer_type}_z{zfmt(zsnap)}_zcut_{fit_range}_clustering.dat.fits'
            cat_fn = ABACUS_DIR+ f'/Cutsky/{tracer_type[:3]}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{mock_id03}/forclustering/'+cutsky_name
            cat = Table.read(cat_fn)
            print(f'[LOAD]: {cat_fn}', flush=True)
            sel = np.isfinite(cat['Z'])
            selz = (cat['Z'] >= zmin) & (cat['Z'] < zmax) 
            # selr  = select_region(catalog['RA'], catalog['DEC'], region=region)
            # print(tracer, zmin, zmax, len(cat), len(cat_sel))
            if use_zerr == 'global':
                selz_obs = (cat['Z_OBS_GLOBAL'] >= zmin) & (cat['Z_OBS_GLOBAL'] < zmax) 
                cat_sel = cat[sel&selz&selz_obs]
                print("[INFO:]", cat_sel['Z_OBS_GLOBAL'].min(), cat_sel['Z_OBS_GLOBAL'].max(), flush=True)
                positions = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, z2d(cat_sel['Z_OBS_GLOBAL'])])
            elif use_zerr == 'bin':
                selz_obs = (cat['Z_OBS_BIN'] >= zmin) & (cat['Z_OBS_BIN'] < zmax) 
                cat_sel = cat[sel&selz&selz_obs]
                print("[INFO:]", cat_sel['Z_OBS_BIN'].min(), cat_sel['Z_OBS_BIN'].max(), flush=True)
                positions = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, z2d(cat_sel['Z_OBS_BIN'])])
            else:
                cat_sel = cat[sel&selz]
                print("[INFO:]", cat_sel['Z'].min(), cat_sel['Z'].max(), flush=True)
                positions = np.array([cat_sel['RA'].data, cat_sel['DEC'].data, z2d(cat_sel['Z'])])
            mask_good = np.all(np.isfinite(positions), axis=0)
            if (~mask_good).sum() > 0:
                print(f"[WARN] Data: dropping {(~mask_good).sum()} non-finite points", flush=True)
                positions = positions[:, mask_good]
            weights = np.ones(len(cat_sel), dtype=bool)
            # if 'default' in weight_type:
                # weights = cat['WEIGHT'].data
    return np.array(positions), np.array(weights)