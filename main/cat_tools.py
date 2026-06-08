import os
import sys
import logging
import numpy as np
from glob import glob
from pathlib import Path
from contextlib import contextmanager
from mockfactory import Catalog
from scipy.interpolate import interp1d

sys.path.append('/global/homes/s/shengyu/Y3/desi_y3_redshift_errors/main/')
from helper import NRAN_Y3, TRACER_CUTSKY_INFO, CSPEED
from utils import setup_logging
from dv_tools import get_repeats_dv, get_cthr, sample_from_cdf_v1, sample_from_cdf_v2

setup_logging()
logger = logging.getLogger('cat_tools') 

GLOBAL_SEED = 123
REPEAT_DIR = Path('/global/cfs/cdirs/desi/users/shengyu/repeats/DA2/loa-v1')
BASE_DIR = Path('/global/cfs/cdirs/desi/users/shengyu/galaxies/catalogs/Y3')
# BASE_DIR = Path('/pscratch/sd/s/shengyu/galaxies/catalogs/Y3')

DESI_PATH = Path('/global/cfs/cdirs/desi/')

def _zfmt(x):
    return f"{x:.3f}".replace(".", "p")

def _rename_LSS(x, tracer=None):
    if x == 'AbacusHF-v2': return "AbacusHF_DR2v2"
    if x == 'holi-v3': 
        if tracer[:3] == 'BGS':
            return "holi_bgs"
        else:
            return "holi_v3"

def _unzip_catalog_options(catalog):
    """Return one catalog option dictionary per tracer."""
    catalog = dict(catalog)
    tracers = catalog.get('tracer', None)
    if tracers is None:
        tracers = ('',)
    elif isinstance(tracers, str):
        tracers = (tracers,)
    else:
        tracers = tuple(tracers)

    def _is_scalar_zrange(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        return all(isinstance(item, (int, float)) for item in value)
    out = {}
    for itracer, tracer in enumerate(tracers):
        options = dict(catalog)
        options['tracer'] = tracer
        for name, value in catalog.items():
            if name == 'tracer' or _is_scalar_zrange(value):
                continue
            if isinstance(value, (list, tuple)) and len(value) == len(tracers):
                options[name] = value[itracer]
        out[tracer] = options
    return out

def _compute_binned_weight(ntile, weight):
    """Compute weights per ntile."""
    sum_ntile = np.bincount(ntile)
    sum_weight = np.bincount(ntile, weights=weight)
    mask_zero_ntile = sum_ntile == 0
    return np.divide(sum_weight, sum_ntile, out=np.ones_like(sum_weight), where=~mask_zero_ntile)

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

def parse_zerr_name(zerr):
    zerr = str(zerr)
    z_evol = zerr.endswith('_zevol')
    use_dv = zerr[:-6] if z_evol else zerr
    valid = {'None', 'False', 'repeat', 'verr_empirical', 'verr_nonparam', 'test'}
    if use_dv not in valid:
        raise ValueError(f"Unsupported zerr label {zerr!r}")
    if use_dv in {'None', 'False'} and z_evol:
        raise ValueError(f"z_evol is not valid with zerr={zerr!r}")
    if use_dv in {'None', 'False'}:
        use_dv = 'None'
        z_evol = False
    return use_dv, z_evol

def normalize_use_dv(use_dv):
    if use_dv in [None, False, 'None', 'False']:
        return False
    if use_dv in ['repeat', 'verr_empirical', 'verr_nonparam']:
        return use_dv
    if use_dv in ['test']:
        return use_dv
    raise ValueError(f"Unrecognized zerr type {use_dv!r}")

def _make_z_evol_edges(zmin, zmax, dz=0.1):
    zmin = float(zmin)
    zmax = float(zmax)
    edges = np.round(np.arange(zmin, zmax + 0.5 * dz, dz), 1)
    if edges.size < 2 or edges[0] >= edges[-1]:
        raise ValueError(f'Invalid z_evol edges for zrange=({zmin}, {zmax})')
    return edges

def sample_cutsky_dv(tracer, redshift, zmin, zmax, use_dv, z_evol=False):
    redshift = np.asarray(redshift, dtype='f8')
    dv = np.zeros(len(redshift), dtype='f8')
    if use_dv is False:
        return dv
    if not z_evol:
        if use_dv in ['repeat', 'verr_empirical']:
            return np.asarray(sample_from_cdf_v1(tracer, zmin, zmax, len(redshift), dv_mode=use_dv), dtype='f8')
        if use_dv in ['verr_nonparam']:
            return np.asarray(sample_from_cdf_v2(tracer, zmin, zmax, len(redshift)), dtype='f8')
        if use_dv in ['test']:
            return np.asarray(sample_from_cdf_v2('QSO', 1.7, 1.8, len(redshift)), dtype='f8')
        raise ValueError(f"not valid dv_mode: {use_dv}")
    zedges = _make_z_evol_edges(zmin, zmax, dz=0.1)
    for iz, (zlo, zhi) in enumerate(zip(zedges[:-1], zedges[1:])):
        if iz == len(zedges) - 2:
            sel = (redshift >= zlo) & (redshift <= zhi)
        else:
            sel = (redshift >= zlo) & (redshift < zhi)
        nz = np.count_nonzero(sel)
        if nz == 0:
            continue
        if use_dv in ['repeat', 'verr_empirical']:
            dv[sel] = sample_from_cdf_v1(tracer, zlo, zhi, nz, dv_mode=use_dv)
        elif use_dv in ['verr_nonparam']:
            dv[sel] = sample_from_cdf_v2(tracer, zlo, zhi, nz)
        else:
            raise ValueError(f"not valid dv_mode: {use_dv}")
    return dv

def get_cutsky_weights(cat, weight_type='WEIGHT_FKP'):
    columns = set(cat.columns())
    if weight_type in [None, False, 'None', 'False']:
        return np.ones(len(cat), dtype='f8')
    if weight_type == 'WEIGHT_FKP':
        if 'WEIGHT' in columns and 'WEIGHT_FKP' in columns:
            return np.asarray(cat['WEIGHT'], dtype='f8') * np.asarray(cat['WEIGHT_FKP'], dtype='f8')
        if 'WEIGHT_FKP' in columns:
            return np.asarray(cat['WEIGHT_FKP'], dtype='f8')
        if 'WEIGHT' in columns:
            return np.asarray(cat['WEIGHT'], dtype='f8')
    if weight_type == 'WEIGHT_FKP_NX13':
        if 'NX' not in columns:
            raise ValueError("NX column is required for WEIGHT_FKP_NX13")
        return (get_cutsky_weights(cat, weight_type='WEIGHT_FKP') * np.asarray(cat['NX'], dtype='f8')**(-1. / 3.))
    if weight_type == 'WEIGHT':
        if 'WEIGHT' in columns:
            return np.asarray(cat['WEIGHT'], dtype='f8')
    if weight_type in columns:
        return np.asarray(cat[weight_type], dtype='f8')
    return np.ones(len(cat), dtype='f8')

def _get_fkp_p0(tracer):
    if 'LRG+ELG' in tracer:
        return 1e4
    if 'BGS' in tracer:
        return 7e3
    if 'LRG' in tracer:
        return 1e4
    if 'ELG' in tracer:
        return 4e3
    if 'QSO' in tracer:
        return 6e3
    raise ValueError(f'No fiducial FKP_P0 configured for tracer={tracer!r}')

def _set_fiducial_weight_fkp(cat, tracer, weight_type='WEIGHT_FKP'):
    if 'WEIGHT_FKP' not in str(weight_type).upper():
        return cat
    if 'NX' not in cat.columns():
        return cat
    cat['WEIGHT_FKP'] = 1. / (1. + np.asarray(cat['NX'], dtype='f8') * _get_fkp_p0(tracer))
    return cat

def _get_box_mattrs(tracer):
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
    return mattrs

def _get_LSS_mattrs(tracer, meshsize=None):
    if meshsize is not None:
        return dict(meshsize=int(meshsize))
    meshsizes = {'BGS': 750, 'LRG': 800, 'ELG': 900, 'LRG+ELG': 750, 'QSO': 1200}
    boxsizes = {'BGS': 6000.0, 'LRG': 6000.0, 'ELG': 6750.0, 'LRG+ELG': 6000.0, 'QSO': 9000.0,}
    return dict(meshsize=meshsizes[tracer], boxsize = boxsizes[tracer])

def get_proposal_mattrs(domain='cubic', **kwargs):
    if domain == 'cubic':
        return _get_box_mattrs(**kwargs)
    if domain in ['cutsky', 'altmtl']:
        return _get_LSS_mattrs(**kwargs)
    raise ValueError(f"Unsupported domain {domain}")

def select_region(ra, dec, region=None):
    """
    Copied from desi-clustering tools
    -------
    mask : array_like
        Boolean mask array indicating the selected region.
    """
    import healpy as hp
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    # North, South, SGC, and NGC footprints
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if region == 'NGCnoN':
        return mask_ngc & (~mask_n)
    if region == 'noN':
        return ~mask_n
    # DES footprint
    def load_footprint():
        #global footprint
        from regressis import footprint
        footprint = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
        return footprint
    north, south, des = load_footprint().get_imaging_surveys()
    mask_des = des[hp.ang2pix(hp.get_nside(des), ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    if region == 'SGCnoDES':
        return (~mask_ngc) & (~mask_des)
    if region == 'noDES':
        return ~mask_des
    raise ValueError('unknown region {}'.format(region))

def get_catalog_fn(version='AbacusHF-v2', domain = 'cubic', tracer='LRG', mock_id=0, random=False, nran=None, region='ALL', **kwargs):
    if domain == 'cubic':
        zsnap = kwargs["zsnap"]
        mock_id03 =  f"{mock_id:03}"
        if random == True: raise ValueError(f"No random needs for cubic mocks")
        if version == 'AbacusHF-v1':
            # load the data
            cubic_dir = BASE_DIR / version / 'Boxes' / tracer / f'sn{_zfmt(zsnap)}' / f'AbacusSummit_base_c000_ph{mock_id03}'
            cubic_name = f'abacus_HF_{tracer}_{_zfmt(zsnap)}_DR2_v1.0_AbacusSummit_base_c000_ph{mock_id03}_clustering.dat.fits'
            return str(cubic_dir / cubic_name)
        if version == 'AbacusHF-v2':
            # load the data
            cubic_dir = BASE_DIR / version / 'Boxes' / tracer / f'sn{_zfmt(zsnap)}' / f'AbacusSummit_base_c000_ph{mock_id03}'
            cubic_name = f'abacus_HF_{tracer}_{_zfmt(zsnap)}_DR2_v2.0_AbacusSummit_base_c000_ph{mock_id03}_base_clustering.dat.h5'
            if tracer[:3] == 'ELG':
                cubic_name = f"abacus_HF_{tracer}_{_zfmt(zsnap)}_DR2_v2.0_AbacusSummit_base_c000_ph{mock_id03}_base_conf_nfwexp_clustering.dat.h5"
            return str(cubic_dir / cubic_name)
    elif domain in ['altmtl', 'cutsky']:
        if version == 'data-dr1-v1.5':
            if tracer == 'ELG': tracer = 'ELG_LOPnotqso'
            cat_dir = DESI_PATH / 'survey' / 'catalogs' / 'Y1' / 'LSS' / 'iron' / 'LSScats' / 'v1.5'
            region_label = '' if region in [None, 'ALL', 'GCcomb'] else f'_{region}'
            if random == False:
                fn = cat_dir / f'{tracer}{region_label}_clustering.dat.fits'
                if not fn.exists():
                    raise FileNotFoundError(f'No data catalog found at {fn}')
                return str(fn)
            if random == True:
                suffix = '_clustering.ran.fits'
                if nran is None:
                    prefix = f'{tracer}{region_label}_'
                    fns = sorted(str(fn) for fn in cat_dir.glob(f'{prefix}*{suffix}'))
                    fns = [fn for fn in fns if os.path.basename(fn).removeprefix(prefix).removesuffix(suffix).isdigit()]
                else:
                    nrans = nran if isinstance(nran, (list, tuple, np.ndarray)) else range(nran)
                    fns = [str(cat_dir / f'{tracer}{region_label}_{iran}_clustering.ran.fits') for iran in nrans]
                if not fns:
                    raise FileNotFoundError(f'No random catalogs found in {cat_dir} for tracer={tracer}, region={region}')
                return fns
            if random == 'parent':
                nrans = range(18) if nran is None else (nran if isinstance(nran, (list, tuple, np.ndarray)) else range(nran))
                fns = [str(cat_dir / f'{tracer}_{iran}_full_noveto.ran.fits') for iran in nrans]
                if not fns:
                    raise FileNotFoundError(f'No parent random catalogs found in {cat_dir} for tracer={tracer}')
                return fns
            raise ValueError(f"No random option {random}")

        if version in ['AbacusHF-v2', 'holi-v3'] and domain == 'altmtl':
            if tracer == 'ELG': tracer = 'ELG_LOPnotqso'
            dr2_mock_dir = DESI_PATH / 'mocks' / 'cai' / 'LSS' / 'DA2' / 'mocks'
            dr2_survey_dir = DESI_PATH / 'survey' / 'catalogs' / 'DA2' / 'LSS' / 'loa-v1' / 'LSScats' / 'v2'
            mock_ls_dir = dr2_mock_dir / _rename_LSS(version, tracer) / f'altmtl{mock_id}' / 'loa-v1' / f'mock{mock_id}' / 'LSScats'
            use_region = region not in [None, 'ALL', 'GCcomb']
            if random == False:
                dat_fns = sorted(str(fn) for fn in mock_ls_dir.glob(f'{tracer}_*_clustering.dat.h5'))
                if use_region:
                    dat_fns = [fn for fn in dat_fns if f'_{region}_' in os.path.basename(fn)]
                if not dat_fns:
                    raise FileNotFoundError(f'No data catalogs found in {mock_ls_dir} matching {tracer}_*_clustering.dat.h5')
                return dat_fns
            elif random in [True, 'parent']:
                if random == True:
                    ran_fns = sorted(str(fn) for fn in mock_ls_dir.glob(f'{tracer}_*_*_clustering.ran.h5'))
                    if use_region:
                        ran_fns = [fn for fn in ran_fns if f'_{region}_' in os.path.basename(fn)]
                    random_suffix = '_clustering.ran.h5'
                elif random == 'parent':
                    program = 'bright' if 'BGS' in tracer else 'dark'
                    ran_fns = sorted(str(fn) for fn in dr2_survey_dir.glob(f'{program}_*_full_noveto.ran.h5'))
                    random_suffix = '_full_noveto.ran.h5'
                if not ran_fns:
                    raise FileNotFoundError(f'No random catalogs found for tracer={tracer}, random={random}, mock_id={mock_id}')
                if nran is not None:
                    grouped_ran_fns = {}
                    for ran_fn in ran_fns:
                        ran_name = os.path.basename(ran_fn).removesuffix(random_suffix)
                        try:
                            _, iran = ran_name.rsplit('_', 1)
                        except ValueError as exc:
                            raise ValueError(f'Unexpected random catalog filename {ran_fn}') from exc
                        grouped_ran_fns.setdefault(iran, []).append(ran_fn)
                    selected_irans = sorted(grouped_ran_fns, key=lambda iran: int(iran))[:nran]
                    ran_fns = [ran_fn for iran in selected_irans for ran_fn in sorted(grouped_ran_fns[iran])]
                return ran_fns
            else:
                raise ValueError(f"No random option {random}")
        raise ValueError(f"Unsupported version {version!r} for domain={domain!r}")
    else:
        raise ValueError(f"No domain option {domain}")

def get_full_hpmapcut_fn(version='AbacusHF-v2', domain='altmtl', tracer='LRG', mock_id=0, **kwargs):
    if domain != 'altmtl':
        raise ValueError(f"full_HPmapcut is only available for altmtl catalogs, got domain={domain!r}")
    if version not in ['AbacusHF-v2', 'holi-v3']:
        raise ValueError(f"Unsupported version {version!r} for full_HPmapcut catalogs")
    if tracer == 'ELG':
        tracer = 'ELG_LOPnotqso'
    dr2_mock_dir = DESI_PATH / 'mocks' / 'cai' / 'LSS' / 'DA2' / 'mocks'
    mock_ls_dir = dr2_mock_dir / _rename_LSS(version, tracer) / f'altmtl{mock_id}' / 'loa-v1' / f'mock{mock_id}' / 'LSScats'
    if tracer == 'BGS': tracer= 'BGS-BRIGHT-21.35'
    fn = mock_ls_dir / f'{tracer}_full_HPmapcut.dat.h5'
    if not fn.exists():
        raise FileNotFoundError(f'No full_HPmapcut catalog found at {fn}')
    return str(fn)

def get_measurement_ready_fn(version='AbacusHF-test', domain='cubic', tracer='QSO', zrange=(0.8, 2.1), zsnap=1.400, mock_id=0, region='GCcomb', weight_type='default', use_dv=False, z_evol=False, use_jax=False, **kwargs):
    tracer = get_simple_tracer(tracer)
    use_dv = normalize_use_dv(use_dv)
    zlabel = f'z{zrange[0]}-{zrange[1]}'
    if domain in ['cutsky', 'altmtl']:
        if 'ELG' in tracer: tracer = 'ELG_LOPnotqso'
        if 'BGS' in tracer: tracer = 'BGS-BRIGHT-21.35'
        summary_path = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base')
        if version == 'AbacusHF-v2':
            mock_dir = summary_path / 'abacus-hf-dr2-v2-altmtl' / f'mock{mock_id}'
            fn = mock_dir / f'{{}}_{tracer}_{zlabel}_{region}_weight-default-FKP.h5'
        elif version == 'holi-v3':
            mock_dir = summary_path / 'holi-v3-altmtl' / f'mock{mock_id}'
            fn = mock_dir / f'{{}}_{tracer}_{zlabel}_{region}_weight-default-FKP.h5'
    elif domain == 'cubic':
        if version == 'AbacusHF-test':
            mock_dir = Path('/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/measurements_abacushf_MC/v2') / tracer
            fn = mock_dir / f'{{}}_{tracer}_z{zsnap:.3f}_c000_hod-base_los-z_{mock_id}.h5'
        elif version == 'EZmocks-test':
            mock_dir = Path('/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/EZmocks_lsstypes/')
            fn = mock_dir / f'{{}}_{tracer}_z{zsnap:.3f}_c000_los-z_{mock_id}.h5'
    else:
        return None
    return str(fn)

def get_measurement_fn(version='AbacusHF-v2', domain = 'cubic', tracer='LRG', zrange=(0.4, 0.6), zsnap = None, mock_id=0, region='ALL', weight_type='default', use_dv = False, z_evol=False, use_jax = True, **kwargs):
    tracer = get_simple_tracer(tracer)
    mock_id03 =  f"{mock_id:03}"
    if domain == 'cubic':
        base_dir = BASE_DIR / version
        mock_dir = base_dir / 'Boxes' / tracer[:3] / f'sn{_zfmt(zsnap)}' / f'AbacusSummit_base_c000_ph{mock_id03}'
        zlabel = f'zp{zsnap:.3f}'
    elif domain == 'cutsky':
        base_dir = BASE_DIR / version
        mock_dir = base_dir / 'Cutsky' / tracer[:3] / f'z{zsnap:.3f}' / f'AbacusSummit_base_c000_ph{mock_id03}' / 'forclustering'
        zlabel = f'zp{zsnap:.3f}'
    elif domain == 'altmtl':
        base_dir = BASE_DIR / version
        if 'data' in version:
            mock_dir = base_dir
        else:
            mock_dir = base_dir / 'altmtl' / tracer[:3] / f'mock{mock_id}'
        zlabel = f'z{zrange[0]}-{zrange[1]}'
    else:
        raise ValueError(f"Not validated domain {domain!r} (expected cubic/cutsky/altmtl)")
    if version == 'AbacusHF-v2':
        vlabel = '_DR2_v2.0'
    elif version == 'AbacusHF-v1':
        vlabel = '_DR2_v1.0'
    elif version == 'holi-v3':
        vlabel = '_holi_v3'
    elif version in ['data-dr1-v1.5']:
        vlabel = ''
    else:
        raise ValueError(f"Unsupported version {version!r}")
    use_dv = normalize_use_dv(use_dv)
    if use_dv is False:
        dv_suffix = ''
    else:
        dv_suffix = f'+dv_{use_dv}'
        if z_evol == True: 
            dv_suffix = f'+dv_{use_dv}_zevol'
    if version in ["AbacusHF-v2", "AbacusHF-v1"] and mock_id>=25:
        logger.warning(f'mock_id {mock_id} is out of range for AbacusHF-v2 (max 25), returning empty path')
        return None
    region_label = f'_{region}' if domain in ['cutsky', 'altmtl'] and region is not None else ''
    if region == 'ALL': region_label=''
    fn_path = mock_dir / 'mpspk'
    os.makedirs(fn_path, exist_ok=True)
    fn = fn_path / f'{{}}_{tracer}_{zlabel}{region_label}{vlabel}{dv_suffix}.npy'
    fn = str(fn)
    if use_jax: fn = os.path.splitext(fn)[0] + '.h5'
    return fn

def get_simple_tracer(tracer):
    """Return a compact tracer label, stripping redshift-bin suffixes such as LRG1."""
    import re
    if isinstance(tracer, (list, tuple)):
        return '+'.join(get_simple_tracer(item) for item in tracer)
    tracer = str(tracer)
    return '+'.join(re.sub(r'\d+$', '', item) for item in tracer.split('+'))

def get_full_tracer_zrange(tracerz=None, zrange=None):
    """Translate compact tracer-bin labels, e.g. LRG1, to tracer and z-range."""
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _translate(one):
        if 'x' in one:
            return list(zip(*[_translate(item) for item in one.split('x')]))
        if one in translate_zrange:
            return one[:-1], translate_zrange[one]
        if zrange is None:
            raise ValueError(f'zrange not found for {one}; choose one from {list(translate_zrange)}')
        return one, zrange

    if isinstance(tracerz, str):
        return _translate(tracerz)
    return type(tracerz)(zip(*map(_translate, tracerz)))

def read_box_positions(version='AbacusHF-v2', tracer='LRG', zrange=(0.4, 0.6), zsnap = 0.5, mock_id=0, weight_type='default', use_dv = False, domain = 'cubic', **kwargs):
    """
    Return the positions of tracer galaxies for either cubic-box or light-cone mocks, formatted for pycorr / Corrfunc two-point estimators.

    Parameters
    ----------
    use_dv : the redshift error distorted position
    use_random: bool, optional
        If True, load the random catalogs for cutsky
    """    
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.rank
    if domain == 'cubic':
        # basic settings
        los = 'z'
        boxsize = 2000
        cubic_fn = get_catalog_fn(version=version, domain=domain, tracer=tracer, zrange=zrange, zsnap=zsnap, mock_id=mock_id) # load the data
        # if rank == 0: logger.info(f'Load {cubic_fn}')
        cat = Catalog.read(cubic_fn, mpicomm=MPI.COMM_SELF)
        if los == 'z':
            use_dv = normalize_use_dv(use_dv)
            if use_dv in ['repeat', 'verr_empirical', 'verr_nonparam']:
                if use_dv == 'repeat':
                    dv_label = '_REP'
                elif use_dv == 'verr_empirical':
                    dv_label = '_ERR_V1'
                elif use_dv == 'verr_nonparam': 
                    dv_label = '_ERR_V2'
                if rank == 0:
                    logger.info(f'use redshift shifted in {use_dv} mode')
                zcol = f'Z{dv_label}'
            elif use_dv is False:
                zcol = 'Z_RSD'
        positions = np.stack([cat['X'], cat['Y'], cat[zcol]], axis=1) % boxsize
        positions = positions - boxsize / 2.0 # move to 0 center
        weights = np.ones(positions.shape[0], dtype=np.float64)
        return np.array(positions), np.array(weights)

def expand_randoms(randoms, parent_randoms, data, from_randoms=('RA', 'DEC'), from_data=('Z',)):
    """
    Copied from desi-clustering
    """
    special_columns = []
    from_data, from_randoms = list(from_data), list(from_randoms)
    for special in ['FRAC_TLOBS_TILES']:
        if special in from_data:
            special_columns.append(special)
            from_data.remove(special)
            if 'NTILE' not in randoms: from_randoms.append('NTILE')
    if len(from_randoms):
        _, randoms_index, parent_index = np.intersect1d(randoms['TARGETID'], parent_randoms['TARGETID'], return_indices=True)
        randoms = randoms[randoms_index]
        for column in from_randoms:
            if column != 'TARGETID':
                randoms[column] = parent_randoms[column][parent_index]
    if len(from_data) or len(special_columns):
        if isinstance(data, (list, tuple)):  # NGC + SGC
            data = Catalog.concatenate(data)
        else:
            data = data.copy()  # shallow copy
        data['TARGETID_DATA'] = data['TARGETID']
        del data['TARGETID']
        if data['TARGETID_DATA'].max() < int(1e9):  # faster method
            lookup = np.full(1 + data['TARGETID_DATA'].max(), -1, dtype='i8')
            lookup[data['TARGETID_DATA']] = np.arange(len(data))
            random_targetid_data = np.asarray(randoms['TARGETID_DATA'])
            in_bounds = (random_targetid_data >= 0) & (random_targetid_data < len(lookup))
            index = np.full(len(random_targetid_data), -1, dtype='i8')
            index[in_bounds] = lookup[random_targetid_data[in_bounds]]
        else:
            sorted_index = np.argsort(data['TARGETID_DATA'])
            data_targetid_data = np.asarray(data['TARGETID_DATA'])
            random_targetid_data = np.asarray(randoms['TARGETID_DATA'])
            index_in_sorted = np.searchsorted(data_targetid_data, random_targetid_data, sorter=sorted_index)
            index = np.full(len(random_targetid_data), -1, dtype='i8')
            found = index_in_sorted < len(data_targetid_data)
            matched = found & (data_targetid_data[sorted_index[np.clip(index_in_sorted, 0, len(data_targetid_data) - 1)]] == random_targetid_data)
            index[matched] = sorted_index[index_in_sorted[matched]]
        if np.any(index < 0):
            nmissing = np.count_nonzero(index < 0)
            raise ValueError(f'Could not match {nmissing} TARGETID_DATA entries between randoms and data')
        for column in from_data:
            randoms[column] = data[column][index]
        if 'FRAC_TLOBS_TILES' in special_columns:
            # Total random weights is FRAC_TLOBS_TILES * (WEIGHT_SYS * WEIGHT_COMP * WEIGHT_ZFAIL coming from z shuffling) * overall region-based normalization factor
            # Correct up to a given region-based normalization factor
            data_wtotp = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']
            randoms['FRAC_TLOBS_TILES'] = randoms.ones()
            for region in ['NGC', 'SGC']:
                mask_region_data = select_region(data['RA'], data['DEC'], region)
                mask_region_randoms = select_region(randoms['RA'], randoms['DEC'], region)
                data_wcomp_ntile = _compute_binned_weight(data['NTILE'][mask_region_data], data_wtotp[mask_region_data] / data['WEIGHT'][mask_region_data])
                randoms['FRAC_TLOBS_TILES'][mask_region_randoms] = randoms['WEIGHT'][mask_region_randoms] / data_wtotp[index[mask_region_randoms]] * data_wcomp_ntile[randoms['NTILE'][mask_region_randoms]]
            #data_wcomp_ntile = _compute_binned_weight(data['NTILE'], data_wtotp / data['WEIGHT'])
            #randoms['FRAC_TLOBS_TILES'] = randoms['WEIGHT'] / data_wtotp[index] * data_wcomp_ntile[randoms['NTILE']]
    return randoms

def _read_catalog(*args, quiet_nonroot=False, **kwargs):
    def _is_fits_path(value):
        if isinstance(value, (list, tuple)):
            return len(value) > 0 and all(_is_fits_path(item) for item in value)
        name = str(value).lower()
        return name.endswith('.fits') or name.endswith('.fits.gz')
    @contextmanager
    def _suppress_nonroot_loggers(*names, level=logging.WARNING):
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank == 0:
            yield
            return
        states = []
        try:
            for name in names:
                log = logging.getLogger(name)
                states.append((log, log.level))
                log.setLevel(level)
            yield
        finally:
            for log, old_level in states:
                log.setLevel(old_level)
    if args and _is_fits_path(args[0]):
        kwargs.pop('group', None)
    if quiet_nonroot:
        with _suppress_nonroot_loggers('FileStack'):
            return Catalog.read(*args, **kwargs)
    return Catalog.read(*args, **kwargs)

def read_cutsky_positions(version='AbacusHF-v2', domain = 'altmtl', tracer='LRG', zrange=(0.4, 0.6), mock_id=0, region='ALL', weight_type='WEIGHT_FKP', use_dv = False, z_evol=False, random=False, nran=None, use_jax = True, extra_columns=(), **kwargs):
    # load the data
    from mpi4py import MPI 
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.rank
    (zmin, zmax) = (zrange[0], zrange[1])
    if random == True: 
        use_dv = False
        rans = _read_catalog(get_catalog_fn(version, domain, tracer, mock_id, random=True, nran=nran, region=region), group='LSS')
        pars = _read_catalog(get_catalog_fn(version, domain, tracer, mock_id, random='parent', nran=nran, region=region), group='LSS', mpicomm=MPI.COMM_SELF, quiet_nonroot=True)
        dats = _read_catalog(get_catalog_fn(version, domain, tracer, mock_id, region='ALL'), group='LSS', mpicomm=MPI.COMM_SELF, quiet_nonroot=True)
        cat = expand_randoms(rans, pars, dats, from_data=('Z',))
    else:
        cat = _read_catalog(get_catalog_fn(version, domain, tracer, mock_id, region=region), group='LSS')
    cat = _set_fiducial_weight_fkp(cat, tracer=tracer, weight_type=weight_type)
    sel = np.isfinite(cat['Z'])
    if 'NX' in cat.columns():
        sel &= (np.asarray(cat['NX']) != 0)
    use_dv = normalize_use_dv(use_dv)
    if use_dv is not False: 
        dv = sample_cutsky_dv(tracer, cat['Z'], zmin, zmax, use_dv=use_dv, z_evol=z_evol)
    elif use_dv is False:
        dv = np.zeros(len(cat))
    cat['Z'] = cat['Z'] + dv/CSPEED*(1+cat['Z'])
    selz = (cat['Z'] >= zmin) & (cat['Z'] < zmax) 
    cat_sel = cat[sel&selz]
    local_nsel = len(cat_sel)
    global_nsel = mpicomm.allreduce(local_nsel)
    if global_nsel == 0:
        if rank == 0:
            logger.warning(f"No objects remain after Z selection")
        empty = np.empty((0, 3), dtype='f8'), np.empty((0,), dtype='f8')
        if extra_columns:
            return (*empty, {})
        return empty
    if local_nsel == 0:
        empty = np.empty((0, 3), dtype='f8'), np.empty((0,), dtype='f8')
        if extra_columns:
            return (*empty, {})
        return empty
    ra = np.radians(np.asarray(cat_sel['RA']))
    dec = np.radians(np.asarray(cat_sel['DEC']))
    dist = comoving_radial_distance(np.asarray(cat_sel['Z']))
    if use_jax ==True:
        cos_dec = np.cos(dec)
        positions = np.stack([
            dist * cos_dec * np.cos(ra),
            dist * cos_dec * np.sin(ra),
            dist * np.sin(dec),
        ], axis=1)
    else:
        positions = np.stack([ra, dec, dist], axis=1)
    mask_good = np.all(np.isfinite(positions), axis=1)
    if (~mask_good).sum() > 0:
        if rank == 0: logger.info(f"Data warning: dropping {(~mask_good).sum()} non-finite points")
    weights = get_cutsky_weights(cat_sel, weight_type=weight_type)[mask_good]
    positions = np.asarray(positions, dtype='f8')
    weights = np.asarray(weights, dtype='f8')
    if extra_columns:
        extra = {}
        if 'IDS' in extra_columns:
            if 'TARGETID' not in cat_sel.columns():
                raise ValueError("TARGETID column is required for extra_columns=('IDS',)")
            extra['IDS'] = np.asarray(cat_sel['TARGETID'])[mask_good]
        return positions, weights, extra
    return positions, weights

def read_positions_weights(version='AbacusHF-v2', domain='cubic', **kwargs):
    if domain == 'cubic':
        return read_box_positions(version=version, domain=domain, **kwargs)
    if domain in ['cutsky', 'altmtl']:
        return read_cutsky_positions(version=version, domain=domain, **kwargs)
    raise ValueError(f"Unsupported domain {domain}")
