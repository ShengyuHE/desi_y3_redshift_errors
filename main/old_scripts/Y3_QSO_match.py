import numpy as np
import os
import fitsio
from astropy.table import Table,join
from astropy import constants

from desitarget.targetmask import desi_mask
from desitarget.targetmask import zwarn_mask as zmtl_zwarn_mask

repeatdir= '/global/cfs/projectdirs/desi/users/jiaxiyu/repeated_observations/EDR_vs_Y3/LSS-scripts_repeats/main-repeats-kibo-dark-pairs.fits'
qsofn    = repeatdir[:-5]+'_QSO'+repeatdir[-5:]
if not os.path.exists(qsofn):
    d        = Table.read(repeatdir)
    tracer   = 'QSO'
    goodkey  = "GOOD_QSO"
    mask, mask_key = desi_mask, "DESI_TARGET"
    effkey, effmin, effmax, effxlim = ("TSNR2_LRG",0.85 * 1000,1.5 * 1000,(500, 1500))
    # efftime_spec calculation for selections
    snr2time = d.meta["{}SNR2T".format(effkey.split("_")[1])]
    efftime0s= snr2time * d["{}_0".format(effkey)]
    efftime1s= snr2time * d["{}_1".format(effkey)]
    # zmtl_zwarn_mask nodata + bad selections
    nodata0  = (d["ZMTL_ZWARN_0"] & zmtl_zwarn_mask["NODATA"]) > 0
    nodata1  = (d["ZMTL_ZWARN_1"] & zmtl_zwarn_mask["NODATA"]) > 0
    badqa0   = (d["ZMTL_ZWARN_0"] & zmtl_zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA")) > 0
    badqa1   = (d["ZMTL_ZWARN_1"] & zmtl_zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA")) > 0
    # Apply the selection criteria to clean the data
    sel      = (d[mask_key] & mask[tracer]) > 0
    sel &= (d["COADD_FIBERSTATUS_0"] == 0) & (d["COADD_FIBERSTATUS_1"] == 0)
    sel &= (~nodata0) & (~nodata1)
    sel &= (~badqa0) & (~badqa1)
    sel &= (efftime0s > effmin) & (efftime1s > effmin)
    sel &= (efftime0s < effmax) & (efftime1s < effmax)
    sel &= (d["{}_0".format(goodkey)]) & (d["{}_1".format(goodkey)])
    sel &= (d['SURVEY_0']=='main')&(d['SURVEY_1']=='main')
    print('write QSO-only file')
    d    = d[sel]
    d.write(qsofn,overwrite=True)
else:
    d        = Table(fitsio.read(qsofn))
    
if np.isnan(d["DV"]).sum() == 0:
    print("QSO redshifts needs to be corrected")
    columns  = ['TARGETID','TILEID','LASTNIGHT','Z']
    d_zacc   = Table(fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/DA2/QSO/kibo/QSO_cat_kibo_cumulative_v3.fits',columns=columns))

    # change the name in the data file to be match with the accurate redshift table
    for rid in range(2):
        for col in columns[1:]:
            if col not in d.colnames:
                d[f'{col}_{rid}'].name = col
        # join the two tables to obtain the true QSO measurement from d_zacc in d_ztrue
        d_ztrue  = []
        d_ztrue  = join(d,d_zacc,keys=columns[:-1],join_type='left',table_names=[f'{rid}', f'{rid}_true'])
        d_ztrue[f'Z_{rid}'] = d_ztrue[f'Z_{rid}_true']*1
        # return to the *_{rid} names
        for col in columns[1:-1]:
            d_ztrue[f'{col}'].name = f'{col}_{rid}'
        # 
        d = d_ztrue.copy()

    # save the new file
    d_ztrue['DV'].name = 'DV_Redrock'
    d_ztrue['DV'] = constants.c.to("km/s").value * (d_ztrue["Z_1"] - d_ztrue["Z_0"]) / (1 + d_ztrue["Z_0"])
    d_ztrue['Z_0'][d_ztrue.mask['Z_0_true']] = np.nan
    d_ztrue['Z_1'][d_ztrue.mask['Z_1_true']] = np.nan
    d_ztrue['DV'][d_ztrue.mask['DV']]= np.nan
    d_ztrue.remove_columns(['Z_0_true','Z_1_true'])

    d_ztrue.write(qsofn,overwrite=True,format='fits')
else:
    print('QSO repeated pairs are ready. Please go ahead for the systematics analysis')