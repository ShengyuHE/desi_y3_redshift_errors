#!/usr/bin/env python
# source /global/homes/s/shengyu/env.sh rc_env
# srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python get_repeat_redshifts.py
#credit: Ashley Ross

import sys
import os
import logging
import shutil
import unittest
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import fitsio
import glob
import argparse
from astropy.table import Table,join,unique,vstack
from matplotlib import pyplot as plt

from LSS import common_tools as common
from LSS.main import cattools as ct
from LSS.globals import main

MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MAIN_DIR))
from utils import setup_logging

setup_logging()
logger=logging.getLogger('get_repeat')

def get_repeats(indat, labels = None):
    if labels is None:
        labels = ['Z']
    tids,cnts = np.unique(indat['TARGETID'],return_counts=True)
    rtids = tids[cnts>1]
    sel_r = np.isin(indat['TARGETID'],rtids)
    specflr = indat[sel_r]
    logger.info("N rows with TARGETID repeated: %d", len(specflr))
    specflr.sort("TARGETID")
    out = {'TARGETID': []}
    for lab in labels:
        out[f'{lab}_1'] = []
        out[f'{lab}_2'] = []
    ind = 0
    n = len(specflr)
    while ind < n:
        row1 = specflr[ind]
        tid = row1['TARGETID']
        ind2 = 1
        while ind + ind2 < n and specflr[ind + ind2]['TARGETID'] == tid:
            row2 = specflr[ind + ind2]
            out['TARGETID'].append(tid)
            for lab in labels:
                out[f'{lab}_1'].append(row1[lab])
                out[f'{lab}_2'].append(row2[lab])
            ind2 += 1
        ind += ind2
        if ind % 100000 == 0:
            logger.info("processed %d", ind)
    return Table(out)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracer",help="version for redshifts",default='ELG')
    parser.add_argument("--survey",help="set of tiles, e.g., Y1 for DR1, DA2 for DR2, main for all of main survey",default='DA2')
    parser.add_argument("--verspec",help="version for redshifts, e.g. loa-v1, iron",default='loa-v1')
    parser.add_argument("--outdir",help="directory for output",default=None)
    args = parser.parse_args()
    logger.info('Received args: %s', args)

    outdir = '/global/cfs/cdirs/desi/users/shengyu/repeats/DA2/loa-v1'

    tracer = args.tracer
    survey = args.survey
    version = args.verspec

    if tracer in ['BGS']:
        mainp = main('BGS', version)#get settings for dark time
        time = 'bright'
    elif tracer in ['LRG', 'ELG', 'QSO']:
        mainp = main('LRG', version)#get settings for dark time
        time = 'dark'

    mt = mainp.mtld
    tiles = mainp.tiles
    imbits = mainp.imbits #mask bits applied to targeting
    ebits = mainp.ebits #extra mask bits we think should be applied

    tsnrcut = mainp.tsnrcut
    dchi2 = mainp.dchi2
    tnsrcol = mainp.tsnrcol        
    zmin = mainp.zmin
    zmax = mainp.zmax
    badfib = mainp.badfib

    #get set of tiles
    wd = mt['SURVEY'] == 'main'
    wd &= mt['ZDONE'] == 'true'
    wd &= mt['FAPRGRM'] == time
    if args.survey == 'Y1':
        wd &= mt['ZDATE'] < 20220900
    if args.survey == 'DA2':
        wd &= mt['ZDATE'] < 20240410
    mtld = mt[wd]

    ldirspec = '/global/cfs/cdirs/desi/survey/catalogs/'+survey+'/LSS/'+version+'/'
    if args.outdir is None:
        outdir = '/global/cfs/cdirs/desi/users/shengyu/repeats/'+survey+'/'+version+'/'
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    specfo = ldirspec+f'datcomb_{time}_spec_zdone.fits'
    specf = Table(fitsio.read(specfo.replace('global','dvs_ro')))
    sel = np.isin(specf['TILEID'],mtld['TILEID'])
    specf = specf[sel]
    specf = Table(specf)
    if time == 'dark':
        specf.keep_columns(['TARGETID','Z','ZWARN','ZERR','DELTACHI2','LOCATION','DESI_TARGET','TILEID','TSNR2_LRG','TSNR2_ELG', 'TSNR2_QSO', 'ZWARN_MTL','COADD_FIBERSTATUS','FIBER','LASTNIGHT'])
    elif time == 'bright':
        specf.keep_columns(['TARGETID','Z','ZWARN','ZERR','DELTACHI2','LOCATION','BGS_TARGET','TILEID','TSNR2_BGS','TSNR2_ELG','ZWARN_MTL','COADD_FIBERSTATUS','FIBER','LASTNIGHT'])
    specf = common.cut_specdat(specf,badfib=mainp.badfib_td,tsnr_min=tsnrcut,tsnr_col=tnsrcol,fibstatusbits=mainp.badfib_status,remove_badfiber_spike_nz=True,mask_petal_nights=True)

    zcol = 'Z_QF' if tracer == 'QSO' else 'Z'
    labels = [zcol, f'TSNR2_{tracer}', 'TILEID', 'ZWARN', 'ZERR', 'DELTACHI2']

    if tracer == 'BGS':
        logger.info("Processing BGS repeats")
        sel_BGS = specf['BGS_TARGET'] > 0
        sel_gz = common.goodz_infull('BGS',specf,zcol='Z')
        specfl = specf[sel_BGS&sel_gz]
        bgsr = get_repeats(specfl, labels=labels)
        bgsr.write(os.path.join(outdir, 'BGSrepeats.fits'),overwrite=True)

    if tracer == 'LRG':
        logger.info("Processing LRG repeats")
        sel_LRG = (specf['DESI_TARGET'] & 1) > 0
        sel_gz = common.goodz_infull('LRG',specf,zcol='Z')
        specfl = specf[sel_LRG&sel_gz]
        lrgr = get_repeats(specfl,labels=labels)
        lrgr.write(os.path.join(outdir, 'LRGrepeats.fits'),overwrite=True)

    if tracer == 'ELG':
        logger.info("Processing ELG repeats")
        elgf = fitsio.read(mainp.elgzf,columns=['TARGETID','LOCATION','TILEID','OII_FLUX','OII_FLUX_IVAR'])
        specf = join(specf,elgf,keys=['TARGETID','TILEID','LOCATION'],join_type='left')
        o2c = np.log10(specf['OII_FLUX'] * np.sqrt(specf['OII_FLUX_IVAR']))+0.2*np.log10(specf['DELTACHI2'])
        w = (o2c*0) != 0
        w |= specf['OII_FLUX'] < 0
        o2c[w] = -20
        specf['o2c'] = o2c
        sel_ELG = (specf['DESI_TARGET'] & 2) > 0
        sel_gz = common.goodz_infull('ELG',specf,zcol='Z')
        specfe = specf[sel_ELG&sel_gz]
        elgr = get_repeats(specfe, labels=labels)
        elgr.write(os.path.join(outdir, 'ELGrepeats.fits'),overwrite=True)

    if tracer == 'QSO':
        logger.info("Processing QSO repeats")
        qsof = fitsio.read(mainp.qsozf,columns=['TARGETID','LOCATION','TILEID','Z'])
        specf = join(specf,qsof,keys=['TARGETID','TILEID','LOCATION'],join_type='left',uniq_col_name='{col_name}{table_name}',table_names=['','_QF'])
        selqso = specf['Z_QF']!=999999
        selqso &= (specf['DESI_TARGET'] & 4) > 0
        specfq = specf[selqso]
        qsor = get_repeats(specfq, labels=labels)
        qsor.write(os.path.join(outdir, 'QSOrepeats.fits'),overwrite=True)

'''
#do BGS
mainp = main('BGS',args.verspec)#get settings for dark time
mt = mainp.mtld
tiles = mainp.tiles
imbits = mainp.imbits #mask bits applied to targeting
ebits = mainp.ebits #extra mask bits we think should be applied

tsnrcut = mainp.tsnrcut
dchi2 = mainp.dchi2
tnsrcol = mainp.tsnrcol        
zmin = mainp.zmin
zmax = mainp.zmax
badfib = mainp.badfib

#get set of tiles
wd = mt['SURVEY'] == 'main'
wd &= mt['ZDONE'] == 'true'
wd &= mt['FAPRGRM'] == 'bright'
if args.survey == 'Y1':
    wd &= mt['ZDATE'] < 20220900
if args.survey == 'DA2':
    wd &= mt['ZDATE'] < 20240410
mtld = mt[wd]

ldirspec = '/global/cfs/cdirs/desi/survey/catalogs/'+args.survey+'/LSS/'+args.verspec+'/'
specfo = ldirspec+'datcomb_bright_spec_zdone.fits'
specf = Table(fitsio.read(specfo.replace('global','dvs_ro')))
sel = np.isin(specf['TILEID'],mtld['TILEID'])
specf = specf[sel]
specf = Table(specf)
specf.keep_columns(['TARGETID','Z','ZWARN','DELTACHI2','LOCATION','BGS_TARGET','TILEID','TSNR2_BGS','TSNR2_ELG','ZWARN_MTL','COADD_FIBERSTATUS','FIBER','LASTNIGHT'])
specf = common.cut_specdat(specf,badfib=mainp.badfib_td,tsnr_min=tsnrcut,tsnr_col=tnsrcol,fibstatusbits=mainp.badfib_status,remove_badfiber_spike_nz=True,mask_petal_nights=True)

#do BGS
sel_BGS = specf['BGS_TARGET'] > 0
sel_gz = common.goodz_infull('BGS',specf,zcol='Z')
specfl = specf[sel_BGS&sel_gz]

bgsr = get_repeats(specfl)
bgsr.write(args.outdir+'/BGSrepeats.fits',overwrite=True)

sel = abs((bgsr['Z1']-bgsr['Z2'])/(1+bgsr['Z1'])) > 0.003
print('fraction of repeat BGS measurements with (Z1-Z2)/(1+Z1) > 0.003:')
print(np.sum(sel)/len(bgsr))
    
#do dark time tracers

mainp = main('LRG',args.verspec)#get settings for dark time

mt = mainp.mtld
tiles = mainp.tiles
imbits = mainp.imbits #mask bits applied to targeting
ebits = mainp.ebits #extra mask bits we think should be applied

tsnrcut = mainp.tsnrcut
dchi2 = mainp.dchi2
tnsrcol = mainp.tsnrcol        
zmin = mainp.zmin
zmax = mainp.zmax
badfib = mainp.badfib

#get set of tiles
wd = mt['SURVEY'] == 'main'
wd &= mt['ZDONE'] == 'true'
wd &= mt['FAPRGRM'] == 'dark'
if args.survey == 'Y1':
    wd &= mt['ZDATE'] < 20220900
if args.survey == 'DA2':
    wd &= mt['ZDATE'] < 20240410
mtld = mt[wd]

specfo = ldirspec+'datcomb_dark_spec_zdone.fits'
specf = Table(fitsio.read(specfo.replace('global','dvs_ro')))
sel = np.isin(specf['TILEID'],mtld['TILEID'])
specf = specf[sel]
specf = Table(specf)
specf.keep_columns(['TARGETID','Z','ZWARN','DELTACHI2','LOCATION','DESI_TARGET','TILEID','TSNR2_LRG','TSNR2_ELG','ZWARN_MTL','COADD_FIBERSTATUS','FIBER','LASTNIGHT'])
specf = common.cut_specdat(specf,badfib=mainp.badfib_td,tsnr_min=tsnrcut,tsnr_col=tnsrcol,fibstatusbits=mainp.badfib_status,remove_badfiber_spike_nz=True,mask_petal_nights=True)

#do LRGs
sel_LRG = (specf['DESI_TARGET']) & 1 > 0
sel_gz = common.goodz_infull('LRG',specf,zcol='Z')
specfl = specf[sel_LRG&sel_gz]

lrgr = get_repeats(specfl)
lrgr.write(args.outdir+'/LRGrepeats.fits',overwrite=True)

sel = abs((lrgr['Z1']-lrgr['Z2'])/(1+lrgr['Z1'])) > 0.003
print('fraction of repeat LRG measurements with (Z1-Z2)/(1+Z1) > 0.003:')
print(np.sum(sel)/len(lrgr))

#do QSO
qsof = fitsio.read(mainp.qsozf,columns=['TARGETID','LOCATION','TILEID','Z'])
specf = join(specf,qsof,keys=['TARGETID','TILEID','LOCATION'],join_type='left',uniq_col_name='{col_name}{table_name}',table_names=['','_QF'])
selqso = specf['Z_QF']!=999999
selqso &= (specf['DESI_TARGET'] & 4) > 0
specfq = specf[selqso]

qsor = get_repeats(specfq,zcol='Z_QF')
qsor.write(args.outdir+'/QSOrepeats.fits',overwrite=True)
sel = abs((qsor['Z1']-qsor['Z2'])/(1+qsor['Z1'])) > 0.01
print('fraction of repeat QSO measurements with (Z1-Z2)/(1+Z1) > 0.01:')
print(np.sum(sel)/len(qsor))

#do elg
elgf = fitsio.read(mainp.elgzf,columns=['TARGETID','LOCATION','TILEID','OII_FLUX','OII_FLUX_IVAR'])
specf = join(specf,elgf,keys=['TARGETID','TILEID','LOCATION'],join_type='left')
o2c = np.log10(specf['OII_FLUX'] * np.sqrt(specf['OII_FLUX_IVAR']))+0.2*np.log10(specf['DELTACHI2'])
w = (o2c*0) != 0
w |= specf['OII_FLUX'] < 0
o2c[w] = -20
specf['o2c'] = o2c
sel_ELG = (specf['DESI_TARGET'] & 2) > 0
sel_gz = common.goodz_infull('ELG',specf,zcol='Z')
specfe = specf[sel_ELG&sel_gz]

elgr = get_repeats(specfe)
elgr.write(args.outdir+'/ELGrepeats.fits',overwrite=True)
sel = abs((elgr['Z1']-elgr['Z2'])/(1+elgr['Z1'])) > 0.001
print('fraction of repeat ELG measurements with (Z1-Z2)/(1+Z1) > 0.001:')
print(np.sum(sel)/len(elgr))
'''
