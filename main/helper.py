import numpy as np

REDSHIFT_OVERALL = dict(BGS = (0.1, 0.4),
                       LRG = (0.4, 1.1), 
                       ELG = (0.8, 1.6),
                       QSO = (0.8, 2.1))

COLOR_OVERALL = dict(BGS = 'yellowgreen',
                    LRG = 'red',
                    ELG = 'blue',
                    QSO = 'darkgreen')

REDSHIFT_LSS_VSMEAR = dict(BGS = [(0.1, 0.4)],
                       LRG = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 
                       ELG = [(0.8, 1.1), (1.1,1.6)],
                       QSO = [(0.8, 2.1)])

REDSHIFT_LSS_CUBICBOX = dict(BGS = [0.200],
                         LRG = [0.500, 0.800, 0.800],
                         ELG= [0.800, 1.100],
                         QSO = [1.100])


REDSHIFT_VSMEAR = dict(BGS = [(0.1, 0.4)],
                       LRG = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 
                       ELG = [(0.8, 1.1), (1.1, 1.3), (1.3, 1.6)],
                       QSO = [(0.8, 1.1), (1.1, 1.4), (1.4, 1.7), (1.7, 2.1)])
                      
REDSHIFT_CUBICBOX = dict(BGS = [0.200],
                         LRG = [0.500, 0.500, 0.500],
                         ELG= [0.800, 0.800, 0.800],
                         QSO = [1.100, 1.100, 1.100, 1.100])

REDSHIFT_ABACUS_Y3 = dict(BGS = None,
                         LRG = [0.500, 0.725, 0.950],
                         ELG= [0.950, 1.175, 1.475],
                         QSO = [0.950, 1.250, 1.550, 1.850])


REDSHIFT_LSS_EZMOCKS = dict(BGS = [0.200],
                         LRG = [0.500, 0.800, 0.800],
                         ELG= [0.950, 1.100],
                         QSO = [1.100])


# REDSHIFT_LSS_CUBICBOX = dict(BGS = [0.200],
#                          LRG = [0.500, 0.500, 0.500],
#                          ELG= [0.800, 0.800],
#                          QSO = [1.100])
# REDSHIFT_CUBICBOX = dict(LRG = [0.500, 0.800, 0.800],
#                          ELG= [0.800, 1.100, 1.100],
#                          QSO = [0.800, 1.100, 1.100, 1.100])

Y3_EFFECTIVE_VOLUME = dict(BGS = [3.8], 
                           LRG = [4.9, 7.6, 9.8],
                           ELG = [5.8, 8.3],
                           QSO = [2.7])

# RSF_COV_ERROR = dict(LRG = [0.319, 0.256, 0.226],
#                      ELG = [0.294, 0.245],
#                      QSO = [0.430])

# RSF_CUBIC_ERROR = dict(BGS = [0.725],
#                        LRG = [0.639, 0.513, 0.452],
#                        ELG = [0.587, 0.491],
#                        QSO = [0.861])

# RSF_EZMOCKS_ERROR = dict(BGS = None,
#                          LRG = [0.553, 0.444, 0.391],
#                          ELG = [0.509, 0.425],
#                          QSO = [0.745])

RSF_COV_ERROR = dict(LRG = [0.0078, 0.0050, 0.0039],
                     ELG = [0.0066, 0.0046],
                     QSO = [0.0141])

RSF_CUBIC_ERROR = dict(BGS = [0.6434],
                       LRG = [0.4990, 0.3217, 0.2495],
                       ELG = [0.4216, 0.2946],
                       QSO = [0.9056])

RSF_EZMOCKS_ERROR = dict(BGS = None,
                         LRG = [0.2105, 0.1357, 0.1053],
                         ELG = [0.1778, 0.1243],
                         QSO = [0.3820])



COLOR_TRACERS = dict(BGS = ['yellowgreen'],
                    LRG = ['orange', 'orangered', 'firebrick'],
                    ELG = ['skyblue', 'steelblue'],
                    QSO = ['purple'])

# NUMBER_Y3 = dict(BGS = 1188526, LRG = 4468483, ELG = 6534844, QSO = 2062839)
# NUMBER_CUBICBOX = dict(BGS = 4309368, LRG = 2982225, ELG= 6747351, QSO = 221666)
# NUMBER_COVBOX = dict(LRG = 63112, ELG= 105268, QSO = 4004)
# RSF_CUBIC_ERROR = dict(BGS= 1.904, LRG = 0.8169, ELG= 1.016, QSO = 0.3278)
# RSF_COV_ERROR = dict(LRG = 0.1187, ELG= 0.1271, QSO = 0.0443)
# NUMBER_CUBICBOX = dict(LRG = [2982225, 2982225, 2982225],
#                          ELG= [6747351, 3367861, 3367861],
#                          QSO = [218809, 221666, 221666, 221666])   
# RSF_ERROR_CUBICBOX = dict(LRG = [0.817, 0.817, 0.817],
#                          ELG= [1.016, 0.718, 0.718],
#                          QSO = [0.326, 0.328, 0.328, 0.328])
# Y3_REDSHIFT_TRACERS_NUMBER = dict(LRG = [1052151, 1613562, 1802770],
#                                   ELG= [2737573, 3797271, 3797271],
#                                   QSO = [2062839, 2062839, 2062839, 2062839])
# LRG = 1052151+1613562+1802770 = 4468483 -- 2982225
# ELG = 2737573 + 3797271 = 6534844  -- 6747351
# QSO = 2062839 -- 3367861

NRAN = {'LRG': 8, 'ELG': 10, 'QSO': 4}

Y3_ZRANGE = {'BGS_ANY-02':[(0.1, 0.4)],
             'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 
             'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 
             'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 
             'QSO': [(0.8, 2.1)]}
Y3_SMOOTHING = {'BGS_ANY-02':[15.], 'LRG': [15.], 'LRG+ELG_LOPnotqso': [15.], 'ELG_LOPnotqso': [15.], 'QSO': [30.]}
Y3_NRAN = {'LRG': 8, 'LRG+ELG_LOPnotqso': 10, 'ELG_LOPnotqso': 10, 'QSO': 4}
Y3_BOXSIZE = {'LRG': 7000., 'LRG+ELG_LOPnotqso': 9000., 'ELG_LOPnotqso': 9000., 'QSO': 10000.}

TRACER_CUBICBOX = dict(LRG = 'lrg', ELG= 'elg', QSO = 'qso')

PLANCK_COSMOLOGY = {
    "Omega_m": 0.315191868,
    "H_0": 67.36,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
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


def GET_RECON_BIAS(tracer='LRG', grid_cosmo=None): # need update for different cosmologies
    if tracer.startswith('BGS'):
        f=  0.682
        bias = {'000': 1.5, '001': 1.7, '002': 1.6, '003': 1.6, '004': 1.8}
        smoothing_radius = 15.
    elif tracer.startswith('LRG+ELG'):
        f = 0.85
        bias = {'000': 1.6, '001': 1.7, '002': 1.6, '003': 1.6, '004': 1.8}
        smoothing_radius = 15.
    elif tracer.startswith('LRG'):
        f =  0.834
        bias = {'000': 2.0, '001': 2.1, '002': 1.9, '003': 1.9, '004': 2.2}
        smoothing_radius = 15.
    elif tracer.startswith('ELG'):
        f= 0.9
        bias = {'000': 1.2, '001': 1.3, '002': 1.2, '003': 1.2, '004': 1.4}
        smoothing_radius = 15.
    elif tracer.startswith('QSO'):
        f= 0.928
        bias = {'000': 2.1, '001': 2.3, '002': 2.1, '003': 2.1, '004': 2.4}
        smoothing_radius = 30.
    else:
        raise ValueError('unknown tracer {}'.format(tracer))
    if grid_cosmo is None:
        bias = bias['000']
    else:
        bias = bias[grid_cosmo]
    return f, bias, smoothing_radius

def SKY_TO_CARTESIAN(rdd, degree=True):
    conversion = 1.
    if degree: conversion = np.pi / 180.
    ra, dec, dist = rdd
    cos_dec = np.cos(dec * conversion)
    x = dist * cos_dec * np.cos(ra * conversion)
    y = dist * cos_dec * np.sin(ra * conversion)
    z = dist * np.sin(dec * conversion)
    return [x, y, z]



def GET_LINE_CONFUSION(tracer):
    # if tracer == 'BGS':
    
    if tracer == 'LRG':
        return 0

    if tracer == 'ELG':
        return 0

    if tracer == 'QSO':
        # line_set = [1215.67, 1549, 1908, 2800, 3727, 3868, 
        #             4101, 4340, 4861, 4958, 5007, 6562]
        # name_set = [r'Ly$\alpha$', 'C[IV]', 'C[III]', 'Mg[II]', 'O[II]', 'Ne[III]',
        #             r'H$\delta$', r'H$\gamma$', r'H$\beta$', 'O[III]1', 'O[III]2', r'H$\alpha$']
        line_set = [2800, 1908, 4340, 1549, 1215.67]
        name_set = ['Mg[II]', 'C[III]', r'H$\gamma$', 'C[IV]',r'Ly$\alpha$',]
    return (line_set, name_set)
