# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:36:09 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt


#::: if only 1 object is contained in dic
def set_nan(dic):
    ###### REMOVE BROKEN ITEMS #######
    #::: nan
    ind_broken = np.where( dic['SYSREM_FLUX3'] == 0. )
    if 'SYSREM_FLUX3' in dic: dic['SYSREM_FLUX3'][ind_broken] = np.nan
#    dic['HJD'][ind_broken] = np.nan #this is not allowed to be set to nan!!! Otherwise the binning will be messed up!!!
    if 'CCDX' in dic: dic['CCDX'][ind_broken] = np.nan
    if 'CCDY' in dic: dic['CCDY'][ind_broken] = np.nan
    if 'CENTDX' in dic: dic['CENTDX'][ind_broken] = np.nan
    if 'CENTDY' in dic: dic['CENTDY'][ind_broken] = np.nan
    return dic
    

#::: if multiple objects are contained in dic
def set_nan_multi(dic):
    ###### REMOVE BROKEN ITEMS #######
    #::: nan
    N_obj = dic['SYSREM_FLUX3'].shape[0]
    for obj_nr in range(N_obj):
        ind_broken = np.where( dic['SYSREM_FLUX3'][obj_nr] == 0. )
        if 'SYSREM_FLUX3' in dic: dic['SYSREM_FLUX3'][obj_nr,ind_broken] = np.nan
    #    dic['HJD'][ind_broken] = np.nan #this is not allowed to be set to nan!!! Otherwise the binning will be messed up!!!
        if 'CCDX' in dic: dic['CCDX'][obj_nr,ind_broken] = np.nan
        if 'CCDY' in dic: dic['CCDY'][obj_nr,ind_broken] = np.nan
        if 'CENTDX' in dic: dic['CENTDX'][obj_nr,ind_broken] = np.nan
        if 'CENTDY' in dic: dic['CENTDY'][obj_nr,ind_broken] = np.nan
    return dic