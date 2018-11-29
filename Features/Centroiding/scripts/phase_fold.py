# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:55:21 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
#import matplotlib.pyplot as plt
from My_Utils import medsig


def rebin_err(t, f, ferr=None, dt = 0.02, ferr_type='medsig', ferr_style='std'):
    """
    @written by Ed Gillen
    The standard rebin function but also dealing with errors
    on the individual data points being binned.
    ferr_type:
        'medsig'
        'meanstd'
    ferr_style:
        'std'
        'sem' = std / sqrt(N)
    """
    treg = np.r_[t.min():t.max():dt]
    nreg = len(treg)
    freg = np.zeros(nreg) + np.nan
#    if ferr!=None:
    freg_err = np.ma.zeros(nreg) + np.nan
    for i in np.arange(nreg):
        l = (t >= treg[i]) * (t < treg[i] + dt)
        if l.any():
            treg[i] = np.ma.mean(t[l])
            if ferr==None:
                if ferr_type == 'medsig':
                    freg[i], freg_err[i] = medsig(f[l])
                else:
                    freg[i] = np.nanmean(f[l])
                    freg_err[i] = np.nanstd(f[l])
                    
                if ferr_style == 'sem':
                    freg_err[i] /= np.sqrt( len(f[l]) )
            else:
                freg[i], freg_err[i] = weighted_avg_and_std( f[l], np.ma.array([1/float(x) for x in ferr[l]]) )
    l = np.isfinite(freg)
#    if ferr==None:
#        return treg[l], freg[l]
    return treg[l], freg[l], freg_err[l]
    
    
    
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.ma.average(values, weights=weights)
    variance = np.ma.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return [average, np.sqrt(variance)]

    
    
def calc_phase(hjd, P, Tprim):
    return ((hjd - Tprim) % P) / P
    
    
    
def phase_fold(time, flux, P, Tprim, bin_width_phase=0.02, ferr_type='medsig', ferr_style='std'):
    phi = calc_phase( time, P, Tprim )
    phi[ phi>0.75 ] -= 1.
    phase, phaseflux, phaseflux_err = rebin_err( phi, flux, None, dt=bin_width_phase, ferr_type=ferr_type, ferr_style=ferr_style )
    return phase, phaseflux, phaseflux_err, phi