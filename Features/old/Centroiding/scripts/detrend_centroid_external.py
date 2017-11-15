# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:57:36 2016

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
from scipy import signal
from astropy.stats import sigma_clip

import pandas as pd
import lightcurve_tools
#import get_scatter_color
#from scipy.optimize import lstsq
#from scipy.stats import sigmaclip
from scipy.optimize import least_squares 
from index_transits import index_transits


def run(dic, dic_nb, flattening='constant'):
    '''
    0) Remove a generall offset of the curve over the full time array
    1) Flatten the curves of the target and all neighbours per night
    2) Find the most correlated neighbours
    3) Detrend the target by the sigma-clipped mean of the most correlated neighbours
    4) remove remaining airmass trends from 1 siderial day phasecurve
    
    _f : flattened (linear trend and offset removed per night) 
    _ref : reference curve computed from neighbouring stars
    _fd : flattened and detrended target star
    _fda: flattend, detrended, and airmass-detrended (1 siderial day) target star
    '''
    
    #TODO: only calculate correlatin / fit out of transit!!!
    
    #::: set parameters
    N_neighbours = len(dic_nb['OBJ_ID'])
#    N_exp = len(dic['HJD'])
    N_top = 5
    ind_bp, exposures_per_night, obstime_per_night = breakpoints(dic)
    ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid = index_transits(dic)
 
    
    #::: 0)
    dic['CENTDX_f'] = detrend_global(dic, dic['CENTDX'], ind_out)
    dic['CENTDY_f'] = detrend_global(dic, dic['CENTDY'], ind_out)
    
    dic_nb['CENTDX_f'] = np.zeros( dic_nb['CENTDX'].shape )
    dic_nb['CENTDY_f'] = np.zeros( dic_nb['CENTDY'].shape )
    
    for i in range(N_neighbours):        
        dic_nb['CENTDX_f'][i,:] = detrend_global(dic, dic_nb['CENTDX'][i,:], ind_out)
        dic_nb['CENTDY_f'][i,:] = detrend_global(dic, dic_nb['CENTDY'][i,:], ind_out)
    
    
    
    #::: 1) 
    
#    dic['CENTDX_f'] = detrend_scipy(dic['CENTDX'], ind_bp, flattening)
#    dic['CENTDY_f'] = detrend_scipy(dic['CENTDY'], ind_bp, flattening)
#    
#    dic_nb['CENTDX_f'] = np.zeros( dic_nb['CENTDX'].shape )
#    dic_nb['CENTDY_f'] = np.zeros( dic_nb['CENTDY'].shape )
#    
#    for i in range(N_neighbours):        
#        dic_nb['CENTDX_f'][i,:] = detrend_scipy(dic_nb['CENTDX'][i,:], ind_bp, flattening)
#        dic_nb['CENTDY_f'][i,:] = detrend_scipy(dic_nb['CENTDY'][i,:], ind_bp, flattening)
    
    dic['CENTDX_f'] = detrend_per_night(dic, dic['CENTDX_f'], ind_out)
    dic['CENTDY_f'] = detrend_per_night(dic, dic['CENTDY_f'], ind_out)
    
#    dic_nb['CENTDX_f'] = np.zeros( dic_nb['CENTDX'].shape )
#    dic_nb['CENTDY_f'] = np.zeros( dic_nb['CENTDY'].shape )
    
    for i in range(N_neighbours):        
        dic_nb['CENTDX_f'][i,:] = detrend_per_night(dic, dic_nb['CENTDX_f'][i,:], ind_out)
        dic_nb['CENTDY_f'][i,:] = detrend_per_night(dic, dic_nb['CENTDY_f'][i,:], ind_out)
    
    
    
    #::: 2)
    
    dic_nb['corrcoeff_x'] = np.zeros(N_neighbours) * np.nan 
    dic_nb['corrcoeff_y'] = np.zeros(N_neighbours) * np.nan
    
    a_x = pd.Series( dic['CENTDX_f'] )
    a_y = pd.Series( dic['CENTDY_f'] )
    for i in range(dic_nb['CENTDX'].shape[0]): 
        b_x = pd.Series( dic_nb['CENTDX_f'][i] )
        b_y = pd.Series( dic_nb['CENTDY_f'][i] )
        dic_nb['corrcoeff_x'][i] = a_x.corr( b_x )
        dic_nb['corrcoeff_y'][i] = a_y.corr( b_y )
    
    #pick the N_top highest corrcoeffs
    ind_x = np.argpartition(dic_nb['corrcoeff_x'], -N_top)[-N_top:]
    ind_y = np.argpartition(dic_nb['corrcoeff_y'], -N_top)[-N_top:]
    
    

    
    #::: 3)
    
    #TODO: remove hardcoding. what is this?!
    dt=0.01
    period=0.9972695787*24.*3600. 
    
    centdx = dic['CENTDX_f']
    centdy = dic['CENTDY_f']
    hjd_phase, centdx_phase, centdx_phase_err, _, _ = lightcurve_tools.phase_fold( dic['HJD'], centdx, period, dic['HJD'][0], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
    _, centdy_phase, centdy_phase_err, _, _ = lightcurve_tools.phase_fold( dic['HJD'], centdy, period, dic['HJD'][0], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
    
    xx_phase = np.zeros((N_top, len(hjd_phase)))
    for i, ind in enumerate(ind_x):
        xx = dic_nb['CENTDX_f'][ ind, : ] 
        _, xx_phase[i], _, _, _ = lightcurve_tools.phase_fold( dic['HJD'], xx, period, dic['HJD'][0], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)

    yy_phase = np.zeros((N_top, len(hjd_phase)))
    for i, ind in enumerate(ind_y):
        yy = dic_nb['CENTDX_f'][ ind, : ] 
        _, yy_phase[i], _, _, _ = lightcurve_tools.phase_fold( dic['HJD'], yy, period, dic['HJD'][0], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)

    def errfct( weights, centdxy_phase, xy_phase ):
        return centdxy_phase - np.average(xy_phase, axis=0, weights=weights)
       
    p0 = np.ones( N_top ) #initial guess, all have the same weight
    params_x = least_squares(errfct, p0[:], bounds=(np.zeros(N_top), np.ones(N_top)), args=(centdx_phase, xx_phase))  
    weights_x = params_x.x #'.x' calls the parameters of the fit, ie. here the weights
    print 'x result:', params_x.x   
    params_y = least_squares(errfct, p0[:], bounds=(np.zeros(N_top), np.ones(N_top)), args=(centdy_phase, yy_phase))  
    weights_y = params_y.x    
    print 'y result:', params_y.x 
        
    dic_nb['CENTDX_ref_mean'] = np.average(dic_nb['CENTDX_f'][ ind_x, : ], axis=0, weights=weights_x)
    dic_nb['CENTDY_ref_mean'] = np.average(dic_nb['CENTDY_f'][ ind_y, : ], axis=0, weights=weights_y)
   
    dic['CENTDX_fd'] = dic['CENTDX_f'] - dic_nb['CENTDX_ref_mean']
    dic['CENTDY_fd'] = dic['CENTDY_f'] - dic_nb['CENTDY_ref_mean']
        
    #::: flatten again per night
#    dic['CENTDX_fd'] = detrend_scipy(dic['CENTDX_fd'], ind_bp, flattening)
#    dic['CENTDY_fd'] = detrend_scipy(dic['CENTDY_fd'], ind_bp, flattening)
    dic['CENTDX_fd'] = detrend_per_night(dic, dic['CENTDX_fd'], ind_out)
    dic['CENTDY_fd'] = detrend_per_night(dic, dic['CENTDY_fd'], ind_out)



          
    #:::4)
          
    dic = fit_phasecurve_1_siderial_day( dic )
    
    #::: flatten again per night
#    dic['CENTDX_fda'] = detrend_scipy(dic['CENTDX_fda'], ind_bp, flattening)
#    dic['CENTDY_fda'] = detrend_scipy(dic['CENTDY_fda'], ind_bp, flattening)
    dic['CENTDX_fd'] = detrend_per_night(dic, dic['CENTDX_fda'], ind_out)
    dic['CENTDY_fd'] = detrend_per_night(dic, dic['CENTDY_fda'], ind_out)
          
          
    return dic, dic_nb
    
    

def breakpoints(dic):
    '''
    Mark breakpoints (start and end of nights)
    '''
    ind_bp = [0]
    exposures_per_night = []
    obstime_per_night = []
    for i, date in enumerate(dic['UNIQUE_NIGHT']):    
        #::: get the exposures of this night
        ind = np.where( dic['NIGHT'] == date )[0] 
        exposures_per_night.append( len(ind) )
        obstime_per_night.append( dic['HJD'][ind[-1]] - dic['HJD'][ind[0]] )
#        ind_bp.append(ind[0])
        ind_bp.append(ind[-1])
    return ind_bp, np.array(exposures_per_night), np.array(obstime_per_night)

    
    
def detrend_scipy(data, ind_bp, flattening):
    '''
    Use scipy to detrend each object and each night individually 
    (seperated by breakpoints ind_bp)
    
    1) flattening='constant': remove constant trend (offset)
    2) flattening='linear': remove linear trend (offset+slope) 
    3) flattening='none': do not remove any trend
     
    Warning: method 2) can cause the transit signal to vanish!
    '''
#    print 'detrend scipy running'
#    print 'breakfpoints'
#    print ind_bp
    
    if flattening=='constant':
        ind_nan = np.isnan(data)
        data[ ind_nan ] = 0
#        data = np.masked_invalid( data )
        data_detrended = signal.detrend(data, type=flattening, bp=ind_bp)
        data_detrended[ ind_nan ] = np.nan
        return data_detrended
    else:
        return data
 
 

def detrend_per_night(dic, data, ind_out):    
    #::: copy data array    
    data_detrended = 1.*data
    
    for i, date in enumerate(dic['UNIQUE_NIGHT']): 
        
        #TODO: this sometimes breaks with error message
        #          File "/appch/data1/mg719/programs/anaconda/lib/python2.7/site-packages/astropy/stats/sigma_clipping.py", line 208, in _sigma_clip
        #    perform_clip(filtered_data, kwargs)
        #  File "/appch/data1/mg719/programs/anaconda/lib/python2.7/site-packages/astropy/stats/sigma_clipping.py", line 180, in perform_clip
        #    _filtered_data.mask |= _filtered_data < min_value
        #TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
        try:
            #::: select one night + select out of transit
            ind_date = np.where( dic['NIGHT'] == date )[0]
            ind = np.intersect1d( ind_date, ind_out )
            
            #::: calculate sigma clipped mean (using astropy)
            offset = np.mean( sigma_clip(data[ ind ]) )
            
            data_detrended[ ind_date ] -= offset     
        
        except:
            pass
        
    return data_detrended
    


def detrend_global(dic, data, ind_out):    
    #::: copy data array    
    data_detrended = 1.*data
    
    #::: select out of transit
    ind = ind_out
    
    #::: calculate sigma clipped mean (using astropy)
    offset = np.mean( sigma_clip(data[ ind ]) )
    
    data_detrended -= offset     
        
    return data_detrended   
    
    
 
def fit_phasecurve_1_siderial_day( dic, dt=0.01, period=0.9972695787*24.*3600. ):
    '''
    1 mean siderial day = 
    ( 23 + 56/60. + 4.0916/3600. ) / 24. = 0.9972695787 days
    '''
    
    #::: phasefold to one siderial day
    t0 = ( np.int( dic['HJD'][0]/24./3600. ) ) * 24.*3600.
    dic['HJD_PHASE_1sidday'], dic['COLOR_PHASE_1sidday'], dic['COLOR_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( dic['HJD'], dic['COLOR'], period, t0, dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
    _, dic['AIRMASS_PHASE_1sidday'], dic['AIRMASS_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( dic['HJD'], dic['AIRMASS'], period, t0, dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
    _, dic['SYSREM_FLUX3_PHASE_1sidday'], dic['SYSREM_FLUX3_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold(dic['HJD'], dic['SYSREM_FLUX3'] / np.nanmedian(dic['SYSREM_FLUX3']), period, t0, dt = dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
    _, dic['CENTDX_fd_PHASE_1sidday'], dic['CENTDX_fd_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( dic['HJD'], dic['CENTDX_fd'], period, t0, dt = dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
    _, dic['CENTDY_fd_PHASE_1sidday'], dic['CENTDY_fd_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( dic['HJD'], dic['CENTDY_fd'], period, t0, dt = dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
    
    
    #::: fit out the trend and save the resulting polyfunction in the dic
    polyfit_params = np.polyfit( dic['HJD_PHASE_1sidday'], dic['CENTDX_fd_PHASE_1sidday'], 2 )
    dic['polyfct_CENTDX'] = np.poly1d( polyfit_params )
        
    polyfit_params = np.polyfit( dic['HJD_PHASE_1sidday'], dic['CENTDY_fd_PHASE_1sidday'], 2 )
    dic['polyfct_CENTDY'] = np.poly1d( polyfit_params )
    
    
    #::: unwrap the phase-folding
    dx = ( (dic['HJD'] - t0) % (period) ) / period 
    dic['poly_CENTDX'] = dic['polyfct_CENTDX'](dx)
    dic['poly_CENTDY'] = dic['polyfct_CENTDY'](dx)
    dic['CENTDX_fda'] = dic['CENTDX_fd'] - dic['poly_CENTDX']
    dic['CENTDY_fda'] = dic['CENTDY_fd'] - dic['poly_CENTDY']
    
    
    return dic
    
    
   
def run_example(dic, dic_nb):
    '''
    For test purposes
    '''
    #::: choose example data
    data = dic['CENTDX']
    time = dic['HJD']
    
    ind_bp = breakpoints(dic)
    data_detrended = detrend_scipy(data, ind_bp)
    plot(data, data_detrended, time)    
    
    
    
def plot(data, data_detrended, time): 
    '''
    For test purposes
    '''   
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(12,4))
    N_exp = 1e4
    
    ax = axes[0]
    ax.scatter( np.arange(N_exp), data[0:N_exp], c=time[0:N_exp].astype(int), rasterized=True, cmap='jet' )
    
    ax = axes[1]
    ax.scatter( np.arange(N_exp), data_detrended[0:N_exp], c=time[0:N_exp].astype(int), rasterized=True, cmap='jet' )

    ax.set( xlim=[0,N_exp] )
    
    