# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:40:28 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
import pandas as pd
import warnings

try:
    import statsmodels.api as sm
except ImportError:
    warnings.warn( "Package 'statsmodels' could not be imported. Omitting some analyses.", ImportWarning )



###########################################################################
#::: define crosscorrelation for pandas df
########################################################################### 
def pandas_lagcorr(datax, datay, lag=0):  
    ''' http://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas '''
    
    return datax.corr(datay.shift(lag))
    
    
    
def pandas_prewhitening(datax, nlags=None, p=10, q=0):
    ''' http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma_0.html '''
    ''' http://statsmodels.sourceforge.net/devel/examples/generated/ex_dates.html '''
        
    #::: index the pandas series with arbitrary dates to comply with statsmodels' tsa format
    dates = pd.date_range('1/1/2000', periods=1, freq='D') 
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            datax_arma_mod = sm.tsa.ARMA(datax, (p,q), dates=dates).fit(disp=0)
        except:
            warnings.warn('ARMA pre-whitening failed. Package "statsmodels" is not available.')
        
    datax_resid = datax_arma_mod.resid
    
    return datax_resid


    
def pandas_crosscorr(datax, datay, nlags=None, prewhitening=False, p=10, q=0):
    ''' http://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas '''
    
    #::: prewhitening
    if prewhitening==True:
        u = pandas_prewhitening(datax, nlags, p, q)
        y = pandas_prewhitening(datay, nlags, p, q)
    else:
        u = datax
        y = datay
    
    #::: lags
    if nlags==None: 
        lags = np.arange( -len(u), len(y) )
    else:
        lags = np.arange( -nlags, nlags )
    N_arr = len(y) - np.abs(lags)
    
    #::: CI
    CI95 = 1.96/np.sqrt(N_arr)
    CI99 = 2.58/np.sqrt(N_arr)
    
    return np.array( [pandas_lagcorr(u, y, lag=i) for i in lags ] ), lags, CI95, CI99
    
    

def pandas_autocorr(datax, nlags=None, prewhitening=False, p=10, q=0):
    
    return pandas_crosscorr(datax, datax, nlags, prewhitening, p, q)
    
    

def pandas_periodic_crosscorr(datax, datay, nlags=None, prewhitening=False, p=10, q=0):
    ''' http://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas '''
    
    #::: prewhitening
    if prewhitening==True:
        u = pandas_prewhitening(datax, nlags, p, q)
        y = pandas_prewhitening(datay, nlags, p, q)
    else:
        u = datax
        y = datay
    
    #::: lags
    if nlags==None: 
        lags = np.arange( 0, len(y) )
    else:
        lags = np.arange( 0, nlags )
    
    #::: CI
    N_arr = len(y)
    CI95 = 1.96/np.sqrt(N_arr) * np.ones(N_arr)
    CI99 = 2.58/np.sqrt(N_arr) * np.ones(N_arr)
    
    outlist = []
    
    ind = range( len(y) )
    for i in range( len(y) ):
        if i > 0:
            ind.append( ind[0] )
            ind.pop( 0 )
            y = y[ind]
            y = y.shift(-1)
        outlist.append( pandas_lagcorr(u, y, lag=0) )
    
    return outlist, lags, CI95, CI99
    
    
    
def pandas_periodic_autocorr(datax, nlags=None, prewhitening=False, p=10, q=0):
    
    return pandas_periodic_crosscorr(datax, datax, nlags, prewhitening, p, q)