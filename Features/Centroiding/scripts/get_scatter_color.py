# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:13:02 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np



###############################################################################
#::: Adjust scatter color according to airmass parabola
###############################################################################
#::: only 1 object in dic
def get_scatter_color(dic):
    print 'Fitting airmass polynomials.'
    dic['COLOR_PER_NIGHT'] = []
    
    for i, date in enumerate( dic['UNIQUE_NIGHT'] ):
        ind = np.where( dic['NIGHT'] == date )[0]

        t = dic['HJD'][ind]
        t = t - np.round(t[0]) #normalize to start of night; use "round" isntead of "int" to avoid shift by 1 day introduced by HJD offsets
        airm0 = dic['AIRMASS'][ind] - 1 #normalize to 0
        
        #::: color coding according to time/airmass
        #::: airmass ranges from 1 to 2
        c = airm0 
        p = np.polyfit( t, airm0, 2 ) #fit a 2D polynomial
        tmin = - p[1] / (2.*p[0]) #get the minimum of the polynomial
        c[ t < tmin ] *= - 1
        
        dic['COLOR_PER_NIGHT'].append(c)
        
    return dic  
    
    
    
#::: multiple objects in dic   
def get_scatter_color_multi(dic):
    print 'Fitting airmass polynomials.'
    dic['COLOR_PER_NIGHT'] = []
    
    for i, date in enumerate( dic['UNIQUE_NIGHT'] ):
        ind = np.where( dic['NIGHT'] == date )[0]

        t = dic['HJD'][0,ind]
        t = t - np.round(t[0]) #normalize to start of night; use "round" isntead of "int" to avoid shift by 1 day introduced by HJD offsets
        airm0 = dic['AIRMASS'][ind] - 1 #normalize to 0
        
        #::: color coding according to time/airmass
        #::: airmass ranges from 1 to 2
        c = airm0 
        p = np.polyfit( t, airm0, 2 ) #fit a 2D polynomial
        tmin = - p[1] / (2.*p[0]) #get the minimum of the polynomial
        c[ t < tmin ] *= - 1
        
        dic['COLOR_PER_NIGHT'].append(c)
        
    return dic  