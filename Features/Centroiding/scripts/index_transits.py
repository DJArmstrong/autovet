# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:55:39 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

# REFACTORED: CHECK

import numpy as np


"""
    DUPLICATE FROM MY_TOOLS - My_Utils
"""
def mask_ranges(x, x_min, x_max):
    """"
        Crop out values and indices out of an array x for multiple given ranges x_min to x_max.
        
        Input:
        x: array,
        x_min: lower limits of the ranges
        x_max: upper limits of the ranges
        
        Output:
        
        
        Example:
        x = np.arange(200)
        x_min = [5, 25, 90]
        x_max = [10, 35, 110]
        """
    
    mask = np.zeros(len(x), dtype=bool)
    for i in range(len(x_min)):
        mask = mask | ((x >= x_min[i]) & (x <= x_max[i]))
    ind_mask = np.arange(len(mask))[mask]
    
    return x[mask], ind_mask, mask



def index_transits(dic, obj_nr=None):
    """
    output:
    ind_tr: indices of points in transit
    ind_tr_half: indices of the 50% innermost transit points
    ind_tr_double: #double the transit duration, includes out of transit parts
    """
    
    
    if obj_nr is not None:
        N = int( 1. * ( dic['HJD'][obj_nr][-1] - dic['EPOCH'][obj_nr] ) / dic['PERIOD'][obj_nr] ) + 1
    #    print 'N', N
        tmid = np.array( [ dic['EPOCH'][obj_nr] + i * dic['PERIOD'][obj_nr] for i in range(N) ] )
    #    print 'tmid', tmid
    #    
    #    print dic['HJD'][obj_nr]
    #    print tmid - dic['WIDTH'][obj_nr]/2.
    #    print tmid + dic['WIDTH'][obj_nr]/2.
        
        _, ind_tr, mask_tr = mask_ranges( dic['HJD'][obj_nr], tmid - dic['WIDTH'][obj_nr]/2., tmid + dic['WIDTH'][obj_nr]/2. )
        _, ind_tr_half, _ = mask_ranges( dic['HJD'][obj_nr], tmid - dic['WIDTH'][obj_nr]/4., tmid + dic['WIDTH'][obj_nr]/4. )
        _, ind_tr_double, _ = mask_ranges( dic['HJD'][obj_nr], tmid - dic['WIDTH'][obj_nr], tmid + dic['WIDTH'][obj_nr] )
              
    #    print 'hjd_tr', hjd_tr
    #    raw_input('...')
        ind_out = np.arange( len(dic['HJD'][obj_nr]) )[ ~mask_tr ]
        
        #::: mark the exposures/bins that lie within the night of a transit and are out of transit
        ind_out_per_tr = ind_tr_double[np.in1d(ind_tr_double,ind_tr,invert=True)]
            
    #    #::: mark outliers
    #    ind_prev = np.where( (np.diff(dic['CCDY'][obj_nr]) > 15.) & (np.diff(dic['CCDX'][obj_nr]) < -15.) )[0]
    #    ind_bug = np.where( (np.diff(dic['CCDY'][obj_nr]) > 15.) & (np.diff(dic['CCDX'][obj_nr]) < -15.) )[0] + 1



    else:
        N = int( 1. * ( dic['HJD'][-1] - dic['EPOCH'] ) / dic['PERIOD'] ) + 1
        
        tmid = np.array( [ dic['EPOCH'] + i * dic['PERIOD'] for i in range(N) ] )
        
        _, ind_tr, mask_tr = mask_ranges( dic['HJD'], tmid - dic['WIDTH']/2., tmid + dic['WIDTH']/2. )
        _, ind_tr_half, _ = mask_ranges( dic['HJD'], tmid - dic['WIDTH']/4., tmid + dic['WIDTH']/4. )
        _, ind_tr_double, _ = mask_ranges( dic['HJD'], tmid - dic['WIDTH'], tmid + dic['WIDTH'] )
              
        ind_out = np.arange( len(dic['HJD']) )[ ~mask_tr ]
        
        #::: mark the exposures/bins that lie within the night of a transit and are out of transit
        ind_out_per_tr = ind_tr_double[np.in1d(ind_tr_double,ind_tr,invert=True)]



    return ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid 
    
    

#::: for binaries, mark the primary and secondary eclipse
def index_eclipses(dic, obj_nr=None):
    """
    output:
    ind_ecl1: indices of points in primary eclipse
    ind_ecl1_half: indices of the 50% innermost eclipse points
    ind_ecl1_double: #double the transit duration, includes out of eclipse parts
    
    ind_ecl2: indices of points in secondary eclipse
    ind_ecl2_half: indices of the 50% innermost eclipse points
    ind_ecl2_double: #double the transit duration, includes out of eclipse parts
    
    ind_out: outside of any eclipse
    
    ! this assumes WIDHT1 == WIDTH2 !
    """
    
    
    if obj_nr is not None:
        print 'Not currently implemented.'
        return None


    else:
        N = int( 1. * ( dic['HJD'][-1] - dic['EPOCH'] ) / dic['PERIOD'] ) + 1
        
        tmid_ecl1 = np.array( [ dic['EPOCH'] +                    i * dic['PERIOD']  for i in range(N) ] )
        tmid_ecl2 = np.array( [ dic['EPOCH'] - dic['PERIOD']/2. + i * dic['PERIOD']  for i in range(N+1) ] )
        
        _, ind_ecl1,        mask_ecl1 = mask_ranges( dic['HJD'], tmid_ecl1 - dic['WIDTH']/2., tmid_ecl1 + dic['WIDTH']/2. )
        _, ind_ecl1_half,   _         = mask_ranges( dic['HJD'], tmid_ecl1 - dic['WIDTH']/4., tmid_ecl1 + dic['WIDTH']/4. )
        _, ind_ecl1_double, _         = mask_ranges( dic['HJD'], tmid_ecl1 - dic['WIDTH'],    tmid_ecl1 + dic['WIDTH']    )
              
        _, ind_ecl2,        mask_ecl2 = mask_ranges( dic['HJD'], tmid_ecl2 - dic['WIDTH']/2., tmid_ecl2 + dic['WIDTH']/2. )
        _, ind_ecl2_half,   _         = mask_ranges( dic['HJD'], tmid_ecl2 - dic['WIDTH']/4., tmid_ecl2 + dic['WIDTH']/4. )
        _, ind_ecl2_double, _         = mask_ranges( dic['HJD'], tmid_ecl2 - dic['WIDTH'],    tmid_ecl2 + dic['WIDTH']    )
        
        ind_out = np.arange( len(dic['HJD']) )[ ~(mask_ecl1 | mask_ecl2) ]
        
        #::: mark the exposures/bins that lie within the night of a transit and are out of transit
#        ind_out_per_tr = ind_tr_double[np.in1d(ind_tr_double,ind_tr,invert=True)]



    return ind_ecl1, ind_ecl1_half, ind_ecl1_double, \
           ind_ecl1, ind_ecl1_half, ind_ecl1_double, \
           ind_out
    
    

    
    
#if __name__ == '__main__':
#    import ngtsio
#    dic = ngtsio.get( 'NG0522-2518', ['CCDX','CCDY','EPOCH','PERIOD','HJD','WIDTH'], obj_id = 'bls' )
#    index_transits(dic, 0)
