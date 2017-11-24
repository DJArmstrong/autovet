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
import numpy.ma as ma
import matplotlib.pyplot as plt
#from scipy import signal
from astropy.stats import sigma_clip
import timeit, warnings

import pandas as pd
import lightcurve_tools
#import get_scatter_color
#from scipy.optimize import lstsq
#from scipy.stats import sigmaclip
from scipy.optimize import least_squares
from index_transits import index_eclipses



class detrend_centroid():
    
    def __init__(self, dic, dic_nb, method, R_min, N_top_max, dt, t_exp=12., silent=True ):
        '''
        method: 'transit' or 'sidday': phase-fold and detrend on which period
        R_min: minimum Pearsons r for selection
        N
        '''
        
#        super(centroid, self).__init__(parent)
    
        self.dic = dic
        self.dic_nb = dic_nb
        self.method = method
        self.R_min = R_min #minimum Pearsons r correlation coefficient; to determine which neighbours will be included in the fit
        self.N_top_max = N_top_max #maximum number of neighbours for fit
        self.silent = silent #print output or not
        
        self.N_exp = len(self.dic['HJD'])
        self.N_neighbours = len(self.dic_nb['OBJ_ID'])
        print 'N_neighbours:',self.N_neighbours
        
        self.t_exp = t_exp #12s exposure
        self.dt = dt
        self.sidereal_day_period = 0.9972695787*24.*3600. #in s
        self.sidereal_day_epoch = int(dic['HJD'][0]/24./3600.)*24.*3600. #in s
        
        self.valid = True
#        self.dic['CENTDX'] += self.dic['CCDX']
#        self.dic['CENTDY'] += self.dic['CCDY']
#        self.dic_nb['CENTDX'] += self.dic_nb['CCDX']
#        self.dic_nb['CENTDY'] += self.dic_nb['CCDY']
        
#        self.run()
#        return self.dic, self.dic_nb  
    
    
    def run(self):
        '''
        1) Flatten the curves of the target and all neighbours per night
            -> use mean of each night (no sigma clipping)
        2) Find the most correlated neighbours
            -> select the 5 highest Pearson r, phase-folded on sidereal day period, ignoring in-transit data
        3) Detrend the target by the most correlated neighbours
            -> least squares fit, phase-folded on sidereal day period, ignoring in-transit data
        4) DO NOT remove remaining airmass trends from 1 sidereal day phasecurve
            -> second order polynomial
            
        _f : flattened (linear trend and offset removed per night) 
        _ref : reference curve computed from neighbouring stars
        _fd : flattened and detrended target star
        _fda: flattend, detrended, and airmass-detrended (1 sidereal day) target star
        '''
        
        #::: calculate breakpoints between different nights
        ind_bp, exposures_per_night, obstime_per_night = self.breakpoints()
        if not self.silent: print 'breakpoints done.'
        
        #::: get indices of in-transit and out-of-transit
        ind_ecl1, ind_ecl1_half, ind_ecl1_double, \
           ind_ecl1, ind_ecl1_half, ind_ecl1_double, \
           self.ind_out = index_eclipses(self.dic)
        if not self.silent: print 'index eclipses done.'
     
     
        
        #::: 0a)
        '''
        mask outliers 
        (all stars, overwrites keys 'CENTDXY' with masked arrays)
        '''
        
        start = timeit.default_timer()
        self.mask_outliers()
        self.mask_outliers_per_night()
        if not self.silent: print 'masking outliers done.'
        if not self.silent: print '\texecution time:', timeit.default_timer() - start, 's'
    
#        plt.figure(figsize=(200,6))
#        plt.plot( self.dic['HJD'], self.dic['CENTDX'], 'k.', rasterized=True)
#        self.mask_outliers()
#        plt.plot( self.dic['HJD'][ self.dic['CENTDX'].mask ], self.dic['CENTDX'].data[ self.dic['CENTDX'].mask ], 'ro', ms=15, rasterized=True)
#        plt.plot( self.dic['HJD'][ self.dic['CENTDX'].mask ], self.dic['CENTDX'].data[ self.dic['CENTDX'].mask ], 'bo', ms=12, rasterized=True)
##        self.mask_outliers_per_night_and_rolling()
##        plt.plot( self.dic['HJD'][ self.dic['CENTDX'].mask ], self.dic['CENTDX'].data[ self.dic['CENTDX'].mask ], 'co', ms=9, rasterized=True)                
#        plt.show()
        
        
        
        #::: 0b)
        '''
        mask the in-transit data for CENTDXY
        (target star only, creates new key 'CENTDXY_out'; out = out of transit)
        '''
        
        self.mask_transit(suffix='')
        if not self.silent: print 'masking transits in CENTDXY done.'
        
        
        #::: 0c)
        '''
        phase-fold all data on a sidereal day period, lunar period, and target period
        '''
        
        self.phase_fold_all_data(suffix='')
        if not self.silent: print 'phase-folding CENTDXY done.'
        
        
        
        
        #::: 1a) 
        '''
        remove night-to-night offsets
        (automatically only regards out-of-transit data for the target)
        '''
        
        self.flatten(suffix='_f')
        if not self.silent: print 'flattening (removing night-to-night offsets) done.'
        
        
        
        #::: 1b)
        '''
        mask the in-transit data for CENTDXY_f
        (target star only, creates new key 'CENTDXY_out_f'; out = out of transit)
        '''
        
        self.mask_transit(suffix='_f')
        if not self.silent: print 'masking transits in CENTDXY_f done.'
        
        
        
        #::: 1c)
        '''
        phase-fold all data on a sidereal day period, lunar period, and target period
        '''
        
        self.phase_fold_all_data(suffix='_f')
        if not self.silent: print 'phase-folding CENTDXY_f done.'
        
        
        
        #::: 2a)
        '''
        calculate correlation coefficients between the phase-folded centroids of target and all neighbours
        (on a sidereal day)
        (using the out-of-transit data only)
        '''
        
        #::: on a sidereal day
        if self.method == 'sidday':
            xkey_for_fit = 'CENTDX_out_f_PHASE_1sidday'
            ykey_for_fit = 'CENTDY_out_f_PHASE_1sidday'
            xkey_nb_for_fit = 'CENTDX_f_PHASE_1sidday'
            ykey_nb_for_fit = 'CENTDY_f_PHASE_1sidday'
        
        #::: on transit period
        if self.method == 'transit':
            xkey_for_fit = 'CENTDX_out_f_PHASE'
            ykey_for_fit = 'CENTDY_out_f_PHASE'
            xkey_nb_for_fit = 'CENTDX_f_PHASE'
            ykey_nb_for_fit = 'CENTDY_f_PHASE'
        
        self.dic_nb['corrcoeff_x'] = np.zeros(self.N_neighbours) * np.nan 
        self.dic_nb['corrcoeff_y'] = np.zeros(self.N_neighbours) * np.nan
        
#        print xkey_for_fit
#        print self.dic[xkey_for_fit]
#        print self.dic['CENTDX']
#        print self.dic['CENTDX_out']
#        print self.dic['CENTDX_f']
#        print self.dic['CENTDX_out_f']
#        print self.dic['CENTDX_f_PHASE']
#        print self.dic['CENTDX_out_f_PHASE']
        
#        err
        a_x = pd.Series( self.dic[xkey_for_fit] )
        a_y = pd.Series( self.dic[ykey_for_fit] )
        for i in range(self.dic_nb[xkey_nb_for_fit].shape[0]): 
            b_x = pd.Series( self.dic_nb[xkey_nb_for_fit][i] )
            b_y = pd.Series( self.dic_nb[ykey_nb_for_fit][i] )
#            print a_x
#            print b_x
#            print len(a_x)
#            print len(b_x)
            self.dic_nb['corrcoeff_x'][i] = a_x.corr( b_x )
#            print self.dic_nb['corrcoeff_x'][i]
            self.dic_nb['corrcoeff_y'][i] = a_y.corr( b_y )
        
        
        #::: choose stronlgy correlated neighbours for reference
        ind_x = np.where( self.dic_nb['corrcoeff_x'] > self.R_min )[0]
        ind_y = np.where( self.dic_nb['corrcoeff_y'] > self.R_min )[0]
        N_top_x = len( ind_x )
        N_top_y = len( ind_y )
#        print self.R_min
#        print self.dic_nb['corrcoeff_x']
#        print self.dic_nb['corrcoeff_y']
#        print 'N_top_x', N_top_x
#        print 'N_top_y', N_top_y
        
        #::: accept a maximum of N_top_max neighbouring objects for reference
        if N_top_x > self.N_top_max:
            N_top_x = self.N_top_max
            ind_x = np.argpartition(self.dic_nb['corrcoeff_x'], -N_top_x)[-N_top_x:]
            
        if N_top_y > self.N_top_max:
            N_top_y = self.N_top_max
            ind_y = np.argpartition(self.dic_nb['corrcoeff_y'], -N_top_y)[-N_top_y:]
#            print ind_y
#            print self.dic_nb['corrcoeff_y']
            
        print 'N_top_x', N_top_x
        print 'N_top_y', N_top_y
        
        if ( (N_top_x == 0) | (N_top_y == 0) ):
            warnings.warn('Not enough reference stars to perform fit!')
            warnings.warn('Reference set to NaN!')
            self.valid = False
        #pick the N_top highest corrcoeffs
#        ind_x = np.argpartition(self.dic_nb['corrcoeff_x'], -N_top)[-N_top:]
#        ind_y = np.argpartition(self.dic_nb['corrcoeff_y'], -N_top)[-N_top:]
        
        print self.dic_nb['corrcoeff_x'][ind_x]
        print self.dic_nb['corrcoeff_y'][ind_y]
        
        print 'corrcoeff analysis done.'
        
    
        
        #::: 2b)
        '''
        find best fit of all neighbours
        (on a sidereal day)
        (using the out-of-transit data only)
        '''
        
        def errfct( params, centdxy_phase, xy_phase ):
            weights = params[:]
#            print xy_phase.shape
#            print weights.shape
            average, sum_of_weights = np.average(xy_phase, axis=0, weights=weights, returned=True)
            reference_curve = sum_of_weights * average
            return centdxy_phase - reference_curve
       
        def fit(N_top, key_for_fit, key_nb_for_fit, ind):
#            print key_for_fit, key_nb_for_fit
            #::: get parameters;  replace NaN bins with 0
            centdxy_phase = self.dic[key_for_fit]
            xy_phase = self.dic_nb[key_nb_for_fit][np.ix_( ind )]
            centdxy_phase[ ~np.isfinite(centdxy_phase) ] = 0.
            xy_phase[ ~np.isfinite(xy_phase) ] = 0.
            
            #::: perform least squars fit to find a local minimum
            p0 = np.ones( N_top ) / N_top #initial guess, one scale parameter plus N_top weight parameters; all initally have scale=1 and the same weight (=1/N_top)
            bounds = (np.zeros(N_top), np.zeros(N_top)+5. ) #bounds: [0,5]
            try:
                result = least_squares(errfct, p0, bounds=bounds, args=(centdxy_phase, xy_phase), loss='cauchy') 
            except:
                result = least_squares(errfct, p0, bounds=bounds, args=(centdxy_phase, xy_phase), loss='soft_l1') 
            
            #::: or perform a DE fit
#            bounds = zip( np.zeros(N_top)-5., np.zeros(N_top)+5. )
#            result = differential_evolution(errfct, bounds=bounds, args=(centdxy_phase, xy_phase)) 
            
            weights = result.x #'.x' calls the parameters of the fit, ie. here the weights
            return weights
        
        
        if self.valid:
            weights_x = fit(N_top_x, xkey_for_fit, xkey_nb_for_fit, ind_x)
            weights_y = fit(N_top_y, ykey_for_fit, ykey_nb_for_fit, ind_y)
        else:
            weights_x = np.nan
            weights_y = np.nan
        
        #::: save in dic_nb
        self.dic_nb['weights_x'] = np.zeros_like(self.dic_nb['corrcoeff_x']) 
        self.dic_nb['weights_x'][ ind_x ] = weights_x
        self.dic_nb['weights_y'] = np.zeros_like(self.dic_nb['corrcoeff_y']) 
        self.dic_nb['weights_y'][ ind_y ] = weights_y
        
        
#        if not self.silent: 
        print 'x sum:', np.sum(weights_x)
        print 'x result:', np.sum(weights_x) * weights_x 
        print 'y sum:', np.sum(weights_y)
        print 'y result:', np.sum(weights_y) * weights_y
        
        
        
        #::: 2c)
        '''
        add info into the dictionaries (non-phase-folded)
        (now for all data again)
        '''        
        
        if self.valid:
            self.dic_nb['CENTDX_ref_mean'] = np.sum(weights_x) * np.average(self.dic_nb['CENTDX_f'][ ind_x, : ], axis=0, weights=weights_x)
            self.dic_nb['CENTDY_ref_mean'] = np.sum(weights_y) * np.average(self.dic_nb['CENTDY_f'][ ind_y, : ], axis=0, weights=weights_y)
        else:
            self.dic_nb['CENTDX_ref_mean'] = np.zeros_like(self.dic['CENTDX_f'])*np.nan
            self.dic_nb['CENTDY_ref_mean'] = np.zeros_like(self.dic['CENTDY_f'])*np.nan
            
        self.dic['CENTDX_out_fd'] = self.dic['CENTDX_out_f'] - self.dic_nb['CENTDX_ref_mean']
        self.dic['CENTDY_out_fd'] = self.dic['CENTDY_out_f'] - self.dic_nb['CENTDY_ref_mean']
        self.dic['CENTDX_fd'] = self.dic['CENTDX_f'] - self.dic_nb['CENTDX_ref_mean']
        self.dic['CENTDY_fd'] = self.dic['CENTDY_f'] - self.dic_nb['CENTDY_ref_mean']

    

        #::: 2d)
        '''
        mask the in-transit data for CENTDXY_fd
        (target star only, creates new key 'CENTDXY_out_fd'; out = out of transit)
        '''
        
        self.mask_transit(suffix='_fd')
        if not self.silent: print 'masking transits in CENTDXY_fd done.'
        
        
        
        #::: 2e)
        '''
        flatten again per night
        '''
        self.flatten(suffix='_fd')
        if not self.silent: print 'flattening 2 (removing night-to-night offsets) done.'
    
    
        
        #::: 2f)
        '''
        phase-fold all data on a sidereal day period, lunar period, and target period
        '''
        
        self.phase_fold_all_data(suffix='_fd')
        self.phase_fold_all_data(suffix='_ref_mean')
        if not self.silent:print 'phase-folding CENTDXY_fd and CENTDXY_ref_mean done.'
        
                  
    
        #::: 3a)
        '''
        creates new key 'CENTDXY_fda'
        '''
              
        self.fit_phasecurve_1_sidereal_day()
        if not self.silent:print 'fitted sidereal day correction; created new key CENTDXY_fda'



        #::: 3b)
        '''
        mask the in-transit data for CENTDXY_fda
        (target star only, creates new key 'CENTDXY_out_fda'; out = out of transit)
        '''
        
        self.mask_transit(suffix='_fda')
        if not self.silent:print 'masking transits in CENTDXY_fda done; created new key CENTDXY_out_fda'
        
        
        #::: 3c)
        '''
        flatten again per night
        '''
        self.flatten(suffix='_fda')
        if not self.silent: print 'flattening 3 (removing night-to-night offsets) done.'
    
        
        
        #::: 3d)
        '''
        phase-fold all data on a sidereal day period, lunar period, and target period
        '''
        
        self.phase_fold_all_data(suffix='_fda')
        if not self.silent:print 'phase-folding CENTDXY_fda done.'
          
            
            
        #::: return    
        return self.dic, self.dic_nb
        
        
        

    def mask_outliers(self, sigma=5):
        self.dic['CENTDX'] = sigma_clip(self.dic['CENTDX'], sigma=sigma, iters=1)
        self.dic['CENTDY'] = sigma_clip(self.dic['CENTDY'], sigma=sigma, iters=1)
        self.dic_nb['CENTDX'] = sigma_clip(self.dic_nb['CENTDX'], sigma=sigma, iters=1, axis=1) #astropy's sigma_clip does work for matrixes per row/column!
        self.dic_nb['CENTDY'] = sigma_clip(self.dic_nb['CENTDY'], sigma=sigma, iters=1, axis=1)
            
            
            
            
    def mask_outliers_per_night(self, sigma=4):
        for i, date in enumerate(self.dic['UNIQUE_NIGHT']): 
            
            #::: select one night + select out of transit
            ind_date = np.where( self.dic['NIGHT'] == date )[0]
#            ind_date_out = np.intersect1d( ind_date, self.ind_out )
            
            #::: catch 'empty' days (where all values have already been flagged / set to NAN / masked)
            def try_sigma_clip(x, sigma, iters, axis=None):
                x = np.ma.masked_invalid(x)
                if ( x.mask.all() ):
                    return x
                else:
                    return sigma_clip(x, sigma=sigma, iters=iters, axis=axis)
                    
            #::: 
            self.dic['CENTDX'][ind_date] = try_sigma_clip(self.dic['CENTDX'][ind_date], sigma=sigma, iters=1)
            self.dic['CENTDY'][ind_date] = try_sigma_clip(self.dic['CENTDY'][ind_date], sigma=sigma, iters=1)
            self.dic_nb['CENTDX'][:,ind_date] = try_sigma_clip(self.dic_nb['CENTDX'][:,ind_date], sigma=sigma, iters=1, axis=1) #astropy's sigma_clip does work for matrixes per row/column!
            self.dic_nb['CENTDY'][:,ind_date] = try_sigma_clip(self.dic_nb['CENTDY'][:,ind_date], sigma=sigma, iters=1, axis=1)
            
      
      
            
    def mask_outliers_per_night_and_rolling(self, sigma=4):

        window = 600./self.t_exp        
        lag_step = window/5.
        
        for i, date in enumerate(self.dic['UNIQUE_NIGHT']): 
            
            #::: select one night + select out of transit
            ind_date = np.where( self.dic['NIGHT'] == date )[0]
#            ind_date_out = np.intersect1d( ind_date, self.ind_out )
            
            #::: go through it in 10 minute steps (not quite rolling, but kinda...) 
            #::: -> more computation time efficient than rolling, and no border-effects
            #::: pick out the data points in intervals
            #            0:50
            #            50:100
            #            100:150 etc
            j0 = 0
            while j0 < len(ind_date):
                j1 = np.min( (len(ind_date), j0+window) )
#                print j0
#                print j1
#                print type(j0)
#                print type(j1)
                ind_select = ind_date[ int(j0):int(j1) ]
                
                #::: 
                try:
                    self.dic['CENTDX'][ind_select] = sigma_clip(self.dic['CENTDX'][ind_select], sigma=sigma, iters=1)
                    self.dic['CENTDY'][ind_select] = sigma_clip(self.dic['CENTDY'][ind_select], sigma=sigma, iters=1)
                    self.dic_nb['CENTDX'][:,ind_select] = sigma_clip(self.dic_nb['CENTDX'][:,ind_select], sigma=sigma, iters=1, axis=1) #astropy's sigma_clip does work for matrixes per row/column!
                    self.dic_nb['CENTDY'][:,ind_select] = sigma_clip(self.dic_nb['CENTDY'][:,ind_select], sigma=sigma, iters=1, axis=1)
                except:
                    print 'sigma clip failure for', date, j0, j1
                
#                j0 += N_pointstep
                j0 += lag_step

            
            
#    def mask_outliers_RM(self, x):
#        x = pd.Series(1.*x)
#        
#        threshold = 3
#        window = 50 #10mins a 12s exposure = 600s/12s = 50 data points
#        buf = pd.rolling_median(x, window=window, center=True).fillna(method='bfill').fillna(method='ffill')
#        difference = np.abs(x - buf)
#        outlier_ind = difference > threshold*np.std(x)
#        x[ outlier_ind ] = np.nan
#        
#        return x
    
        
        
    def mask_transit(self, suffix):
        mask = np.zeros(self.N_exp, dtype=int)
#        mask[self.ind_tr] = 1 #mask the in-transit data
        self.dic['CENTDX_out'+suffix] = ma.masked_array(self.dic['CENTDX'+suffix], mask=mask, copy=True)
        self.dic['CENTDY_out'+suffix] = ma.masked_array(self.dic['CENTDY'+suffix], mask=mask, copy=True)
        
        
    
    def breakpoints(self):
        '''
        Mark breakpoints (start and end of nights)
        '''
        ind_bp = [0]
        exposures_per_night = []
        obstime_per_night = []
        for i, date in enumerate(self.dic['UNIQUE_NIGHT']):    
            #::: get the exposures of this night
            ind = np.where( self.dic['NIGHT'] == date )[0] 
            exposures_per_night.append( len(ind) )
            obstime_per_night.append( self.dic['HJD'][ind[-1]] - self.dic['HJD'][ind[0]] )
    #        ind_bp.append(ind[0])
            ind_bp.append(ind[-1])
        return ind_bp, np.array(exposures_per_night), np.array(obstime_per_night)
    
        
        
     
    def phase_fold_all_data(self, suffix=''):
        
        def pf( y, mode ):
            if mode=='transit':  return lightcurve_tools.phase_fold(self.dic['HJD'], y, self.dic['PERIOD'], self.dic['EPOCH'],             dt=self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=False ) 
            elif mode=='sidday': return lightcurve_tools.phase_fold(self.dic['HJD'], y, self.sidereal_day_period, self.sidereal_day_epoch, dt=self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=False ) 

        def pf_matrix( y, mode ):
            if mode=='transit':  return lightcurve_tools.phase_fold_matrix(self.dic['HJD'], y, self.dic['PERIOD'], self.dic['EPOCH'],             dt=self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=False ) 
            elif mode=='sidday': return lightcurve_tools.phase_fold_matrix(self.dic['HJD'], y, self.sidereal_day_period, self.sidereal_day_epoch, dt=self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=False ) 

                
        if suffix in ['', '_f', '_fd', '_fda']:
            
            _, self.dic['CENTDX'+suffix+'_PHASE'], _, _, _ = pf( self.dic['CENTDX'+suffix], 'transit' )
            _, self.dic['CENTDY'+suffix+'_PHASE'], _, _, _ = pf( self.dic['CENTDY'+suffix], 'transit' )
            _, self.dic['CENTDX'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic['CENTDX'+suffix], 'sidday' )
            _, self.dic['CENTDY'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic['CENTDY'+suffix], 'sidday' )  
            
            _, self.dic['CENTDX_out'+suffix+'_PHASE'], _, _, _ = pf( self.dic['CENTDX_out'+suffix], 'transit' )
            _, self.dic['CENTDY_out'+suffix+'_PHASE'], _, _, _ = pf( self.dic['CENTDY_out'+suffix], 'transit' )          
            _, self.dic['CENTDX_out'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic['CENTDX_out'+suffix], 'sidday' )
            _, self.dic['CENTDY_out'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic['CENTDY_out'+suffix], 'sidday' )
           
        if suffix in ['_f']:
            
            _, self.dic_nb['CENTDX'+suffix+'_PHASE'], _, _, _ = pf_matrix( self.dic_nb['CENTDX'+suffix], 'transit' ) 
            _, self.dic_nb['CENTDY'+suffix+'_PHASE'], _, _, _ = pf_matrix( self.dic_nb['CENTDY'+suffix], 'transit' )    
            _, self.dic_nb['CENTDX'+suffix+'_PHASE_1sidday'], _, _, _ = pf_matrix( self.dic_nb['CENTDX'+suffix], 'sidday' )    
            _, self.dic_nb['CENTDY'+suffix+'_PHASE_1sidday'], _, _, _ = pf_matrix( self.dic_nb['CENTDY'+suffix], 'sidday' )
           
        if suffix in ['_ref_mean']:
            
            _, self.dic_nb['CENTDX'+suffix+'_PHASE'], _, _, _ = pf( self.dic_nb['CENTDX'+suffix], 'transit' ) 
            _, self.dic_nb['CENTDY'+suffix+'_PHASE'], _, _, _ = pf( self.dic_nb['CENTDY'+suffix], 'transit' )    
            _, self.dic_nb['CENTDX'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic_nb['CENTDX'+suffix], 'sidday' )    
            _, self.dic_nb['CENTDY'+suffix+'_PHASE_1sidday'], _, _, _ = pf( self.dic_nb['CENTDY'+suffix], 'sidday' )
           
           
    
    
    def flatten(self, suffix='_f'):    
        
        #::: create the dictionary keys if needed
        if suffix == '_f':
            self.dic['CENTDX'+suffix] = self.dic['CENTDX']*1.
            self.dic['CENTDY'+suffix] = self.dic['CENTDY']*1.
            self.dic_nb['CENTDX'+suffix] = self.dic_nb['CENTDX']*1.
            self.dic_nb['CENTDY'+suffix] = self.dic_nb['CENTDY']*1.
        
        #::: then go through night-by-night
        for i, date in enumerate(self.dic['UNIQUE_NIGHT']): 
            
            #::: select one night + select out of transit
            ind_date = np.where( self.dic['NIGHT'] == date )[0]
            ind_date_out = np.intersect1d( ind_date, self.ind_out )
            
            #::: subtract nightly offsets
            self.dic['CENTDX'+suffix][ ind_date ] = self.dic['CENTDX'+suffix][ ind_date ] - np.nanmean( self.dic['CENTDX'+suffix][ ind_date_out ] )
            self.dic['CENTDY'+suffix][ ind_date ] = self.dic['CENTDY'+suffix][ ind_date ] - np.nanmean( self.dic['CENTDY'+suffix][ ind_date_out ] )
            try:
                self.dic_nb['CENTDX'+suffix][ :, ind_date ] = self.dic_nb['CENTDX'+suffix][ :, ind_date ] - np.nanmean( self.dic_nb['CENTDX'+suffix][ :, ind_date ], axis=1 )[:, np.newaxis]
                self.dic_nb['CENTDY'+suffix][ :, ind_date ] = self.dic_nb['CENTDY'+suffix][ :, ind_date ] - np.nanmean( self.dic_nb['CENTDY'+suffix][ :, ind_date ], axis=1 )[:, np.newaxis]
            except:
                pass
            
#        fig, axes = plt.subplots(2,1,figsize=(80,6))
#
##        axes[0].plot( self.dic['HJD'], self.dic['CENTDX'], 'g.', rasterized=True  )
#        axes[0].plot( self.dic['HJD'], self.dic['CENTDX_f'], 'b.', rasterized=True  )
##        axes[0].plot( self.dic['HJD'][self.ind_tr], self.dic['CENTDX_f'][self.ind_tr], 'r.', rasterized=True )
#        
##        axes[1].plot( self.dic['HJD'], self.dic_nb['CENTDX'][1, :], 'g.', rasterized=True  )            
#        axes[1].plot( self.dic['HJD'], self.dic_nb['CENTDX_f'][0, :], 'b.', rasterized=True  )
#            
#        err
        
        
        
#    def detrend_per_night(dic, data, ind_out):    
#        #::: copy data array    
#        data_detrended = 1.*data
#        
#        for i, date in enumerate(dic['UNIQUE_NIGHT']): 
#            
#            #TODO: this sometimes breaks with error message
#            #          File "/appch/data1/mg719/programs/anaconda/lib/python2.7/site-packages/astropy/stats/sigma_clipping.py", line 208, in _sigma_clip
#            #    perform_clip(filtered_data, kwargs)
#            #  File "/appch/data1/mg719/programs/anaconda/lib/python2.7/site-packages/astropy/stats/sigma_clipping.py", line 180, in perform_clip
#            #    _filtered_data.mask |= _filtered_data < min_value
#            #TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
#            try:
#                #::: select one night + select out of transit
#                ind_date = np.where( dic['NIGHT'] == date )[0]
#                ind = np.intersect1d( ind_date, ind_out )
#                
#                #::: calculate sigma clipped mean (using astropy)
#                offset = np.mean( sigma_clip(data[ ind ]) )
#                
#                data_detrended[ ind_date ] -= offset     
#            
#            except:
#                pass
#            
#        return data_detrended
#        
#    
#    
#    def detrend_global(dic, data, ind_out):    
#        #::: copy data array    
#        data_detrended = 1.*data
#        
#        #::: select out of transit
#        ind = ind_out
#        
#        #::: calculate sigma clipped mean (using astropy)
#        offset = np.mean( sigma_clip(data[ ind ]) )
#        
#        data_detrended -= offset     
#            
#        return data_detrended   
        
        
     
    def fit_phasecurve_1_sidereal_day(self, polyorder=None):
        '''
        use out-of-transit data only
        
        1 mean sidereal day = 
        ( 23 + 56/60. + 4.0916/3600. ) / 24. = 0.9972695787 days
        '''
        
        #::: phasefold to one sidereal day
#         self.dic['HJD_PHASE_1sidday'], self.dic['COLOR_PHASE_1sidday'], self.dic['COLOR_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['COLOR'], self.sidereal_day_period, self.sidereal_day_epoch, dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
#        _, self.dic['AIRMASS_PHASE_1sidday'], self.dic['AIRMASS_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['AIRMASS'], self.sidereal_day_period, self.sidereal_day_epoch, dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['SYSREM_FLUX3_PHASE_1sidday'], self.dic['SYSREM_FLUX3_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['SYSREM_FLUX3'] / np.nanmedian(self.dic['SYSREM_FLUX3']), self.sidereal_day_period, self.sidereal_day_epoch, dt = self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
        
        #::: for CENTDXY use out of transit data only        
        self.dic['HJD_PHASE_1sidday'], self.dic['CENTDX_out_fd_PHASE_1sidday'], self.dic['CENTDX_out_fd_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['CENTDX_out_fd'], self.sidereal_day_period, self.sidereal_day_epoch, dt = self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
        _,                             self.dic['CENTDY_out_fd_PHASE_1sidday'], self.dic['CENTDY_out_fd_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['CENTDY_out_fd'], self.sidereal_day_period, self.sidereal_day_epoch, dt = self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
        
        
        
        #::: detrend via polyfit
        if polyorder is not None:
            
            def polyfit(x, y):
    #            ind_infinite = ~ np.isfinite(y)
    #            x[ ind_infinite ] = 0
    #            y[ ind_infinite ] = 0
                ind_finite = np.isfinite(x) & np.isfinite(y)
                x = x[ ind_finite ]
                y = y[ ind_finite ]
                return np.polyfit( x, y, polyorder )
                
            #::: fit out the trend and save the resulting polyfunction in the self.dic
            polyfit_params = polyfit( self.dic['HJD_PHASE_1sidday'], self.dic['CENTDX_out_fd_PHASE_1sidday'] )
            self.dic['polyfct_CENTDX'] = np.poly1d( polyfit_params )
                
            polyfit_params = polyfit( self.dic['HJD_PHASE_1sidday'], self.dic['CENTDY_out_fd_PHASE_1sidday'] )
            self.dic['polyfct_CENTDY'] = np.poly1d( polyfit_params )
        
            #::: unwrap the phase-folding
            dx = ( (self.dic['HJD'] - self.sidereal_day_epoch) % (self.sidereal_day_period) ) / self.sidereal_day_period 
            self.dic['poly_CENTDX'] = self.dic['polyfct_CENTDX'](dx)
            self.dic['poly_CENTDY'] = self.dic['polyfct_CENTDY'](dx)
        
            self.dic['CENTDX_fda'] = self.dic['CENTDX_fd'] - self.dic['poly_CENTDX']
            self.dic['CENTDY_fda'] = self.dic['CENTDY_fd'] - self.dic['poly_CENTDY']
            self.dic['CENTDX_out_fda'] = self.dic['CENTDX_out_fd'] - self.dic['poly_CENTDX']
            self.dic['CENTDY_out_fda'] = self.dic['CENTDY_out_fd'] - self.dic['poly_CENTDY']
        
        
        #::: detrend via moving average (median) fit 
        #TODO: currently this removes data points on the edges (not covered in the rolling window) by setting them NAN; try to extrapolate those?
        else:
        
            def ma_fit(y):
                rolling_mean = pd.Series(y).rolling(window=int(len(y)/50.), center=True).mean()
#                rolling_mean[ np.isnan(rolling_mean) ] = 0.
                return rolling_mean.as_matrix()
                
            def ma_fct(t, tp, yp):
                #::: if the phase-folding is not starting at phase 0 (but e.g. at 0.25),
                #::: offset it to phase 0
                if (tp[0] < 0.):
                    ind = np.where( tp<0. )[0]
                    tp = np.append( tp[ind[-1]+1:], tp[ind]+1. )
                    yp = np.append( yp[ind[-1]+1:], yp[ind] )
                return np.interp(t, tp, yp)
                
            #::: fit out the trend and save the resulting polyfunction in the self.dic
            self.dic['ma_CENTDX_PHASE_1sidday'] = ma_fit( self.dic['CENTDX_out_fd_PHASE_1sidday'] )
            self.dic['ma_CENTDY_PHASE_1sidday'] = ma_fit( self.dic['CENTDY_out_fd_PHASE_1sidday'] )
        
            #::: unwrap the phase-folding
            dx = ( (self.dic['HJD'] - self.sidereal_day_epoch) % (self.sidereal_day_period) ) / self.sidereal_day_period 
            self.dic['ma_CENTDX'] = ma_fct( dx, self.dic['HJD_PHASE_1sidday'], self.dic['ma_CENTDX_PHASE_1sidday'] )
            self.dic['ma_CENTDY'] = ma_fct( dx, self.dic['HJD_PHASE_1sidday'], self.dic['ma_CENTDY_PHASE_1sidday'] )
        
        
            self.dic['CENTDX_fda'] = self.dic['CENTDX_fd'] - self.dic['ma_CENTDX']
            self.dic['CENTDY_fda'] = self.dic['CENTDY_fd'] - self.dic['ma_CENTDY']
            self.dic['CENTDX_out_fda'] = self.dic['CENTDX_out_fd'] - self.dic['ma_CENTDX']
            self.dic['CENTDY_out_fda'] = self.dic['CENTDY_out_fd'] - self.dic['ma_CENTDY']

#        _, self.dic['CENTDX_fda_PHASE_1sidday'], self.dic['CENTDX_fda_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['CENTDX_fda'], self.sidereal_day_period, t0, dt = self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
#        _, self.dic['CENTDY_fda_PHASE_1sidday'], self.dic['CENTDY_fda_PHASE_1sidday_ERR'], _, _ = lightcurve_tools.phase_fold( self.dic['HJD'], self.dic['CENTDY_fda'], self.sidereal_day_period, t0, dt = self.dt, ferr_type='meansig', ferr_style='std', sigmaclip=True)
        
        
#        print self.dic['poly_CENTDX']
#        plt.figure(figsize=(12,6))
#        plt.plot( self.dic['HJD'], self.dic['CENTDX_fd_PHASE'], 'g.', rasterized=True )
#        plt.plot( self.dic['HJD'], self.dic['CENTDX_fda_PHASE'], 'b.', rasterized=True )
#        plt.title('target period')
#        plt.show()
#        
#        plt.figure(figsize=(12,6))
#        plt.plot( self.dic['HJD'], self.dic['CENTDX_fd_PHASE_1sidday'], 'g.', rasterized=True )
#        plt.plot( self.dic['HJD'], self.dic['CENTDX_fda_PHASE_1sidday'], 'b.', rasterized=True )
#        plt.title('1 sid day')
#        plt.show()
        
#        dxx = np.arange( self.dic['HJD'][0], self.dic['HJD'][-1], 600 )
#        dxx_wrapped = ( (dxx - t0) % (self.sidereal_day_period) ) / self.sidereal_day_period 
#        plt.plot( dxx, self.dic['polyfct_CENTDX'](dxx_wrapped)*100, 'r-', rasterized=True )
#        plt.show()
        
        
       
    def run_example(self):
        '''
        For test purposes
        '''
        #::: choose example data
        data = self.dic['CENTDX']
        time = self.dic['HJD']
        
        ind_bp = self.breakpoints()
#        data_detrended = detrend_scipy(data, ind_bp)
#        plot(data, data_detrended, time)    
        
        
        
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
    
    
