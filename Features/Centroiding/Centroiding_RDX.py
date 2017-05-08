# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 18:25:52 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import sys, os, glob
import pandas as pd
import timeit
from scipy.stats import binom_test, ttest_1samp

try:
    import statsmodels.api as sm
except ImportError:
    warnings.warn( "Package 'statsmodels' could not be imported. Omitting some analyses.", ImportWarning )

try:
    import seaborn as sns
    sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
except ImportError:
    warnings.warn( "Package 'seaborn' could not be imported. Use standard matplotlib instead.", ImportWarning )


from scripts import ngtsio_v1_1_1_centroiding as ngtsio
from scripts import index_transits, lightcurve_tools, \
                    stacked_images, analyse_neighbours, \
                    detrend_centroid_external_RDX, \
                    helper_functions, get_scatter_color, \
                    pandas_tsa, set_nan
from scripts.helper_functions import mystr    
#from scripts import simulate_signal

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium',
         'font.size': 18.}
pylab.rcParams.update(params)

#import matplotlib
#::: MNRAS STYLE 
#matplotlib.rc('font',**{'family':'serif', 
#             'serif':['Times'], 
#             'weight':'normal', 
#             'size':18})
#matplotlib.rc('legend',**{'fontsize':18})
#matplotlib.rc('text', usetex=True)




###########################################################################
#::: centroid class
########################################################################### 
class centroid():
    '''
    USER INPUT:
    period: in seconds
    epoch: in seconds
    if user wants to manually overwrite the BLS or CANVAS period
    '''
    
    def __init__(self, fieldname, obj_id, ngts_version = 'TEST18', source = 'CANVAS', bls_rank = 1, user_period = None, user_epoch = None, user_width=None, user_flux = None, user_centdx = None, user_centdy = None, time_hjd = None, pixel_radius = 200., flux_min = 500., flux_max = 10000., method='transit', R_min=0., N_top_max=20, bin_width=300., min_time=1800., dt=0.005, roots=None, outdir=None, parent=None, show_plot=False, flagfile=None, dic=None, nancut=None):
#        super(centroid, self).__init__(parent)
            
        self.roots = roots
        
#        self.exposure = 12. #(in s)
        self.fieldname = fieldname
        self.obj_id = obj_id
        self.ngts_version = ngts_version
        self.source = source
        self.bls_rank = bls_rank
        self.user_period = user_period
        self.user_epoch = user_epoch
        self.user_width = user_width
        self.user_flux = user_flux
        self.user_centdx = user_centdx
        self.user_centdy = user_centdy
        self.time_hjd = time_hjd
        self.pixel_radius = pixel_radius
        self.flux_min = flux_min
        self.flux_max = flux_max
        self.method = method
        self.R_min = R_min
        self.N_top_max = N_top_max
        self.bin_width = bin_width #(in s)
        self.min_time = min_time #(in s) minimum coverage of innermost and out-of-transit required for including a night
        self.dt = dt
        self.show_plot = show_plot
        self.flagfile = flagfile
        self.dic = dic
        self.nancut = nancut
        
        self.crosscorr = {}
        
        if outdir is None: 
            self.outdir = os.path.join( 'output', self.ngts_version, self.fieldname, str(self.pixel_radius)+'_'+str(self.flux_min)+'_'+str(self.flux_max)+'_'+self.method+'_'+str(self.R_min)+'_'+str(self.N_top_max)+'_'+str(self.bin_width)+'_'+str(self.min_time)+'_'+str(self.dt), '' )
        else:
            self.outdir = outdir
        if not os.path.exists(self.outdir): 
            os.makedirs(self.outdir)
          
        try:
            self.catalogfname = glob.glob( 'input/catalog/'+self.fieldname+'*'+self.ngts_version+'_cat_master.dat')[0]
        except:
            self.catalogfname = None
            
#        self.run()
        
        
        
    ###########################################################################
    #::: load data (all nights) of the target
    ###########################################################################
    def load_object(self):
        
        #::: if no basic dic has been given, load it via ngtsio
        if self.dic is None:
            #::: get infos for the target object
            keys = ['OBJ_ID','FLUX_MEAN','RA','DEC', \
                    'NIGHT','AIRMASS', \
                    'HJD','CCDX','CCDY','CENTDX','CENTDY','SYSREM_FLUX3', \
                    'PERIOD','WIDTH','EPOCH','DEPTH','NUM_TRANSITS', \
                    'CANVAS_PERIOD','CANVAS_WIDTH','CANVAS_EPOCH','CANVAS_DEPTH']
            self.dic = ngtsio.get(self.fieldname, keys, obj_id = self.obj_id, time_hjd = self.time_hjd, ngts_version = self.ngts_version, bls_rank = self.bls_rank, silent = True, set_nan = True)

            if self.source == 'CANVAS':
                #::: overwrite BLS with CANVAS infos (if existing) -> copy the keys
                for key in ['PERIOD','WIDTH','EPOCH','DEPTH']:
                    #::: if the CANVAS data exists, use it
                    if ('CANVAS_'+key in self.dic) and (~np.isnan( self.dic['CANVAS_'+key] )): 
                        self.dic[key] = self.dic['CANVAS_'+key].copy()
                        #::: convert CANVAS depth (in mmag) into the standard BLS depth (float)
                        if key == 'DEPTH': self.dic[key] *= -1E-3
                    #::: if the CANVAS data is missing, make BLS the source
                    else:
                        self.source = 'BLS'
        else:
            pass #(self.dic does already exist)
                    
        #::: set FLUX=0 entries to NaN
        # not needed anymore, now handled within ngtsio
#        self.dic = set_nan.set_nan(self.dic) 
        
        #::: calculate unique nights
        self.dic['UNIQUE_NIGHT'] = np.unique(self.dic['NIGHT']) 
        
        #::: calcualte median CCD position
        self.dic['CCDX_0'] = np.nanmedian(self.dic['CCDX'])
        self.dic['CCDY_0'] = np.nanmedian(self.dic['CCDY'])
        
        #::: overwrite flux, period, epoch and width if given by user
        #::: raise error if user_flux is not the correct length
        if self.user_flux is not None:
            if self.user_flux.shape != self.dic['SYSREM_FLUX3'].shape:
                raise ValueError( "user_flux must be the same length as the ngtsio_flux. self.user_flux.shape = "  + str(self.user_flux.shape) + ", self.dic['SYSREM_FLUX3'].shape = " + str(self.dic['SYSREM_FLUX3'].shape) )
            self.dic['SYSREM_FLUX3'] = self.user_flux
            
        if self.user_centdx is not None:
            if self.user_centdx.shape != self.dic['CENTDX'].shape:
                raise ValueError( "user_centdx must be the same length as the ngtsio_centdx. self.user_centdx.shape = "  + str(self.user_centdx.shape) + ", self.dic['CENTDX'].shape = " + str(self.dic['CENTDX'].shape) )
            self.dic['CENTDX'] = self.user_centdx
            
        if self.user_centdy is not None:
            if self.user_centdy.shape != self.dic['CENTDY'].shape:
                raise ValueError( "user_centdy must be the same length as the ngtsio_centdy. self.user_centdy.shape = "  + str(self.user_centdy.shape) + ", self.dic['CENTDY'].shape = " + str(self.dic['CENTDY'].shape) )
            self.dic['CENTDY'] = self.user_centdy
                
        if ( (self.user_flux is not None) | (self.user_centdx is not None) | (self.user_centdy is not None) ):
            self.dic = set_nan.set_nan(self.dic, key='SYSREM_FLUX3')
            
        if (self.user_period is not None) & (self.user_period > 0):
            self.dic['PERIOD'] = self.user_period #in s
        if (self.user_epoch is not None) & (self.user_epoch > 0):
            self.dic['EPOCH'] = self.user_epoch #in s
        if (self.user_width is not None) & (self.user_width > 0):
            self.dic['WIDTH'] = self.user_width #in s
            
                
        

    ###########################################################################
    #::: identify the neighbouring objects for reference
    ########################################################################### 
    def load_neighbours(self):
        
        #::: load position and flux of all objects in the field for reference
        self.dic_all = ngtsio.get(self.fieldname, ['CCDX','CCDY','FLUX_MEAN'], time_index=0, ngts_version = self.ngts_version, bls_rank = self.bls_rank, silent=True)
        
        #::: find neighbouring objects
        ind_neighbour = np.where( (np.abs(self.dic_all['CCDX'] - self.dic['CCDX_0']) < self.pixel_radius) & \
                                 (np.abs(self.dic_all['CCDY'] - self.dic['CCDY_0']) < self.pixel_radius) & \
                                 (self.dic_all['FLUX_MEAN'] > self.flux_min) & \
                                 (self.dic_all['FLUX_MEAN'] < self.flux_max) & \
                                 (self.dic_all['OBJ_ID'] != self.dic['OBJ_ID']) \
                                 )[0] 
        obj_id_nb = self.dic_all['OBJ_ID'][ind_neighbour]             
        
        #::: get infos of neighbouring objects
        self.dic_nb = ngtsio.get( self.fieldname, ['OBJ_ID','HJD','CCDX','CCDY','CENTDX','CENTDY','FLUX_MEAN'], obj_id = obj_id_nb, time_hjd = self.time_hjd, ngts_version = self.ngts_version, bls_rank = self.bls_rank, silent = True) 
        self.dic_nb['CCDX_0'] = np.nanmedian( self.dic_nb['CCDX'], axis=1 )
#        self.dic_all['CCDX'][ind_neighbour]
        self.dic_nb['CCDY_0'] = np.nanmedian( self.dic_nb['CCDY'], axis=1 )
#        self.dic_all['CCDY'][ind_neighbour]
#        del self.dic_nb['CCDX']
#        del self.dic_nb['CCDY']
        
        #::: delete the dictionary of all objects
        del self.dic_all
        
#        print 'N_neighbours =', len(ind_neighbour)
#        
#        print '***********************************'
#        print self.dic['HJD'].shape
#        print self.dic['SYSREM_FLUX3'].shape
#        print self.dic_nb['CENTDX'].shape
#        print self.dic_nb['CENTDY'].shape
#        print self.dic_nb['CCDX'].shape
#        print self.dic_nb['CCDY'].shape
        
        #::: apply same nancut as for target object to all neighbours 
        #::: ("autovet" specific only)
        if self.nancut is not None:
            for key in self.dic_nb: 
                if isinstance(self.dic_nb[key], np.ndarray):
                    if (self.dic_nb[key].ndim==2) :
                        self.dic_nb[key] = self.dic_nb[key][slice(None), ~self.nancut]
    


    ###########################################################################
    #::: import crossmatched catalog (Ed's version with 2MASS)
    ########################################################################### 
    def load_catalog(self):
        #::: if the catalogue for this object exists
        if self.catalogfname is not None:
            catdata = np.genfromtxt( self.catalogfname, names=True )
            
            self.dic['B-V'] = catdata['BV'][ catdata['NGTS_ID'] == float(self.dic['OBJ_ID']) ]   
            self.dic['Vmag'] = catdata['Vmag'][ catdata['NGTS_ID'] == float(self.dic['OBJ_ID']) ]  
            self.dic['Jmag'] = catdata['Jmag'][ catdata['NGTS_ID'] == float(self.dic['OBJ_ID']) ]  
            self.dic['Bmag'] = catdata['Bmag'][ catdata['NGTS_ID'] == float(self.dic['OBJ_ID']) ]  
            
            self.dic_nb['B-V'] = np.zeros( len(self.dic_nb['OBJ_ID']) ) * np.nan
            self.dic_nb['Vmag'] = np.zeros( len(self.dic_nb['OBJ_ID']) ) * np.nan
            for i, obj_id in enumerate(self.dic_nb['OBJ_ID']):
                self.dic_nb['B-V'][i] = catdata['BV'][ catdata['NGTS_ID'] == float(obj_id) ]  
                self.dic_nb['Vmag'][i] = catdata['Vmag'][ catdata['NGTS_ID'] == float(obj_id) ]  
               
            del catdata
        
        #::: otherwise
        else:
            self.dic['B-V'] = np.nan
            self.dic['Vmag'] = np.nan
            self.dic['Jmag'] = np.nan
            self.dic['Bmag'] = np.nan
            self.dic_nb['B-V'] = np.zeros( len(self.dic_nb['OBJ_ID']) ) * np.nan
            self.dic_nb['Vmag'] = np.zeros( len(self.dic_nb['OBJ_ID']) ) * np.nan




    def mark_eclipses(self):
        self.ind_out = index_transits.index_eclipses(self.dic)[-1]
        

    def assign_airmass_colorcode(self):
        #::: assign colors for different nights; dconvert HJD from seconds into days
        self.dic = get_scatter_color.get_scatter_color(self.dic)
        self.dic['COLOR'] = np.concatenate( self.dic['COLOR_PER_NIGHT'], axis=0 )

        

    def binning(self):
        self.dic['HJD_BIN'], \
            [ self.dic['CENTDX_fda_BIN'], self.dic['CENTDY_fda_BIN'], self.dic['ma_CENTDX_BIN'], self.dic['ma_CENTDY_BIN'], self.dic['COLOR_BIN'], self.dic['AIRMASS_BIN'], self.dic['SYSREM_FLUX3_BIN'], self.dic['CENTDX_f_BIN'], self.dic['CENTDY_f_BIN'], self.dic['CENTDX_fd_BIN'], self.dic['CENTDY_fd_BIN'], self.dic_nb['CENTDX_ref_mean_BIN'], self.dic_nb['CENTDY_ref_mean_BIN'] ], \
            [ self.dic['CENTDX_fda_BIN_ERR'], self.dic['CENTDY_fda_BIN_ERR'], self.dic['ma_CENTDX_BIN_ERR'], self.dic['ma_CENTDY_BIN_ERR'], self.dic['COLOR_BIN_ERR'], self.dic['AIRMASS_BIN_ERR'], self.dic['SYSREM_FLUX3_BIN_ERR'], self.dic['CENTDX_f_BIN_ERR'], self.dic['CENTDY_f_BIN_ERR'], self.dic['CENTDX_fd_BIN_ERR'], self.dic['CENTDY_fd_BIN_ERR'], self.dic_nb['CENTDX_ref_mean_BIN_ERR'], self.dic_nb['CENTDY_ref_mean_BIN_ERR'] ], \
            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic['CENTDX_fda'], self.dic['CENTDY_fda'], self.dic['ma_CENTDX'], self.dic['ma_CENTDY'], self.dic['COLOR'], self.dic['AIRMASS'], self.dic['SYSREM_FLUX3'], self.dic['CENTDX_f'], self.dic['CENTDY_f'], self.dic['CENTDX_fd'], self.dic['CENTDY_fd'], self.dic_nb['CENTDX_ref_mean'], self.dic_nb['CENTDY_ref_mean'] )), dt=600, sigmaclip=False, ferr_style='std' )

#        self.dic['HJD_BIN'], \
#            [ self.dic['CENTDX_fda_BIN'], self.dic['CENTDY_fda_BIN'], self.dic['poly_CENTDX_BIN'], self.dic['poly_CENTDY_BIN'], self.dic['COLOR_BIN'], self.dic['AIRMASS_BIN'], self.dic['SYSREM_FLUX3_BIN'], self.dic['CENTDX_f_BIN'], self.dic['CENTDY_f_BIN'], self.dic['CENTDX_fd_BIN'], self.dic['CENTDY_fd_BIN'] ], \
#            [ self.dic['CENTDX_fda_BIN_ERR'], self.dic['CENTDY_fda_BIN_ERR'], self.dic['poly_CENTDX_BIN_ERR'], self.dic['poly_CENTDY_BIN_ERR'], self.dic['COLOR_BIN_ERR'], self.dic['AIRMASS_BIN_ERR'], self.dic['SYSREM_FLUX3_BIN_ERR'], self.dic['CENTDX_f_BIN_ERR'], self.dic['CENTDY_f_BIN_ERR'], self.dic['CENTDX_fd_BIN_ERR'], self.dic['CENTDY_fd_BIN_ERR'] ], \
#            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic['CENTDX_fda'], self.dic['CENTDY_fda'], self.dic['poly_CENTDX'], self.dic['poly_CENTDY'], self.dic['COLOR'], self.dic['AIRMASS'], self.dic['SYSREM_FLUX3'], self.dic['CENTDX_f'], self.dic['CENTDY_f'], self.dic['CENTDX_fd'], self.dic['CENTDY_fd'] )), dt=600, sigmaclip=False, ferr_style='std' )
#        
#        _, \
#            [ self.dic_nb['CENTDX_ref_mean_BIN'], self.dic_nb['CENTDY_ref_mean_BIN'] ], \
#            [ self.dic_nb['CENTDX_ref_mean_BIN_ERR'], self.dic_nb['CENTDY_ref_mean_BIN_ERR'] ], \
#            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic_nb['CENTDX_ref_mean'], self.dic_nb['CENTDY_ref_mean'] )), dt=600 )



    ###########################################################################
    #::: 
    ########################################################################### 
    def phase_fold(self):
        self.N_phasepoints = int( 1./self.dt )
        
        self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], self.dic['SYSREM_FLUX3_PHASE_ERR'], self.dic['N_PHASE'], self.dic['PHI'] = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['SYSREM_FLUX3'] / np.nanmedian(self.dic['SYSREM_FLUX3'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_PHASE'], self.dic['CENTDX_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX'] - np.nanmedian(self.dic['CENTDX'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_PHASE'], self.dic['CENTDY_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY'] - np.nanmedian(self.dic['CENTDY'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_f_PHASE'], self.dic['CENTDX_f_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_f'] - np.nanmedian(self.dic['CENTDX_f'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_f_PHASE'], self.dic['CENTDY_f_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_f'] - np.nanmedian(self.dic['CENTDY_f'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_fd_PHASE'], self.dic['CENTDX_fd_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_fd'] - np.nanmedian(self.dic['CENTDX_fd'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_fd_PHASE'], self.dic['CENTDY_fd_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_fd'] - np.nanmedian(self.dic['CENTDY_fd'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_fda_PHASE'], self.dic['CENTDX_fda_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_fda'] - np.nanmedian(self.dic['CENTDX_fda'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_fda_PHASE'], self.dic['CENTDY_fda_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_fda'] - np.nanmedian(self.dic['CENTDY_fda'][self.ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)

        
        
    ###########################################################################
    #::: 
    ########################################################################### 
    def cross_correlate(self):
        #::: create pandas df from parts of self.dic
        self.phasedf = pd.DataFrame( {k: self.dic[k] for k in ('HJD_PHASE', 'SYSREM_FLUX3_PHASE', 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE')}, columns=['HJD_PHASE', 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'SYSREM_FLUX3_PHASE'] )
   
        self.fig_corrfx, flags_fx = self.ccfct( 'SYSREM_FLUX3_PHASE', 'CENTDX_fda_PHASE', 'FLUX vs CENTDX' )
        self.fig_corrfy, flags_fy = self.ccfct( 'SYSREM_FLUX3_PHASE', 'CENTDY_fda_PHASE', 'FLUX vs CENTDY' )
        self.fig_corrxy, flags_xy = self.ccfct( 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'CENTDX vs CENTDY' )

        #self.fig_autocorr = self.acfct( ['SYSREM_FLUX3_PHASE','CENTDX_fda_PHASE','CENTDY_fda_PHASE'], ['FLUX','CENTDX','CENTDY'] )            
             
 
    #::: autocorrelation fct
    def acfct(self, xkeys, titles):
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        fig, axes = plt.subplots(2,3,figsize=(15,8))
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
               
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#        for i, xkey in enumerate(xkeys):
#            x = self.phasedf[xkey]
#            ax = axes[0,i]
#            
#            autocorr, lags, autocorr_CI95, autocorr_CI99 = pandas_tsa.pandas_autocorr(x)
#            ax.plot( lags, autocorr )
#            ax.plot( lags[10:-10], autocorr_CI99[10:-10], 'k--' )
#            ax.plot( lags[10:-10], -autocorr_CI99[10:-10], 'k--' )
#            ax.set( xlim=[lags[0]-50, lags[-1]+50], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='acf' )
#            ax.set_xticklabels( [(j*self.dt) for j in ax.get_xticks()] )
#            ax.set_title( titles[i] )
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#        for i, xkey in enumerate(xkeys):
#            x = self.phasedf[xkey]
#            ax = axes[1,i]
#            
#            kwargs = { 'marker':None, 'linestyle':'-'}
#            sm.graphics.tsa.plot_acf(x, ax=ax, alpha=0.05, lags=len(x)-1, unbiased=True, use_vlines=False, **kwargs)
#            lags = np.arange( len(x) )
#            autocorr_CI95 = 1.96/np.sqrt( len(x) - lags )
#            autocorr_CI99 = 2.58/np.sqrt( len(x) - lags )
#            ax.plot( lags[0], autocorr[0], 'o' )
#            ax.plot( lags[10:-10], autocorr_CI99[10:-10], 'k--' )
#            ax.plot( lags[10:-10], -autocorr_CI99[10:-10], 'k--' )
#            ax.set( xlim=[lags[0]-50, lags[-1]+50], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='acf', title='' )
#            ax.set_xticklabels( [(j*self.dt) for j in ax.get_xticks()] )
#            ax.legend()
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for i, xkey in enumerate(xkeys):
            ax = axes[0,i]
            x = self.phasedf[xkey]
            autocorr, lags, autocorr_CI95, autocorr_CI99 = pandas_tsa.pandas_periodic_autocorr(x)
            N_a = len(autocorr)   
            xlags = np.linspace(-0.25,0.75,N_a)   
            autocorr = np.concatenate( (autocorr[ int(3*N_a/4): ], autocorr[ :int(3*N_a/4) ]) )
            autocorr_CI95 = np.concatenate( (autocorr_CI95[ int(3*N_a/4): ], autocorr_CI95[ :int(3*N_a/4) ]) )
            autocorr_CI99 = np.concatenate( (autocorr_CI99[ int(3*N_a/4): ], autocorr_CI99[ :int(3*N_a/4) ]) )
            ax.plot( xlags, autocorr, 'g-' )
            ax.plot( xlags[10:-10], autocorr_CI99[10:-10], 'k--' )
            ax.plot( xlags[10:-10], -autocorr_CI99[10:-10], 'k--' )
            ax.set( title=titles[i], xlim=[xlags[0], xlags[-1]], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='acf (periodic)' )
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::        
            
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for i, xkey in enumerate(xkeys):
            x = self.phasedf[xkey]
            ax = axes[1,i]
            
            kwargs = { 'marker':None, 'linestyle':'-'}
            nlags = 50
            try:
                sm.graphics.tsa.plot_pacf(x, ax=ax, alpha=0.05, lags=nlags, use_vlines=False, **kwargs) #lags=len(x)-1
            except:
                pass
            lags = np.arange( nlags )
            ax.set( xlim=[lags[0]-2, lags[-1]+2], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='pacf', title='' )
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                   
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        plt.tight_layout()
        return fig
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
       
        
               
             
    def ccfct(self, xkey, ykey, title):
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        x = self.phasedf[xkey]
        y = self.phasedf[ykey]
        
        #fig, axes = plt.subplots(1,2,figsize=(10,4))
        #fig.suptitle(self.obj_id + ' ' + title)
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
               
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#        win = [0.1, 0.25, 0.33]
        win = [0.25]
        windows= ( np.array(win) * (1/self.dt) ).astype(int)
        correls = [ self.phasedf.rolling(window=windows[i], center=True).corr() for i,_ in enumerate(windows) ]
        
#        color = ['b','g','r']
        for i,_ in enumerate(windows): 
            self.dic['RollCorr_'+xkey+'_'+ykey] = correls[i].loc[ :, xkey, ykey ]
        #    axes[0].plot( self.phasedf['HJD_PHASE'], correls[i].loc[ :, xkey, ykey ], label=str(windows[i] * self.dt) )
        #    axes[0].axhline( 2.58/np.sqrt(windows[i]), color='k', linestyle='--')#, color=color[i] )
        #    axes[0].axhline( - 2.58/np.sqrt(windows[i]), color='k', linestyle='--')#, color=color[i] )
       # axes[0].set( xlim=[-0.25,0.75], ylim=[-1,1], xlabel='phase', ylabel='rolling correlation')
       # axes[0].legend()
                
#        crosscorr, lags, crosscorr_CI95, crosscorr_CI99 = pandas_tsa.pandas_crosscorr(x, y)
#        axes[1].plot( lags, crosscorr )
#        axes[1].plot( lags[10:-10], crosscorr_CI99[10:-10], 'k--' )
#        axes[1].plot( lags[10:-10], -crosscorr_CI99[10:-10], 'k--' )
#        axes[1].set( xlim=[lags[0]-50, lags[-1]+50], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='ccf' )
#        axes[1].set_xticklabels( [(i*self.dt) for i in axes[1].get_xticks()] )
            
        crosscorr, lags, crosscorr_CI95, crosscorr_CI99 = pandas_tsa.pandas_periodic_crosscorr(x, y)
        N_c = len(crosscorr)   
        self.ccf_lags = np.linspace(-0.25,0.75,N_c)
        self.crosscorr[title] = np.concatenate( (crosscorr[ int(3*N_c/4): ], crosscorr[ :int(3*N_c/4) ]) )
        self.dic['CrossCorr_'+xkey+'_'+ykey] = self.crosscorr[title]
        crosscorr_CI95 = np.concatenate( (crosscorr_CI95[ int(3*N_c/4): ], crosscorr_CI95[ :int(3*N_c/4) ]) )
        crosscorr_CI99 = np.concatenate( (crosscorr_CI99[ int(3*N_c/4): ], crosscorr_CI99[ :int(3*N_c/4) ]) )
       # axes[1].plot( self.ccf_lags, self.crosscorr[title] )
      #  axes[1].plot( self.ccf_lags, crosscorr_CI99, 'k--' )
      #  axes[1].plot( self.ccf_lags, -crosscorr_CI99, 'k--' )
      #  axes[1].set( xlim=[self.ccf_lags[0], self.ccf_lags[-1]], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='ccf (periodic)' )
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                  
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
      #  plt.tight_layout() 
        fig = 0
        return fig, None
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

            

       
    def do_stats(self):
        
        #::: center on out-of-transit
        phi = self.dic['HJD_PHASE']
        ind_0 = np.argmin(np.abs(phi)) #i.e. where PHASE==0
        #TODO: very rough approximation
        ind_out_phase = np.where( (phi<-0.15) | ((phi>0.15) & (phi<0.35)) | (phi>0.65) )[0] #::: very rough approx.
#            ind_in_transit = np.where( self.dic['SYSREM_FLUX3_PHASE']<0.99 )[0] #::: very rough approx.
        centdx = self.dic['CENTDX_fda_PHASE']*1000.
        centdy = self.dic['CENTDY_fda_PHASE']*1000.
        centdx -= np.nanmedian(centdx[ ind_out_phase ])
        centdy -= np.nanmedian(centdy[ ind_out_phase ])

        #::: open dictionary on correlation stats
        self.stats = {}   
        
        #::: Signal-to-noise ratio in rolling correlation (noise == standard deviation out of transit) 
        for key in ['X','Y']:
            self.stats['RollCorrSNR_'+key] = np.abs( self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTD'+key+'_fda_PHASE'][ ind_0 ] / np.nanstd( self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTD'+key+'_fda_PHASE'][ ind_out_phase ] ) )
#            print SNR_RC
        
        #::: Signal-to-noise ratio in cross-correlation  
        for key in ['X','Y']:
            self.stats['CrossCorrSNR_'+key] = np.abs( self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTD'+key+'_fda_PHASE'][ ind_0 ] / np.nanstd( self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTD'+key+'_fda_PHASE'][ ind_out_phase ] ) )
#            print SNR_CC            
        
        #::: Hypothesis tests on rain plots
        for centd, key in zip([centdx, centdy], ['X','Y']):
            x_hyptest = len( centd[ (self.dic['SYSREM_FLUX3_PHASE']<0.99) & (centd<0.) ] )
            N_hyptest = len( centd[ self.dic['SYSREM_FLUX3_PHASE']<0.99 ] )
            self.stats['Binom_'+key] = binom_test( x_hyptest, N_hyptest, p=0.5 ) 
#                print 'p-value(Binom_' + key + ') = ' + str(p_value['Binom_'+key])   
#                if p_value['Binom_'+key] <= 0.01:
#                    print '\tREJECT the Null Hypothesis that there is no centroid shift.'
#                else:
#                    print '\tNo conclusion.'
                
            t_statistic, self.stats['Ttest_'+key] = ttest_1samp( centd[ self.dic['SYSREM_FLUX3_PHASE']<0.99 ], 0 )   
#                print 'p-value(Ttest_' + key + ') = ' + str(p_value['Ttest_'+key])   
#                if p_value['Ttest_'+key] <= 0.01:
#                    print '\tREJECT the Null Hypothesis that there is no centroid shift.'
#                else:
#                    print '\tNo conclusion.'
                         
                         
                
    def plot_scatter_matrix(self):  
        
        axes = pd.tools.plotting.scatter_matrix(self.phasedf[ ['CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'SYSREM_FLUX3_PHASE'] ], alpha=0.6, figsize=(10, 10), diagonal='kde') # diagonal='hist'
        self.fig_matrix = plt.gcf()
        
        #Change label notation
        labels = ['CENTDX','CENTDY','FLUX']
        [s.xaxis.label.set_text(labels[i%3]) for i,s in enumerate(axes.reshape(-1))]
        [s.yaxis.label.set_text(labels[i/3]) for i,s in enumerate(axes.reshape(-1))]
    
        #::: remove dublicates form plot
        corr = self.phasedf[ ['CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'SYSREM_FLUX3_PHASE'] ].corr().as_matrix()
        for n in range( axes.shape[0] * axes.shape[1] ):
            i, j = np.unravel_index(n, axes.shape)
            if j==0:
                axes[i,j].axvline(0, linestyle='--', color='grey')
                axes[i,j].axvline(0.5, linestyle='--', color='grey')
            if j>0:
                axes[i,j].axvline(0, linestyle='--', color='grey')
            # remove dublicates form plot
            if j>i: 
                axes[i,j].cla()
                axes[i,j].set_axis_off()
            # log-scale the kde plots
            if i == j:  
                axes[i,j].set_yscale('log')
            # annotate corr coefficients
            if j<i:
                axes[i,j].annotate("%.3f" %corr[i,j], (0.85, 0.9), xycoords='axes fraction', ha='center', va='center')



    ###########################################################################
    #::: look at phase-folded lightcurve and centroid curve
    ###########################################################################        
    def plot_phase_folded_curves(self):
        
        #::: detrended curves
        self.fig_phasefold, axes = plt.subplots( 4, 1, sharex=True, figsize=(16,16) )
        
        axes[0].plot( self.dic['PHI'], self.dic['SYSREM_FLUX3']/np.nanmedian(self.dic['SYSREM_FLUX3']), 'k.', alpha=0.1, rasterized=True )
        axes[0].errorbar( self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], yerr=self.dic['SYSREM_FLUX3_PHASE_ERR'], fmt='o', color='r', ms=10, rasterized=True )
        axes[0].set( ylabel='FLUX', ylim=[ np.min(self.dic['SYSREM_FLUX3_PHASE']-self.dic['SYSREM_FLUX3_PHASE_ERR']), np.max(self.dic['SYSREM_FLUX3_PHASE']+self.dic['SYSREM_FLUX3_PHASE_ERR']) ])
        
        axes[1].plot( self.dic['PHI'], self.dic['CENTDX'], 'k.', alpha=0.1, rasterized=True )
        axes[1].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDX_fda_PHASE'], yerr=self.dic['CENTDX_fda_PHASE_ERR'], fmt='o', ms=10, rasterized=True ) #, color='darkgrey')
        axes[1].set( ylabel='CENTDX (in pixel)', ylim=[ np.min(self.dic['CENTDX_fda_PHASE']-self.dic['CENTDX_fda_PHASE_ERR']), np.max(self.dic['CENTDX_fda_PHASE']+self.dic['CENTDX_fda_PHASE_ERR']) ])
        
        axes[2].plot( self.dic['PHI'], self.dic['CENTDY'], 'k.', alpha=0.1, rasterized=True )
        axes[2].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDY_fda_PHASE'], yerr=self.dic['CENTDY_fda_PHASE_ERR'], fmt='o', ms=10, rasterized=True ) #, color='darkgrey')
        axes[2].set( ylabel='CENTDY (in pixel)', xlabel='Phase', ylim=[ np.min(self.dic['CENTDY_fda_PHASE']-self.dic['CENTDY_fda_PHASE_ERR']), np.max(self.dic['CENTDY_fda_PHASE']+self.dic['CENTDY_fda_PHASE_ERR']) ], xlim=[-0.25,0.75])
        
        axes[3].plot( self.dic['HJD_PHASE'], self.dic['N_PHASE'], 'go', ms=10, rasterized=True ) #, color='darkgrey')
        axes[3].set( ylabel='Nr of exposures', xlabel='Phase', xlim=[-0.25,0.75])
        
        plt.tight_layout()
        
           
         
    ###########################################################################
    #::: plot info page
    ###########################################################################  
    def plot_info_page(self):
        #::: plot object
        self.fig_info_page = plt.figure(figsize=(16,3.6))
        gs = gridspec.GridSpec(1, 4)
        
        #::: plot locations on CCD    
        ax = plt.subplot(gs[0, 0])
        xtext = 50
        if self.dic['CCDY_0'] < 1000:
            ytext = 1800
        else:
            ytext = 350
        ax.text(xtext,ytext,'Flux ' + mystr(self.flux_min) + '-' + mystr(self.flux_max))
        ax.text(xtext,ytext-250, str(self.pixel_radius) + ' px')
        ax.plot( self.dic_nb['CCDX_0'], self.dic_nb['CCDY_0'], 'k.', rasterized=True )
        ax.plot( self.dic['CCDX_0'], self.dic['CCDY_0'], 'r.', rasterized=True ) 
        ax.add_patch( patches.Rectangle(
                        (self.dic['CCDX_0']-self.pixel_radius, self.dic['CCDY_0']-self.pixel_radius),
                        2*self.pixel_radius,
                        2*self.pixel_radius,
                        fill=False, color='r', lw=2) )
    #    ax.axis('equal')
        ax.set(xlim=[0,2048], ylim=[0,2048], xlabel='CCDX', ylabel='CCDY') 
        
        #::: plot lightcurve
        ax = plt.subplot(gs[0, 1:3])
        time = self.dic['HJD'] / 3600. / 24.
        ax.plot( time, self.dic['SYSREM_FLUX3'], 'k.', rasterized=True )
        ax.set(xlim=[time[0], time[-1]], xlabel='HJD', ylabel='Flux (counts)')
        
        #::: add second axis to lightcurve
        conversion = 1./np.nanmedian(self.dic['SYSREM_FLUX3'])
        ax2 = ax.twinx()
        mn, mx = ax.get_ylim()
        ax2.set_ylim(mn*conversion, mx*conversion)
        ax2.set_ylabel('Flux (norm.)')

        #::: plot info text
        ax = plt.subplot(gs[0, 3])
        self.plot_info_text(ax)    
        
        plt.tight_layout()
    
    
    
    def plot_info_text(self, ax):
        if 'DEPTH' not in self.dic: depth = ''
        else: depth = mystr(np.abs(self.dic['DEPTH'])*1000.,2)
        if 'NUM_TRANSITS' not in self.dic: num_transits = ''
        else: depth = mystr(self.dic['NUM_TRANSITS'],0)
            
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.axis('off')
        hmsdms = helper_functions.deg2hmsdms(self.dic['RA']*180/np.pi, self.dic['DEC']*180/np.pi)
#        ra, dec = deg2HMS.deg2HMS(ra=self.dic['RA'], dec=self.dic['DEC'])
        ax.text(0,1.0,self.fieldname+' '+self.dic['OBJ_ID']+' '+self.source)
        ax.text(0,0.9,hmsdms)
        ax.text(0,0.8,'FLUX: '+str(self.dic['FLUX_MEAN']))
        ax.text(0,0.7,'J-mag: '+mystr(self.dic['Jmag'],3))
        ax.text(0,0.6,'V-mag: '+mystr(self.dic['Vmag'],3))
        ax.text(0,0.5,'B-mag: '+mystr(self.dic['Bmag'],3))
        ax.text(0,0.4,'B-V color: '+mystr(self.dic['B-V'],3))
#        ax.text(0,0.6,'PERIOD (s): '+mystr(self.dic['PERIOD'],2))
        ax.text(0,0.3,'PERIOD (d): '+mystr(self.dic['PERIOD']/3600./24.,3))
#        ax.text(0,0.4,'Width (s): '+mystr(self.dic['WIDTH'],2))
        ax.text(0,0.2,'Width (h): '+mystr(self.dic['WIDTH']/3600.,2))
#        ax.text(0,0.2,'EPOCH (s): '+mystr(self.dic['EPOCH'],2))
        ax.text(0,0.1,'Depth (mmag): '+depth)
        ax.text(0,0.0,'Num Transits: '+num_transits)
        


    def plot_stacked_image(self):
        try:
            self.fig_stacked_image = stacked_images.plot(self.fieldname, self.ngts_version, self.dic['CCDX_0'], self.dic['CCDY_0'], r=15) 
        except:
            self.fig_stacked_image = None
            


    ###########################################################################
    #::: save all plots in one pdf per target object
    ###########################################################################   
    def save_pdf(self):
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_analysis.pdf' )          
        with PdfPages( outfilename ) as pdf:
            pdf.savefig( self.fig_corrfx  )
            pdf.savefig( self.fig_corrfy  )
            pdf.savefig( self.fig_corrxy  )
            pdf.savefig( self.fig_phasefold  )
            pdf.savefig( self.fig_matrix  )
            pdf.savefig( self.fig_autocorr  )
            pdf.savefig( self.fig_info_page  )
            if self.fig_stacked_image is not None: 
                pdf.savefig( self.fig_stacked_image  )
            print 'Plots saved as ' + outfilename
            
        if self.show_plot == False: plt.close('all')
            
            
            
    ###########################################################################
    #::: save the phasecurve data for further external fitting
    ########################################################################### 
    def save_data(self):
        #::: phase-folded data:
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_data_PHASE.txt' )
        X = np.c_[ self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], self.dic['SYSREM_FLUX3_PHASE_ERR'], 
                   self.dic['CENTDX_fda_PHASE'], self.dic['CENTDX_fda_PHASE_ERR'], self.dic['CENTDY_fda_PHASE'], self.dic['CENTDY_fda_PHASE_ERR'], 
                   self.dic['CENTDX_fd_PHASE'], self.dic['CENTDX_fd_PHASE_ERR'], self.dic['CENTDY_fd_PHASE'], self.dic['CENTDY_fd_PHASE_ERR'], 
                   self.dic['CENTDX_f_PHASE'], self.dic['CENTDX_f_PHASE_ERR'], self.dic['CENTDY_f_PHASE'], self.dic['CENTDY_f_PHASE_ERR'], 
                   self.dic['CENTDX_PHASE'], self.dic['CENTDX_PHASE_ERR'], self.dic['CENTDY_PHASE'], self.dic['CENTDY_PHASE_ERR'],
                   self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'], self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'], self.dic['RollCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'],
                   self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'], self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'], self.dic['CrossCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'] ]
        header = 'HJD_PHASE'+'\t'+'SYSREM_FLUX3_PHASE'+'\t'+'SYSREM_FLUX3_PHASE_ERR'+'\t'+\
                 'CENTDX_fda_PHASE'+'\t'+'CENTDX_fda_PHASE_ERR'+'\t'+'CENTDY_fda_PHASE'+'\t'+'CENTDY_fda_PHASE_ERR'+'\t'+\
                 'CENTDX_fd_PHASE'+'\t'+'CENTDX_fd_PHASE_ERR'+'\t'+'CENTDY_fd_PHASE'+'\t'+'CENTDY_fd_PHASE_ERR'+'\t'+\
                 'CENTDX_f_PHASE'+'\t'+'CENTDX_f_PHASE_ERR'+'\t'+'CENTDY_f_PHASE'+'\t'+'CENTDY_f_PHASE_ERR'+'\t'+\
                 'CENTDX_PHASE'+'\t'+'CENTDX_PHASE_ERR'+'\t'+'CENTDY_PHASE'+'\t'+'CENTDY_PHASE_ERR'+'\t'+\
                 'RollCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'+'\t'+'RollCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'+'\t'+'RollCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'+'\t'+\
                 'CrossCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'+'\t'+'CrossCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'+'\t'+'CrossCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'
        np.savetxt(outfilename, X, delimiter='\t', header=header)
        
        #::: full time series data, binned to 10min
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_data_BIN.txt' )
        X = np.c_[ self.dic['HJD_BIN'], self.dic['SYSREM_FLUX3_BIN'], self.dic['SYSREM_FLUX3_BIN_ERR'], 
                   self.dic['CENTDX_fda_BIN'], self.dic['CENTDX_fda_BIN_ERR'], self.dic['CENTDY_fda_BIN'], self.dic['CENTDY_fda_BIN_ERR'] ]
        header = 'HJD_BIN'+'\t'+'SYSREM_FLUX3_BIN'+'\t'+'SYSREM_FLUX3_BIN_ERR'+'\t'+\
                 'CENTDX_fda_BIN'+'\t'+'CENTDX_fda_BIN_ERR'+'\t'+'CENTDY_fda_BIN'+'\t'+'CENTDY_fda_BIN_ERR'
        np.savetxt(outfilename, X, delimiter='\t', header=header)                                  
        
        #::: full time series data, every exposure
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_data_ALL.txt' )
        X = np.c_[ self.dic['HJD'], self.dic['SYSREM_FLUX3'],
                   self.dic['CENTDX_fda'], self.dic['CENTDY_fda'] ]
        header = 'HJD_ALL'+'\t'+'SYSREM_FLUX3_ALL'+'\t'+ \
                 'CENTDX_fda_ALL'+'\t'+'CENTDY_fda_ALL'
        np.savetxt(outfilename, X, delimiter='\t', header=header)   
        
        
        
        
    ###########################################################################
    #::: save an info file for further external fitting
    ########################################################################### 
    def save_info(self):
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_info.txt' )
        ind_out_phase = np.where( (self.dic['HJD_PHASE'] < -0.15) | ( (self.dic['HJD_PHASE'] > 0.15) & (self.dic['HJD_PHASE'] > 0.35) ) | (self.dic['HJD_PHASE'] > 0.65) )              
        header = 'FIELDNAME' + '\t' +\
                 'OBJ_ID' + '\t' +\
                 'NGTS_VERSION' + '\t' +\
                 'RA' + '\t' +\
                 'DEC' + '\t' +\
                 'CCDX_0' + '\t' +\
                 'CCDY_0' + '\t' +\
                 'FLUX_MEAN' + '\t' +\
                 'CENTDX_fda_PHASE_RMSE' + '\t' +\
                 'CENTDY_fda_PHASE_RMSE' + '\t' +\
                 'CENTDX_fd_PHASE_RMSE' + '\t' +\
                 'CENTDY_fd_PHASE_RMSE' + '\t' +\
                 'CENTDX_f_PHASE_RMSE' + '\t' +\
                 'CENTDY_f_PHASE_RMSE' + '\t' +\
                 'CENTDX_PHASE_RMSE' + '\t' +\
                 'CENTDY_PHASE_RMSE' + '\t' +\
                 'RollCorrSNR_X' + '\t' +\
                 'RollCorrSNR_Y' + '\t' +\
                 'CrossCorrSNR_X' + '\t' +\
                 'CrossCorrSNR_Y' + '\t' +\
                 'Ttest_X' + '\t' +\
                 'Ttest_Y' + '\t' +\
                 'Binom_X' + '\t' +\
                 'Binom_Y'
                 
        with open(outfilename, 'w') as f:
            f.write(header)
            f.write( self.fieldname+'\t'+\
                     self.obj_id+'\t'+\
                     self.ngts_version+'\t'+\
                     str(self.dic['RA'])+'\t'+\
                     str(self.dic['DEC'])+'\t'+\
                     str(self.dic['CCDX_0'])+'\t'+\
                     str(self.dic['CCDY_0'])+'\t'+\
                     str(self.dic['FLUX_MEAN'])+'\t'+\
                     str(np.nanstd(self.dic['CENTDX_fda_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDY_fda_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDX_fd_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDY_fd_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDX_f_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDY_f_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDX_PHASE'][ind_out_phase]))+'\t'+\
                     str(np.nanstd(self.dic['CENTDY_PHASE'][ind_out_phase]))+'\t'+\
                     str(self.stats['RollCorrSNR_X'])+'\t'+\
                     str(self.stats['RollCorrSNR_Y'])+'\t'+\
                     str(self.stats['CrossCorrSNR_X'])+'\t'+\
                     str(self.stats['CrossCorrSNR_Y'])+'\t'+\
                     str(self.stats['Ttest_X'])+'\t'+\
                     str(self.stats['Ttest_Y'])+'\t'+\
                     str(self.stats['Binom_X'])+'\t'+\
                     str(self.stats['Binom_Y'])+'\n' )
                         
                
                
                
    ###########################################################################
    #::: save an info file for further external fitting
    ########################################################################### 
    def save_flagfile(self):
        if self.flagfile is not None:
            
            with open(self.flagfile, 'a') as f:
                f.write( self.fieldname+'\t'+\
                         self.obj_id+'\t'+\
                         self.ngts_version+'\t'+\
                         str(self.stats['RollCorrSNR_X'])+'\t'+\
                         str(self.stats['RollCorrSNR_Y'])+'\t'+\
                         str(self.stats['CrossCorrSNR_X'])+'\t'+\
                         str(self.stats['CrossCorrSNR_Y'])+'\t'+\
                         str(self.stats['Ttest_X'])+'\t'+\
                         str(self.stats['Ttest_Y'])+'\t'+\
                         str(self.stats['Binom_X'])+'\t'+\
                         str(self.stats['Binom_Y'])+'\n' )
 
#        else:
#            print "Note: flagfile is 'None'."
            

 
                    
            
#    def detrend(self, method=1):
#        if method==1:
#            detrender = detrend_centroid_external_RDX_1.detrend_centroid(self.dic, self.dic_nb)
#        if method==2:
#            detrender = detrend_centroid_external_RDX_2.detrend_centroid(self.dic, self.dic_nb)
#        if method==3:
#            detrender = detrend_centroid_external_RDX_3.detrend_centroid(self.dic, self.dic_nb)
#        if method==4:
#            detrender = detrend_centroid_external_RDX_4.detrend_centroid(self.dic, self.dic_nb)
#            
#        self.dic, self.dic_nb = detrender.run()        
    
    
    def detrend(self):
        detrender = detrend_centroid_external_RDX.detrend_centroid(self.dic, self.dic_nb, method=self.method, R_min=self.R_min, N_top_max=self.N_top_max, dt=self.dt)
        self.dic, self.dic_nb = detrender.run()   
        
        
        
    def check_object(self):
        self.valid = True
        
        for key in ['X','Y']:
            N_nan = np.count_nonzero(np.isnan(self.dic['CENTD'+key]))
            N_tot = len(self.dic['CENTD'+key])
            #if more than half of all entries are NaN, declare the object as invalid
            if (1.*N_nan/N_tot > 0.5): self.valid = False
            
            
            
    ###########################################################################
    #::: run (for individual targets)
    ###########################################################################    
    def run(self):
        
        #::: to load data
        self.load_object()
        print 'loaded object.'
        
        self.check_object()
        print 'object is valid.'
        
        if self.valid:
            #::: overwrite transit parameters with a simulated transit
    #        self.dic = simulate_signal.simulate( self.dic, tipo='EB_FGKM', plot=False )
            
            #::: load reference stars
            self.load_neighbours()
            print 'loaded neighbours.'
            
            #::: load crossmatching information
            self.load_catalog()
            print 'loaded catalog.'
            
            self.mark_eclipses()
            print 'marked eclipses and out-of-eclipse.'
            
            #::: assign colorcode corresponding to airmass
            self.assign_airmass_colorcode()
            print 'assigned airmass color codes.'
            
            #::: detrend the centroid
            self.detrend()
            print 'detrended externally.'
            
            #::: bin the dictionary (_BIN and _BIN_ERR)
            self.binning()
            print 'binned.'
            
            #::: plot neighbours for comparison
     #       start = timeit.default_timer()
        #    analyse_neighbours.plot( self.dic, self.dic_nb, self.outdir, self.fieldname, self.obj_id, self.ngts_version, dt=self.dt )
     #       print 'plotted neighbours.'
     #       print '(time needed for this):', timeit.default_timer() - start
    
            #TODO shift the plot parts of the following functions into seperate script
    
            #::: to study a target in detail
            self.phase_fold()
            
            #::: calculate and plot ccf and acf
            self.cross_correlate()
            
            #::: calculate stats (SNR, Ttest, Bimod-test)
            self.do_stats()
            
            #plots
   #         self.plot_scatter_matrix()
            
    #        self.plot_phase_folded_curves()
            
    ##        self.plot_rainplot_summary() #exclude
    ##        
    ##        self.plot_rainplot_per_night() #exclude
    ##        
    ##        self.plot_detrending_over_time() #exclude
    #        
    #        self.plot_detrending() #exclude
            
     #       self.plot_info_page()
            
     #       self.plot_stacked_image()
            
     #       self.save_pdf()
            
            self.save_data()
            
            self.save_info()
                      
            self.save_flagfile()
            
        else:
            warnings.warn('Object '+self.obj_id+' skipped: too many NaNs in CENTD array.')
            
            
        
###########################################################################
#::: MAIN
###########################################################################             
if __name__ == '__main__':
    
    start = timeit.default_timer()
    if len(sys.argv) == 1:
        #::: TEST candidate
        C = centroid('NG0304-1115', '019780', ngts_version='TEST18', user_period=69260.3241759, user_epoch=58102726.652, user_width=0.1*69260.3241759, dt=0.001)  
        C.run()
        
    elif len(sys.argv) == 3:
        print 'Input:', sys.argv[1], sys.argv[2]
        C = centroid(str(sys.argv[1]), str(sys.argv[2]), show_plot=True)  
        C.run()
        
    elif len(sys.argv) == 4:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3]
        C = centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]) )  
        C.run()
    elif len(sys.argv) == 5:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        C = centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), user_period=float(sys.argv[4]) )  
        C.run()
        
    elif len(sys.argv) == 6:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        C = centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), user_period=float(sys.argv[4]), user_epoch=float(sys.argv[5]) )
        C.run()
        
    elif len(sys.argv) == 7:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        C = centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), user_period=float(sys.argv[4]), user_epoch=float(sys.argv[5]), dt=float(sys.argv[6]) )
        C.run()        
        
    print 'execution time:', timeit.default_timer() - start
