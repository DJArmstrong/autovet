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

try:
    from ngtsio import ngtsio
except ImportError:
    from scripts import ngtsio_v1_1_1_autovet as ngtsio
    warnings.warn( "Package 'ngtsio' not installed. Use version ngtsio v1.1.1 from 'scripts/' instead.", ImportWarning )
       

from scripts import index_transits, lightcurve_tools, \
                    stacked_images, analyse_neighbours, detrend_centroid_external, \
                    helper_functions, get_scatter_color, \
                    pandas_tsa
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
    
    def __init__(self, fieldname, obj_id, ngts_version='TEST18', source = 'CANVAS', bls_rank = 1, period = None, epoch = None, width = None, time_hjd = None, pixel_radius = 150., flux_min = 1000., flux_max = 10000., bin_width=300., min_time=1800., dt=0.005, roots=None, outdir=None, parent=None, show_plot=False, flagfile=None, dic=None):
#        super(centroid, self).__init__(parent)
        
        self.fieldname = fieldname
        self.obj_id = obj_id
        self.ngts_version = ngts_version
        self.source = source
        self.bls_rank = bls_rank
        self.user_period = period
        self.user_epoch = epoch
        self.user_width = width
        self.time_hjd = time_hjd
        self.pixel_radius = pixel_radius
        self.flux_min = flux_min
        self.flux_max = flux_max
        self.bin_width = bin_width #(in s)
        self.min_time = min_time #(in s) minimum coverage of innermost and out-of-transit required for including a night
        self.dt = dt
        self.roots = roots
        self.outdir = outdir
        self.parent = parent
        self.show_plot = show_plot
        self.flagfile = flagfile
        self.dic = dic
        
        self.crosscorr = {}
        
        if outdir is None: 
            self.outdir = os.path.join( 'output', self.ngts_version, self.fieldname, '' )
        else:
            self.outdir = outdir
        if not os.path.exists(self.outdir): 
            os.makedirs(self.outdir)
          
        try:
            self.catalogfname = glob.glob( 'input/catalog/'+self.fieldname+'*'+self.ngts_version+'_cat_master.dat')[0]
        except:
            self.catalogfname = None
            
        self.run()
        
        
        
        
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
        
            #::: get infos of the transit signal from CANVAS or BLS (if not given)
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
        
        #::: overwrite period and epoch if given by user
        if self.user_period is not None:
            self.dic['PERIOD'] = self.user_period #in s
        if self.user_epoch is not None:
            self.dic['EPOCH'] = self.user_epoch #in s
        if self.user_width is not None:
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
        self.dic_nb = ngtsio.get( self.fieldname, ['OBJ_ID','CCDX','CCDY','CENTDX','CENTDY','FLUX_MEAN'], obj_id = obj_id_nb, time_hjd = self.time_hjd, ngts_version = self.ngts_version, bls_rank = self.bls_rank, silent = True) 
        self.dic_nb['CCDX_0'] = np.nanmedian( self.dic_nb['CCDX'], axis=1 )
#        self.dic_all['CCDX'][ind_neighbour]
        self.dic_nb['CCDY_0'] = np.nanmedian( self.dic_nb['CCDY'], axis=1 )
#        self.dic_all['CCDY'][ind_neighbour]
        del self.dic_nb['CCDX']
        del self.dic_nb['CCDY']
        
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



    def assign_airmass_colorcode(self):
        #::: assign colors for different nights; dconvert HJD from seconds into days
        self.dic = get_scatter_color.get_scatter_color(self.dic)
        self.dic['COLOR'] = np.concatenate( self.dic['COLOR_PER_NIGHT'], axis=0 )

        

    def binning(self):
        self.dic['HJD_BIN'], \
            [ self.dic['CENTDX_fda_BIN'], self.dic['CENTDY_fda_BIN'], self.dic['poly_CENTDX_BIN'], self.dic['poly_CENTDY_BIN'], self.dic['COLOR_BIN'], self.dic['AIRMASS_BIN'], self.dic['SYSREM_FLUX3_BIN'], self.dic['CENTDX_f_BIN'], self.dic['CENTDY_f_BIN'], self.dic['CENTDX_fd_BIN'], self.dic['CENTDY_fd_BIN'], self.dic_nb['CENTDX_ref_mean_BIN'], self.dic_nb['CENTDY_ref_mean_BIN'] ], \
            [ self.dic['CENTDX_fda_BINERR'], self.dic['CENTDY_fda_BINERR'], self.dic['poly_CENTDX_BINERR'], self.dic['poly_CENTDY_BINERR'], self.dic['COLOR_BINERR'], self.dic['AIRMASS_BINERR'], self.dic['SYSREM_FLUX3_BINERR'], self.dic['CENTDX_f_BINERR'], self.dic['CENTDY_f_BINERR'], self.dic['CENTDX_fd_BINERR'], self.dic['CENTDY_fd_BINERR'], self.dic_nb['CENTDX_ref_mean_BINERR'], self.dic_nb['CENTDY_ref_mean_BINERR'] ], \
            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic['CENTDX_fda'], self.dic['CENTDY_fda'], self.dic['poly_CENTDX'], self.dic['poly_CENTDY'], self.dic['COLOR'], self.dic['AIRMASS'], self.dic['SYSREM_FLUX3'], self.dic['CENTDX_f'], self.dic['CENTDY_f'], self.dic['CENTDX_fd'], self.dic['CENTDY_fd'], self.dic_nb['CENTDX_ref_mean'], self.dic_nb['CENTDY_ref_mean'] )), dt=600, sigmaclip=False, ferr_style='std' )

#        self.dic['HJD_BIN'], \
#            [ self.dic['CENTDX_fda_BIN'], self.dic['CENTDY_fda_BIN'], self.dic['poly_CENTDX_BIN'], self.dic['poly_CENTDY_BIN'], self.dic['COLOR_BIN'], self.dic['AIRMASS_BIN'], self.dic['SYSREM_FLUX3_BIN'], self.dic['CENTDX_f_BIN'], self.dic['CENTDY_f_BIN'], self.dic['CENTDX_fd_BIN'], self.dic['CENTDY_fd_BIN'] ], \
#            [ self.dic['CENTDX_fda_BINERR'], self.dic['CENTDY_fda_BINERR'], self.dic['poly_CENTDX_BINERR'], self.dic['poly_CENTDY_BINERR'], self.dic['COLOR_BINERR'], self.dic['AIRMASS_BINERR'], self.dic['SYSREM_FLUX3_BINERR'], self.dic['CENTDX_f_BINERR'], self.dic['CENTDY_f_BINERR'], self.dic['CENTDX_fd_BINERR'], self.dic['CENTDY_fd_BINERR'] ], \
#            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic['CENTDX_fda'], self.dic['CENTDY_fda'], self.dic['poly_CENTDX'], self.dic['poly_CENTDY'], self.dic['COLOR'], self.dic['AIRMASS'], self.dic['SYSREM_FLUX3'], self.dic['CENTDX_f'], self.dic['CENTDY_f'], self.dic['CENTDX_fd'], self.dic['CENTDY_fd'] )), dt=600, sigmaclip=False, ferr_style='std' )
#        
#        _, \
#            [ self.dic_nb['CENTDX_ref_mean_BIN'], self.dic_nb['CENTDY_ref_mean_BIN'] ], \
#            [ self.dic_nb['CENTDX_ref_mean_BINERR'], self.dic_nb['CENTDY_ref_mean_BINERR'] ], \
#            _ = lightcurve_tools.rebin_err_matrix(self.dic['HJD'], np.vstack(( self.dic_nb['CENTDX_ref_mean'], self.dic_nb['CENTDY_ref_mean'] )), dt=600 )



    ###########################################################################
    #::: 
    ########################################################################### 
    def phase_fold(self):
        self.N_phasepoints = int( 1./self.dt )
        
        ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid = index_transits.index_transits(self.dic)
        
        self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], self.dic['SYSREM_FLUX3_PHASE_ERR'], self.dic['N_PHASE'], self.dic['PHI'] = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['SYSREM_FLUX3'] / np.nanmedian(self.dic['SYSREM_FLUX3'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_PHASE'], self.dic['CENTDX_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX'] - np.nanmedian(self.dic['CENTDX'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_PHASE'], self.dic['CENTDY_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY'] - np.nanmedian(self.dic['CENTDY'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_f_PHASE'], self.dic['CENTDX_f_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_f'] - np.nanmedian(self.dic['CENTDX_f'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_f_PHASE'], self.dic['CENTDY_f_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_f'] - np.nanmedian(self.dic['CENTDY_f'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_fd_PHASE'], self.dic['CENTDX_fd_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_fd'] - np.nanmedian(self.dic['CENTDX_fd'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_fd_PHASE'], self.dic['CENTDY_fd_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_fd'] - np.nanmedian(self.dic['CENTDY_fd'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDX_fda_PHASE'], self.dic['CENTDX_fda_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDX_fda'] - np.nanmedian(self.dic['CENTDX_fda'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
        _, self.dic['CENTDY_fda_PHASE'], self.dic['CENTDY_fda_PHASE_ERR'], _, _ = lightcurve_tools.phase_fold(self.dic['HJD'], self.dic['CENTDY_fda'] - np.nanmedian(self.dic['CENTDY_fda'][ind_out]), self.dic['PERIOD'], self.dic['EPOCH'], dt = self.dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)

        
        
    ###########################################################################
    #::: 
    ########################################################################### 
    def cross_correlate(self):
        #::: create pandas df from parts of self.dic
        self.phasedf = pd.DataFrame( {k: self.dic[k] for k in ('HJD_PHASE', 'SYSREM_FLUX3_PHASE', 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE')}, columns=['HJD_PHASE', 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'SYSREM_FLUX3_PHASE'] )
   
        self.fig_corrfx, flags_fx = self.ccfct( 'SYSREM_FLUX3_PHASE', 'CENTDX_fda_PHASE', 'FLUX vs CENTDX' )
        self.fig_corrfy, flags_fy = self.ccfct( 'SYSREM_FLUX3_PHASE', 'CENTDY_fda_PHASE', 'FLUX vs CENTDY' )
        self.fig_corrxy, flags_xy = self.ccfct( 'CENTDX_fda_PHASE', 'CENTDY_fda_PHASE', 'CENTDX vs CENTDY' )

        self.fig_autocorr = self.acfct( ['SYSREM_FLUX3_PHASE','CENTDX_fda_PHASE','CENTDY_fda_PHASE'], ['FLUX','CENTDX','CENTDY'] )            
             
 
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
        
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        fig.suptitle(self.obj_id + ' ' + title)
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
               
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#        win = [0.1, 0.25, 0.33]
        win = [0.25]
        windows= ( np.array(win) * (1/self.dt) ).astype(int)
        correls = [ self.phasedf.rolling(window=windows[i], center=True).corr() for i,_ in enumerate(windows) ]
        
#        color = ['b','g','r']
        for i,_ in enumerate(windows): 
            self.dic['RollCorr_'+xkey+'_'+ykey] = correls[i].loc[ :, xkey, ykey ]
            axes[0].plot( self.phasedf['HJD_PHASE'], correls[i].loc[ :, xkey, ykey ], label=str(windows[i] * self.dt) )
            axes[0].axhline( 2.58/np.sqrt(windows[i]), color='k', linestyle='--')#, color=color[i] )
            axes[0].axhline( - 2.58/np.sqrt(windows[i]), color='k', linestyle='--')#, color=color[i] )
        axes[0].set( xlim=[-0.25,0.75], ylim=[-1,1], xlabel='phase', ylabel='rolling correlation')
        axes[0].legend()
                
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
        axes[1].plot( self.ccf_lags, self.crosscorr[title] )
        axes[1].plot( self.ccf_lags, crosscorr_CI99, 'k--' )
        axes[1].plot( self.ccf_lags, -crosscorr_CI99, 'k--' )
        axes[1].set( xlim=[self.ccf_lags[0], self.ccf_lags[-1]], ylim=[-1,1], xlabel=r'lag $\tau$ (phase shift)', ylabel='ccf (periodic)' )
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                  
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        plt.tight_layout() 
        return fig, None
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

            
            
                
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
        if 'DEPTH' in self.dic: ax.text(0,0.1,'Depth (mmag): '+mystr(np.abs(self.dic['DEPTH'])*1000.,2))
        if 'NUM_TRANSITS' in self.dic: ax.text(0,0.0,'Num Transits: '+mystr(self.dic['NUM_TRANSITS'],0))
        


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
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_data.txt' )
        X = np.c_[ self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], self.dic['SYSREM_FLUX3_PHASE_ERR'], 
                   self.dic['CENTDX_fda_PHASE'], self.dic['CENTDX_fda_PHASE_ERR'], self.dic['CENTDY_fda_PHASE'], self.dic['CENTDY_fda_PHASE_ERR'], 
                   self.dic['CENTDX_f_PHASE'], self.dic['CENTDX_f_PHASE_ERR'], self.dic['CENTDY_f_PHASE'], self.dic['CENTDY_f_PHASE_ERR'], 
                   self.dic['CENTDX_fd_PHASE'], self.dic['CENTDX_fd_PHASE_ERR'], self.dic['CENTDY_fd_PHASE'], self.dic['CENTDY_fd_PHASE_ERR'], 
                   self.dic['CENTDX_PHASE'], self.dic['CENTDX_PHASE_ERR'], self.dic['CENTDY_PHASE'], self.dic['CENTDY_PHASE_ERR'],
                   self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'], self.dic['RollCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'], self.dic['RollCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'],
                   self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'], self.dic['CrossCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'], self.dic['CrossCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'] ]
        header = 'HJD_PHASE'+'\t'+'SYSREM_FLUX3_PHASE'+'\t'+'SYSREM_FLUX3_PHASE_ERR'+'\t'+\
                 'CENTDX_fda_PHASE'+'\t'+'CENTDX_fda_PHASE_ERR'+'\t'+'CENTDY_fda_PHASE'+'\t'+'CENTDY_fda_PHASE_ERR'+'\t'+\
                 'CENTDX_f_PHASE'+'\t'+'CENTDX_f_PHASE_ERR'+'\t'+'CENTDY_f_PHASE'+'\t'+'CENTDY_f_PHASE_ERR'+'\t'+\
                 'CENTDX_fd_PHASE'+'\t'+'CENTDX_fd_PHASE_ERR'+'\t'+'CENTDY_fd_PHASE'+'\t'+'CENTDY_fd_PHASE_ERR'+'\t'+\
                 'CENTDX_PHASE'+'\t'+'CENTDX_PHASE_ERR'+'\t'+'CENTDY_PHASE'+'\t'+'CENTDY_PHASE_ERR'+'\t'+\
                 'RollCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'+'\t'+'RollCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'+'\t'+'RollCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'+'\t'+\
                 'CrossCorr_SYSREM_FLUX3_PHASE_CENTDX_fda_PHASE'+'\t'+'CrossCorr_SYSREM_FLUX3_PHASE_CENTDY_fda_PHASE'+'\t'+'CrossCorr_CENTDX_fda_PHASE_CENTDY_fda_PHASE'
                 
        np.savetxt(outfilename, X, delimiter='\t', header=header)
        
        
        
    ###########################################################################
    #::: save an info file for further external fitting
    ########################################################################### 
    def save_info(self):
        outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_' + self.ngts_version + '_centroid_info.txt' )
        ind_out = np.where( (self.dic['HJD_PHASE'] < -0.15) | (self.dic['HJD_PHASE'] > 0.15) )      
        X = np.c_[ self.dic['RA'], self.dic['DEC'], self.dic['CCDX_0'], self.dic['CCDY_0'], self.dic['FLUX_MEAN'], np.nanstd(self.dic['CENTDX_fda_PHASE'][ind_out]), np.nanstd(self.dic['CENTDY_fda_PHASE'][ind_out]), np.nanstd(self.dic['CENTDX_PHASE'][ind_out]), np.nanstd(self.dic['CENTDY_PHASE'][ind_out]) ]
        np.savetxt(outfilename, X, delimiter='\t', header='RA\tDEC\tCCDX_0\tCCDY_0\tFLUX_MEAN\tCENTDX_fda_PHASE_RMSE\tCENTDY_fda_PHASE_RMSE\tCENTDX_PHASE_RMSE\tCENTDY_PHASE_RMSE')
                
                
                
    ###########################################################################
    #::: save an info file for further external fitting
    ########################################################################### 
    def save_flagfile(self):
        if self.flagfile is not None:
            ccf_signal = {}
            ccf_noise = {}
            ind_peak = np.argmin( np.abs(self.ccf_lags) ) #where is lag 0 on the x axis of the CCF
            ind_offpeak = np.where( (self.ccf_lags < -0.1) | ((self.ccf_lags > 0.1) & (self.ccf_lags > 0.4)) | (self.ccf_lags > 0.6) )[0] #savely outside of lag 0 and lag 0.5 (i.e. primary and secondary eclipse)       
            ccf_signal['FLUX vs CENTDX'] = np.nanmean( self.crosscorr['FLUX vs CENTDX'][ [ind_peak-1, ind_peak, ind_peak+1] ] ) #signal = mean of the three CCF values around lag 0
            ccf_signal['FLUX vs CENTDY'] = np.nanmean( self.crosscorr['FLUX vs CENTDY'][ [ind_peak-1, ind_peak, ind_peak+1] ] ) #signal = mean of the three CCF values around lag 0
            ccf_noise['FLUX vs CENTDX'] = np.nanstd( self.crosscorr['FLUX vs CENTDX'][ind_offpeak] ) #noise = mean of all other CCF values excluding the peak around lag 0 and lag 0.5
            ccf_noise['FLUX vs CENTDY'] = np.nanstd( self.crosscorr['FLUX vs CENTDY'][ind_offpeak] ) #noise = mean of all other CCF values excluding the peak around lag 0 and lag 0.5
            
#            ccf_SNR = ccf_signal / ccf_noise
            
            with open(self.flagfile, 'a') as f:
                f.write( self.fieldname+'\t'+self.obj_id+'\t'+str(ccf_signal['FLUX vs CENTDX']/ccf_noise['FLUX vs CENTDX'])+'\t'+str(ccf_signal['FLUX vs CENTDY']/ccf_noise['FLUX vs CENTDY']) + '\n' )
        else:
            print "Warning: flagfile is 'None'."
            
        
    ###########################################################################
    #::: run (for individual targets)
    ###########################################################################    
    def run(self):
        
        #::: to load data
        self.load_object()
        
        #::: overwrite transit parameters with a simulated transit
#        self.dic = simulate_signal.simulate( self.dic, tipo='EB_FGKM', plot=False )
        
        #::: load reference stars
        self.load_neighbours()
        
        #::: load crossmatching information
        self.load_catalog()
        
        #::: assign colorcode corresponding to airmass
        self.assign_airmass_colorcode()
        
        #::: detrend the centroid
        self.dic, self.dic_nb = detrend_centroid_external.run( self.dic, self.dic_nb )
        
        #::: bin the dictionary (_BIN and _BINERR)
        self.binning()
        
        #::: plot neighbours for comparison
        analyse_neighbours.plot( self.dic, self.dic_nb, self.outdir, self.fieldname, self.obj_id, self.ngts_version, dt=self.dt )
        

        #TODO shift the plot parts of the following functions into seperate script

        #::: to study a target in detail
        self.phase_fold()
        
        #::: calculate and plot ccf and acf
        self.cross_correlate()
        
        #plots
        self.plot_scatter_matrix()
        
        self.plot_phase_folded_curves()
        
##        self.plot_rainplot_summary() #exclude
##        
##        self.plot_rainplot_per_night() #exclude
##        
##        self.plot_detrending_over_time() #exclude
#        
#        self.plot_detrending() #exclude
        
        self.plot_info_page()
        
        self.plot_stacked_image()
        
        self.save_pdf()
        
        self.save_data()
        
        self.save_info()
                  
        self.save_flagfile()
        
            
            
        
###########################################################################
#::: MAIN
###########################################################################             
if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        #::: TEST candidates
        centroid('NG0304-1115', '009861', ngts_version='TEST18', dt=0.005, pixel_radius = 150., flux_min = 500., flux_max = 10000., show_plot=True)  
#        centroid('NG0409-1941', '020057', ngts_version='TEST18', period=138847.72673086426, epoch=59376222.0, time_hjd=np.arange(680,760+1,dtype=int), dt=0.005, pixel_radius = 150., flux_min = 500., flux_max = 10000., show_plot=True)  
#        centroid('NG0409-1941', '020057', bls_rank=1, time_hjd=np.arange(680,760+1,dtype=int), dt=0.005, pixel_radius = 150., flux_min = 500., flux_max = 10000., show_plot=True)  

    elif len(sys.argv) == 3:
        print 'Input:', sys.argv[1], sys.argv[2]
        centroid(str(sys.argv[1]), str(sys.argv[2]), dt=0.005, pixel_radius = 150., flux_min = 500., flux_max = 10000., show_plot=True)  

    elif len(sys.argv) == 4:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3]
        centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), show_plot=True)  

    elif len(sys.argv) == 5:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), period=float(sys.argv[4]), show_plot=True)  

    elif len(sys.argv) == 6:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), period=float(sys.argv[4]), epoch=float(sys.argv[5]), show_plot=True)
        
    elif len(sys.argv) == 7:
        print 'Input:', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        centroid(str(sys.argv[1]), str(sys.argv[2]), ngts_version=str(sys.argv[3]), period=float(sys.argv[4]), epoch=float(sys.argv[5]), dt=float(sys.argv[6]), show_plot=True)