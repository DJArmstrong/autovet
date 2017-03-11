# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:44:11 2016

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
from matplotlib.backends.backend_pdf import PdfPages
import lightcurve_tools
import stacked_images
import pandas as pd
#from astropy.stats import LombScargle
#from scipy.signal import lombscargle
#import lightcurve_tools, get_scatter_color
import os
from helper_functions import mystr



def plot(dic, dic_nb, outdir, fieldname, obj_id, ngts_version, dt=0.01, show_plot=False):
    figs = {}
    figs[0] = plot_phasecurve_1_siderial_day( dic )
    figs[1] = plot_detrending_steps_evaluation( dic, dic_nb, dt )
    figs[2] = plot_neighbours_phasecurve_and_location( fieldname, ngts_version, dic, dic_nb, dt )
    figs[3] = plot_hjd_curves( dic, dic_nb )
    save_pdf( figs, outdir, fieldname, obj_id, ngts_version, show_plot )
    
    
    
###########################################################################
#::: save all plots in one pdf per target object
###########################################################################   
def save_pdf( figs, outdir, fieldname, obj_id, ngts_version, show_plot ):
    outfilename = os.path.join( outdir, fieldname + '_' + obj_id + '_' + ngts_version + '_centroid_appendix.pdf' )    
    with PdfPages( outfilename ) as pdf:
        for ind in figs: 
            pdf.savefig( figs[ind] )
        print 'Plots saved as ' + outfilename
        
    if show_plot == False: plt.close('all')
   


def plot_detrending_steps_evaluation( dic, dic_nb, dt ):   
    fig, axes = plt.subplots(1,2,figsize=(16,16))
    offset = 0.02
    xtext = -0.25    
    ytext = 0.01
    
    #::: phasefolded curves
    centdx = dic['CENTDX']
    centdy = dic['CENTDY']
    plot_phasecurves(dic, dt, centdx, centdy, color='g', axes=axes, offset=0*offset )
    axes[0].text( xtext, ytext-0*offset, 'target (raw)')
    
    centdx = dic['CENTDX_f']
    centdy = dic['CENTDY_f']
    plot_phasecurves( dic, dt, centdx, centdy, color='orange',  axes=axes, offset=1*offset )
    axes[0].text( xtext, ytext-1*offset, 'target (flattened externally)')
        
    centdx = np.nanmean( dic_nb['CENTDX'], axis=0 )
    centdy = np.nanmean( dic_nb['CENTDY'], axis=0 )
    plot_phasecurves( dic, dt, centdx, centdy, color='b',  axes=axes, offset=2*offset )
    axes[0].text( xtext, ytext-2*offset, 'neighbours (mean of all)')

    centdx = dic['CENTDX_f'] - np.nanmean( dic_nb['CENTDX'], axis=0 )
    centdy = dic['CENTDY_f'] - np.nanmean( dic_nb['CENTDY'], axis=0 )
    plot_phasecurves( dic, dt, centdx, centdy, color='orange',  axes=axes, offset=3*offset )
    axes[0].text( xtext, ytext-3*offset, 'target - neighbours (mean of all)')
    
    centdx = dic_nb['CENTDX_ref_mean']
    centdy = dic_nb['CENTDY_ref_mean']
    plot_phasecurves( dic, dt, centdx, centdy, color='b',  axes=axes, offset=4*offset )
    axes[0].text( xtext, ytext-4*offset, 'reference stars (mean of best fit)')
    
    centdx = dic['CENTDX_fd']
    centdy = dic['CENTDY_fd']
    plot_phasecurves( dic, dt, centdx, centdy, color='orange',  axes=axes, offset=5*offset )
    axes[0].text( xtext, ytext-5*offset, 'target (flattened and detrended, best fit)')
    
    centdx = dic['CENTDX_fda']
    centdy = dic['CENTDY_fda']
    plot_phasecurves( dic, dt, centdx, centdy, color='orange',  axes=axes, offset=6*offset )
    axes[0].text( xtext, ytext-6*offset, 'target (flattened, detrended and 1 day siderial airmass correction)')
    
    axes[0].set( xlim=[-0.25,0.75], ylim=[-0.02-6*offset,0.02] )
    axes[1].set( xlim=[-0.25,0.75], ylim=[-0.02-6*offset,0.02] )
    
    plt.tight_layout()
    return fig
    


def plot_hjd_curves( dic, dic_nb ):
    #::: set y offsets
    offset = 0.1
    
    #::: set plotting range
    N_points = len(dic['HJD_BIN'])
#    N_points = 1000
    
    #::: set x-axis
    x = np.arange(N_points)
#    x = dic['HJD_BIN'][slice(None)]
    
#    print '***********************************'
#    print x
#    print dic_nb['CENTDX_ref_mean_BIN'][slice(None)]
#    print '***********************************'
#    print x.shape
#    print dic_nb['CENTDX_ref_mean_BIN'][slice(None)].shape
#    print '***********************************'
#    print dic['SYSREM_FLUX3'].shape
#    print dic_nb['CENTDX'].shape
#    print dic_nb['CENTDY'].shape
    
    
    #::: set scatter color
    c = dic['COLOR_BIN']
    cmap = 'jet'
    
    #::: plot
    fig, axes = plt.subplots(4,1, sharex=True, sharey=False, figsize=(100,16))    
    texts = ['raw','reference stars','flattened + detrended','siderial day airmass correction', 'result']    

    ax = axes[0]
    ax.scatter( x, dic['SYSREM_FLUX3_BIN'][slice(None)], c=c, rasterized=True, cmap=cmap )
    ax.set( ylabel='FLUX (BINNED)' )    
    
    ax = axes[1]
    ax.scatter( x, dic['CENTDX_f_BIN'][slice(None)], c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic_nb['CENTDX_ref_mean_BIN'][slice(None)] - offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic['CENTDX_fd_BIN'][slice(None)] - 2*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    for i, text in enumerate( texts ): 
        ax.text( x[0], -i*offset, text )
        ax.axhline( -i*offset, color='k' )
    ax.scatter( x, dic['poly_CENTDX_BIN'][slice(None)] - 3*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic['CENTDX_fda_BIN'][slice(None)] - 4*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.set( ylim=[-0.1-4*offset,0.1], ylabel='CENTDX (BINNED)' )
        
    ax = axes[2]
    ax.scatter( x, dic['CENTDY_f_BIN'][slice(None)], c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic_nb['CENTDY_ref_mean_BIN'][slice(None)] - offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic['CENTDY_fd_BIN'][slice(None)] - 2*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    for i, text in enumerate( texts ): 
        ax.text( x[0], -i*offset, text )
        ax.axhline( -i*offset, color='k' )
    ax.scatter( x, dic['poly_CENTDY_BIN'][slice(None)] - 3*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.scatter( x, dic['CENTDY_fda_BIN'][slice(None)] - 4*offset, c=c, rasterized=True, cmap=cmap, vmin=-1, vmax=1 )
    ax.set( ylim=[-0.1-4*offset,0.1], ylabel='CENTDY (BINNED)' )

    ax = axes[3]
    ax.scatter( x, dic['AIRMASS_BIN'][slice(None)], c=c, rasterized=True, cmap=cmap )
    ax.set( ylim=[1.,2.], ylabel='AIRMASS (BINNED)' )
    
#    ax = axes[4]
#    ax.scatter( x, dic['COLOR_BIN'][slice(None)], c=c, rasterized=True, cmap=cmap )
#    ax.set( ylim=[-1.,1.], ylabel='COLOR_BIN' )
    
    ax.set( xlim=[x[0],x[-1]] )
    
    plt.tight_layout()
    return fig
    
 
 
def plot_phasecurve_1_siderial_day( dic ):
    '''
    1 mean siderial day = 
    ( 23 + 56/60. + 4.0916/3600. ) / 24. = 0.9972695787 days
    see e.g. https://en.wikipedia.org/wiki/Sidereal_time
    
    Note: dic['poly_CENTDX'] is a function!
    '''

    #::: show airmass as proof of concept for the siderial day phase folding
#    fig, axes = plt.subplots(1,2,figsize=(16,8))
#    
#    axes[0].scatter( dic['HJD_PHASE_1sidday'], dic['COLOR_PHASE_1sidday'], c=dic['COLOR_PHASE_1sidday'], rasterized=True, cmap='jet', vmin=-1, vmax=1)
#    axes[0].set( ylim=[-1.,1.] )
#
#    axes[1].scatter( dic['HJD_PHASE_1sidday'], dic['AIRMASS_PHASE_1sidday'], c=dic['COLOR_PHASE_1sidday'], rasterized=True, cmap='jet', vmin=-1, vmax=1)
#    axes[1].set( ylim=[1.,2.] )


    #::: show FLUX and CENTDXY
    fig, axes = plt.subplots(3,1,figsize=(8,6), sharex=True)
#    axes[0].scatter( dic['PHI'][::10], dic['CENTDX_fd'][::10], c=dic['COLOR'][::10], rasterized=True, cmap='jet' )
    axes[0].scatter( dic['HJD_PHASE_1sidday'], dic['CENTDX_fd_PHASE_1sidday'], c=dic['COLOR_PHASE_1sidday'], rasterized=True, cmap='jet', vmin=-1, vmax=1 )
#    axes[0].errorbar( dic['HJD_PHASE_1sidday'], dic['CENTDX_fd_PHASE_1sidday'], yerr=dic['CENTDX_fd_PHASE_1sidday_ERR'], fmt='.', color='k' )
    
    axes[1].scatter( dic['HJD_PHASE_1sidday'], dic['CENTDY_fd_PHASE_1sidday'], c=dic['COLOR_PHASE_1sidday'], rasterized=True, cmap='jet', vmin=-1, vmax=1 )
#    axes[1].errorbar( dic['HJD_PHASE_1sidday'], dic['CENTDY_fd_PHASE_1sidday'], yerr=dic['CENTDY_fd_PHASE_1sidday_ERR'], fmt='.', color='k' )

    axes[2].scatter( dic['HJD_PHASE_1sidday'], dic['SYSREM_FLUX3_PHASE_1sidday'], c=dic['COLOR_PHASE_1sidday'], rasterized=True, cmap='jet', vmin=-1, vmax=1 )
#    axes[2].errorbar( dic['HJD_PHASE_1sidday'], dic['SYSREM_FLUX3_PHASE_1sidday'], yerr=dic['SYSREM_FLUX3_PHASE_1sidday_ERR'], fmt='.', color='k' )
    
    
    #::: show FLUX and CENTDXY trends / polyfits
    axes[0].plot( dic['HJD_PHASE_1sidday'], dic['polyfct_CENTDX'](dic['HJD_PHASE_1sidday']), 'r-' )
    axes[0].scatter( dic['HJD_PHASE_1sidday'], dic['CENTDX_fd_PHASE_1sidday'] - dic['polyfct_CENTDX'](dic['HJD_PHASE_1sidday']), c='r', rasterized=True )
        
    axes[1].plot( dic['HJD_PHASE_1sidday'], dic['polyfct_CENTDY'](dic['HJD_PHASE_1sidday']), 'r-' )
    axes[1].scatter( dic['HJD_PHASE_1sidday'], dic['CENTDY_fd_PHASE_1sidday'] - dic['polyfct_CENTDY'](dic['HJD_PHASE_1sidday']), c='r', rasterized=True )
    
    plt.tight_layout()
    return fig

    
    
def plot_neighbours_phasecurve_and_location( fieldname, ngts_version, dic, dic_nb, dt ):
        
    N_nb = len(dic_nb['OBJ_ID'])
    fig, axes = plt.subplots(N_nb, 5, figsize=(20,N_nb*4))

    for i in range(N_nb):
        centdx = dic_nb['CENTDX'][i,:]
        centdy = dic_nb['CENTDY'][i,:]
        plot_phasecurves_extended( fieldname, ngts_version, dic, dic_nb, i, dt, centdx, centdy, axes=axes[i,:])
        
    plt.tight_layout()
    return fig
        


def plot_phasecurves( dic, dt, centdx, centdy, title=None, color='b', axes=None, offset=None ):
    
    hjd_phase, centdx_c_phase, centdx_c_phase_err, _, _ = lightcurve_tools.phase_fold( dic['HJD'], centdx - np.nanmean(centdx), dic['PERIOD'], dic['EPOCH'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)
    hjd_phase, centdy_c_phase, centdy_c_phase_err, _, _ = lightcurve_tools.phase_fold( dic['HJD'], centdy - np.nanmean(centdy), dic['PERIOD'], dic['EPOCH'], dt = dt, ferr_type='meansig', ferr_style='sem', sigmaclip=True)

    if offset is not None:
        centdx_c_phase -= offset
        centdy_c_phase -= offset

    if axes is None: 
        fig, axes = plt.subplots(1,2, sharex=True, sharey=True, figsize=(12,4))

    axes[0].errorbar( hjd_phase, centdx_c_phase, yerr=centdx_c_phase_err, fmt='o', color=color, rasterized=True ) #, color='darkgrey')
    axes[0].set_ylabel('CENTDX (in pixel)')
#        axes[0].set_ylim([ np.min(centdx_c_phase - centdx_c_phase_err), np.max(centdx_c_phase + centdx_c_phase_err) ])
    
    axes[1].errorbar( hjd_phase, centdy_c_phase, yerr=centdy_c_phase_err, fmt='o', color=color, rasterized=True ) #, color='darkgrey')
    axes[1].set_ylabel('CENTDY (in pixel)')
#        axes[1].set_ylim([ np.min(centdy_c_phase - centdy_c_phase_err), np.max(centdy_c_phase + centdy_c_phase_err) ])

    axes[0].set( xlim=[-0.25,0.75], ylim=[-0.02,0.02] )
    axes[1].set( xlim=[-0.25,0.75], ylim=[-0.02,0.02] )
    
    if title is not None:
        plt.suptitle( title )
    
    

def plot_phasecurves_extended( fieldname, ngts_version, dic, dic_nb, i, dt, centdx, centdy, color='b', axes=None ):
    
    if axes is None:
        fig, axes = plt.subplots(1,5, figsize=(20,4))
    
    plot_phasecurves( dic, dt, centdx, centdy, color='b', axes=axes )
    
    axes[2].plot( dic['CCDX'][0], dic['CCDY'][0], 'bo', ms=12 )
    axes[2].plot( dic_nb['CCDX_0'], dic_nb['CCDY_0'], 'k.' )
    axes[2].plot( dic_nb['CCDX_0'][i], dic_nb['CCDY_0'][i], 'ro', ms=12 )
    axes[2].set( xlim=[ dic['CCDX'][0]-150, dic['CCDX'][0]+150 ], ylim=[ dic['CCDY'][0]-150, dic['CCDY'][0]+150 ] )
    
    stacked_images.plot(fieldname, ngts_version, dic_nb['CCDX_0'][i], dic_nb['CCDY_0'][i], r=15, ax=axes[3], show_apt=True, show_cbar=True)
    
    plot_neighbour_info_text(axes[4], dic, dic_nb, i)



###########################################################################
#::: plot info page
###########################################################################  
def plot_neighbour_info_text(ax, dic, dic_nb, i):
        
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.axis('off')
    ax.text(0,1.0,'OBJ_ID: '+dic_nb['OBJ_ID'][i])
    ax.text(0,0.9,'FLUX: '+mystr(dic_nb['FLUX_MEAN'][i],2))
    ax.text(0,0.8,'CCDX_0: '+mystr(dic_nb['CCDX_0'][i],2))
    ax.text(0,0.7,'CCDY_0: '+mystr(dic_nb['CCDY_0'][i],2))
    ax.text(0,0.6,'CCD distance: '+mystr(np.sqrt( (dic['CCDX'][0] - dic_nb['CCDX_0'][i])**2 + (dic['CCDY'][0] - dic_nb['CCDY_0'][i])**2 ),2))
    ax.text(0,0.5,'CCD_X distance: '+mystr(( dic['CCDX'][0] - dic_nb['CCDX_0'][i] ),2))
    ax.text(0,0.4,'CCD_Y distance: '+mystr(( dic['CCDY'][0] - dic_nb['CCDX_0'][i] ),2))
    ax.text(0,0.3,'B-V color: '+mystr(dic_nb['B-V'][i],2))
    ax.text(0,0.2,'B-V color difference: '+mystr(dic['B-V'] - dic_nb['B-V'][i],2))
    ax.text(0,0.1,'V Mag: '+mystr(dic_nb['Vmag'][i],2))
    ax.text(0,0.0,'Corr Coeff X / Y: '+mystr(dic_nb['corrcoeff_x'][i],2) + ' / ' + mystr(dic_nb['corrcoeff_x'][i],2))
        
    
