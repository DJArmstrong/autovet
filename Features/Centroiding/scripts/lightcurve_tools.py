# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:45:15 2016

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
import sys, socket
from astropy.stats import sigma_clip

import ngtsio_v1_2_0_centroiding as ngtsio
from My_Utils import mystr, medsig
import index_transits, deg2HMS
from binning import binning1D_per_night
from set_nan import set_nan  
import timeit


def rebin_err(t, f, ferr=None, dt = 0.02, phasefolded=False, ferr_type='medsig', ferr_style='std', sigmaclip=False):
    """
    @written by Ed Gillen, extended by Maximilian N. Guenther
    The standard rebin function but also dealing with errors
    on the individual data points being binned.
    ferr_type:
        'medsig'
        'meanstd'
    ferr_style:
        'std'
        'sem' = std / sqrt(N)
    """
    #::: sigma clip
    if sigmaclip is True:
        try:
            f = sigma_clip(f, sigma=5, iters=3)
        except:
            pass
    
    #::: make masked values to NaNs if applicable
    try:
        f[ f.mask ] = np.nan
    except:
        pass
    
    #::: bin
    #::: detect if it's phase-folded data or not
    if phasefolded is False:
        treg = np.r_[t.min():t.max():dt]
    else:
        treg = np.r_[-0.25:0.75:dt]
    nreg = len(treg)
    freg = np.zeros(nreg) + np.nan
    freg_err = np.zeros(nreg) + np.nan
    N = np.zeros(nreg)
    for i in np.arange(nreg):
        l = (t >= treg[i]) * (t < treg[i] + dt)
        if l.any():
            treg[i] = np.nanmean(t[l])
            N[i] = len(t[l])
            if ferr==None:
                if ferr_type == 'medsig':
                    freg[i], freg_err[i] = medsig(f[l])
                else:
                    try:
                        freg[i] = np.nanmean(f[l])
                        freg_err[i] = np.nanstd(f[l])
                    except: #e.g. in case of an empty or completely masked array
                        freg[i] = np.nan
                        freg_err[i] = np.nan
                    
                if ferr_style == 'sem':
                    freg_err[i] /= np.sqrt( len(f[l]) )
            else:
                freg[i], freg_err[i] = weighted_avg_and_std( f[l], np.ma.array([1/float(x) for x in ferr[l]]) )

    if phasefolded is False:
        k = np.isfinite(freg) #only return finite bins
    else:
        k = slice(None) #return the entire phase, filled with NaN replacements
    
    return treg[k], freg[k], freg_err[k], N[k]
    
    
    
def rebin_err_matrix(t, fmatrix, fmatrixerr=None, dt = 0.02, phasefolded=False, ferr_type='meanstd', ferr_style='sem', sigmaclip=True):
    '''
    f is a matrix, each row contains a 1D array (e.g. Flux, CENTDX, CENTDY in one array)
    '''
    """
    @written by Ed Gillen, extended by Maximilian N. Guenther
    The standard rebin function but also dealing with errors
    on the individual data points being binned.
    ferr_type:
        'medsig'
        'meanstd'
    ferr_style:
        'std'
        'sem' = std / sqrt(N)
    """
    N_items = fmatrix.shape[0]
    
    #::: sigma clip
    if sigmaclip is True:
        for j in range(N_items):
            try:
                f = sigma_clip( fmatrix[ j, : ], sigma=5, iters=3 )
                f [ f.mask ] = np.nan
                fmatrix[ j, : ] = f
            except:
                pass
    
    #::: bin
    #::: detect if it's phase-folded data or not
    if phasefolded is False:
        treg = np.r_[t.min():t.max():dt]
    else:
        treg = np.r_[-0.25:0.75:dt]
    nreg = len(treg)
    fmatrixreg = np.zeros( (N_items, nreg) ) + np.nan
    fmatrixreg_err = np.zeros( (N_items, nreg) ) + np.nan
    N = np.zeros(nreg)
    for i in np.arange(nreg):
        l = (t >= treg[i]) * (t < treg[i] + dt)
#        print treg[i]
#        print treg[i] + dt
#        print l
        if l.any():
            treg[i] = np.nanmean(t[l])
            N[i] = len(t[l])
            if fmatrixerr==None:
                if ferr_type == 'medsig':
                    #TODO
                    fmatrixreg[:,i] = np.nanmedian(fmatrix[:,l], axis=1)
                    fmatrixreg_err[:,i] = np.median( np.abs(fmatrix[:,l] - np.median(fmatrix[:,l], axis=1)), axis=1 )
#                    error('not implemented yet')
#                    freg[i], freg_err[i] = medsig(f[l])
                else:
#                    print '-----'
#                    print i
#                    print fmatrix[0,l]
#                    print np.nanmean(fmatrix[:,l], axis=1)[0]
#                    if i>50: print err
                    fmatrixreg[:,i] = np.nanmean(fmatrix[:,l], axis=1)
                    fmatrixreg_err[:,i] = np.nanstd(fmatrix[:,l], axis=1)
                    
                if ferr_style == 'sem':
                    fmatrixreg_err[:,i] /= np.sqrt( N[i] )
            else:
                #TODO
                error('not implemented yet')
#                freg[i], freg_err[i] = weighted_avg_and_std( f[l], np.ma.array([1/float(x) for x in ferr[l]]) )            
    
    if phasefolded is False:
        k = np.isfinite(fmatrixreg[0]) #only return finite bins
    else:
        k = slice(None) #return the entire phase, filled with NaN replacements
        
    return treg[k], fmatrixreg[:,k], fmatrixreg_err[:,k], N[k]
    
    

    
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
    
    
    
def phase_fold(time, flux, P, Tprim, dt = 0.02, ferr_type='medsig', ferr_style='std', sigmaclip=False):
    phi = calc_phase( time, P, Tprim )
    phi[ phi>0.75 ] -= 1.
    phase, phaseflux, phaseflux_err, N = rebin_err( phi, flux, None, dt=dt, phasefolded=True, ferr_type=ferr_type, ferr_style=ferr_style, sigmaclip=sigmaclip )    
    return phase, phaseflux, phaseflux_err, N, phi
    


def phase_fold_matrix(time, flux_matrix, P, Tprim, dt = 0.02, ferr_type='medsig', ferr_style='std', sigmaclip=False):
    phi = calc_phase( time, P, Tprim )
    phi[ phi>0.75 ] -= 1.
    phase, phasefluxmatrix, phasefluxmatrix_err, N = rebin_err_matrix( phi, flux_matrix, None, dt=dt, phasefolded=True, ferr_type=ferr_type, ferr_style=ferr_style, sigmaclip=sigmaclip )
    return phase, phasefluxmatrix, phasefluxmatrix_err, N, phi
    
    
    
def plot_phase_folded_lightcurve(ax, time, P, Tprim, flux, ferr=None, ferr_type='meansig', ferr_style='std', normalize=True, title='', period_factor=1.):

#    f = sigma_clip(f, sigma=5, iters=5)
#    f [ f.mask ] = np.nan
    
    if normalize: flux /= np.nanmedian(flux)
        
    P *= period_factor

#TODO: BINNING_1D_PER_NIGHT CRASHES THE CODE - WHY IS IT INTORDUCING NAN VALUES???    
#    t10min, f10min, f10min_err = rebin_err( time, f, ferr, dt = 600, ferr_type=ferr_type, ferr_style=ferr_style ) #bin to 10 minutes = 600 sec 
#    t10min, f10min, f10min_err = binning1D_per_night( time, f, bin_width = 50, timegap=3600, setting='median', normalize=True ) #bin to 10 minutes = 600 sec
#    print np.where(np.isnan(time))
#    print np.where(np.isnan(f))
#    print t10min, f10min, f10min_err     
#    print np.where(np.isnan(t10min))
#    print np.where(np.isnan(f10min))
    
#    print f10min_err
#    f10min_err[f10min_err == 0] = np.nan
    
#    t10min = time
#    f10min = f 
#    f10min_err= ferr
    
    
#    phi10min = phase( t10min, P, Tprim )
#    phi10min[ phi10min>0.75 ] -= 1.
#    print phi10min

#    treg, freg, freg_err = binning1D_per_night( phi10min, f10min, bin_width = 1, timegap=0, setting='median', normalize=True ) #bin to 10 minutes = 600 sec            
#    treg, freg, freg_err = rebin_err( phi10min, f10min, None, dt = 0.02, ferr_type=ferr_type, ferr_style=ferr_style )
#    print treg, freg, freg_err

#    ax.plot( phi10min, f10min, '.', c='lightgrey', ms=4, lw=0, rasterized=True, zorder = -1 )
#    ax.scatter( phi10min, f10min, c='lightgrey', s=10, lw=0, rasterized=True )

    phase, phaseflux, phaseflux_err, N, phi = phase_fold(time, flux, P, Tprim, ferr_type='medsig', ferr_style='std')

    def set_ax(ax):
        ax.plot( phi, flux, '.', c='lightgrey', ms=4, lw=0, rasterized=True, zorder = -1 )
        ax.errorbar( phase, phaseflux, yerr=phaseflux_err, color='r', fmt='o', rasterized=True )
        ax.set_title( title )
        ax.set_ylabel( 'Flux' )
        ax.set_xlabel( 'Phase' )
        ax.set_xlim([-0.25,0.75])
        ax.set_ylim([ np.nanmin(phaseflux-2*phaseflux_err), np.nanmax(phaseflux+2*phaseflux_err) ])
        ax.axvline(0,color='k')
        ax.axvline(0.5,color='k')
    
    if isinstance(ax, list):
        set_ax(ax[0])
        set_ax(ax[1])
        ax[1].set_xlim([-0.2,0.2])
    else:
        set_ax(ax)

    
    
    
def plot_phase_folded_lightcurve_dic(ax, dic, obj_id=None, ferr_type='medsig', ferr_style='std', period_factor=1.):
    #::: multiple objects in dic
    if not isinstance( dic['OBJ_ID'], basestring ): 
        ind = np.where( dic['OBJ_ID'] == obj_id )[0]
        if 'SYSREM_FLUX3_ERR' in dic: ferr =  dic['SYSREM_FLUX3_ERR'][ind]
        else: ferr = None        
        plot_phase_folded_lightcurve( ax, dic['HJD'][ind], dic['PERIOD'][ind], dic['EPOCH'][ind], dic['SYSREM_FLUX3'][ind], ferr, dic['FIELDNAME']+', '+dic['OBJ_ID'][ind], ferr_type=ferr_type, ferr_style=ferr_style )
    
    #::: single object in dic 
    else: 
        if 'SYSREM_FLUX3_ERR' in dic: ferr =  dic['SYSREM_FLUX3_ERR']
        else: ferr = None     
        ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid = index_transits.index_transits(dic) #ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid                 
        dic['SYSREM_FLUX3'] /= np.nanmedian( dic['SYSREM_FLUX3'][ind_out] )
        plot_phase_folded_lightcurve( ax, dic['HJD'], dic['PERIOD'], dic['EPOCH'], dic['SYSREM_FLUX3'], ferr = ferr, normalize = False, title = dic['FIELDNAME']+', '+dic['OBJ_ID'], ferr_type=ferr_type, ferr_style=ferr_style, period_factor=period_factor )



def plot_binned_lightcurve(ax, time, f, ferr=np.nan, bin_time=0, show_transit_regions=False, period=None, epoch=None, width=None, obj_id=None, normalize=True, title='', exposure=12., debug=False):

    if bin_time!=0:
        start = timeit.default_timer()        
        #use either my binning method (bin_time in s)
        bin_width = 1. * bin_time / exposure
        
        #TODO: IF TIME IS IN HOURS, TIMEGAP HAS TO BE ADJUSTED!!!!
        time, f, ferr = binning1D_per_night(time, f, bin_width, timegap=TODO*3600, setting='median', normalize=normalize)
        
        #or Ed's binning method (SLOW!) (bin_time required in units of hjd, i.e. here in days)
#        time, f, ferr  = rebin_err(time, f, ferr=None, dt = bin_time/(24. * 3600.), ferr_type='medsig', ferr_style='std')
        stop = timeit.default_timer()
        if debug: print 'Binning succesfully created in', stop-start, 's.'

    
    start = timeit.default_timer()    
#    ax.scatter( time, f, c='lightgrey', s=10, lw=0, rasterized=True )
    ax.plot( time, f,  '.', color='grey', rasterized=True ) #plot is faster than scatter for large data
#    ax.errorbar( time, f, yerr=ferr, color='grey', fmt='.', rasterized=True )
    ax.set_title( title )
    ax.set_ylabel( 'Flux' )
    ax.set_xlabel( 'HJD' )
    ax.set_xlim([ np.int(np.min(time))-1, np.int(np.max(time))+1 ])
#    ax.set_ylim([ np.nanmedian(f) - 6*np.nanstd(f), np.nanmedian(f) + 6*np.nanstd(f) ])
    stop = timeit.default_timer()
    if debug: print 'Scatter succesfully created in', stop-start, 's.'

    start = timeit.default_timer() 
    for i in np.arange( np.int(time[0]), np.int(time[-1])+1 ): 
        ax.axvline(i, color='lightgrey', zorder=-2)
    stop = timeit.default_timer()
    if debug: print 'Lines succesfully created in', stop-start, 's.'
    
    
    start = timeit.default_timer() 
    if (show_transit_regions==True) and (period is not None) and (epoch is not None) and (width is not None):  
        T_ingress = ( epoch - (width/2.) )
        T_egress = ( epoch + (width/2.) ) 
        j = 0            
        while T_egress < time[-1]:
            ax.axvspan(T_ingress, T_egress, facecolor='g', alpha=0.5, zorder=-1)
            j += 1
            T_ingress = j*period + ( epoch - (width/2.) )
            T_egress = j*period + ( epoch + (width/2.) )
    stop = timeit.default_timer()
    if debug: print 'Greens succesfully created in', stop-start, 's.'

 
 
def plot_binned_lightcurve_dic(ax, dic, obj_id=None, bin_time=0, normalize=True, show_transit_regions=False, debug=False):
#    dic['HJD'] /= (24. * 3600.)
    
    #::: multiple objects in dic
    if not isinstance( dic['OBJ_ID'], basestring ): 
        ind = np.where( dic['OBJ_ID'] == obj_id )[0]
        if 'SYSREM_FLUX3_ERR' in dic: ferr =  dic['SYSREM_FLUX3_ERR'][ind]
        else: ferr = np.nan       
        plot_binned_lightcurve( ax, dic['HJD'][ind]/(24. * 3600.), dic['SYSREM_FLUX3'][ind], ferr=ferr, bin_time=bin_time, normalize=normalize, title = dic['FIELDNAME']+', '+dic['OBJ_ID'][ind] )
    
    #::: single object in dic 
    else: 
        if 'SYSREM_FLUX3_ERR' in dic: ferr =  dic['SYSREM_FLUX3_ERR']
        else: ferr = np.nan     
        ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid = index_transits.index_transits(dic) #ind_tr, ind_tr_half, ind_tr_double, ind_out, ind_out_per_tr, tmid                         
        if normalize==True: dic['SYSREM_FLUX3'] /= np.nanmedian( dic['SYSREM_FLUX3'][ind_out] )
        plot_binned_lightcurve( ax, dic['HJD']/(24. * 3600.), dic['SYSREM_FLUX3'], ferr=ferr, bin_time=bin_time, show_transit_regions=show_transit_regions, period=dic['PERIOD']/(24. * 3600.), epoch=dic['EPOCH']/(24. * 3600.), width=dic['WIDTH']/(24. * 3600.), normalize=True, title = dic['FIELDNAME']+', '+dic['OBJ_ID'], debug=debug )

    
   
   
def plot_info_text(ax, dic):
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.axis('off')
#        ax.text(0,0.9,'RA (deg): '+mystr(dic['RA'][obj_nr]*180./np.pi,2))
#        ax.text(0,0.8,'DEC (deg): '+mystr(dic['DEC'][obj_nr]*180./np.pi,2))
    ra, dec = deg2HMS.deg2HMS(ra=dic['RA'], dec=dic['DEC'])
    ax.text(0,1.0,'OBJ_ID: '+dic['OBJ_ID'])
    ax.text(0,0.9,'FLUX: '+str(dic['FLUX_MEAN']))
    ax.text(0,0.8,'RA (deg): '+ra)
    ax.text(0,0.7,'DEC (deg): '+dec)
    ax.text(0,0.6,'PERIOD: ' + mystr(dic['PERIOD'],2) + ' s, ' + mystr(dic['PERIOD']/3600./24.,2) + ' d')
    ax.text(0,0.5,'WIDTH: ' + mystr(dic['WIDTH'],2) + ' s, ' + mystr(dic['WIDTH']/3600.,2) + ' h')
    ax.text(0,0.4,'EPOCH: ' + mystr(dic['EPOCH'],2) + ' s, ' + mystr(dic['WIDTH']/3600./24.,2) + ' d')
    ax.text(0,0.3,'DEPTH: ' + mystr(np.abs(dic['DEPTH'])*1000.,2) + ' mmag')
    ax.text(0,0.2,'NUM TRANSITS: '+mystr(dic['NUM_TRANSITS'],0))

    
    

def plot_lightcurve_analysis( dic, obj_id, bin_time, period_factor=1., normalize=True, show_transit_regions=False, fig_scale=1, plotting=True, debug=False ): 
    if plotting==True:
        plt.figure(figsize=(fig_scale*20,fig_scale*8))
        ax1 = plt.subplot2grid((2,4), (0,0), colspan=2)
        ax2 = plt.subplot2grid((2,4), (0,2))
        ax3 = plt.subplot2grid((2,4), (0,3))
        ax2b = plt.subplot2grid((2,4), (1,2))
        
        plot_binned_lightcurve_dic(ax1, dic, obj_id=obj_id, bin_time=bin_time, normalize=normalize, show_transit_regions=show_transit_regions)
        
        plot_phase_folded_lightcurve_dic([ax2,ax2b], dic, obj_id='011494', ferr_type='medsig', ferr_style='sem', period_factor=period_factor)
        ax2.text(0.98, 0.02, 'P='+str(period_factor * dic['PERIOD']/(24. * 3600.))[0:6]+'d',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax2.transAxes
            )
            
        plot_info_text(ax3, dic)
            
        plt.tight_layout()
    
    else:
        pass
    


def run(fieldname, obj_id, bin_time, ngts_version='TEST16A', period_factor=1., normalize=True, show_transit_regions=False, fig_scale=1, plotting=True, debug=False):
    plt.close('all')
    start = timeit.default_timer()
    dic = ngtsio.get( fieldname, ['RA','DEC','FLUX_MEAN','HJD','SYSREM_FLUX3','PERIOD','EPOCH','WIDTH','DEPTH','NUM_TRANSITS'], obj_id = obj_id, ngts_version=ngts_version, silent=True )
    stop = timeit.default_timer()
    
    if dic is not None:
        dic = set_nan(dic)
        if debug: print 'Data for', fieldname, obj_id, 'succesfully loaded in', stop-start, 's.'
        start = timeit.default_timer()
        plot_lightcurve_analysis( dic, obj_id, bin_time, period_factor=period_factor, normalize=normalize, show_transit_regions=show_transit_regions, fig_scale=fig_scale, plotting=plotting, debug=debug )
        stop = timeit.default_timer()
        if debug: print 'Plot for', fieldname, obj_id, 'succesfully created in', stop-start, 's.'
#        plt.show()
    

    
def test():
    start = timeit.default_timer()
    fieldname = 'NG0304-1115'
    obj_id = '011494'
    bin_time = 0 #in seconds
    run(fieldname, obj_id, bin_time)
    print timeit.default_timer() - start, 's'

    
if __name__=='__main__':
    test()
