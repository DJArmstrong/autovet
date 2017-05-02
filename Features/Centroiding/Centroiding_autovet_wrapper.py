# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:06:16 2017

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""


import warnings
from Features.Centroiding.Centroiding import centroid



###########################################################################
#::: load data (all nights) of the target - AUTOVET version
###########################################################################
def centroid_autovet(candidate, pixel_radius = 150., flux_min = 1000., flux_max = 10000., bin_width=300., min_time=1800., dt=0.005, roots=None, outdir=None, parent=None, show_plot=False, flagfile=None):
    '''
    Amendments for autovet implementation
    '''
    
    if ( candidate.candidate_data['per'] > 0 ) and ( 't0' in candidate.candidate_data )  and ( 'tdur' in candidate.candidate_data ):
      
        period = candidate.candidate_data['per'] * 3600. * 24. #from days to seconds
        epoch = candidate.candidate_data['t0'] * 3600. * 24. #from days to seconds
        width = candidate.candidate_data['tdur'] * 3600. * 24. #from days to seconds
        
        fieldname = candidate.info['FIELDNAME']
        obj_id = candidate.info['OBJ_ID']
        ngts_version = candidate.info['NGTS_VERSION']
        
        dic = {}
        info_keys = ['OBJ_ID','FLUX_MEAN','RA','DEC','NIGHT','AIRMASS','CCDX','CCDY','CENTDX','CENTDY']
        for info_key in info_keys: dic[info_key] = candidate.info[info_key]        
        dic['HJD'] = candidate.lightcurve['time']
        dic['SYSREM_FLUX3'] = candidate.lightcurve['flux']
    
        centroid( fieldname, obj_id, ngts_version = ngts_version, source = '', bls_rank = None, period = period, epoch = epoch, width = width, time_hjd = None, pixel_radius = pixel_radius, flux_min = flux_min, flux_max = flux_max, bin_width=bin_width, min_time=min_time, dt=dt, roots=roots, outdir=outdir, parent=parent, show_plot=show_plot, flagfile=flagfile, dic=dic )

    else:
        warnings.warn('Centroiding aborted and skipped. Analysis requires a planet period and epoch.')
               
   