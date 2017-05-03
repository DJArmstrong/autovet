#NGTS setup

import glob
import numpy as np
import fitsio
import os

#set up to be run from autovet directory
from Features.Centroiding.scripts import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate
from Features import Featureset

def NGTS_Setup():
    mloader = 'multiloader_input_TEST18.txt'
    orionfeatfile = 'orionfeatures.txt'
    BLSdirs = glob.glob('/ngts/prodstore/02/BLSPipe*TEST18')

    with open(mloader,'w') as f:
        f.write('#fieldname ngts_version obj_id label per t0 tdur\n')

    with open(orionfeatfile,'w') as f:
        f.write('OBJ_ID, RANK, DELTA_CHISQ, NPTS_TRANSIT, NUM_TRANSITS, NBOUND_IN_TRANS, AMP_ELLIPSE, SN_ELLIPSE, GAP_RATIO, SN_ANTI, SDE\n')

    for indir in BLSdirs:
        infile = glob.glob(os.path.join(indir,'*.fits'))[0]
        dat = fitsio.FITS(infile)
    
        for cand in dat[4]:
            obj_id = cand['OBJ_ID'].strip(' ')
        
            field = dat[2]['FIELD'].read()[0].strip(' ')
            version = 'TEST18'
            label = 'real_candidate'
            per = cand['PERIOD']/86400.
            t0 = cand['EPOCH']/86400.
            tdur = cand['WIDTH']/86400.
            depth = cand['DEPTH']
        
            #fieldname ngts_version obj_id label per t0 tdur
            with open(mloader,'a') as f:
                f.write(field+' '+version+' '+obj_id+' '+label+' '+str(per)+' '+str(t0)+' '+str(tdur)+'\n')
    
            diags = np.array([cand['RANK'],cand['DELTA_CHISQ'],cand['NPTS_TRANSIT'],cand['NUM_TRANSITS'],cand['NBOUND_IN_TRANS'],cand['AMP_ELLIPSE'],cand['SN_ELLIPSE'],cand['GAP_RATIO'],cand['SN_ANTI'],cand['SDE']])

            with open(orionfeatfile,'a') as f:
                f.write(obj_id+','+field+','+version+','+label+',')
                for entry in diags[:-1]:
                    f.write(str(entry)+',')
                f.write(str(diags[-1])+'\n')
        

def centroid_autovet(candidate, pixel_radius = 150., flux_min = 1000., flux_max = 10000., bin_width=300., min_time=1800., dt=0.005, roots=None, outdir=None, parent=None, show_plot=False, flagfile=None):
    '''
    Amendments for autovet implementation
    '''
    from Features.Centroiding.Centroiding_RDX import centroid
    
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
    
        C = centroid( fieldname, obj_id, ngts_version = ngts_version, source = '', bls_rank = None, period = period, epoch = epoch, width = width, time_hjd = None, pixel_radius = pixel_radius, flux_min = flux_min, flux_max = flux_max, bin_width=bin_width, min_time=min_time, dt=dt, roots=roots, outdir=outdir, parent=parent, show_plot=show_plot, flagfile=flagfile, dic=dic )
        C.run()

    else:
        warnings.warn('Centroiding aborted and skipped. Analysis requires a planet period and epoch.')

    
#with autovet code
def NGTS_MultiLoader(infile):
    '''
    infile (string): link to a file containing the columns
       fieldname    ngts_version    obj_id    label    per   t0   tdur
    '''
    
    #::: read list of all fields
    indata = np.genfromtxt(infile, names=True, dtype=None)
    
    field_ids = [ x+'_'+y for (x,y) in zip(indata['fieldname'], indata['ngts_version']) ]
    
    unique_field_ids = np.unique(field_ids)
    
    
    #:::: loop over all fields
    for field_id in unique_field_ids:
        
        ind = np.where( np.array(field_ids) == field_id)[0]
        fieldname = field_id[0:11]
        ngts_version = field_id[12:]
        
        #::: extract all candidate obj_ids in this field
        target_obj_ids_in_this_field = indata['obj_id'][ ind ]
        target_candidates_in_this_field = indata[ ind ]
        
        #::: load this field into memory with ngtsio
        field_dic = ngtsio.get(fieldname, ['OBJ_ID','HJD','FLUX','FLUX_ERR','CCDX','CCDY','CENTDX','CENTDY','FLUX_MEAN','RA','DEC','NIGHT','AIRMASS'], obj_id=target_obj_ids_in_this_field, ngts_version=ngts_version)
        
        
        #::: loop over all candidates in this field
        for candidate in target_candidates_in_this_field:
        
            can = Candidate('{:06d}'.format(candidate['obj_id']), filepath=None, observatory='NGTS', field_dic=field_dic, label=candidate['label'], candidate_data={'per':candidate['per'], 't0':candidate['t0'], 'tdur':candidate['tdur']} )
       
            print candidate['obj_id']
            #print candidate['per']
            #print can.lightcurve
            #Centroiding
            #run centroid_autovet wrapper - think it saves an output file
            #centroid_autovet(can,outdir='/home/dja/Autovetting/Centroid/')

            #set up candidate, featureset objects
            feat = Featureset(can)
            
            featurestocalc = 	{'SOM_Stat':[],'SOM_Distance':[],'SOM_IsRamp':[],'SOM_IsVar':[],
            					'Skew':[],'Kurtosis':[],'NZeroCross':[],'P2P_mean':[],'P2P_98perc':[],
            					'Peak_to_peak':[],'std_ov_error':[],'MAD':[],'RMS':[],'MaxSecDepth':[],
            					'MaxSecPhase':[],'MaxSecSig':[],'Even_Odd_depthratio':[],'Even_Odd_depthdiff_fractional':[],
            					'RPlanet':[],'TransitSNR':[],'PointDensity_ingress':[],
            					'SingleTransitEvidence':[],
            					'Fit_period':[],'Fit_chisq':[],'Fit_depthSNR':[],'Fit_t0':[],'Fit_aovrstar':[],'Fit_rprstar':[],
            					'Even_Fit_period':[],'Even_Fit_chisq':[],'Even_Fit_depthSNR':[],'Even_Fit_t0':[],'Even_Fit_aovrstar':[],'Even_Fit_rprstar':[],
            					'Odd_Fit_period':[],'Odd_Fit_chisq':[],'Odd_Fit_depthSNR':[],'Odd_Fit_t0':[],'Odd_Fit_aovrstar':[],'Odd_Fit_rprstar':[],
            					'Trapfit_t0':[],'Trapfit_t23phase':[],'Trapfit_t14phase':[],'Trapfit_depth':[],
            					'Even_Trapfit_t0':[],'Even_Trapfit_t23phase':[],'Even_Trapfit_t14phase':[],'Even_Trapfit_depth':[],
            					'Odd_Trapfit_t0':[],'Odd_Trapfit_t23phase':[],'Odd_Trapfit_t14phase':[],'Odd_Trapfit_depth':[],
            					'Even_Odd_trapdurratio':[],'Full_partial_tdurratio':[],'Even_Full_partial_tdurratio':[],'Odd_Full_partial_tdurratio':[]}
            					
   
            feat.CalcFeatures(featuredict=featurestocalc)
            
            #calculate features beyond the already in place ones

            
    
            #load in the already read features, and the centroid output file features
    
            #combine with writeout features to give large features array ready for classification
    
    
#repeat for synthetic transits