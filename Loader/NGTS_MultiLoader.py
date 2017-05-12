import numpy as np
import os
import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate
from Features.Centroiding.Centroiding_autovet_wrapper import centroid_autovet


# NGTS specific loader for multiple sources from various fields
def NGTS_MultiLoader(infile, outdir=None, docentroid=False, dofeatures=False):
    '''
    infile (string): link to a file containing the columns
       fieldname    ngts_version    obj_id    label    per   t0   tdur rank
       
    outdir (string): directory to save centroid outputs to
    
    docentroid (bool): perform centroid operation on candidates
    
    dofeatures (dic): features to calculate, for format see Featureset.py
    '''
    
    #::: read list of all fields
    indata = np.genfromtxt(infile, names=True, dtype=None)
    
    field_ids = [ x+'_'+y for (x,y) in zip(indata['fieldname'], indata['ngts_version']) ]
    
    unique_field_ids = np.unique(field_ids)
    
    output_per = []
    output_pmatch = []
    output_epochs = []
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
        
        #get the full list of periods in this field
        field_periods = indata['per'][ ind ]
        field_epochs = indata['t0'][ ind ]
        
        #::: loop over all candidates in this field
        for candidate in target_candidates_in_this_field:
            
            #apply pmatch cut. This is being put in early to reduce the numbers we have to deal with
            print candidate['obj_id']
            pmatch = np.sum((field_periods/candidate['per']>0.998) & (field_periods/candidate['per']<1.002))
            if pmatch <= 5:#cuts to ~27115 total in TEST18 (down from 96716 after cutting same object same per peaks)
            
                candidate_data = {'per':candidate['per'], 't0':candidate['t0'], 'tdur':candidate['tdur']}
                can = Candidate('{:06d}'.format(candidate['obj_id']), filepath=None, observatory='NGTS', field_dic=field_dic, label=candidate['label'], candidate_data=candidate_data, field_periods=field_periods, field_epochs=field_epochs)
            
                '''
                now do the main stuff with this candidate...
                or save all candidates into a dcitionary/list of candidates and then go on from there...
                for now just do centroiding :
                '''
                if docentroid:
                    canoutdir = os.path.join(outdir,fieldname+'_'+'{:06d}'.format(candidate['obj_id'])+'_'+str(candidate['rank']))
                    centroid_autovet( can, outdir=canoutdir)
                    
                if dofeatures:
                    feat = Featureset(can)
                    feat.CalcFeatures(featuredict=dofeatures)
                    #TODO WRITEOUT FEATURES TO FILE
