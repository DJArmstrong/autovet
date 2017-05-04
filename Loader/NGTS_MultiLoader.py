import numpy as np
        
import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate




# NGTS specific loader for multiple sources from various fields
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
        
        #::: load this field into memory with ngtsio
        field_dic = ngtsio.get(fieldname, ['OBJ_ID','HJD','FLUX','FLUX_ERR','CCDX','CCDY','CENTDX','CENTDY'], obj_id=target_obj_ids_in_this_field, ngts_version=ngts_version)
        
        
        #::: loop over all candidates in this field
        for obj_id in target_obj_ids_in_this_field:
            
            can = Candidate(obj_id, filepath=None, observatory='NGTS', field_dic=field_dic, label=indata['label'], hasplanet={'per':indata['per'], 't0':indata['t0'], 'tdur':indata['tdur']} )
       
            '''
            now do the main stuff with this candidate...
            or save all candidates into a dcitionary/list of candidates and then go on from there...
            for now:
            '''
            print can.lightcurve
