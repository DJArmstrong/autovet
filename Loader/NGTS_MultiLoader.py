import numpy as np
import os
import glob
import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate
from Features.Centroiding.Centroiding_autovet_wrapper import centroid_autovet
from Features import Featureset


def FeatFile_Setup(featoutfile, dofeatures):
    keystowrite = np.sort(dofeatures.keys())
    with open(featoutfile,'w') as f:
        f.write('#')
        f.write('ID,label,')
        for key in keystowrite:
            f.write(str(key)+',')
        f.write('\n')

# NGTS specific loader for multiple sources from various fields
def NGTS_MultiLoader(infile, outdir=None, docentroid=False, dofeatures=False, featoutfile='featurefile.txt', overwrite=True, prepSOM=False, SOMoutfile='SOM'):
    '''
    infile (string): 	link to a file containing the columns
       				fieldname    ngts_version    obj_id    label    per   t0   tdur rank
       
    outdir (string): 	directory to save centroid outputs to
    
    docentroid (bool): 	perform centroid operation on candidates. 
    					NB if both dofeatures and docentroid are called, only 
    					previously processed candidates with features calculated
    					will be ignored
    
    dofeatures (dic): 	features to calculate, for format see Featureset.py. 
    
    featoutfile (str): 	filepath to save calculated features to
    
    overwrite (bool): 	overwrite feature output file and start again (if True), 
    					or skip already processed candidates (if False)
    					
    prepSOM (bool):     produce lightcurves ready for SOM
    
    SOMoutfilestem (str):	file to save SOMarray to (stem, will have field and version added to it)
    '''
    
    #::: read list of all fields
    indata = np.genfromtxt(infile, names=True, dtype=None)
    
    field_ids = [ x+'_'+y for (x,y) in zip(indata['fieldname'], indata['ngts_version']) ]
    
    unique_field_ids = np.unique(field_ids)
    
    output_per = []
    output_pmatch = []
    output_epochs = []
    
    #set up output files
    processed_ids = []
    if dofeatures:
        if not os.path.exists(featoutfile):
            FeatFile_Setup(featoutfile, dofeatures)
        else:
            if overwrite:
                FeatFile_Setup(featoutfile, dofeatures)
            else:
                featfile = np.genfromtxt(featoutfile,names=True,delimiter=',',dtype=None)
                for cand in featfile:
                    processed_ids.append(cand['ID'])
        keystowrite = np.sort(dofeatures.keys())
        
    elif docentroid: #if dofeatures has been called, that will overwrite the processed_ids determination.
        centroiddirs = glob.glob(os.path.join(outdir,'NG*'))
        for dir in centroiddirs:
            if len(glob.glob(os.path.join(dir,'*_centroid_info.txt')))==1:
                processed_ids.append(os.path.basename(os.path.normpath(dir)))
    
    if prepSOM:
        from Features.TransitSOM import TransitSOM_release as TSOM
        SOMarray = []
        som_ids = []
        SOMerrors = []
           
    #:::: loop over all fields
    for field_id in unique_field_ids:

        
        ind = np.where( np.array(field_ids) == field_id)[0]
        fieldname = field_id[0:11]
        ngts_version = field_id[12:]
        
        SOMouttest = SOMoutfile+'_'+ngts_version+'_'+fieldname+'_array.txt'
        if not os.path.exists(SOMouttest): #prevents reloading SOM files

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
                save_id = fieldname+'_'+'{:06d}'.format(candidate['obj_id'])+'_'+str(candidate['rank'])
                print save_id
                if save_id not in processed_ids:
                    #apply pmatch cut. This is being put in early to reduce the numbers we have to deal with
                    pmatch = np.sum((field_periods/candidate['per']>0.998) & (field_periods/candidate['per']<1.002))
                    if pmatch <= 5:#cuts to ~27115 total in TEST18 (down from 96716 after cutting same object same per peaks)
            
                        candidate_data = {'per':candidate['per'], 't0':candidate['t0'], 'tdur':candidate['tdur']}
                        can = Candidate('{:06d}'.format(candidate['obj_id']), filepath=None, observatory='NGTS', field_dic=field_dic, label=candidate['label'], candidate_data=candidate_data, field_periods=field_periods, field_epochs=field_epochs)
                        if len(can.lightcurve['time'])>0:
                            if can.lightcurve['time'][0] != -10: #signal that loading candidate didn't work
                                '''
                                now do the main stuff with this candidate...
                                or save all candidates into a dictionary/list of candidates and then go on from there...
                                '''
                                if docentroid:
                                    canoutdir = os.path.join(outdir,save_id,'')
                                    try:
                                        centroid_autovet( can, outdir=canoutdir)
                                    except ValueError: #catching the airmass poly error
                                        pass
                    
                                if dofeatures:
                                    feat = Featureset(can)
                                    feat.CalcFeatures(featuredict=dofeatures)
                                    features = feat.Writeout(keystowrite)
                                    with open(featoutfile,'a') as f:
                                        f.write(save_id+','+candidate['label']+',')
                                        for fe in features[2]:
                                            f.write(str(fe)+',')
                                        f.write('\n')
                                    
                                if prepSOM:
                                    lc = np.array([can.lightcurve['time'],can.lightcurve['flux'],can.lightcurve['error']]).T
                                    try:
                                        SOMarray_single, SOMerrors_single = TSOM.PrepareOneLightcurve(lc,candidate['per'],candidate['t0'],candidate['tdur'],nbins=20,clip_outliers=10)     
                                        SOMarray.append(SOMarray_single)
                                        SOMerrors.append(SOMerrors_single)
                                    except UnboundLocalError:
                                        print 'Skipping '

            if prepSOM:
                #save total file
                np.savetxt(SOMoutfile+'_'+ngts_version+'_'+fieldname+'_array.txt',np.array(SOMarray))
                np.savetxt(SOMoutfile+'_'+ngts_version+'_'+fieldname+'_error.txt',np.array(SOMerrors))
 