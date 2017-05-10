#NGTS setup

import glob
import numpy as np
import fitsio
import os

#set up to be run from autovet directory
from Features.Centroiding.scripts import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate
from Features import Featureset
from Features.Centroiding.Centroiding_autovet_wrapper import centroid_autovet

def NGTS_Setup():
    mloader = 'multiloader_input_TEST18_v2.txt'
    orionfeatfile = 'orionfeatures_v2.txt'
    BLSdirs = glob.glob('/ngts/prodstore/02/BLSPipe*TEST18')
   
    with open(mloader,'w') as f:
        f.write('#fieldname ngts_version obj_id label per t0 tdur rank\n')

    with open(orionfeatfile,'w') as f:
        f.write('OBJ_ID, RANK, DELTA_CHISQ, NPTS_TRANSIT, NUM_TRANSITS, NBOUND_IN_TRANS, AMP_ELLIPSE, SN_ELLIPSE, GAP_RATIO, SN_ANTI, SDE\n')

    for indir in BLSdirs:
        infile = glob.glob(os.path.join(indir,'*.fits'))[0]
        dat = fitsio.FITS(infile)
        
        for cand in dat[4]:
        
            obj_id = cand['OBJ_ID'].strip(' ')
        
            candidates_on_same_id = np.where(dat[4]['obj_id'].read()==cand['OBJ_ID'])[0]  #will include same candidate

            cut = False
            if len(candidates_on_same_id)>1:  #there's a match aside from this candidate itself
                for matchcand in dat[4][candidates_on_same_id]:
                    if (matchcand['PERIOD']/cand['PERIOD']>0.998) & (matchcand['PERIOD']/cand['PERIOD']<1.002):
                        if matchcand['RANK'] < cand['RANK']:
                            cut = True
            if not cut:
                field = dat[2]['FIELD'].read()[0].strip(' ')
                version = 'TEST18'
                label = 'real_candidate'
                per = cand['PERIOD']/86400.
                t0 = cand['EPOCH']/86400.
                tdur = cand['WIDTH']/86400.
                depth = cand['DEPTH']
        
                #fieldname ngts_version obj_id label per t0 tdur
                with open(mloader,'a') as f:
                    f.write(field+' '+version+' '+obj_id+' '+label+' '+str(per)+' '+str(t0)+' '+str(tdur)+' '+str(int(cand['RANK']))+'\n')
    
                diags = np.array([cand['RANK'],cand['DELTA_CHISQ'],cand['NPTS_TRANSIT'],cand['NUM_TRANSITS'],cand['NBOUND_IN_TRANS'],cand['AMP_ELLIPSE'],cand['SN_ELLIPSE'],cand['GAP_RATIO'],cand['SN_ANTI'],cand['SDE']])

                with open(orionfeatfile,'a') as f:
                    f.write(obj_id+','+field+','+version+','+label+',')
                    for entry in diags[:-1]:
                        f.write(str(entry)+',')
                    f.write(str(diags[-1])+'\n')

from Loader.NGTS_MultiLoader import NGTS_MultiLoader

infile = '/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2_0.txt'
outdir = '/home/dja/Autovetting/Centroid/'
NGTS_MultiLoader(infile, outdir=outdir, docentroid=True)  #to just run the centroids

            #featurestocalc = 	{'SOM_Stat':[],'SOM_Distance':[],'SOM_IsRamp':[],'SOM_IsVar':[],
           # 					'Skew':[],'Kurtosis':[],'NZeroCross':[],'P2P_mean':[],'P2P_98perc':[],
           # 					'Peak_to_peak':[],'std_ov_error':[],'MAD':[],'RMS':[],'MaxSecDepth':[],
           # 					'MaxSecPhase':[],'MaxSecSig':[],'Even_Odd_depthratio':[],'Even_Odd_depthdiff_fractional':[],
           # 					'RPlanet':[],'TransitSNR':[],'PointDensity_ingress':[],
           # 					'SingleTransitEvidence':[],
           # 					'Fit_period':[],'Fit_chisq':[],'Fit_depthSNR':[],'Fit_t0':[],'Fit_aovrstar':[],'Fit_rprstar':[],
           # 					'Even_Fit_period':[],'Even_Fit_chisq':[],'Even_Fit_depthSNR':[],'Even_Fit_t0':[],'Even_Fit_aovrstar':[],'Even_Fit_rprstar':[],
           # 					'Odd_Fit_period':[],'Odd_Fit_chisq':[],'Odd_Fit_depthSNR':[],'Odd_Fit_t0':[],'Odd_Fit_aovrstar':[],'Odd_Fit_rprstar':[],
           # 					'Trapfit_t0':[],'Trapfit_t23phase':[],'Trapfit_t14phase':[],'Trapfit_depth':[],
           # 					'Even_Trapfit_t0':[],'Even_Trapfit_t23phase':[],'Even_Trapfit_t14phase':[],'Even_Trapfit_depth':[],
           # 					'Odd_Trapfit_t0':[],'Odd_Trapfit_t23phase':[],'Odd_Trapfit_t14phase':[],'Odd_Trapfit_depth':[],
           # 					'Even_Odd_trapdurratio':[],'Full_partial_tdurratio':[],'Even_Full_partial_tdurratio':[],'Odd_Full_partial_tdurratio':[]}
NGTS_MultiLoader(infile, dofeatures=featurestocalc)  #to just run the features (currently won't save!)

		
def NGTS_FeatureCombiner():
            print 'empty'
            #load in the already read features, the centroid output file features, and featureset features
    
            #combine to give large features array ready for classification
    
            #save
    
#repeat for synthetic transits
def Synth_Iterator():
    synthorionlist = '/ngts/pipeline/output/synthetics/*TEST18/ORION*'
    synthpostsysremlist = '/ngts/pipeline/output/synthetics/*TEST18/POST-SYSREM*'

    for synthfile in synthfilelist:
        field = os.path.split(synthfile)[1][19:30]
        for orionfile in synthorionlist:
            if os.path.split(orionfile)[1][6:17] == field:
                break
    
        lcf = fitsio.FITS(synthfile)
        obj_id_list = lcf['catalogue'].read()['OBJ_ID']
       
        blsdat = fitsio.FITS(orionfile)
        blsbase = blsdat['catalogue'].read()
        bls_objids = blsbase['OBJ_ID']
        peakmatch = blsbase['FAKE_PEAK_MATCH']
    
        blscatalogue = blsdat['candidates'].read()
        periods = blscatalogue['PERIOD']
        tdurs = blscatalogue['WIDTH']
        t0s = blscatalogue['EPOCH']
    
        for i,obj_id in enumerate(bls_objids):
           
            print 'Preparing '+obj_id
            try: 
                if peakmatch[i] > 0:  #orion found injected candidate
    
                    #look up candidate with correct rank (are there more than these in here?)
                    
                    #extract diags
    
                   diags = np.array([cand['RANK'],cand['DELTA_CHISQ'],cand['NPTS_TRANSIT'],cand['NUM_TRANSITS'],cand['NBOUND_IN_TRANS'],cand['AMP_ELLIPSE'],cand['SN_ELLIPSE'],cand['GAP_RATIO'],cand['SN_ANTI'],cand['SDE']])

                    #extract and save lc
 
                    obj_index = np.where(obj_id_list==obj_id.strip(' '))[0][0]

                    time = lcf['hjd'][obj_index,:][0]
                    flux = lcf['flux'][obj_index,:][0]
                    flux_err = lcf['flux_err'][obj_index,:][0]
                    flags = lcf['flags'][obj_index,:][0]
 
                    #calculate features (make sure field_periods is from the usual field, NOT the synths)