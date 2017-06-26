#NGTS setup

import glob
import numpy as np
import fitsio
import os
import sys

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

def NGTS_CentroidRun(inputs):
    from Loader.NGTS_MultiLoader import NGTS_MultiLoader
    infilelist = np.sort(glob.glob('/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2_*.txt'))
    for input in inputs:
        infile = infilelist[int(input)]
        outdir = '/home/dja/Autovetting/Centroid/Run0/'
        NGTS_MultiLoader(infile, outdir=outdir, docentroid=True)  #to just run the centroids

def NGTS_LoaderTest():
    from Loader.NGTS_MultiLoader_loadtest import NGTS_MultiLoader
    infilelist = glob.glob('/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2_*.txt')
    for infile in infilelist:
        NGTS_MultiLoader(infile,dofeatures=False)
        
def NGTS_FeatureCalc(inputs):
    from Loader.NGTS_MultiLoader import NGTS_MultiLoader
    infilelist = np.sort(glob.glob('/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2_*.txt'))
    for input in inputs:
        infile = infilelist[int(input)]
        #infile = '/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2_0.txt'

        featurestocalc = {'tdur_phase':[],'pmatch':[],'ntransits':[],'missingDataFlag':[],'SOM_Stat':[],'SOM_Distance':[],'SOM_IsRamp':[],'SOM_IsVar':[],
            		'Skew':[],'Kurtosis':[],'NZeroCross':[],'P2P_mean':[],'P2P_98perc':[],
            		'Peak_to_peak':[],'std_ov_error':[],'MAD':[],'RMS':[],'RMS_TDur':[],'MaxSecDepth':[],
            		'MaxSecPhase':[],'MaxSecSig':[],'MaxSecSelfSig':[],'Even_Odd_depthratio':[],'Even_Odd_depthdiff_fractional':[],
            		'TransitSNR':[],'PointDensity_ingress':[],
            		'Fit_period':[],'Fit_chisq':[],'Fit_depthSNR':[],'Fit_t0':[],'Fit_aovrstar':[],'Fit_rprstar':[],
            		'Even_Fit_chisq':[],'Even_Fit_depthSNR':[],'Even_Fit_aovrstar':[],'Even_Fit_rprstar':[],
            		'Odd_Fit_chisq':[],'Odd_Fit_depthSNR':[],'Odd_Fit_aovrstar':[],'Odd_Fit_rprstar':[],
            		'Trapfit_t0':[],'Trapfit_t23phase':[],'Trapfit_t14phase':[],'Trapfit_depth':[],
            		'Even_Trapfit_t23phase':[],'Even_Trapfit_t14phase':[],'Even_Trapfit_depth':[],
            		'Odd_Trapfit_t23phase':[],'Odd_Trapfit_t14phase':[],'Odd_Trapfit_depth':[],
            		'Even_Odd_trapdurratio':[],'Even_Odd_trapdepthratio':[],'Full_partial_tdurratio':[],
            		'Even_Full_partial_tdurratio':[],'Odd_Full_partial_tdurratio':[]}
        featoutfile = os.path.join('/home/dja/Autovetting/Dataprep/Featurerun_v0/','features_v0'+os.path.split(infile)[1])
        NGTS_MultiLoader(infile, dofeatures=featurestocalc, featoutfile=featoutfile, overwrite=False)

		
def ScanCentroids(centroiddir='/home/dja/Autovetting/Centroid/Run0/',outfile='/home/dja/Autovetting/Centroid/Run0/centroid_features_run0.txt'):
    dirlist = glob.glob(os.path.join(centroiddir,'NG*'))
    centroidkeys = ['CENTDX_fda_PHASE_RMSE','CENTDY_fda_PHASE_RMSE','RollCorrSNR_X',
    				'RollCorrSNR_Y','CrossCorrSNR_X','CrossCorrSNR_Y','Ttest_X',
    				'Ttest_Y','Binom_X','Binom_Y']
    with open(outfile,'w') as f:
        f.write('#')
        f.write('ID,')
        for key in centroidkeys:
            f.write(str(key)+',')
        f.write('\n')    				
    for dir in dirlist:
        infofile = glob.glob(os.path.join(dir,'*centroid_info.txt'))
        dat = np.genfromtxt(infofile,names=True,dtype=None)
        with open(outfile,'a') as f:
            f.write(os.path.basename(os.path.normpath(dir))+',')
            for key in centroidkeys:
                val = dat[key]
                f.write(str(val)+',')
            f.write('\n')

def Synth_FeatureCalc():
    from Loader import Candidate
    from Features import Featureset

    loaderdat = np.genfromtxt('/home/dja/Autovetting/Dataprep/SynthLCs_alex/synth_input_TEST18_alex.txt',names=True,dtype=None)
    featdat = np.genfromtxt('/home/dja/Autovetting/Dataprep/SynthLCs_alex/synthorionfeatures_alex.txt',names=True,delimiter=',',dtype=None)
    lcdir = '/home/dja/Autovetting/Dataprep/SynthLCs_alex/'

    featurestocalc = {'tdur_phase':[],'ntransits':[],'missingDataFlag':[],'SOM_Stat':[],'SOM_Distance':[],'SOM_IsRamp':[],'SOM_IsVar':[],
            		'Skew':[],'Kurtosis':[],'NZeroCross':[],'P2P_mean':[],'P2P_98perc':[],
            		'Peak_to_peak':[],'std_ov_error':[],'MAD':[],'RMS':[],'RMS_TDur':[],'MaxSecDepth':[],
            		'MaxSecPhase':[],'MaxSecSig':[],'MaxSecSelfSig':[],'Even_Odd_depthratio':[],'Even_Odd_depthdiff_fractional':[],
            		'TransitSNR':[],'PointDensity_ingress':[],
            		'Fit_period':[],'Fit_chisq':[],'Fit_depthSNR':[],'Fit_t0':[],'Fit_aovrstar':[],'Fit_rprstar':[],
            		'Even_Fit_chisq':[],'Even_Fit_depthSNR':[],'Even_Fit_aovrstar':[],'Even_Fit_rprstar':[],
            		'Odd_Fit_chisq':[],'Odd_Fit_depthSNR':[],'Odd_Fit_aovrstar':[],'Odd_Fit_rprstar':[],
            		'Trapfit_t0':[],'Trapfit_t23phase':[],'Trapfit_t14phase':[],'Trapfit_depth':[],
            		'Even_Trapfit_t23phase':[],'Even_Trapfit_t14phase':[],'Even_Trapfit_depth':[],
            		'Odd_Trapfit_t23phase':[],'Odd_Trapfit_t14phase':[],'Odd_Trapfit_depth':[],
            		'Even_Odd_trapdurratio':[],'Even_Odd_trapdepthratio':[],'Full_partial_tdurratio':[],
            		'Even_Full_partial_tdurratio':[],'Odd_Full_partial_tdurratio':[]}  
            		
    outfile = '/home/dja/Autovetting/Dataprep/SynthLCs_alex/synth_features_alex.txt'
    keystowrite = np.sort(featurestocalc.keys())
    orionkeys = ['RANK', 'DELTA_CHISQ', 'NPTS_TRANSIT', 'NUM_TRANSITS', 'NBOUND_IN_TRANS', 'AMP_ELLIPSE', 'SN_ELLIPSE', 'GAP_RATIO', 'SN_ANTI', 'SDE']
    with open(outfile,'w') as f:
        f.write('#')
        f.write('ID,label,')
        for key in keystowrite:
            f.write(str(key)+',')
        for key in orionkeys:
            f.write(str(key)+',')
        f.write('\n')
    
    for candidate in loaderdat:
        print candidate['fieldname']+'_'+candidate['obj_id']
      #if candidate['fieldname']+'_'+candidate['obj_id'] == 'NG0304-1115_F00177':
        filepath = os.path.join(lcdir,candidate['fieldname']+'_'+candidate['obj_id']+'_lc.txt')
        candidate_data = {'per':candidate['per'], 't0':candidate['t0'], 'tdur':candidate['tdur']}
        can = Candidate(candidate['obj_id'], filepath=filepath, observatory='NGTS_synth', label=candidate['label'], candidate_data=candidate_data)
        #import pylab as p
        #p.ion()
        #phase = np.mod(can.lightcurve['time']-can.candidate_data['t0'],can.candidate_data['per'])/can.candidate_data['per']
        #p.plot(phase,can.lightcurve['flux'],'r.')
        #p.pause(2)
        #raw_input()
        feat = Featureset(can)
        feat.CalcFeatures(featuredict=featurestocalc)      
        features = feat.Writeout(keystowrite)
        orionidx = np.where((featdat['FIELD']==candidate['fieldname'])&(featdat['OBJ_ID']==candidate['obj_id']))[0]
        
        orionfeatures = featdat[['RANK', 'DELTA_CHISQ', 'NPTS_TRANSIT', 'NUM_TRANSITS', 'NBOUND_IN_TRANS', 'AMP_ELLIPSE', 'SN_ELLIPSE', 'GAP_RATIO', 'SN_ANTI', 'SDE']][orionidx]
        with open(outfile,'a') as f:
            f.write(candidate['fieldname']+'_'+candidate['obj_id']+','+candidate['label']+',')
            for fe in features[2]:
                f.write(str(fe)+',')
            for fe in orionfeatures[0]:
                f.write(str(fe)+',')
            f.write('\n')

#repeat for synthetic transits
def Synth_Iterator():
    synthdir = '/wasp/scratch/alexsmith/synthetics/'
    synthruns = ['NG0522-2518-802-TEST18_4','NG0304-1115-809-TEST18_shallow','NG2047-0248-810-TEST18_shallow']
    synthorionlist = []
    synthpostsysremlist = []
    for run in synthruns:
        synthorionlist.append(glob.glob(os.path.join(synthdir,run)+'/ORION*')[0])
        synthpostsysremlist.append(glob.glob(os.path.join(synthdir,run)+'/POST-SYSREM*')[0])
        
    outdir = '/home/dja/Autovetting/Dataprep/SynthLCs_alex/'
    mloader = 'synth_input_TEST18_alex.txt'
    orionfeatfile = 'synthorionfeatures_alex.txt'

    with open(os.path.join(outdir,mloader),'w') as f:
        f.write('#fieldname ngts_version obj_id label per t0 tdur rank\n')

    with open(os.path.join(outdir,orionfeatfile),'w') as f:
        f.write('#OBJ_ID, FIELD, VERSION, LABEL, RANK, DELTA_CHISQ, NPTS_TRANSIT, NUM_TRANSITS, NBOUND_IN_TRANS, AMP_ELLIPSE, SN_ELLIPSE, GAP_RATIO, SN_ANTI, SDE\n')

    
    for synthfile in synthpostsysremlist:
        field = os.path.split(synthfile)[1][17:28]
        for orionfile in synthorionlist:
            if os.path.split(orionfile)[1][6:17] == field:
                break
    
        lcf = fitsio.FITS(synthfile)
        obj_id_list = lcf['catalogue'].read()['OBJ_ID']
       
        blsdat = fitsio.FITS(orionfile)
        blsbase = blsdat['catalogue'].read()
        bls_objids = blsbase['OBJ_ID']
        peakmatch = blsbase['FAKE_PEAK_MATCH']
        candmatch = blsbase['FAKE_CAND_MATCH']
        
        blscatalogue = blsdat['candidates'].read()
        periods = blscatalogue['PERIOD']
        tdurs = blscatalogue['WIDTH']
        t0s = blscatalogue['EPOCH']
        
        fieldname = os.path.split(synthfile)[1][17:28]
        version = os.path.split(synthfile)[1][-11:-5]
        label = 'synth'
        
        for i,obj_id in enumerate(bls_objids):
            
            print 'Preparing '+obj_id
            if peakmatch[i] > 0:  #orion found injected candidate
              outfile = os.path.join(outdir,fieldname+'_'+obj_id.strip(' ')+'_lc.txt')
              if not os.path.exists(outfile):
                #look up candidate with correct rank (are there more than these in here?)
                candidx = candmatch[i]-1 #they start from 1 (0s are no detn), hence the -1
                cand = blscatalogue[candidx]
                per = cand['PERIOD']/86400.
                t0 = cand['EPOCH']/86400.
                tdur = cand['WIDTH']/86400.
                depth = cand['DEPTH']
    
                #fieldname ngts_version obj_id label per t0 tdur
                with open(os.path.join(outdir,mloader),'a') as f:
                    f.write(fieldname+' '+version+' '+obj_id+' '+label+' '+str(per)+' '+str(t0)+' '+str(tdur)+' '+str(int(cand['RANK']))+'\n')
    
                diags = np.array([cand['RANK'],cand['DELTA_CHISQ'],cand['NPTS_TRANSIT'],cand['NUM_TRANSITS'],cand['NBOUND_IN_TRANS'],cand['AMP_ELLIPSE'],cand['SN_ELLIPSE'],cand['GAP_RATIO'],cand['SN_ANTI'],cand['SDE']])

                with open(os.path.join(outdir,orionfeatfile),'a') as f:
                    f.write(obj_id.strip(' ')+','+fieldname+','+version+','+label+',')
                    for entry in diags[:-1]:
                        f.write(str(entry)+',')
                    f.write(str(diags[-1])+'\n')

                #extract and save lc
 
                obj_index = np.where(obj_id_list==obj_id.strip(' '))[0][0]
                time = lcf['hjd'][obj_index,:][0]
                flux = lcf['flux'][obj_index,:][0]
                flux_err = lcf['flux_err'][obj_index,:][0]
                flags = lcf['flags'][obj_index,:][0]

                output = np.array([time,flux,flux_err,flags]).T
                np.savetxt(outfile,output)


if __name__=='__main__':
    #Synth_Iterator()
    #Synth_FeatureCalc()
    #NGTS_CentroidRun()
    Scan_Centroids()
    #inputs = sys.argv[1:]
    #NGTS_FeatureCalc(inputs)
    #NGTS_LoaderTest()