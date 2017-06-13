import numpy as np
import os
import ngtsio_v1_1_1_autovet as ngtsio
from Loader import Candidate
from Features.Centroiding.Centroiding_autovet_wrapper import centroid_autovet
from Features import Featureset
import batman
from Features import TransitSOM as TSOM
from Features import utils

def batman_model(t0,per,rprstar,aovrstar,exp_time,time):
    fix_e = 0.
    fix_w = 90.
    ldlaw = 'quadratic'
    fix_ld = [0.1,0.3]
    bparams = batman.TransitParams()       #object to store transit parameters
    bparams.t0 = t0                      #time of inferior conjunction
    bparams.per = per                      #orbital period
    bparams.rp = rprstar                      #planet radius (in units of stellar radii)
    bparams.a = aovrstar                        #semi-major axis (in units of stellar radii)
    bparams.inc = 90.
    bparams.ecc = fix_e                      #eccentricity
    bparams.w = fix_w                        #longitude of periastron (in degrees)
    bparams.limb_dark = ldlaw        #limb darkening model
    bparams.u = fix_ld      #limb darkening coefficients
    m = batman.TransitModel(bparams, time, exp_time=exp_time, supersample_factor=7)    #initializes model
    return m, bparams

def Trapezoidmodel(t0_phase,t23,t14,depth,phase_data):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    return model

# NGTS specific loader for multiple sources from various fields
def NGTS_MultiLoader(infile, outdir=None, docentroid=False, dofeatures=False, featoutfile='featurefile.txt'):
    '''
    infile (string): link to a file containing the columns
       fieldname    ngts_version    obj_id    label    per   t0   tdur rank
       
    outdir (string): directory to save centroid outputs to
    
    docentroid (bool): perform centroid operation on candidates
    
    dofeatures (dic): features to calculate, for format see Featureset.py
    
    featoutfile (str): filepath to save calculated features to
    '''
    
    #::: read list of all fields
    indata = np.genfromtxt(infile, names=True, dtype=None)
    
    field_ids = [ x+'_'+y for (x,y) in zip(indata['fieldname'], indata['ngts_version']) ]
    
    unique_field_ids = np.unique(field_ids)
    
    output_per = []
    output_pmatch = []
    output_epochs = []
    
    #set up output files
    if dofeatures:
        keystowrite = np.sort(dofeatures.keys())
        with open(featoutfile,'w') as f:
            f.write('#')
            f.write('ID,label,')
            for key in keystowrite:
                f.write(str(key)+',')
            f.write('\n')


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
                or save all candidates into a dictionary/list of candidates and then go on from there...
                '''
                if docentroid:
                    canoutdir = os.path.join(outdir,fieldname+'_'+'{:06d}'.format(candidate['obj_id'])+'_'+str(candidate['rank']))
                    centroid_autovet( can, outdir=canoutdir)
                    
                if dofeatures:
                  #if candidate['obj_id'] == 3294 and candidate['rank']==1:
                
                  feat = Featureset(can)
                  lc = can.lightcurve
                  som = TSOM.TSOM.LoadSOM(os.path.join(os.path.join(os.getcwd(),'Features/TransitSOM/'),'NGTSOM_bin20_iter100.txt'),20,20,20,0.1)
                  lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                  SOMarray,SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,can.candidate_data['per'],can.candidate_data['t0'],can.candidate_data['tdur'],nbins=20)
                  polygrad = np.polyfit(np.arange(20)/20.,SOMarray,1)[1]
                  print polygrad
                  if np.abs(polygrad)>0.7:
                  #if SOMarray[-1]<0.6 and SOMarray[0]>0.9:
                  #feat.CalcFeatures(featuredict={'ntransits':[]})
                  #if feat.features['ntransits']<=2:
                    feat.CalcFeatures(featuredict=dofeatures)
                    features = feat.Writeout(keystowrite)
                    print candidate['obj_id']
                    print candidate['rank']
                    #print feat.features
                    import pylab as p
                    p.ion()
                    print 'Features: '
                    print 'ntransits: '
                    print feat.features['ntransits']
                    print 'MissingDataFlag'
                    print feat.features['missingDataFlag']
                    print 'PointDensity_ingress'
                    print feat.features['PointDensity_ingress']
                    print 'TransitSNR'
                    print feat.features['TransitSNR']
    
                    print 'Fit Diagnostic'
                    print 'Fit_chisq: '+str(feat.features['Fit_chisq'])
                    print 'Fit_depthSNR: '+str(feat.features['Fit_depthSNR'])
                    per = feat.features['Fit_period']
                    t0 = feat.features['Fit_t0']
                    aovrstar = feat.features['Fit_aovrstar']
                    rprstar = feat.features['Fit_rprstar']
    
                    #BATMAN MODEL
                    exp_time = np.median(np.diff(can.lightcurve['time']))
                    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time,can.lightcurve['time'])
                    modelflux = m.light_curve(bparams)

                    phase = np.mod(can.lightcurve['time']-can.candidate_data['t0'],can.candidate_data['per'])/can.candidate_data['per']
                    p.figure(1)
                    p.clf()
                    p.plot(phase,can.lightcurve['flux'],'r.')
    
                    p.figure(2)
                    p.clf()
                    phase_model_orig = np.mod(can.lightcurve['time']-t0,per)/per
                    p.plot(phase_model_orig,can.lightcurve['flux'],'b.')
                    p.plot(phase_model_orig,modelflux,'r.')
    
                    #BATMAN EVEN
                    print 'Even/Odd Fit Diagnostic'
                    print 'Even Fit_chisq: '+str(feat.features['Even_Fit_chisq'])
                    print 'Even Fit_depthSNR: '+str(feat.features['Even_Fit_depthSNR'])
                    per = feat.features['Fit_period']
                    t0 = feat.features['Fit_t0']
                    aovrstar = feat.features['Even_Fit_aovrstar']
                    rprstar = feat.features['Even_Fit_rprstar']
    
                    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time,can.lightcurve['time'])
                    modelflux = m.light_curve(bparams)
    
                    p.figure(3)
                    p.clf()
                    phase_model = np.mod(can.lightcurve['time']-t0,per)/per
                    p.plot(phase_model_orig,can.lightcurve['flux'],'b.')
                    p.plot(phase_model,modelflux,'r.')

    
                    #BATMAN ODD
                    print 'Odd Fit_chisq: '+str(feat.features['Odd_Fit_chisq'])
                    print 'Odd Fit_depthSNR: '+str(feat.features['Odd_Fit_depthSNR'])
                    per = feat.features['Fit_period']
                    t0 = feat.features['Fit_t0']
                    aovrstar = feat.features['Odd_Fit_aovrstar']
                    rprstar = feat.features['Odd_Fit_rprstar']
                    phase_model = np.mod(can.lightcurve['time']-t0,per)/per
    
                    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time,can.lightcurve['time'])
                    modelflux = m.light_curve(bparams)
                    p.plot(phase_model,modelflux,'g.')
    
                    print 'Even_Odd_depthratio: '+str(feat.features['Even_Odd_depthratio'])
                    print 'Even_Odd_depthdiff_fractional: '+str(feat.features['Even_Odd_depthdiff_fractional'])
    
                    #TRAPEZOID MODEL
                    t0 = feat.features['Trapfit_t0']
                    t23 = feat.features['Trapfit_t23phase']
                    t14 = feat.features['Trapfit_t14phase']
                    depth = feat.features['Trapfit_depth']
                    print 'TrapFit Diags:'
                    print 't0: '+str(t0)
                    print 't23: '+str(t23)
                    print 't14: '+str(t14)
                    print 'depth: '+str(depth)
                    phase_orig = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
                    model = Trapezoidmodel(0.5,t23,t14,depth,phase_orig)
                    p.figure(4)
                    p.clf()
                    p.plot(phase_orig,can.lightcurve['flux'],'b.')
                    p.plot(phase_orig,model,'g.')

                    #EVEN TRAPFIT
                    t0 = feat.features['Trapfit_t0']
                    t23 = feat.features['Even_Trapfit_t23phase']
                    t14 = feat.features['Even_Trapfit_t14phase']
                    depth = feat.features['Even_Trapfit_depth']
                    print 'Even_TrapFit Diags:'
                    print 't0: '+str(t0)
                    print 't23: '+str(t23)
                    print 't14: '+str(t14)
                    print 'depth: '+str(depth)
                    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
                    model = Trapezoidmodel(0.5,t23,t14,depth,phase)
                    p.figure(5)
                    p.clf()
                    p.plot(phase_orig,can.lightcurve['flux'],'b.')
                    p.plot(phase,model,'r.')

                    #ODD TRAPFIT
                    t0 = feat.features['Trapfit_t0']
                    t23 = feat.features['Odd_Trapfit_t23phase']
                    t14 = feat.features['Odd_Trapfit_t14phase']
                    depth = feat.features['Odd_Trapfit_depth']
                    print 'Odd_TrapFit Diags:'
                    print 't0: '+str(t0)
                    print 't23: '+str(t23)
                    print 't14: '+str(t14)
                    print 'depth: '+str(depth)
                    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
                    model = Trapezoidmodel(0.5,t23,t14,depth,phase)
                    p.figure(5)
                    p.plot(phase,model,'g.')
    
                    print 'Even_Odd_trapdurratio: '+str(feat.features['Even_Odd_trapdurratio'])
                    print 'Full_partial_tdurratio: '+str(feat.features['Full_partial_tdurratio'])
                    print 'Even_Full_partial_tdurratio: '+str(feat.features['Even_Full_partial_tdurratio'])
                    print 'Odd_Full_partial_tdurratio: '+str(feat.features['Odd_Full_partial_tdurratio'])

                    #SECONDAY ZOOM PLOT
                    print 'Secondary Features: '
                    print 'MaxSecDepth: '+str(feat.features['MaxSecDepth'])
                    print 'MaxSecPhase: '+str(feat.features['MaxSecPhase'])
                    print 'MaxSecSig: '+str(feat.features['MaxSecSig'])
                    print 'MaxSecSelfSig: '+str(feat.features['MaxSecSelfSig'])
                    p.figure(6)
                    p.clf()
                    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],can.candidate_data['t0'])  #transit at phase 0.5 
                    p.plot(phase,can.lightcurve['flux'],'b.')
                    p.plot([feat.features['MaxSecPhase'],feat.features['MaxSecPhase']],[1-feat.features['MaxSecDepth'],1.],'g-')    
    
                    #SOM ZOOM PLOT
                    print 'SOM Features: '
                    print 'SOM_Stat: '+str(feat.features['SOM_Stat'])
                    print 'SOM_Distance: '+str(feat.features['SOM_Distance'])
                    print 'SOM_IsRamp: '+str(feat.features['SOM_IsRamp'])
                    print 'SOM_IsVar: '+str(feat.features['SOM_IsVar'])
    
                    lc = can.lightcurve
                    som = TSOM.TSOM.LoadSOM(os.path.join(os.path.join(os.getcwd(),'Features/TransitSOM/'),'NGTSOM_bin20_iter100.txt'),20,20,20,0.1)
                    lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                    SOMarray,SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,can.candidate_data['per'],can.candidate_data['t0'],can.candidate_data['tdur'],nbins=20)
  
                    p.figure(7)
                    p.clf()
                    p.plot(np.arange(20),SOMarray,'b.')
        
                    SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray))))
                    print 'SOM_loc: '+str(som(SOMarray)[0,:])
    
                    p.pause(5)
                    raw_input()
                    with open(featoutfile,'a') as f:
                        f.write(fieldname+'_'+'{:06d}'.format(candidate['obj_id'])+'_'+str(candidate['rank'])+','+candidate['label']+',')
                        for fe in features[2]:
                            f.write(str(fe)+',')
                        f.write('\n')
