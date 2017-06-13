#synth feature checker

import numpy as np
import batman
import Features.TransitSOM as TSOM
import Features.utils as utils
import os
from Loader import Candidate
import Features

def batman_model(t0,per,rprstar,aovrstar,exp_time):
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
    m = batman.TransitModel(bparams, can.lightcurve['time'],exp_time=exp_time,supersample_factor=7)    #initializes model
    return m, bparams

def Trapezoidmodel(t0_phase,t23,t14,depth,phase_data):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    return model

lcdir = '/Users/davidarmstrong/Software/Python/NGTS/Autovetting/Synth_Tests/LCs'
featfilepath = '/Users/davidarmstrong/Software/Python/NGTS/Autovetting/Synth_Tests/synth_featurestest.txt'
featfile = np.genfromtxt(featfilepath,names=True,dtype=None,delimiter=',')
loaderdat = np.genfromtxt('/Users/davidarmstrong/Software/Python/NGTS/Autovetting/Synth_Tests/synth_input_TEST18_v2_all.txt',names=True,dtype=None)
loaderidx = []
for cand in loaderdat:
    loaderidx.append(cand['fieldname']+'_'+cand['obj_id'])
loaderidx = np.array(loaderidx)

for i,ID in enumerate(featfile['ID']):
 if ID=='NG0304-1115_F00013': 
  filepath = os.path.join(lcdir,ID+'_lc.txt')
  if os.path.exists(filepath):
    idx = np.where(loaderidx==ID)[0]
    candidate = loaderdat[idx]
    candidate_data = {'per':candidate['per'][0], 't0':candidate['t0'][0], 'tdur':candidate['tdur'][0]}
    can = Candidate(candidate['obj_id'], filepath=filepath, observatory='NGTS_synth', label=candidate['label'], candidate_data=candidate_data)
    feat=Features.Featureset(can,useflatten=False,testplots=False)
    feat.CalcFeatures(featuredict={'Full_partial_tdurratio':[]})
    print feat.features
    raw_input()
    import pylab as p
    p.ion()

#plot lightcurve with fit

    print 'Features: '
    print 'MissingDataFlag'
    print featfile['MissingDataFlag'][i]
    print 'PointDensity_ingress'
    print featfile['PointDensity_ingress'][i]
    print 'SingleTransitEvidence'
    print featfile['SingleTransitEvidence'][i]
    print 'TransitSNR'
    print featfile['TransitSNR'][i]
    
    print 'Fit Diagnostic'
    print 'Fit_chisq: '+str(featfile['Fit_chisq'][i])
    print 'Fit_depthSNR: '+str(featfile['Fit_depthSNR'][i])
    per = featfile['Fit_period'][i]
    t0 = featfile['Fit_t0'][i]
    aovrstar = featfile['Fit_aovrstar'][i]
    rprstar = featfile['Fit_rprstar'][i]
    
    #BATMAN MODEL
    exp_time = np.median(np.diff(can.lightcurve['time']))
    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time)
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
    print 'Even Fit_chisq: '+str(featfile['Even_Fit_chisq'][i])
    print 'Even Fit_depthSNR: '+str(featfile['Even_Fit_depthSNR'][i])
    per = featfile['Fit_period'][i]
    t0 = featfile['Fit_t0'][i]
    aovrstar = featfile['Even_Fit_aovrstar'][i]
    rprstar = featfile['Even_Fit_rprstar'][i]
    
    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time)
    modelflux = m.light_curve(bparams)
    
    p.figure(3)
    p.clf()
    phase_model = np.mod(can.lightcurve['time']-t0,per)/per
    p.plot(phase_model_orig,can.lightcurve['flux'],'b.')
    p.plot(phase_model,modelflux,'r.')

    
    #BATMAN ODD
    print 'Odd Fit_chisq: '+str(featfile['Odd_Fit_chisq'][i])
    print 'Odd Fit_depthSNR: '+str(featfile['Odd_Fit_depthSNR'][i])
    per = featfile['Fit_period'][i]
    t0 = featfile['Fit_t0'][i]
    aovrstar = featfile['Odd_Fit_aovrstar'][i]
    rprstar = featfile['Odd_Fit_rprstar'][i]
    phase_model = np.mod(can.lightcurve['time']-t0,per)/per
    
    m, bparams = batman_model(t0,per,rprstar,aovrstar,exp_time)
    modelflux = m.light_curve(bparams)
    p.plot(phase_model,modelflux,'g.')
    
    print 'Even_Odd_depthratio: '+str(featfile['Even_Odd_depthratio'][i])
    print 'Even_Odd_depthdiff_fractional: '+str(featfile['Even_Odd_depthdiff_fractional'][i])
    
    #TRAPEZOID MODEL
    t0 = featfile['Trapfit_t0'][i]
    t23 = featfile['Trapfit_t23phase'][i]
    t14 = featfile['Trapfit_t14phase'][i]
    depth = featfile['Trapfit_depth'][i] 
    print 'TrapFit Diags:'
    print 't0: '+str(t0)
    print 't23: '+str(t23)
    print 't14: '+str(t14)
    print 'depth: '+str(depth)
    phase_orig = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
    model = Trapezoidmodel(0.5,t23,t14,depth,phase_orig)
    p.figure(4)
    p.plot(phase_orig,can.lightcurve['flux'],'b.')
    p.plot(phase_orig,model,'g.')

    #EVEN TRAPFIT
    t0 = featfile['Trapfit_t0'][i]
    t23 = featfile['Even_Trapfit_t23phase'][i]
    t14 = featfile['Even_Trapfit_t14phase'][i]
    depth = featfile['Even_Trapfit_depth'][i] 
    print 'Even_TrapFit Diags:'
    print 't0: '+str(t0)
    print 't23: '+str(t23)
    print 't14: '+str(t14)
    print 'depth: '+str(depth)
    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
    model = Trapezoidmodel(0.5,t23,t14,depth,phase)
    p.figure(5)
    p.plot(phase_orig,can.lightcurve['flux'],'b.')
    p.plot(phase,model,'r.')

    #ODD TRAPFIT
    t0 = featfile['Trapfit_t0'][i]
    t23 = featfile['Odd_Trapfit_t23phase'][i]
    t14 = featfile['Odd_Trapfit_t14phase'][i]
    depth = featfile['Odd_Trapfit_depth'][i] 
    print 'Odd_TrapFit Diags:'
    print 't0: '+str(t0)
    print 't23: '+str(t23)
    print 't14: '+str(t14)
    print 'depth: '+str(depth)
    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],t0+can.candidate_data['per']/2.)  #transit at phase 0.5
     
    model = Trapezoidmodel(0.5,t23,t14,depth,phase)
    p.figure(5)
    p.plot(phase,model,'g.')
    
    print 'Even_Odd_trapdurratio: '+str(featfile['Even_Odd_trapdurratio'][i])
    print 'Full_partial_tdurratio: '+str(featfile['Full_partial_tdurratio'][i])
    print 'Even_Full_partial_tdurratio: '+str(featfile['Even_Full_partial_tdurratio'][i])
    print 'Odd_Full_partial_tdurratio: '+str(featfile['Odd_Full_partial_tdurratio'][i])

    #SECONDAY ZOOM PLOT
    print 'Secondary Features: '
    print 'MaxSecDepth: '+str(featfile['MaxSecDepth'][i])
    print 'MaxSecPhase: '+str(featfile['MaxSecPhase'][i])
    print 'MaxSecSig: '+str(featfile['MaxSecSig'][i])

    p.figure(6)
    p.clf()
    phase = utils.phasefold(can.lightcurve['time'],can.candidate_data['per'],can.candidate_data['t0'])  #transit at phase 0.5
    p.plot(phase,can.lightcurve['flux'],'b.')
    p.plot([featfile['MaxSecPhase'][i],featfile['MaxSecPhase'][i]],[1-featfile['MaxSecDepth'][i],1.],'g-')    
    
    #SOM ZOOM PLOT
    print 'SOM Features: '
    print 'SOM_Stat: '+str(featfile['SOM_Stat'][i])
    print 'SOM_Distance: '+str(featfile['SOM_Distance'][i])
    print 'SOM_IsRamp: '+str(featfile['SOM_IsRamp'][i])
    print 'SOM_IsVar: '+str(featfile['SOM_IsVar'][i])
    
    lc = can.lightcurve
    som = TSOM.TSOM.LoadSOM(os.path.join(os.path.join(os.getcwd(),'Features/TransitSOM/'),'NGTSOM_bin20_iter100.txt'),20,20,20,0.1)
    lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
    SOMarray,SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,can.candidate_data['per'],can.candidate_data['t0'],can.candidate_data['tdur'],nbins=20)
  
    p.figure(7)
    p.clf()
    p.plot(np.arange(20),SOMarray,'b.')
        
    SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray))))
    print 'SOM_loc: '+str(som(SOMarray)[0,:])
    
    p.pause(2)
    raw_input()