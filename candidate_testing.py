
import Loader
from Classify import Learner
import numpy as np
import pandas as pd
import pylab as p
p.ion()

def Trapezoidmodel(t0_phase,t23,t14,depth,phase_data):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    return model


candidate_ids = ['NG0348-3345_020642_1', 'NG0509-3345_000541_1',
       'NG0509-3345_031345_4', 'NG0509-3345_032371_4',
       'NG0509-3345_034057_1', 'NG0509-3345_035384_1',
       'NG0522-2518_010634_2', 'NG0522-2518_010891_2',
       'NG0522-2518_034480_4', 'NG0524-3056_001165_3',
       'NG0524-3056_034443_4', 'NG0537-3056_000401_1',
       'NG0537-3056_000401_5', 'NG0537-3056_005364_2',
       'NG0537-3056_012633_1', 'NG0549-3345_007144_1',
       'NG0549-3345_011779_3', 'NG0549-3345_026173_1',
       'NG0549-3345_028552_1', 'NG0549-3345_045356_3',
       'NG0603-3345_001737_1', 'NG0618-6441_012493_2',
       'NG1315-2807_013215_5', 'NG2047-0248_067711_2',
       'NG2047-0248_070871_1', 'NG2126-1652_024896_4',
       'NG2132+0248_006775_1', 'NG2132+0248_009658_1',
       'NG2132+0248_043342_1', 'NG2142+0826_004101_1',
       'NG2142+0826_030951_1', 'NG2150-3922_008216_1',
       'NG2152-1403_000629_1', 'NG2152-1403_008546_4',
       'NG2152-1403_018334_5', 'NG2152-1403_033794_2',
       'NG2152-1403_035203_4', 'NG2346-3633_000887_5',
       'NG2346-3633_018799_3', 'NG2346-3633_019630_1',
       'NG2346-3633_021533_1', 'NG2346-3633_021533_4']

trainingfile = '/home/dja/Autovetting/Classify/TrainingSets_withsimCentroid/trainset.txt'
tset=Learner.TrainingSet(trainingfile)
candidatedat = np.genfromtxt('/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2.txt',dtype=None,names=True)
dat_nodrop = pd.read_csv(trainingfile,index_col=0)

for o in range(len(candidatedat['fieldname'])):
  objid = candidatedat['fieldname'][o]+'_'+'{:06d}'.format(candidatedat['obj_id'][o])+'_'+str(candidatedat['rank'][o])
  if objid in candidate_ids:
    field = objid[:11]
    id = int(objid[13:18])
    candidate_data = {'per':candidatedat['per'][o], 't0':candidatedat['t0'][o], 'tdur':candidatedat['tdur'][o]}
    can=Loader.Candidate(id,[field,'TEST18'],observatory='NGTS',candidate_data=candidate_data)
    tidx = np.where(tset.ids==objid)[0]
    feat = tset.features[tidx,:]
    print objid
    p.figure(1)
    p.clf()
    p.plot(can.lightcurve['time'],can.lightcurve['flux'],'b.')
    p.figure(2)
    p.clf()
    phase = np.mod(can.lightcurve['time']-(can.candidate_data['t0']+can.candidate_data['per']/2.),can.candidate_data['per'])/can.candidate_data['per']
    p.plot(phase,can.lightcurve['flux'],'r.')
    
    #TRAPEZOID MODEL
    
    t0 = dat_nodrop.loc[objid,'Trapfit_t0']
    t23 = dat_nodrop.loc[objid,'Trapfit_t23phase']
    t14 = dat_nodrop.loc[objid,'Trapfit_t14phase']
    depth = dat_nodrop.loc[objid,'Trapfit_depth']
    print 'TrapFit Diags:'
    print 't0: '+str(t0)
    print 't23: '+str(t23)
    print 't14: '+str(t14)
    print 'depth: '+str(depth)

    model = Trapezoidmodel(0.5,t23,t14,depth,phase)
    p.figure(3)
    p.clf()
    p.plot(phase,can.lightcurve['flux'],'b.')
    p.plot(phase,model,'g.')
    
    for f in range(len(tset.featurenames)):
        print tset.featurenames[f]+': '+str(tset.features[tidx,f])
    
    p.pause(0.1)
    raw_input()