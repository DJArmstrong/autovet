
import glob
import os
import numpy as np
from autovet.Features import utils as futils


flist = glob.glob('/Users/davidarmstrong/Software/Python/Autovet/PASTIStest/Injections/*.txt')
metadata = np.genfromtxt('/Users/davidarmstrong/Software/Python/Autovet/PASTIStest/injection_metadata.txt',delimiter=',')

#set up X input

planetX = []
ebX = []
bgebX = []
hebX = []
psbX = []
btpX = []

planetX_ph = []
ebX_ph = []
bgebX_ph = []
hebX_ph = []
psbX_ph = []
btpX_ph = []

planetX_loc = []
ebX_loc = []
bgebX_loc = []
hebX_loc = []
psbX_loc = []
btpX_loc = []

nbins = 1000
npoints = 4614
nbins_loc = 100

for infile in flist:
    
    dat = np.genfromtxt(infile)
    injid = os.path.split(infile)[1][:5]
    injtype = int(injid[0])
    print(injid)
    
    metaidx = np.where(metadata[:,0]==int(injid))[0]
    per = metadata[metaidx,3]
    t0 = metadata[metaidx,4]
    tdur = metadata[metaidx,5]

    phase = futils.phasefold(dat[:,0],per,t0-per*0.5)
    idx = np.argsort(phase)
    
    #save stacked lcs
    if dat.shape[0] < npoints:
        fillarray = np.ones(npoints-dat.shape[0]) * np.median(dat[:,1])
        flux = np.hstack((dat[:,1],fillarray))
    else:
        flux = dat[:,1]
       
    #create 'global' view
    binphaselc,stds,emptybins = futils.BinPhaseLC(np.array([phase[idx],dat[idx,1]]).T,nbins,fill_value='interp')

    #create 'local' view
    tdur_phase = tdur/per
    phasesort = phase[idx]
    local_cut = (phasesort>=(0.5-1.5*tdur_phase)) & (phasesort<=(0.5+1.5*tdur_phase))
    bin_local,stds_local,emptybins_local = futils.BinPhaseLCSegment(np.array([phasesort[local_cut],dat[idx,1][local_cut]]).T,nbins_loc,fill_value='interp')
    
    #normalise local view
    sortbins = np.sort(bin_local[:,1])
    maxlevel = np.median(sortbins[int(np.ceil(2*nbins_loc/3.)):]) #highest third
    minlevel = np.mean(sortbins[:10]) #lowest 10 points
    norm_flux_loc = (bin_local[:,1]-minlevel) / (maxlevel-minlevel)
    
    if injtype==0:
        planetX.append(flux)
        planetX_ph.append(binphaselc[:,1])
        planetX_loc.append(norm_flux_loc)
    elif injtype==1:
        ebX.append(flux)
        ebX_ph.append(binphaselc[:,1]) 
        ebX_loc.append(norm_flux_loc)   
    elif injtype==2:
        hebX.append(flux)
        hebX_ph.append(binphaselc[:,1])
        hebX_loc.append(norm_flux_loc)
    elif injtype==3:
        psbX.append(flux)
        psbX_ph.append(binphaselc[:,1])    
        psbX_loc.append(norm_flux_loc)
    elif injtype==4:
        bgebX.append(flux)
        bgebX_ph.append(binphaselc[:,1])  
        bgebX_loc.append(norm_flux_loc)  
    elif injtype==5:
        btpX.append(flux)
        btpX_ph.append(binphaselc[:,1])    
        btpX_loc.append(norm_flux_loc)
    
    #import pylab as p
    #p.ion()
    #p.figure(1)
    #p.clf()
    #p.plot(phase,dat[:,1],'b.')
    #p.pause(1)
    #p.figure(2)
    #p.clf()
    #p.plot(binphaselc[:,1],'m.')
    #p.pause(1)
    #p.figure(3)
    #p.clf()
    #p.plot(bin_local[:,1],'m.')
    #p.pause(1)
    #input()    
       
    
planetX = np.array(planetX)
ebX = np.array(ebX)
bgebX = np.array(bgebX)
hebX = np.array(hebX)
psbX = np.array(psbX)
btpX = np.array(btpX)

planetX_ph = np.array(planetX_ph)
ebX_ph = np.array(ebX_ph)
bgebX_ph = np.array(bgebX_ph)
hebX_ph = np.array(hebX_ph)
psbX_ph = np.array(psbX_ph)
btpX_ph = np.array(btpX_ph)

planetX_loc = np.array(planetX_loc)
ebX_loc = np.array(ebX_loc)
bgebX_loc = np.array(bgebX_loc)
hebX_loc = np.array(hebX_loc)
psbX_loc = np.array(psbX_loc)
btpX_loc = np.array(btpX_loc)

np.savetxt('run2norm/planetXph_1000bin_kepQ9.txt',planetX_ph)
np.savetxt('run2norm/ebXph_1000bin_kepQ9.txt',ebX_ph)
np.savetxt('run2norm/bgebXph_1000bin_kepQ9.txt',bgebX_ph)
np.savetxt('run2norm/hebXph_1000bin_kepQ9.txt',hebX_ph)
np.savetxt('run2norm/psbXph_1000bin_kepQ9.txt',psbX_ph)
np.savetxt('run2norm/btpXph_1000bin_kepQ9.txt',btpX_ph)

np.savetxt('run2norm/planetXloc_100bin_kepQ9.txt',planetX_loc)
np.savetxt('run2norm/ebXloc_100bin_kepQ9.txt',ebX_loc)
np.savetxt('run2norm/bgebXloc_100bin_kepQ9.txt',bgebX_loc)
np.savetxt('run2norm/hebXloc_100bin_kepQ9.txt',hebX_loc)
np.savetxt('run2norm/psbXloc_100bin_kepQ9.txt',psbX_loc)
np.savetxt('run2norm/btpXloc_100bin_kepQ9.txt',btpX_loc)

np.savetxt('run2norm/planetX_kepQ9.txt',planetX)
np.savetxt('run2norm/ebX_kepQ9.txt',ebX)
np.savetxt('run2norm/bgebX_kepQ9.txt',bgebX)
np.savetxt('run2norm/hebX_kepQ9.txt',hebX)
np.savetxt('run2norm/psbX_kepQ9.txt',psbX)
np.savetxt('run2norm/btpX_kepQ9.txt',btpX)

