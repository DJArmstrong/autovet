import numpy as np
try:
    import pyfits
except ImportError:
    import astropy.io.fits as pyfits
import sys


def dopolyfit(win,d,ni,sigclip):
    '''
    Iterative polynomial fit. Each iteration, discrepant points are cut from the next fit.
    
    Arguments:
    win		--	np array (npoints,3). Format [time, flux, error]
    			Lightcurve segment to fit.
    d		--	int
    			Polynomial degree
    ni		--	int
    			Number of iterations
    sigclip	--	float
    			Sigma clipping threshold.
    
    Returns:
    base 	--	np polynomial object
    			Final fitted polynomial parameters
    '''
    base = np.polyfit(win[:,0],win[:,1],w=1.0/win[:,2],deg=d)
    #for n iterations, clip sigma, redo polyfit
    for iter in range(ni):
        offset = np.abs(win[:,1]-np.polyval(base,win[:,0]))/win[:,2]  
        if (offset<sigclip).sum()>int(0.8*len(win[:,0])):
            clippedregion = win[offset<sigclip,:]
        else:
            clippedregion = win[offset<np.average(offset)]   
        base = np.polyfit(	clippedregion[:,0],clippedregion[:,1],
        					w=1.0/np.power(clippedregion[:,2],2),deg=d)
    return base

def CheckForGaps(dat,centidx,winlowbound,winhighbound,gapthresh):
    diffshigh = np.diff(dat[centidx:winhighbound,0])
    gaplocshigh = np.where(diffshigh>gapthresh)[0]
    highgap = len(gaplocshigh)>0
    diffslow = np.diff(dat[winlowbound:centidx,0])
    gaplocslow = np.where(diffslow>gapthresh)[0]
    lowgap = len(gaplocslow)>0
    return lowgap, highgap, gaplocslow,gaplocshigh

def formwindow(datcut,dat,cent,size,boxsize,gapthresh,expectedpoints,cadence):
    winlowbound = np.searchsorted(datcut[:,0],cent-size/2.)
    winhighbound = np.searchsorted(datcut[:,0],cent+size/2.)
    boxlowbound = np.searchsorted(dat[:,0],cent-boxsize/2.)
    boxhighbound = np.searchsorted(dat[:,0],cent+boxsize/2.)
    centidx = np.searchsorted(datcut[:,0],cent)

    if centidx==boxlowbound:
        centidx += 1
    if winhighbound == len(datcut[:,0]):
        winhighbound -= 1
    flag = 0

    lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)

    if winlowbound == 0:
        lowgap = True
        gaplocslow = [-1]
    if winhighbound == len(datcut[:,0]):
         highgap = True
         gaplocshigh = [len(datcut[:,0]) -centidx]
    
    if highgap:
        if lowgap:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = winlowbound + 1 + gaplocslow[-1]
        else:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = np.searchsorted(datcut[:,0],datcut[winhighbound,0]-size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if lowgap:
                winlowbound = winlowbound + 1 + gaplocslow[-1] #uses reduced fitting section
    else:
        if lowgap:
            winlowbound = winlowbound + 1 + gaplocslow[-1]            
            winhighbound =  np.searchsorted(datcut[:,0],datcut[winlowbound,0]+size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if highgap:
                winhighbound = centidx + gaplocshigh[0] #uses reduced fitting section

    #window = np.concatenate((dat[winlowbound:boxlowbound,:],dat[boxhighbound:winhighbound,:]))
    window = datcut[winlowbound:winhighbound,:]
    if len(window[:,0]) < 20:
        flag = 1
    box = dat[boxlowbound:boxhighbound,:]

    return window,boxlowbound,boxhighbound,flag

#def ReadLCFITS(infile,inputcol,inputcol_err):
#    dat = pyfits.open(infile)
#    time = dat[1].data['TIME']
#    t0 = time[0]
#    time -= t0
#    nanstrip = time==time
#    time = time[nanstrip]
#    flux = dat[1].data[inputcol][nanstrip]
#    err = dat[1].data[inputcol_err][nanstrip]
#    quality = dat[1].data['SAP_QUALITY'][nanstrip]
#    fluxnanstrip = flux==flux
#    time = time[fluxnanstrip]
#    flux = flux[fluxnanstrip]
#    err = err[fluxnanstrip]
#    quality = quality[fluxnanstrip]
#    return time, flux, err, t0, quality

#def ReadLCTXT(infile):
#    dat = np.genfromtxt(infile)
#    idx = np.argsort(dat[:,0])
#    t0 = dat[0,0]
#    return dat[idx,0]-t0,dat[idx,1],dat[idx,2],t0,dat[idx,3]




def Kepflatten(time,flux,err,quality,winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,t0,plot,transitcut,tc_per,tc_t0,tc_tdur,outfile=False):
    lc = np.zeros([len(time),4])
    lc[:,0] = time
    lc[:,1] = flux
    lc[:,2] = err
    lc[:,3] = quality
    qualcheck = quality==0
    lc = lc[qualcheck,:]

    lcdetrend = np.zeros(len(lc[:,0]))

    #general setup
    lenlc = lc[-1,0]
    lcbase = np.median(lc[:,1])
    lc[:,1] /= lcbase
    lc[:,2] = lc[:,2]/lcbase
    nsteps = np.ceil(lenlc/stepsize).astype('int')
    stepcentres = np.arange(nsteps)/float(nsteps) * lenlc + stepsize/2.
    cadence = np.median(np.diff(lc[:,0]))
    
    expectedpoints = winsize/2./cadence

    if transitcut and tc_per>0:
        timecut, fluxcut, errcut, qualitycut = CutTransits(lc[:,0]+t0,lc[:,1],lc[:,2],lc[:,3],tc_t0,tc_per,tc_tdur)
        lc_tofit = np.zeros([len(timecut),4])
        lc_tofit[:,0] = timecut-t0
        lc_tofit[:,1] = fluxcut
        lc_tofit[:,2] = errcut
        lc_tofit[:,3] = qualitycut
    else:
        lc_tofit = lc


    #for each step centre:
    #actual flattening
    for s in range(nsteps):
        stepcent = stepcentres[s]
        winregion,boxlowbound,boxhighbound,flag = formwindow(lc_tofit,lc,stepcent,winsize,stepsize,gapthreshold,expectedpoints,cadence)  #should return window around box not including box

        if not flag:
            baseline = dopolyfit(winregion,polydegree,niter,sigmaclip)
            lcdetrend[boxlowbound:boxhighbound] = lc[boxlowbound:boxhighbound,1] / np.polyval(baseline,lc[boxlowbound:boxhighbound,0])
        else:
            lcdetrend[boxlowbound:boxhighbound] = np.ones(boxhighbound-boxlowbound)
        #if winregion[0,0]+2454833> 0:
        #p.figure(2)
        #p.clf()
        #    #print winregion.shape
        #    #print boxhighbound
        #    #print boxlowbound
        #p.plot(winregion[:,0],winregion[:,1],'b.')
        #raw_input()
        #p.plot(lc[boxlowbound:boxhighbound,0],lc[boxlowbound:boxhighbound,1],'r.')
        #x = np.arange(1000)/1000. *(winregion[-1,0]-winregion[0,0])+winregion[0,0]
        #p.plot(x,np.polyval(baseline,x),'g-')
        #raw_input()
    
    output = np.zeros_like(lc)
    output[:,0] = lc[:,0] + t0
    output[:,1] = lcdetrend
    output[:,2] = lc[:,2]
    output[:,3] = lc[:,3]
    
#    if plot:
#        p.figure(1)
#        p.clf()
#        p.plot(lc[:,0]+t0,lc[:,1],'b.')
#        p.plot(output[:,0],output[:,1],'g.')
#        raw_input('Press any key to continue')
    if outfile:
        np.savetxt(outfile,output)
    else:
        return output

def CutTransits(time,flux,err,qual,transitt0,transitper,transitdur):

    phase = np.mod(time-transitt0,transitper)/transitper
    
    intransit = (phase<transitdur/transitper) | (phase>1-transitdur/transitper)
    #p.figure(2)
    #p.clf()
    #p.plot(phase,flux,'g.')
    #raw_input()
    return time[~intransit],flux[~intransit],err[~intransit],qual[~intransit]


def Run(infile,outfile,inputcol='PDCSAP_FLUX',inputcol_err='PDCSAP_FLUX_ERR',winsize=10.,stepsize=0.3,polydegree=3,niter=10,sigmaclip=8.,gapthreshold=0.8,plot=0,transitcut=False,tc_period=0,tc_t0=0,tc_tdur=0):
    
    if infile[-5:] == '.fits':
	    time, flux, err, t0, quality = ReadLCFITS(infile,inputcol,inputcol_err) #standard kepler fits files
    else:
	    time, flux, err, t0, quality = ReadLCTXT(infile)  #txt files should be time, flux, err, quality. It will cut everything where quality not equal to 0.
    
    #parse user inputs
    optionalinputs = {}
    optionalinputs['inputcol'] = inputcol
    optionalinputs['inputcol_err'] = inputcol_err
    optionalinputs['winsize'] = winsize   #days, size of polynomial fitting region
    optionalinputs['stepsize'] = stepsize #days, size of region within polynomial region to detrend
    optionalinputs['polydegree'] = polydegree  #degree of polynomial to fit to local curve
    optionalinputs['niter'] = niter      #number of iterations to fit polynomial, clipping points significantly deviant from curve each time.
    optionalinputs['sigmaclip'] = sigmaclip   #significance at which points are clipped (see niter comment)
    optionalinputs['gapthreshold'] = gapthreshold  #days, threshold at which a gap in the time series is detected and the local curve is adjusted to not run over it
    optionalinputs['plot'] = plot
    optionalinputs['transitcut'] = transitcut
    optionalinputs['tc_period'] = tc_period
    optionalinputs['tc_t0'] = tc_t0
    optionalinputs['tc_tdur'] = tc_tdur
    
    #Parse actual user inputs
    for inputval in sys.argv[3:]:
        key,val = inputval.split('=')
        if key == 'polydegree' or key == 'niter' or key=='plot':
            val = int(val)
        elif key == 'winsize' or key == 'stepsize' or key == 'sigmaclip' or key == 'gapthreshold':
            val = float(val)
        optionalinputs[key] = val

    Kepflatten(time,flux,err,quality,optionalinputs['winsize'],optionalinputs['stepsize'],optionalinputs['polydegree'],optionalinputs['niter'],optionalinputs['sigmaclip'],optionalinputs['gapthreshold'],t0,optionalinputs['plot'],optionalinputs['transitcut'],optionalinputs['tc_period'],optionalinputs['tc_t0'],optionalinputs['tc_tdur'],outfile)
    
if __name__ == '__main__':
    Run(sys.argv[1],sys.argv[2])
