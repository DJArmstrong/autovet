import numpy as np
from scipy import interpolate,stats
import os

def SaveFeatureSets(FeatureSets,activefeatures,outfile):
    """
    Takes a list of Featuresets and saves their features in a txt format.

    Inputs:
    FeatureSets: list of FeatureSet objects
    activefeatures: list of the names of features to save. If any of these haven't been calculated, that object will be ignored.
    outfile: filepath 
    """
    nfeatures = len(activefeatures)
    nobjects = len(FeaturesSets)
    outarray = np.zeros([nobjects,nfeatures])
    outindex = []
    outlabels = []
    for f in range(nobjects):
        try:
            linetosave = np.array([FeatureSets[f].features[a] for a in activefeatures])
            outarray[f,:] = np.array([FeatureSets[f].features[a] for a in activefeatures])
            outindex.append(FeatureSets[f].target.id)
            outlabels.append(FeatureSets[f].target.label)
        except KeyError:  #one of the activefeatures has not been calculated
            outarray[f,:] = np.ones(len(activefeatures))-101 #makes a row with all values=-100
    
    outarray = outarray[(outarray!=-100).all(axis=1)]  #removes rows where all entries are -100
    np.savetxt(outfile,outarray,delimiter=',',header='#'+(' ').join(activefeatures))
    
    #save an index file containing ids and known classifications
    filepaths = os.path.split(outfile)
    with open(os.path.join(filepaths[0],'index_'+filepaths[1]),'w') as f:
        for i in range(len(outindex)):
            f.write(str(outindex[i])+','+str(outlabels[i])+'\n')

def LoadFeatureFile(self):
    if self.featurefile[:-4]=='.txt':
        features = np.genfromtxt(self.featurefile,delimiter=',')
        filepaths = os.path.split(self.featurefile)
        indexfile = os.path.join(filepaths[0],'index_'+filepaths[1])
        ids = np.genfromtxt(indexfile,delimiter=',')[:,0]
    return ids,features
        
def Scatter(lc,window,cut_outliers=False):
    """
    STD around a smoothed lightcurve.
    
    Inputs:
    lc      --  dict containing keys 'time', 'flux'
    window  --  number of datapoints to smooth over
    cut_outliers  --  remove outliers before processing. Uses the more conservative of a 98th percentile or 5*MAD clipping threshold.
    """
    if cut_outliers:
        time_cut, flux_cut = CutOutliers(lc['time'],lc['flux'])
    else:
        flux_cut = lc['flux']
        time_cut = lc['time']
        
    #interpolate gaps
    interp_time,interp_flux = FillGaps_Linear(time_cut,flux_cut)
    
    #Do smoothing
    flux_smooth = MovingAverage(interp_flux,window)
    flux_smooth = flux_smooth[window/2:-window/2]
    interp_smooth = interpolate.interp1d(interp_time[window/2:-window/2],flux_smooth,kind='linear',fill_value='extrapolate')

    return np.std(flux_cut - interp_smooth(time_cut))


def SPhot(lc,P_rot,k=5):
    """
    S_phot,k diagnostic (see Mathur et al 2014)
    """
    while k*P_rot > (lc['time'][-1] - lc['time'][0])/3.:
        k -= 1
        
    if k == 0:
        return np.zeros(10)
            
    #make time segments
    segwidth = k * P_rot
    
    nsegments = np.floor((lc['time'][-1]-lc['time'][0])/segwidth)   #will skip data at the end if segwidth doesn't exactly fit lc length
    tboundaries = np.arange(nsegments) * segwidth + lc['time'][0]
    
    #set up output array and index count
    
    pointindex = 0
    sphot = []
    npoints = []
    segtimes = []
    expectednpoints = segwidth/np.median(np.diff(lc['time']))
    while lc['time'][-1]-lc['time'][pointindex] > segwidth:
        stopindex = np.searchsorted(lc['time'],lc['time'][pointindex]+segwidth)
        if stopindex - pointindex > 0.75*expectednpoints:
            sphot.append(np.std(lc['flux'][pointindex:stopindex]))
            npoints.append(stopindex-pointindex)
            segtimes.append(np.mean(lc['time'][pointindex:stopindex]))
        pointindex += 1
        
    sphot = np.array(sphot)
    return sphot


def CalcContrast(SPhotseg,SPhotglob):  #SPhotglob is std of whole lightcurve
    """
    Contrast (see Mathur et al 2014)
        
    Will be returned as part of SPhot.
    """
    nhigh = SPhotseg>=SPhotglob
    SPhothigh = np.mean(SPhotseg[nhigh])
    SPhotlow = np.mean(SPhotseg[~nhigh])
    return SPhothigh/SPhotlow

def CutOutliers(time,flux):
    threshold = np.percentile(flux,[2,98])
    mednorm = np.median(flux)
    MAD_calc = 1.4826 * np.median(np.abs(flux - mednorm))
    sigmathreshold = [5*MAD_calc + mednorm,mednorm - 5*MAD_calc]
    if threshold[1] > sigmathreshold[1]:  #takes the least active cut option
        cut = (flux<threshold[1])&(flux>threshold[0])
    else:
        cut = (flux<sigmathreshold[1])&(flux>sigmathreshold[0])

    return time[cut],flux[cut]

def MovingAverage(interval, window_size): #careful with start and end. Also requires regular grid.
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def FillGaps_Linear(time,flux):       
    cadence = np.median(np.diff(time))
    npoints = np.floor((time[-1]-time[0])/cadence)
    interp_times = np.arange(npoints)*cadence + time[0]
    if interp_times[-1] > time[-1]:
        interp_times = interp_times[:-1]
    interp_obj = interpolate.interp1d(time,flux,kind='linear')
    interp_flux = interp_obj(interp_times)
    return interp_times,interp_flux

def phasefold(time,per,t0=0):
    return np.mod(time-t0,per)/per
    
def BinPhaseLC(phaselc,nbins,fill_value=None):
    bin_edges = np.arange(nbins)/float(nbins)
    bin_indices = np.digitize(phaselc[:,0],bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
    binnedstds = np.zeros(nbins)
    emptybins = 0
    for bin in range(nbins):
        if np.sum(bin_indices==bin) > 0:
            binnedlc[bin,1] = np.mean(phaselc[bin_indices==bin,1])  #doesn't make use of sorted phase array, could probably be faster?
            binnedstds[bin] = np.std(phaselc[bin_indices==bin,1])
        else:
            emptybins += 1
            if fill_value is not None:
                binnedlc[bin,1] = fill_value
            else:
                binnedlc[bin,1] = np.mean(phaselc[:,1])  #bit awkward this, but only alternative is to interpolate?
            binnedstds[bin] = np.std(phaselc[:,1])
    return binnedlc, binnedstds, emptybins
    
def FindSecondary(lc,per,t0,tdur):
    """
    Returns dict of information on maximum possible secondary eclipse.
        
    Scans a box of width the transit duration over phases 0.3 to 0.7. Returns maximum significance of box relative to local phasecurve, normalised by point errors.
    """
    tdur_phase = tdur/per
    flux = lc['flux'].copy()
    err = lc['error'].copy()
    phase = phasefold(lc['time'],per,t0)
    idx = np.argsort(phase)
    flux = flux[idx]
    phase = phase[idx]
    err = err[idx]
        
    #scan a box of width tdur over phase 0.3-0.7.
    scanresolution = tdur_phase/5.
    ntests = int(0.4/scanresolution)
    centphases = np.linspace(0.3,0.7,ntests)
    minphases = centphases - tdur_phase*0.5*0.3
    maxphases = centphases + tdur_phase*0.5*0.3   #gives a box width of 0.6 tdur - should give better trade off between resolution and sensitivity.
    secdepths = np.zeros(ntests) - 10
    secdepthsigs = np.zeros(ntests) - 10
    for t in range(ntests):
        lolim = np.searchsorted(phase,minphases[t])
        hilim = np.searchsorted(phase,maxphases[t])
        npoints = hilim-lolim
        if npoints > 20:
            #base1 = np.median(flux[lolim-npoints*2:lolim])
            #base2 = np.median(flux[hilim:hilim+npoints*2])
            #base = (base1+base2)*0.5
            #print base
            base = np.median(flux)
            depth = base - np.mean(flux[lolim:hilim])
            deptherror = np.std(flux[lolim:hilim])#/np.sqrt(npoints)
            #deptherror = np.mean(err[lolim:hilim])/np.sqrt(npoints)
            #print depth
            secdepthsigs[t] = depth/deptherror
            secdepths[t] = depth
    maxidx = np.argmax(secdepthsigs)
    secondary = {}
    secondary['phase'] = centphases[maxidx]
    secondary['depth'] = secdepths[maxidx]
    secondary['significance'] = secdepthsigs[maxidx]
    MAD = 1.4826 * np.median(np.abs(secdepths - np.median(secdepths)))
    secondary['selfsignificance'] = secdepths[maxidx] / MAD
    #import pylab as p
    #p.ion()
    #p.figure(10)
    #p.clf()
    #p.plot(centphases,secdepths,'b.')
    #p.figure(11)
    #p.clf()
    #p.plot(centphases,secdepthsigs,'r.')
    #p.pause(2)
    #raw_input()
    return secondary
    
def SplitOddEven(lc,per,t0,oddeven):
    time = lc['time']
    flux = lc['flux']
    err = lc['error']
    phase = phasefold(time,per*2,t0)
    if oddeven=='even':
        split = (phase <= 0.25) | (phase > 0.75)
    else:
        split = (phase > 0.25) & (phase <= 0.75)
    splitlc = {'time':time[split],'flux':flux[split],'error':err[split]}
    return splitlc
    
def BinTransitDuration(lc,per,t0,tdur):
    #idea is to have one bin per T23 transit duration, with the first bin covering that region of the phase curve.
    tdur_phase = tdur/per
    nbins = np.ceil(1./tdur_phase).astype('int')
    phase = phasefold(lc['time']+tdur/2.,per,t0)  #+tdur/2. means that phase curve starts at the beginning of tdur
    idx = np.argsort(phase)
    flux = lc['flux'][idx]
    time = lc['time'][idx]
    phase = phase[idx]
    bin_edges = np.arange(nbins)/float(nbins)
    bin_indices = np.digitize(phase,bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
    for bin in range(nbins):
        if np.sum(bin_indices==bin) > 0:
            binnedlc[bin,1] = np.mean(flux[bin_indices==bin])  #doesn't make use of sorted phase array, could probably be faster?
        else:
            binnedlc[bin,1] = -10.  #bit awkward this, but only alternative is to interpolate?
    return binnedlc
    
def BinSegment(lcseg,nbins,fill_value=None):
    flux = lcseg['flux']
    bin_edges = np.linspace(np.min(lcseg['time']),np.max(lcseg['time']),nbins)
    bin_indices = np.digitize(lcseg['time'],bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    pointsinbin = np.zeros(nbins)
    for bin in range(nbins):
        pointsinbin[bin] = np.sum(bin_indices==bin)
    avgpointsinbin = np.median(pointsinbin)
    for bin in range(nbins):
        binnedlc[bin,0] = np.mean(lcseg['time'][bin_indices==bin])
        if pointsinbin[bin] > avgpointsinbin*0.5:
            binnedlc[bin,1] = np.mean(flux[bin_indices==bin])  #doesn't make use of sorted phase array, could probably be faster?
        else:
            if fill_value is not None:
                binnedlc[bin,1] = fill_value
            else:
                binnedlc[bin,1] = np.mean(flux)  #bit awkward this, but only alternative is to interpolate?
    return binnedlc
   
    