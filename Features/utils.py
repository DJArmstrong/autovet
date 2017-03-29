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
    
def BinPhaseLC(phaselc,nbins):
    bin_edges = np.arange(nbins)/float(nbins)
    bin_indices = np.digitize(phaselc[:,0],bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
    binnedstds = np.zeros(nbins)
    for bin in range(nbins):
        if np.sum(bin_indices==bin) > 0:
            binnedlc[bin,1] = np.mean(phaselc[bin_indices==bin,1])  #doesn't make use of sorted phase array, could probably be faster?
            binnedstds[bin] = np.std(phaselc[bin_indices==bin,1])
        else:
            binnedlc[bin,1] = np.mean(phaselc[:,1])  #bit awkward this, but only alternative is to interpolate?
            binnedstds[bin] = np.std(phaselc[:,1])
    return binnedlc,binnedstds
    
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
    scanresolution = tdur_phase/10.
    ntests = int(0.4*per/scanresolution)
    centphases = np.linspace(0.3,0.7,ntests)
    minphases = centphases - tdur_phase*0.5
    maxphases = centphases + tdur_phase*0.5
    secdepths = np.zeros(ntests)
    secdepthsigs = np.zeros(ntests)
    for t in range(ntests):
        lolim = np.searchsorted(phase,minphases[t])
        hilim = np.searchsorted(phase,maxphases[t])
        npoints = hilim-lolim
        if npoints > 0:
            base1 = np.median(flux[lolim-npoints*2:lolim])
            base2 = np.median(flux[hilim:hilim+npoints*2])
            base = (base1+base2)*0.5
            depth = base - np.mean(flux[lolim:hilim])
            deptherror = np.mean(err[lolim:hilim])/np.sqrt(npoints)
            secdepthsigs[t] = depth/deptherror
            secdepths[t] = depth
    maxidx = np.argmax(secdepthsigs)
    secondary = {}
    secondary['phase'] = centphases[maxidx]
    secondary['depth'] = secdepths[maxidx]
    secondary['significance'] = secdepthsigs[maxidx]
    return secondary

