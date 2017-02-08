import PeriodLS
import numpy as np
from scipy import interpolate,stats

class Featureset(object):

    def __init__(self,Candidate):
        """
        Calculate features for a given candidate or candidate list.
        
        Arguments:
        Candidate   -- instance of Candidate class.
        """
        self.features = {}
        self.target = Candidate
        
                        
    def CalcFeatures(self,featurelist=[]):
        """
        User facing function to calculate features and populate features dict.
        
        Inputs:
        featurelist -- list of features to calculate. None to use observatory defaults.
        """

        if len(featurelist)==0:
            if self.target.obs == 'NGTS':
                self.featurelist = []
            elif self.target.obs=='Kepler' or self.target.obs=='K2':
                self.featurelist = []
            else:
                print 'Observatory not supported, please input desired feature list'

        for featurename in featurelist:
            if featurename not in self.features.keys():  #avoid recalculating features
                feature = getattr(self,featurename)()
                if type(feature) == np.ndarray: #assumes if the feature is now an array, it's good
                    self.features[featurename] = feature
                elif feature:   #if function failed, should be 0
                    self.features[featurename] = feature

    def LSPeriod(self):  #really inefficient at present. Finds a peak, removes it, reruns periodogram 4 times.
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        lc   -- numpy array, column 0 time, column 1 flux
        
        Returns:
        period -- first 4 peaks from Lomb-Scargle periodogram
        ampratios -- ratio of each of first 10 peaks amplitude to maximum peak amplitude
        """
        if self.target.obs == 'K2':
            a = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
        else:
            a = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        a.fit()
        periods = a.periods
        ampratios = a.ampratios
        return np.array([periods,ampratios])

    def EBPeriod(self):
        """
        Tests for phase variation at double the period to correct EB periods.
        
        Inputs:
        lc   -- numpy array, column 0 time, column 1 flux
        period -- period to test
        
        Returns:
        corrected period, either initial period or double      
        """
        lc = self.target.lightcurve
        period = self.features['LSPeriod'][0,0]  #assumes most significant LS period hit, and that LSPeriod was set to return more than one (default returns 4)
     
        phaselc2P = np.zeros([len(lc['time']),2])
        phaselc2P[:,0] = self.phasefold(lc['time'],period*2)
        phaselc2P = phaselc2P[np.argsort(phaselc2P[:,0]),:] #now in phase order
        binnedlc2P,binstd = self.BinPhaseLC(phaselc2P,64)

        minima = np.argmin(binnedlc2P[:,1])
        posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
        posssecondary = np.where((posssecondary<0.05) | (posssecondary > 0.95))[0]   #within 0.05 either side of phase 0.5 from minima

        pointsort = np.sort(lc['flux'])
        top10points = np.median(pointsort[-30:])
        bottom10points = np.median(pointsort[:30])
        
        if self.target.obs=='K2':
            if lc[-1,0]-lc[0,0] >= 60:
                periodlim= 20.  #20 day general K2 limit. May be low. Seems to be highest reliable period though
            else: #for C0
                periodlim = 10.
        else:
            periodlim = 100000. #no effective limit
            
        if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.03*(top10points-bottom10points) and period*2<=periodlim:  
            return period * 2
        else:
            return period

    #BELOW HERE NEEDS TESTING
    def Skew(self):
        return stats.skew(self.target.lightcurve['flux'])
    
    def Kurtosis(self):
        return stats.kurtosis(self.target.lightcurve['flux'])
    
    def NZeroCross(self):
        lc = self.target.lightcurve
        
        time_cut, flux_cut = CutOutliers(lc['time'],lc['flux'])
        
        #interpolate gaps
        interp_time,interp_flux = FillGaps_Linear(time_cut,flux_cut)
    
        #Do smoothing
        flux_smooth = MovingAverage(interp_flux,window)
        flux_smooth = flux_smooth[window/2:-window/2]

        norm = np.median(flux_smooth)
        flux_zeroed = flux_smooth - norm
        return ((flux_zeroed[:-1] * flux_zeroed[1:]) < 0).sum()

    def P2P_mean(self):
        """
        Mean point-to-point difference across flux. Does not take account of gaps.
        """
        return np.mean(np.abs(np.diff(self.target.lightcurve['flux'])))

    def P2P_98perc(self):
        """
        98th percentile of point-to-point difference across flux. Does not take account of gaps.
        """
        return np.percentile(np.abs(np.diff(self.target.lightcurve['flux'])),98)

    def F8(self):
        """
        'Flicker' on 8-hour timescale. Calculated through std around an 8-hour moving average.
        """
        return Scatter(16,cut_outliers=True) #16points is 8 hours

    def CDPP_6(self):
        """
        CDPP on 6-hour timescale. Calculated through std around a 6-hour moving average.
        """
        return Scatter(12,cut_outliers=True)  #12 for 6 hours

    def Peak_to_peak(self):
        flux = self.target.lightcurve['flux']
        return np.percentile(flux,98)-np.percentile(flux,2)

    def std_ov_error(self):
        """
        STD over mean error. Measures significance of variability
        """
        return np.std(self.target.lightcurve['flux'])/np.mean(self.target.lightcurve['error'])

    def MAD(self):
        """
        Median Average Deviation
        """
        flux = self.target.lightcurve['flux']
        mednorm = np.median(flux)
        return 1.4826 * np.median(np.abs(flux - mednorm))

   # def PontRedNoise(self,cut_outliers=False):
   #     lc = self.target.lightcurve
   
   #     if cut_outliers:
   #         time_cut, flux_cut = CutOutliers(lc['time'],lc['flux'])
   #     else:
#            time_cut = lc['time']
#            flux_cut = lc['flux']
   #     
   #     #interpolate gaps
   #     interp_time,interp_flux = FillGaps_Linear(time_cut,flux_cut)
   # 
   #     sigmas = []
   # 
    #    for window in range(29)+1:  #up to 15 hours
    #        #Do smoothing
    #        flux_smooth = MovingAverage(interp_flux,window)
    #        flux_smooth = flux_smooth[window/2:-window/2]
    #        interp_smooth = interpolate.interp1d(interp_time[window/2:-window/2],flux_smooth,kind='linear',fill_value='extrapolate')
    #        sigmas.append(np.std(flux_cut - interp_smooth(time_cut)))
    #
    #    sigmas = np.array(sigmas)
    #
    #    p.figure(1)
    #    p.clf()
    #    p.plot(range(29)+1,sigmas,'b')
    #
    #    whtnoise = sigmas[0] * 1./np.sqrt(np.arange(29)+1)
    #    p.plot(range(29)+1,whtnoise,'g')
    #    p.show()

    def Scatter(self,window,cut_outliers=False):
        """
        STD around a smoothed lightcurve.
     
        Inputs:
        window  --  number of datapoints to smooth over
        cut_outliers  --  remove outliers before processing. Uses the more conservative of a 98th percentile or 5*MAD clipping threshold.
        """
        lc = self.target.lightcurve
    
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

    def CutOutliers(self,time,flux):
        threshold = np.percentile(flux,[2,98])
        mednorm = np.median(flux)
        MAD_calc = 1.4826 * np.median(np.abs(flux - mednorm))
        sigmathreshold = [5*MAD_calc + mednorm,mednorm - 5*MAD_calc]
        if threshold[1] > sigmathreshold[1]:  #takes the least active cut option
            cut = (flux<threshold[1])&(flux>threshold[0])
        else:
            cut = (flux<sigmathreshold[1])&(flux>sigmathreshold[0])

        return time[cut],flux[cut]

    def MovingAverage(self,interval, window_size): #careful with start and end. Also requires regular grid.
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    def FillGaps_Linear(self,time,flux):       
        cadence = np.median(np.diff(time))
        npoints = np.floor(time[-1]-time[0]/cadence)
        interp_times = np.arange(npoints)*cadence
        interp_obj = interpolate.interp1d(time,flux,kind='linear')
        interp_flux = interp_obj(interp_times)
        return interp_times,interp_flux


    def phasefold(self,time,per):
        return np.mod(time,per)/per
    
    def BinPhaseLC(self,phaselc,nbins):
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
