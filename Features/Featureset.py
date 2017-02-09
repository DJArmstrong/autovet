import PeriodLS
import numpy as np
from scipy import interpolate,stats
import utils

class Featureset(object):

    def __init__(self,Candidate,useflatten=False):
        """
        Calculate features for a given candidate or candidate list.
        
        Arguments:
        Candidate   -- instance of Candidate class.
        useflatten  -- bool, true to use flattened lightcurve for scatter based features
        """
        self.features = {}
        self.target = Candidate
        self.useflatten = useflatten
        
                        
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
                print featurename
                feature = getattr(self,featurename)()
                print feature
                if type(feature)==np.ndarray or type(feature)==dict: #assumes if the feature is now an array, it's good
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
        phaselc2P[:,0] = utils.phasefold(lc['time'],period*2)
        phaselc2P = phaselc2P[np.argsort(phaselc2P[:,0]),:] #now in phase order
        binnedlc2P,binstd = utils.BinPhaseLC(phaselc2P,64)

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
    
    def NZeroCross(self,window=16):
        lc = self.target.lightcurve
        
        time_cut, flux_cut = utils.CutOutliers(lc['time'],lc['flux'])
        
        #interpolate gaps
        interp_time,interp_flux = utils.FillGaps_Linear(time_cut,flux_cut)
        
        #Do smoothing
        flux_smooth = utils.MovingAverage(interp_flux,window)
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
        return self.Scatter(16,cut_outliers=True) #16points is 8 hours

    def CDPP_6(self):
        """
        CDPP on 6-hour timescale. Calculated through std around a 6-hour moving average.
        """
        return self.Scatter(12,cut_outliers=True)  #12 for 6 hours

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

    def RMS(self):
        return np.sqrt(np.mean(np.power(self.target.lightcurve['flux']-np.median(self.target.lightcurve['flux']),2)))
    
    def SPhot(self,k=5):
        """
        S_phot,k diagnostic (see Mathur et al 2014)
        """
        
        lc = self.target.lightcurve
        if 'LSPeriod' not in self.features.keys():
            print 'Calculating LS Period first'
            self.features['LSPeriod'] = LSPeriod()
        P_rot = self.features['LSPeriod'][0,0]
        
        while k*P_rot > lc['time'][-1] - lc['time'][0]:
            k -= 1
        
        if k == 0:
            return {'SPhot_mean':-10,'SPhot_median':-10,'SPhot_max':-10,'SPhot_min':-10,'Contrast':-10,'k':0}
            
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
        contrast = utils.CalcContrast(sphot,np.std(lc['flux']))
        
        return {'SPhot_mean':np.mean(sphot),'SPhot_median':np.median(sphot),'SPhot_max':np.max(sphot),'SPhot_min':np.min(sphot),'Contrast':contrast,'k':k}
  

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
        if self.useflatten:
            lc = self.target.lightcurve_f  #uses flattened lightcurve
        else:
            lc = self.target.lightcurve
    
        if cut_outliers:
            time_cut, flux_cut = utils.CutOutliers(lc['time'],lc['flux'])
        else:
            flux_cut = lc['flux']
            time_cut = lc['time']
        
        #interpolate gaps
        interp_time,interp_flux = utils.FillGaps_Linear(time_cut,flux_cut)
    
        #Do smoothing
        flux_smooth = utils.MovingAverage(interp_flux,window)
        flux_smooth = flux_smooth[window/2:-window/2]
        interp_smooth = interpolate.interp1d(interp_time[window/2:-window/2],flux_smooth,kind='linear',fill_value='extrapolate')

        return np.std(flux_cut - interp_smooth(time_cut))

