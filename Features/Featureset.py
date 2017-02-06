import PeriodLS
import numpy as np

class FeatureSet(object):

    def __init__(self,Candidate,observatory):
        """
        Calculate features for a given candidate or candidate list.
        
        Arguments:
        Candidate   -- instance of Candidate class.
        observatory -- source of candidate. Accepted values are: [NGTS,Kepler,K2]
        """
        self.features = {}
        self.target = Candidate
        
                        
    def CalcFeatures(self,featurelist=None):
        """
        User facing function to calculate features and populate features dict.
        
        Inputs:
        featurelist -- list of features to calculate. None to use observatory defaults.
        """
        if not featurelist:
            if self.target.obs == 'NGTS':
                self.featurelist = []
            elif self.target.obs=='Kepler' or self.target.obs=='K2':
                self.featurelist = []
            else:
                print 'Observatory not supported, please input desired feature list'

        for featurename in featurelist:
            if featurename not in self.features.keys:  #avoid recalculating features
                feature = FeatureSet.featurename()  #NEED TO TURN STRING INTO FUNCTION NAME HERE
                if feature:   #if function failed, should be None
                    self.features[featurename] = feature

    def GetPeriod(self):
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        lc   -- numpy array, column 0 time, column 1 flux
        
        Returns:
        period -- first 10 peaks from Lomb-Scargle periodogram
        ampratios -- ratio of each of first 10 peaks amplitude to maximum peak amplitude
        """
        if self.target.obs == 'K2':
            a = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
        else:
            a = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        a.fit()
        periods = a.periods
        ampratios = a.ampratios
        return periods,ampratios

    def EBtest(self):
        """
        Tests for phase variation at double the period to correct EB periods.
        
        Inputs:
        lc   -- numpy array, column 0 time, column 1 flux
        period -- period to test
        
        Returns:
        corrected period, either initial period or double      
        """
        lc = self.target.lightcurve
        period = self.features['GetPeriod']
        phaselc2P = lc.copy()
        phaselc2P[:,0] = FeatureSet.phasefold(lc[:,0],period*2)
        phaselc2P = phaselc2P[np.argsort(phaselc2P[:,0]),:] #now in phase order
        binnedlc2P,binstd = FeatureSet.BinPhaseLC(phaselc2P,64)

        minima = np.argmin(binnedlc2P[:,1])
        posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
        posssecondary = np.where((posssecondary<0.05) | (posssecondary > 0.95))[0]   #within 0.05 either side of phase 0.5 from minima

        pointsort = np.sort(lc[:,1])
        top10points = np.median(pointsort[-30:])
        bottom10points = np.median(pointsort[:30])
        
        if self.target.obs=='K2':
            if lc[-1,0]-lc[0,0] >= 60:
                periodlim= 20.
            else: #for C0
                periodlim = 10.
        else:
            periodlim = 100000. #no effective limit
            
        if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.03*(top10points-bottom10points) and period*2<=periodlim:  
            return period * 2
        else:
            return period

    def phasefold(time,per):
        return np.mod(time,per)/per
    
    def BinPhaseLC(lc,nbins):
        bin_edges = np.arange(nbins)/float(nbins)
        bin_indices = np.digitize(lc[:,0],bin_edges) - 1
        binnedlc = np.zeros([nbins,2])
        binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
        binnedstds = np.zeros(nbins)
        for bin in range(nbins):
            if np.sum(bin_indices==bin) > 0:
                binnedlc[bin,1] = np.mean(lc[bin_indices==bin,1])  #doesn't make use of sorted phase array, could probably be faster?
                binnedstds[bin] = np.std(lc[bin_indices==bin,1])
            else:
                binnedlc[bin,1] = np.mean(lc[:,1])  #bit awkward this, but only alternative is to interpolate?
                binnedstds[bin] = np.std(lc[:,1])
        return binnedlc,binnedstds
