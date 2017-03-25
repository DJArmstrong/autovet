import PeriodLS
import TransitSOM as TSOM
import numpy as np
from scipy import interpolate,stats
import utils
import sys
import os

class Featureset(object):

    def __init__(self,Candidate,useflatten=False,sompath=None):
        """
        Calculate features for a given candidate or candidate list.
        
        Arguments:
        Candidate   -- instance of Candidate class.
        useflatten  -- bool, true to use flattened lightcurve for scatter based features
        sompath     -- string, filepath to saved SOM Kohonen Layer. Can be ignored if no SOM features will be used.
        """
        self.features = {}
        self.target = Candidate
        self.useflatten = useflatten
        self.periodls = 0
        self.sphotarray = np.array([0])
        self.tsfresh = None
        self.sompath = sompath
        self.som = None
        self.SOMarray = None
        self.__somlocation__ = os.path.join(os.getcwd(),'Features/TransitSOM/')
                        
    def CalcFeatures(self,featuredict={}):
        """
        User facing function to calculate features and populate features dict.
        
        Inputs:
        featuredict -- dict of features to calculate. {} to use observatory defaults. Feature name should be key, value should be necessary arguments (will often be empty list)

        Returns
        self.features -- a dict containing all calculated features
        """
        if len(featuredict.keys())==0:
            if self.target.obs == 'NGTS':
                self.featuredict = {} #should be default for NGTS
            elif self.target.obs=='Kepler':
                self.featuredict = {} #should be default for Kepler
            elif self.target.obs=='K2':
                self.featuredict = {} #should be default for K2
            else:
                print 'Observatory not supported, please input desired feature list'

        for featurename in featuredict.keys():
            if len(featuredict[featurename])>0:
                testkey = featurename+str(featuredict[featurename][0])
            else:
                testkey = featurename
            if testkey not in self.features.keys():  #avoid recalculating features
                feature = getattr(self,featurename)(featuredict[featurename])
                self.features[testkey] = feature

    def Writeout(self,activefeatures):
        featurelist = [self.features[a] for a in activefeatures]
        return self.target.id,self.target.label, featurelist

    def TSFresh(self,args):
        """
        Calculates the complete set of tsfresh features (all generic timeseries features).
        
        The number of features TSFresh returns seems to be somewhat variable. Care may have to be taken with the rarer ones.
        
        Set up to run on one candidate, but could be changed to run on an entire field/set at once.
        """
        import tsfresh
        from tsfresh import extract_features
        import pandas as pd
        
        if self.tsfresh is None:  #will only calculate set the first time
            tsinput = pd.DataFrame(self.target.lightcurve)
            tsinput['index'] = np.ones(len(self.target.lightcurve['time']))
            self.tsfresh = extract_features(tsinput,column_id='index',column_sort='time',column_value='flux')

        return self.tsfresh[args[0]][1.0]

    def SOM_Stat(self,args):
        """
        SOM Theta2 statistic from Armstrong et al 2017. Uses pre-trained SOMs, produces statistic ranking candidate's transit shape.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f
        else:
            lc = self.target.lightcurve
        if self.target.obs == 'Kepler':
            self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'snrcut_30_lr01_300_20_20_bin50.txt'),20,20,50,0.1)
            if self.SOMarray is None:
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=50)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=0,case=2,flocation=self.__somlocation__)
            
        elif self.target.obs == 'K2':
            self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'k2all_lr01_500_8_8_bin20.txt'),8,8,20,0.1)
            if self.SOMarray is None:
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=1,case=2,flocation=self.__somlocation__)
            
        elif self.target.obs == 'NGTS':
            if self.som is None:
                if self.sompath is None:
                    print 'Need sompath to be set'
                    return -10
                else:      
                    lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                    self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
                    self.som = TSOM.TSOM.LoadSOM(self.sompath,20,20,20) #limited to 20x20 SOMs, with 20 points binned acros the transit, for NGTS
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,som=self.som,case=2,flocation=self.__somlocation__)
            
        return planet_prob
        
    def SOM_Distance(self,args):
        """
        Euclidean distance of transit shape from closest matching point on SOM.
        """
        if self.som is None:
            if self.useflatten:
                lc = self.target.lightcurve_f
            else:
                lc = self.target.lightcurve
            if self.target.obs=='Kepler':
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'snrcut_30_lr01_300_20_20_bin50.txt'),20,20,50,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=50)
            elif self.target.obs=='K2':
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'k2all_lr01_500_8_8_bin20.txt'),8,8,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            elif self.target.obs=='NGTS':
                if self.sompath is None:
                    print 'Need sompath to be set'
                    return -10
                else:
                    self.som = TSOM.TSOM.LoadSOM(self.sompath,20,20,20) #limited to 20x20 SOMs, with 20 points binned acros the transit.
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)

        #pretending we have more than 1 transit - otherwise have to rewrite bits of pymvpa
        self.SOMarray = np.vstack((self.SOMarray,np.ones(len(self.SOMarray))))
        self.SOMerrors = np.vstack((self.SOMerrors,np.ones(len(self.SOMerrors))))
        map = self.som(self.SOMarray)
        map = map[0,:]
        distance = np.sqrt(np.sum( np.power( self.SOMarray - self.som.K[map[0],map[1]] , 2 ) ))
        return distance

    def SOM_IsRamp(self,args):
        """
        Boolean, does candidate's transit shape match the SOM pixels corresponding to ramps. NGTS only.
        """
        if self.target.obs != 'NGTS':
            print 'Only valid for NGTS'
            return -10
        else:
            if self.som is None:
                if self.useflatten:
                    lc = self.target.lightcurve_f
                else:
                    lc = self.target.lightcurve
                if self.sompath is None:
                    print 'Need sompath to be set'
                    return -10
                else:
                    self.som = TSOM.TSOM.LoadSOM(self.sompath,20,20,20) #limited to 20x20 SOMs, with 20 points binned acros the transit.
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)

            #pretending we have more than 1 transit - otherwise have to rewrite bits of pymvpa
            self.SOMarray = np.vstack((self.SOMarray,np.ones(len(self.SOMarray))))
            self.SOMerrors = np.vstack((self.SOMerrors,np.ones(len(self.SOMerrors))))
            map = self.som(self.SOMarray)
            map = map[0,:]
            flag = (map[0]==1 and map[1]==4) or (map[0]==4 and map[1]==4)
            return int(flag)
            
    def SOM_IsVar(self,args):
        """
        Boolean, does candidate's transit shape match the SOM pixel corresponding to periodic variables. NGTS only.
        """
        if self.target.obs != 'NGTS':
            print 'Only valid for NGTS'
            return -10
        else:
            if self.som is None:
                if self.useflatten:
                    lc = self.target.lightcurve_f
                else:
                    lc = self.target.lightcurve
                if self.sompath is None:
                    print 'Need sompath to be set'
                    return -10
                else:
                    self.som = TSOM.TSOM.LoadSOM(self.sompath,20,20,20) #limited to 20x20 SOMs, with 20 points binned acros the transit.
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)

            #pretending we have more than 1 transit - otherwise have to rewrite bits of pymvpa
            self.SOMarray = np.vstack((self.SOMarray,np.ones(len(self.SOMarray))))
            self.SOMerrors = np.vstack((self.SOMerrors,np.ones(len(self.SOMerrors))))
            map = self.som(self.SOMarray)
            map = map[0,:]
            flag = (map[0]==11 and map[1]==19)
            return int(flag)
                
    
    def LSPeriod(self,args):
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        args -- [peak_number]
        peak_number  --  which LS peak to extract (starts at 0 for largest peak). If multiple, should call this function with the largest peak_number desired first.
        
        Returns:
        period -- peak peak_number from Lomb-Scargle periodogram
        """
        peak_number = args[0]
        if not self.periodls: #checks it hasn't been defined before
            if self.target.obs == 'K2':
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
            else:
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        self.periodls.fit(peak_number) #will only run fit if peak_number greater than any previously run. However, will run fit from start (fit once, remove peak, refit, etc) otherwise.
        period = self.periodls.periods[peak_number]
        return period

    def LSAmp(self,args): 
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        args -- [amp_number]
        peak_number  --  which LS peak to extract
        
        Returns:
        ampratio -- ratio of peak peak_number amplitude to maximum peak amplitude
        """
        peak_number = args[0]
        if not self.periodls: #checks it hasn't been defined before
            if self.target.obs == 'K2':
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
            else:
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        self.periodls.fit(peak_number)
        ampratio = self.periodls.ampratios[peak_number]
        return ampratio

    def EBPeriod(self,args):
        """
        Tests for phase variation at double the period to correct EB periods.
        
        Inputs:
        period -- period to test. Currently uses LSPeriod0
        
        Returns:
        corrected period, either initial period or double      
        """
        lc = self.target.lightcurve
        if 'LSPeriod0' not in self.features.keys():
            print 'Calculating LS Period first'
            self.features['LSPeriod0'] = LSPeriod([0])
        period = self.features['LSPeriod0']
     
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

    def Skew(self,args):
        return stats.skew(self.target.lightcurve['flux'])
    
    def Kurtosis(self,args):
        return stats.kurtosis(self.target.lightcurve['flux'])
    
    def NZeroCross(self,args,window=16):
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

    def P2P_mean(self,args):
        """
        Mean point-to-point difference across flux. Does not take account of gaps.
        """
        return np.mean(np.abs(np.diff(self.target.lightcurve['flux'])))

    def P2P_98perc(self,args):
        """
        98th percentile of point-to-point difference across flux. Does not take account of gaps.
        """
        return np.percentile(np.abs(np.diff(self.target.lightcurve['flux'])),98)

    def F8(self,args):
        """
        'Flicker' on 8-hour timescale. Calculated through std around an 8-hour moving average.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f  #uses flattened lightcurve
        else:
            lc = self.target.lightcurve
        return utils.Scatter(lc,16,cut_outliers=True) #16points is 8 hours

    def CDPP_6(self,args):
        """
        CDPP on 6-hour timescale. Calculated through std around a 6-hour moving average.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f  #uses flattened lightcurve
        else:
            lc = self.target.lightcurve
        return utils.Scatter(lc,12,cut_outliers=True)  #12 for 6 hours

    def Peak_to_peak(self,args):
        flux = self.target.lightcurve['flux']
        return np.percentile(flux,98)-np.percentile(flux,2)

    def std_ov_error(self):
        """
        STD over mean error. Measures significance of variability
        """
        return np.std(self.target.lightcurve['flux'])/np.mean(self.target.lightcurve['error'])

    def MAD(self,args):
        """
        Median Average Deviation
        """
        flux = self.target.lightcurve['flux']
        mednorm = np.median(flux)
        return 1.4826 * np.median(np.abs(flux - mednorm))

    def RMS(self,args):
        return np.sqrt(np.mean(np.power(self.target.lightcurve['flux']-np.median(self.target.lightcurve['flux']),2)))

    def SPhot_mean(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                print 'Calculating LS Period first'
                self.features['LSPeriod0'] = self.LSPeriod([0])
                P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.mean(self.sphotarray)
        
    def SPhot_median(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                print 'Calculating LS Period first'
                self.features['LSPeriod0'] = self.LSPeriod([0])
                P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.median(self.sphotarray)
        
    def SPhot_max(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                print 'Calculating LS Period first'
                self.features['LSPeriod0'] = self.LSPeriod([0])
                P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.max(self.sphotarray)    

    def SPhot_min(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                print 'Calculating LS Period first'
                self.features['LSPeriod0'] = self.LSPeriod([0])
                P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.max(self.sphotarray)  
    
    def Contrast(self,args):
        lc = self.target.lightcurve
        if len(self.sphotarray)==1:
            if 'LSPeriod0' not in self.features.keys():
                print 'Calculating LS Period first'
                self.features['LSPeriod0'] = self.LSPeriod([0])
                P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        contrast = utils.CalcContrast(self.sphotarray,np.std(lc['flux']))
        return contrast
              


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

