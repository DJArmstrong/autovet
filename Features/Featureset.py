import PeriodLS
import TransitFit
import TransitSOM as TSOM
import numpy as np
from scipy import interpolate,stats
import utils
import sys
import os
import pylab as p
p.ion()

class Featureset(object):

    def __init__(self,Candidate,useflatten=False,testplots=False):
        """
        Calculate features for a given candidate or candidate list.
        
        Arguments:
        Candidate   -- instance of Candidate class.
        useflatten  -- bool, true to use flattened lightcurve for scatter based features
        """
        self.features = {}
        self.target = Candidate
        self.useflatten = useflatten
        self.periodls = None
        self.sphotarray = np.array([0])
        self.tsfresh = None
        self.som = None
        self.SOMarray = None
        self.__somlocation__ = os.path.join(os.getcwd(),'Features/TransitSOM/')
        self.secondary = None
        self.transitfit = None
        self.trapfit = None
        self.eventrapfit = None
        self.oddtrapfit = None
        self.evenlc = None
        self.eventransitfit = None
        self.oddlc = None
        self.oddtransitfit = None
        self.fit_initialguess = np.array([self.target.candidate_data['per'],self.target.candidate_data['t0'],10.,0.005])
        self.trapfit_initialguess = np.array([self.target.candidate_data['t0'],self.target.candidate_data['tdur']*0.9/self.target.candidate_data['per'],self.target.candidate_data['tdur']/self.target.candidate_data['per'],0.005])
        self.testplots = testplots
        if self.testplots:
            import pylab as p
            p.ion()
        
    def CalcFeatures(self,featuredict={}):
        """
        User facing function to calculate features and populate features dict.
        
        Inputs:
        featuredict -- dict of features to calculate. {} to use observatory defaults. Feature name should be key, value should be necessary arguments (will often be empty list)

        Results go in self.features -- a dict containing all calculated features
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
            flag = 0
            if len(featuredict[featurename])>0:  #avoid recalculating features
                keys = [featurename+str(x) for x in featuredict[featurename]]
                for key in keys:
                    if key not in self.features.keys():
                        flag = 1  #will recalculate all if one entry is new
            else:
                if featurename not in self.features.keys():  
                    flag = 1
            if flag:
                try:
                    feature = getattr(self,featurename)(featuredict[featurename])
                except:  #any errors lead to populating the feature with -10. Very catch-all, but worth it for the large numbers this is used on.
                    if len(featuredict[featurename])>0:
                        feature = np.zeros(len(keys)) -10
                    else:
                        feature = -10 
                if len(featuredict[featurename])>0:
                    for k,key in enumerate(keys):
                        self.features[key] = feature[k]
                else:
                    self.features[featurename] = feature   
            
        if self.testplots: #allows pause for plots to load
            p.pause(5)
            raw_input('Press return to continue')   

    def Writeout(self,activefeatures):
        featurelist = [self.features[a] for a in activefeatures]
        return self.target.id,self.target.label, featurelist

    def ReadIn(self,featurearray,featurenames):
        for feature,featurename in zip(featurearray,featurenames):
            self.features[featurenames] = feature

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

    def SOM_Theta1(self,args):
        """
        SOM Theta1 statistic from Armstrong et al 2017. Uses pre-trained SOMs, produces statistic ranking candidate's transit shape.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f
        else:
            lc = self.target.lightcurve
        if self.target.obs == 'Kepler':
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'snrcut_30_lr01_300_20_20_bin50.txt'),20,20,50,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=50)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=0,case=1,flocation=self.__somlocation__)
            
        elif self.target.obs == 'K2':
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'k2all_lr01_500_8_8_bin20.txt'),8,8,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=1,case=1,flocation=self.__somlocation__)
            
        elif (self.target.obs == 'NGTS') or (self.target.obs == 'NGTS_synth'):
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'NGTSOM_20_20_100_0.1.txt'),20,20,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=2,case=1,flocation=self.__somlocation__)
        if type(planet_prob)=='numpy.ndarray':  #the faked extra transit from a different SOM function was present
            planet_prob = planet_prob[0]
        if self.testplots:
            p.figure()
            if len(self.SOMarray.shape)!=1:
                SOMtoplot = self.SOMarray[0,:]
            else:
                SOMtoplot = self.SOMarray
            p.plot(np.arange(len(SOMtoplot)),SOMtoplot,'b.')
            p.title('SOMarray')
            print 'SOM_Stat: '+str(planet_prob)
            print self.SOMarray
            print 'lc:'
            print lc
        return planet_prob

    def SOM_Theta2(self,args):
        """
        SOM Theta2 statistic from Armstrong et al 2017. Uses pre-trained SOMs, produces statistic ranking candidate's transit shape.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f
        else:
            lc = self.target.lightcurve
        if self.target.obs == 'Kepler':
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'snrcut_30_lr01_300_20_20_bin50.txt'),20,20,50,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=50)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=0,case=2,flocation=self.__somlocation__)
            
        elif self.target.obs == 'K2':
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'k2all_lr01_500_8_8_bin20.txt'),8,8,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=1,case=2,flocation=self.__somlocation__)
            
        elif (self.target.obs == 'NGTS') or (self.target.obs == 'NGTS_synth'):
            if self.som is None:
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'NGTSOM_20_20_100_0.1.txt'),20,20,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)
            planet_prob = TSOM.TSOM.ClassifyPlanet(self.SOMarray,self.SOMerror,missionflag=2,case=2,flocation=self.__somlocation__)
        if type(planet_prob)=='numpy.ndarray':  #the faked extra transit from a different SOM function was present
            planet_prob = planet_prob[0]
        if self.testplots:
            p.figure()
            if len(self.SOMarray.shape)!=1:
                SOMtoplot = self.SOMarray[0,:]
            else:
                SOMtoplot = self.SOMarray
            p.plot(np.arange(len(SOMtoplot)),SOMtoplot,'b.')
            p.title('SOMarray')
            print 'SOM_Stat: '+str(planet_prob)
            print self.SOMarray
            print 'lc:'
            print lc
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
            elif (self.target.obs == 'NGTS') or (self.target.obs == 'NGTS_synth'):
                self.som = TSOM.TSOM.LoadSOM(os.path.join(self.__somlocation__,'NGTSOM_20_20_100_0.1.txt'),20,20,20,0.1)
                lc_sominput = np.array([lc['time'],lc['flux'],lc['error']]).T
                self.SOMarray,self.SOMerror = TSOM.TSOM.PrepareOneLightcurve(lc_sominput,self.target.candidate_data['per'],self.target.candidate_data['t0'],self.target.candidate_data['tdur'],nbins=20)

        #pretending we have more than 1 transit - otherwise have to rewrite bits of pymvpa
        if len(self.SOMarray.shape)==1:
            self.SOMarray = np.vstack((self.SOMarray,np.ones(len(self.SOMarray))))
            self.SOMerror = np.vstack((self.SOMerror,np.ones(len(self.SOMerror))))
        map = self.som(self.SOMarray)
        map = map[0,:]
        distance = np.sqrt(np.sum( np.power( self.SOMarray - self.som.K[map[0],map[1]] , 2 ) ))
        self.SOMarray = self.SOMarray[0,:]
        self.SOMerror = self.SOMerror[0,:]
        return distance
    
    def LSPeriod(self,args):
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        args -- [peak_number1,peak_number2,...]
        peak_number  --  which LS peak to extract (starts at 0 for largest peak). If multiple, should call this function with the largest peak_number desired first.
        
        Returns:
        [period1,period2,...] -- peak peak_number from Lomb-Scargle periodogram
        """
        if self.periodls is None: #checks it hasn't been defined before
            if self.target.obs == 'K2':
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
            else:
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        output = []
        for peak_number in args:
            self.periodls.fit(peak_number) #will only run fit if peak_number greater than any previously run. However, will run fit from start (fit once, remove peak, refit, etc) otherwise.
            output.append(self.periodls.periods[peak_number])
        if len(output)==1:
            return output[0]
        else:
            return output

    def LSAmp(self,args): 
        """
        Get dominant periods and ratio of Lomb-Scargle amplitudes for each.
        
        Inputs:
        args -- [amp_number1,amp_number2,...]
        amp_number  --  which LS peak amplitude ratio to extract
        
        Returns:
        [ampratio1,ampratio2,...] -- ratio of peak peak_number amplitude to maximum peak amplitude
        """
        if not self.periodls: #checks it hasn't been defined before
            if self.target.obs == 'K2':
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs)        
            else:
                self.periodls = PeriodLS.PeriodLS(self.target.lightcurve,observatory=self.target.obs,removethruster=False,removecadence=False)
        output = []
        for peak_number in args:
            self.periodls.fit(peak_number)
            output.append(self.periodls.ampratios[peak_number])
        if len(output)==1:
            return output[0]
        else:
            return output

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
            self.features['LSPeriod0'] = self.LSPeriod([0])
        period = self.features['LSPeriod0']
     
        phaselc2P = np.zeros([len(lc['time']),2])
        phaselc2P[:,0] = utils.phasefold(lc['time'],period*2)
        phaselc2P[:,1] = lc['flux']
        phaselc2P = phaselc2P[np.argsort(phaselc2P[:,0]),:] #now in phase order
        binnedlc2P, binstd, emptybins = utils.BinPhaseLC(phaselc2P,64)

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
        '''
        Scipy skew function on lightcurve.
        '''
        return stats.skew(self.target.lightcurve['flux'])
    
    def Kurtosis(self,args):
        '''
        Scipy Kurtosos function on lightcurve.
        '''
        return stats.kurtosis(self.target.lightcurve['flux'])
    
    def NZeroCross(self,args,window=16):
        '''
        Number of zero crossings of 16-point smoothed lightcurve.
        '''
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

    def RMS_6(self,args):
        """
        RMS on 6-hour timescale. Calculated through std around a 6-hour moving average.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f  #uses flattened lightcurve
        else:
            lc = self.target.lightcurve
        return utils.Scatter(lc,12,cut_outliers=True)  #12 for 6 hours

    def Peak_to_peak(self,args):
        '''
        98th - 2nd percentile of lightcurve flux.
        '''
        flux = self.target.lightcurve['flux']
        return np.percentile(flux,98)-np.percentile(flux,2)

    def std_ov_error(self,args):
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
        '''
        Lightcurve root-mean-square.
        '''
        return np.sqrt(np.mean(np.power(self.target.lightcurve['flux']-np.median(self.target.lightcurve['flux']),2)))

    def SPhot_mean(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                self.features['LSPeriod0'] = self.LSPeriod([0])
            P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.mean(self.sphotarray)
        
    def SPhot_median(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                self.features['LSPeriod0'] = self.LSPeriod([0])
            P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.median(self.sphotarray)
        
    def SPhot_max(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                self.features['LSPeriod0'] = self.LSPeriod([0])
            P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.max(self.sphotarray)    

    def SPhot_min(self,args):
        if len(self.sphotarray)==1:
            lc = self.target.lightcurve
            if 'LSPeriod0' not in self.features.keys():
                self.features['LSPeriod0'] = self.LSPeriod([0])
            P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        return np.max(self.sphotarray)  
    
    def Contrast(self,args):
        lc = self.target.lightcurve
        if len(self.sphotarray)==1:
            if 'LSPeriod0' not in self.features.keys():
                self.features['LSPeriod0'] = self.LSPeriod([0])
            P_rot = self.features['LSPeriod0']
            self.sphotarray = utils.SPhot(lc,P_rot)
        contrast = utils.CalcContrast(self.sphotarray,np.std(lc['flux']))
        return contrast
              
    def MaxSecDepth(self,args):
        """
        Returns maximum depth of secondary eclipse, in relative flux.
        
        Scans a box of width the transit duration over phases 0.3 to 0.7. Returns maximum significance of box relative to local phasecurve, normalised by point errors.
        """
        if self.secondary is None:
            if self.useflatten:
                lc = self.target.lightcurve_f
            else:
                lc = self.target.lightcurve 
            per = self.target.candidate_data['per']
            t0 = self.target.candidate_data['t0']
            tdur = self.target.candidate_data['tdur']
            self.secondary = utils.FindSecondary(lc,per,t0,tdur)
        #p.figure()
        #phase = utils.phasefold(self.target.lightcurve['time'],self.target.candidate_data['per'],self.target.candidate_data['t0'])
        #p.plot(phase,self.target.lightcurve['flux'],'b.')
        #p.plot([self.secondary['phase'],self.secondary['phase']],[1-self.secondary['depth'],1],'r--')
        #p.title('Secondary Test')
        #print 'Secondary Diags:'
        #print self.target.candidate_data['tdur']
        #print self.target.candidate_data['per']
        #print self.secondary
        return self.secondary['depth']

    def MaxSecPhase(self,args):
        """
        Returns phase location of maximum secondary eclipse.
        
        Scans a box of width the transit duration over phases 0.3 to 0.7. Returns maximum significance of box relative to local phasecurve, normalised by point errors.
        """
        if self.secondary is None:
            if self.useflatten:
                lc = self.target.lightcurve_f
            else:
                lc = self.target.lightcurve 
            per = self.target.candidate_data['per']
            t0 = self.target.candidate_data['t0']
            tdur = self.target.candidate_data['tdur']
            self.secondary = utils.FindSecondary(lc,per,t0,tdur)
        return self.secondary['phase']

    def MaxSecSig(self,args):
        """
        Returns significance of maximum secondary eclipse, normalised by errors.
        
        Scans a box of width the transit duration over phases 0.3 to 0.7. Returns maximum significance of box relative to local phasecurve, normalised by point errors.
        """
        if self.secondary is None:
            if self.useflatten:
                lc = self.target.lightcurve_f
            else:
                lc = self.target.lightcurve 
            per = self.target.candidate_data['per']
            t0 = self.target.candidate_data['t0']
            tdur = self.target.candidate_data['tdur']
            self.secondary = utils.FindSecondary(lc,per,t0,tdur)
        return self.secondary['significance']

    def MaxSecSelfSig(self,args):
        """
        Returns significance of maximum secondary eclipse, normalised by errors.
        
        Scans a box of width the transit duration over phases 0.3 to 0.7. Returns maximum significance of box relative to local phasecurve, normalised by point errors.
        """
        if self.secondary is None:
            if self.useflatten:
                lc = self.target.lightcurve_f
            else:
                lc = self.target.lightcurve 
            per = self.target.candidate_data['per']
            t0 = self.target.candidate_data['t0']
            tdur = self.target.candidate_data['tdur']
            self.secondary = utils.FindSecondary(lc,per,t0,tdur)
        return self.secondary['selfsignificance']

    def LSPhase_amp(self,args):
        lc = self.target.lightcurve
        if 'EBPeriod' not in self.features.keys():
            self.features['EBPeriod'] = self.EBPeriod([])
        per = self.features['EBPeriod']
        phase = utils.phasefold(lc['time'],per)
        phasedat = np.array([phase,lc['flux']]).T
        phasedat = phasedat[np.argsort(phase),:]
        if len(lc['time']) > 500:
            nbins = 200
        else:
            nbins = int(len(lc['time'])/3.)
        binflux, binerrs, emptybins = utils.BinPhaseLC(phasedat,nbins)
        return np.max(binflux[:,1]) - np.min(binflux[:,1])
        
    def LSPhase_p2pmean(self,args):
        lc = self.target.lightcurve
        if 'EBPeriod' not in self.features.keys():
            self.features['EBPeriod'] = self.EBPeriod([])
        per = self.features['EBPeriod']
        phase = utils.phasefold(lc['time'],per)
        p2p = np.diff(lc['flux'][np.argsort(phase)])
        return np.mean(p2p)

    def LSPhase_p2pmax(self,args):        
        lc = self.target.lightcurve
        if 'EBPeriod' not in self.features.keys():
            self.features['EBPeriod'] = self.EBPeriod([])
        per = self.features['EBPeriod']
        phase = utils.phasefold(lc['time'],per)
        p2p = np.diff(lc['flux'][np.argsort(phase)])
        return np.max(p2p)

    def Fit_period(self,args):  #change to useflatten option?
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.params[0]

    def Fit_t0(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.params[1]

    def Fit_aovrstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.params[2]
        
    def Fit_rprstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.params[3]

    def Fit_duration(self,args):  #assumes circular at the moment, as fit doesn't allow for varying e
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        #b = self.transitfit.params[2]*np.cos(self.transitfit.params[4])
        b = 0
        tdur = self.transitfit.params[0]/math.pi * np.arcsin(np.sqrt((1+self.transitfit.params[3])**2-b**2)/(self.transitfit.params[2]*1.))    
        return tdur

    def Fit_ingresstime(self,args):  #in the e=0, Rp<<Rstar<<a, b<<1-k limit
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        #b = self.transitfit.params[2]*np.cos(self.transitfit.params[4])
        b = 0
        return self.transitfit.params[0]/math.pi / self.transitfit.params[2] * self.transitfit.params[3]/np.sqrt(1-b**2)       

    def Fit_chisq(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.chisq

    def Fit_depthSNR(self,args): #errors are currently shoddy as anything, so don't trust this feature
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        return self.transitfit.params[3]/self.transitfit.errors[3]
         
    def Even_Fit_aovrstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.eventransitfit.params[0]
        
    def Even_Fit_rprstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.eventransitfit.params[1]

    def Even_Fit_chisq(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.eventransitfit.chisq
    
    def Even_Fit_depthSNR(self,args): #errors are currently shoddy as anything, so don't trust this feature
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.eventransitfit.params[1]/self.eventransitfit.errors[1]

    def Odd_Fit_aovrstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.oddtransitfit.params[0]
        
    def Odd_Fit_rprstar(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.oddtransitfit.params[1]

    def Odd_Fit_chisq(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.oddtransitfit.chisq
    
    def Odd_Fit_depthSNR(self,args):  #errors are currently shoddy as anything, so don't trust this feature
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.oddtransitfit.params[1]/self.oddtransitfit.errors[1]

    def Even_Odd_depthratio(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        return self.eventransitfit.params[1]/self.oddtransitfit.params[1]
                
    def Even_Odd_depthdiff_fractional(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtransitfit is None:
            self.oddtransitfit = TransitFit.TransitFit(self.oddlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')
        if self.eventransitfit is None:
            self.eventransitfit = TransitFit.TransitFit(self.evenlc,self.fit_initialguess,self.target.exp_time,sfactor=7,fixper=self.transitfit.params[0],fixt0=self.transitfit.params[1])
        #err = np.max([self.eventransitfit.errors[3],self.oddtransitfit.errors[3]])
        return np.abs(self.eventransitfit.params[1] - self.oddtransitfit.params[1])/self.transitfit.params[3]

    def RPlanet(self,args):
        if self.transitfit is None:
            self.transitfit = TransitFit.TransitFit(self.target.lightcurve,self.fit_initialguess,self.target.exp_time,sfactor=7)
        rsun = 6957000000.
        rearth = 6371000.
        return self.transitfit.params[3]*self.target.stellar_radius* rsun/rearth  #in earth radii
           
    def TransitSNR(self,args):
        per = self.target.candidate_data['per']    
        tdur = self.target.candidate_data['tdur']
        t0 = self.target.candidate_data['t0']

        if self.useflatten:
            lc = self.target.lightcurve_f
        else:
            lc = self.target.lightcurve 

        #get depth of transit (using period and data, not fit).
        binnedlc = utils.BinTransitDuration(lc,per,t0,tdur*0.8) #0.8 is to avoid ingress and egress
        binnedlc = binnedlc[binnedlc[:,1]>=0]
        transitbin = binnedlc[0,1]
        noise = np.std(binnedlc[2:-2,1]) #avoids duration bins containing ingress and egress
        if np.isnan(noise):
            return -10
        else:
            return (1-transitbin)/noise

    def SingleTransitEvidence(self,args):
        per = self.target.candidate_data['per']    
        tdur = self.target.candidate_data['tdur']
        t0 = self.target.candidate_data['t0']
        tdur_phase = tdur/per
        #print t0
        if self.useflatten:
            lc = self.target.lightcurve_f
        else:
            lc = self.target.lightcurve 
        
        phase = utils.phasefold(lc['time'],per,t0)
        transits = np.where((np.abs(np.diff(phase))>0.9)&((phase[:-1]<tdur_phase/2.)|(phase[:-1]>1-tdur_phase/2.)))[0]
        segwidth = 9  #SHOULD BE ODD. MEASURED IN TRANSIT DURATIONS
        if tdur*segwidth >= per:
            return -10
        else:
            sesratios = []
            #print 'STE_DIAG'
            #print transits
            for transit in transits:
                ttime = lc['time'][transit]
                #print ttime
                segstart = np.searchsorted(lc['time'],ttime-segwidth/2.*tdur)
                segend = np.searchsorted(lc['time'],ttime+segwidth/2.*tdur)
                lcseg = {}
                lcseg['time'] = lc['time'][segstart:segend]
                lcseg['flux'] = lc['flux'][segstart:segend]
                if segend-segstart > segwidth*3:  #implies 3 points per transit duration 
                    nbins = np.ceil((np.max(lcseg['time'])-np.min(lcseg['time']))/tdur).astype('int')
                    binnedlc = utils.BinSegment(lcseg,nbins,fill_value=1.)
                    transitidx = np.argmin(np.abs(binnedlc[:,0]-ttime))
                    transitval = binnedlc[transitidx,1]
                    binnedlc = np.ma.array(binnedlc,mask=False)
                    binnedlc.mask[transitidx]= True
                    sesratios.append((1-transitval)/np.std(binnedlc[:,1]))
                #print sesratios
                #import pylab as p
                #p.ion()
                #p.figure(1)
                #p.clf()
                #p.plot(lcseg['time'],lcseg['flux'],'b.')
                #p.plot(binnedlc[:,0],binnedlc[:,1],'r.-')
                #p.plot([t0+per*count,t0+per*count],[0.99,1.],'g-')
                #p.pause(2)
                #raw_input()
            return np.median(np.array(sesratios))
                    
    def RMS_TDur(self,args):
        """
        RMS on transit duration timescale. Calculated through std around a transit duration moving average.
        """
        if self.useflatten:
            lc = self.target.lightcurve_f  #uses flattened lightcurve
        else:
            lc = self.target.lightcurve
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])            
        tdur = self.trapfit.params[2]*self.target.candidate_data['per']
        cadence = np.median(np.diff(lc['time']))
        npoints = np.round(tdur/cadence)  
        return utils.Scatter(lc,npoints,cut_outliers=True)

    def Trapfit_t0(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        return self.trapfit.params[0]

    def Trapfit_t23phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        return self.trapfit.params[1]

    def Trapfit_t14phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        return self.trapfit.params[2]

    def Full_partial_tdurratio(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        return self.trapfit.params[1]/self.trapfit.params[2]
        
    def Trapfit_depth(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        return self.trapfit.params[3]

    def Even_Trapfit_t23phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[0]

    def Even_Trapfit_t14phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[1]

    def Even_Full_partial_tdurratio(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[0]/self.eventrapfit.params[1]
        
    def Even_Trapfit_depth(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[2]

    def Odd_Trapfit_t23phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.oddtrapfit.params[0]

    def Odd_Trapfit_t14phase(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.oddtrapfit.params[1]

    def Odd_Full_partial_tdurratio(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.oddtrapfit.params[0]/self.oddtrapfit.params[1]
        
    def Odd_Trapfit_depth(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.oddtrapfit.params[2]
    
    def Even_Odd_trapdurratio(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[1]/self.oddtrapfit.params[1]        

    def Even_Odd_trapdepthratio(self,args):
        if self.trapfit is None:
            self.trapfit = TransitFit.TransitFit(self.target.lightcurve,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'])
        if self.oddlc is None:
            self.oddlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'odd')
        if self.oddtrapfit is None:
            self.oddtrapfit = TransitFit.TransitFit(self.oddlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        if self.evenlc is None:
            self.evenlc = utils.SplitOddEven(self.target.lightcurve,self.target.candidate_data['per'],self.target.candidate_data['t0'],'even')            
        if self.eventrapfit is None:
            self.eventrapfit = TransitFit.TransitFit(self.evenlc,self.trapfit_initialguess,self.target.exp_time,sfactor=7,fittype='trap',fixper=self.target.candidate_data['per'],fixt0=self.trapfit.params[0])
        return self.eventrapfit.params[2]/self.oddtrapfit.params[2]        
        
    def PointDensity_ingress(self,args):
        '''
        Density of points in ingress/egress relative to average point density of lightcurve.
        Ingress/egress calculates as within +-10% from time implied by transit duration.
        Calculates average density ignoring empty regions of phase.
        '''
        #phase lc on planet period
        per = self.target.candidate_data['per']
        t0 = self.target.candidate_data['t0']
        tdur_phase = self.target.candidate_data['tdur']/per
        phase = utils.phasefold(self.target.lightcurve['time'],per,t0+per/2.)
        phasediffs = np.abs(phase - 0.5)
        points_in_ingress = (phasediffs<(tdur_phase/2.)*1.1) & (phasediffs>(tdur_phase/2.)*0.9)       
        density_in_ingress = np.sum(points_in_ingress)/((tdur_phase/2.)*0.2)
        nbins = np.floor(2./tdur_phase).astype('int')  #2 bins per transit duration
        phaselc = np.zeros([len(phase),2])
        phaselc[:,0] = phase
        phaselc[:,1] = self.target.lightcurve['flux']
        phaselc = phaselc[np.argsort(phaselc[:,0]),:] #now in phase order
        binnedlc, binstd, emptybins = utils.BinPhaseLC(phaselc,nbins,fill_value=-10)
        populated_fraction_of_lc = 1-np.sum(binnedlc[:,1]==-10)/nbins
        return density_in_ingress / (len(self.target.lightcurve['time'])/populated_fraction_of_lc)
        #return: range of densities?, ingress/egress density over average?, std of densities? ingress/egress density over nearby bin density?

    def PointDensity_transit(self,args):
        '''
        Density of points in transit relative to average point density of lightcurve.
        Calculates average density ignoring empty regions of phase.
        '''
        per = self.target.candidate_data['per']
        t0 = self.target.candidate_data['t0']
        tdur_phase = self.target.candidate_data['tdur']/per
        phase = utils.phasefold(self.target.lightcurve['time'],per,t0+per/2.)
        phasediffs = np.abs(phase - 0.5)
        points_in_transit = phasediffs<(tdur_phase/2.)     
        density_in_transit = np.sum(points_in_transit)/(tdur_phase)
        nbins = np.floor(1./tdur_phase).astype('int')  #1 bin per transit duration
        phaselc = np.zeros([len(phase),2])
        phaselc[:,0] = phase
        phaselc[:,1] = self.target.lightcurve['flux']
        phaselc = phaselc[np.argsort(phaselc[:,0]),:] #now in phase order
        binnedlc, binstd, emptybins = utils.BinPhaseLC(phaselc,nbins,fill_value=-10)
        populated_fraction_of_lc = 1-np.sum(binnedlc[:,1]==-10)/nbins
        return density_in_transit / (len(self.target.lightcurve['time'])/populated_fraction_of_lc)
    
    def Scatter_transit(self,args):
        '''
        Scatter of points in transit relative to average scatter of lightcurve.
        '''
        per = self.target.candidate_data['per']
        t0 = self.target.candidate_data['t0']
        tdur_phase = self.target.candidate_data['tdur']/per
        phase = utils.phasefold(self.target.lightcurve['time'],per,t0+per/2.)
        phasediffs = np.abs(phase - 0.5)
        points_in_transit = phasediffs<(tdur_phase/2.)     
        scatter_transit = np.std(self.target.lightcurve['flux'][points_in_transit])
        avg_scatter = np.std(self.target.lightcurve['flux'])
        return scatter_transit/avg_scatter
     
    def missingDataFlag(self,args):
        '''
        Fraction of empty tdur/5 width bins within 5 transit durations centred on transit, in phase
        '''
        per = self.target.candidate_data['per']
        t0 = self.target.candidate_data['t0']+per/2.  #transit at phase 0.5
        tdur_phase = self.target.candidate_data['tdur']/per
        phase = utils.phasefold(self.target.lightcurve['time'],per,t0)
        phaselc = np.zeros([len(phase),2])
        phaselc[:,0] = phase
        phaselc[:,1] = self.target.lightcurve['flux']
        phaselc = phaselc[np.argsort(phaselc[:,0]),:] #now in phase order
        resolution = 5./tdur_phase
        if resolution<40:  #means very long durations don't automatically pass this test.
            resolution = 40
        nbins = np.floor(resolution).astype('int')  #5 bins across transit duration
        binnedlc,binstd, emptybins = utils.BinPhaseLC(phaselc,nbins,fill_value=-10)
        neartransit = (binnedlc[:,0] > 0.5-5*tdur_phase/2.) & (binnedlc[:,0] < 0.5+5*tdur_phase/2.)
        return float(np.sum(binnedlc[neartransit,1]==-10))/float(np.sum(neartransit))

    def pmatch(self,args):
        '''
        Number of other candidates in this field with matching period and epoch
        '''
        per = self.target.candidate_data['per']
        plist = self.target.field_periods
        epoch = self.target.candidate_data['t0']
        elist = self.target.field_epochs
        match = (per/plist>0.99) & (per/plist<1.01) & (np.abs(epoch-elist)<3600/86400.)
        return np.sum(match)

    def ntransits(self,args):
        '''
        Number of transits with data
        '''
        per = self.target.candidate_data['per']    
        tdur = self.target.candidate_data['tdur']
        t0 = self.target.candidate_data['t0']
        tdur_phase = tdur/per
        phase = utils.phasefold(self.target.lightcurve['time'],per,t0)
        transits = np.where((np.abs(np.diff(phase))>0.9)&((phase[:-1]<tdur_phase/2.)|(phase[:-1]>1-tdur_phase/2.)))[0]
        return len(transits)
        
    def tdur_phase(self,args):
        '''
        transit width in phase
        '''
        return self.target.candidate_data['tdur']/self.target.candidate_data['per']       
         