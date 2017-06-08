# This file is part of the TransitSOM code accompanying the paper Armstrong et al 2016
# Copyright (C) 2016 David Armstrong
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
from types import *

def PrepareLightcurves(filelist,periods,t0s,tdurs,nbins=50,clip_outliers=5):
    """
        Takes a list of lightcurve files and produces a SOM array suitable for classification or training. Single lightcurves should use PrepareOneLightcurve().
    
            Args:
                filelist: List of lightcurve files. Files should be in format time (days), flux, error. 
                periods: Array of orbital periods in days, one per input file.
                t0s: Array of epochs in days, one per input file
                tdurs: Transit durations (T14) in days, one per input file. It is important these are calculated directly from the transit - small errors due to ignoring eccentricity for example will cause bad classifications. 
                nbins: Number of bins to make across 3 transit duration window centred on transit. Empty bins will be interpolated linearly. (int)
                clip_outliers: if non-zero, data more than clip_outliers*datapoint error from the bin mean will be ignored.
                
            Returns:
                SOMarray: Array of binned normalised lightcurves
                SOMarray_errors: Array of binned lightcurve errors
                som_ids: The index in filelist of each lightcurve file used (files with no transit data are ignored)
    """
    try:
        import somutils
    except ImportError:
        print 'Accompanying libraries not in PYTHONPATH or current directory'
        return 0,0,0
    
    try:
        assert len(filelist)==len(periods)==len(t0s)==len(tdurs)
    except AssertionError:
        print 'Filelist, periods, epochs and transit duration arrays must be 1D arrays or lists of the same size'
        return 0,0,0
    
    try:
        assert nbins>0
        assert type(nbins) is IntType
    except AssertionError:
        print 'nbins must be positive integer'

        
    SOMarray_bins = []
    som_ids = []
    SOMarray_binerrors = []
    for i,infile in enumerate(filelist):
        print 'Preparing '+infile
        if os.path.exists(infile):
            try:
                lc = np.genfromtxt(infile)
                lc = lc[~np.isnan(lc[:,1]),:]

                #get period, T0, Tdur
                per = float(periods[i])
                t0 = float(t0s[i])
                tdur = float(tdurs[i])
                
                SOMtransit_bin,binerrors = PrepareOneLightcurve(lc,per,t0,tdur,nbins,clip_outliers)         

                #append to SOMarray:
                SOMarray_bins.append(SOMtransit_bin)
                som_ids.append(i)
                SOMarray_binerrors.append(binerrors)
            except:
                print 'Error loading or binning '+infile
                print 'Skipping '+infile   
        else:
            print 'File not found: '+infile 
                
    SOMarray = np.array(SOMarray_bins)
    SOMarray_errors = np.array(SOMarray_binerrors)
    return SOMarray, SOMarray_errors, np.array(som_ids)


def PrepareOneLightcurve(lc,per,t0,tdur,nbins=50,clip_outliers=5):
    """
        Takes one lightcurve array and bins it to format suitable for classification.
    
            Args:
                lc: Lightcurve array. Columns should be time (days), flux, error. Nans should be removed prior to calling function. 
                per: Orbital period in days (float)
                t0: Epoch in days (float)
                tdur: Transit duration (T14) in days (float). It is important this is calculated directly from the transit - small errors due to ignoring eccentricity for example will cause bad classifications. 
                nbins: Number of bins to make across 3 transit duration window centred on transit. Empty bins will be interpolated linearly. (int)
                clip_outliers: if non-zero, data more than clip_outliers*datapoint error from the bin mean will be ignored.
                
            Returns:
                SOMtransit_bin: Binned normalised lightcurve
                binerrors: Binned lightcurve errors
    """
    
    try:
        import somutils
    except ImportError:
        print 'Accompanying libraries not in PYTHONPATH or current directory'
        return 0,0,0
    
    try:
        assert nbins>0
        assert type(nbins) is IntType
    except AssertionError:
        print 'nbins must be positive integer'

    
    #phase fold (transit at 0.5)
    phase = somutils.phasefold(lc[:,0],per,t0-per*0.5)
    idx = np.argsort(phase)
    lc = lc[idx,:]
    phase = phase[idx]

    #cut to relevant region
    tdur_phase = tdur/per
    lowidx = np.searchsorted(phase,0.5-tdur_phase*1.5)
    highidx = np.searchsorted(phase,0.5+tdur_phase*1.5)
    lc = lc[lowidx:highidx,:]
    phase = phase[lowidx:highidx]
    bin_edges = np.linspace(0.5-tdur_phase*1.5,0.5+tdur_phase*1.5,nbins+1)
    bin_edges[-1]+=0.0001                               #avoids edge problems
    
    #perform binning
    if len(lc[:,0]) != 0:
        binphases,SOMtransit_bin,binerrors = somutils.GetBinnedVals(phase,lc[:,1],lc[:,2],lc[:,2],nbins,bin_edges,clip_outliers=clip_outliers)

    #normalise arrays, and interpolate nans where necessary
    SOMarray_single,SOMarray_errors_single = somutils.PrepareArray(SOMtransit_bin,binerrors)

    return SOMarray_single,SOMarray_errors_single
        
  

def ClassifyPlanet(SOMarray,SOMerrors,n_mc=1000,som=None,groups=None,missionflag=0,case=1,map_all=np.zeros([5,5,5])-1,flocation=''):
    """
        Produces Theta1 or Theta2 statistic to classify transit shapes using a SOM.
        Either uses the trained SOM from Armstrong et al (2016), or a user-provided SOM object.
        If using Armstrong et al (2016) SOM, only Kepler or K2 transits can be classified.
    
            Args:
                SOMarray: Array of normalised inputs (e.g. binned transits), of shape [n_inputs,n_bins]. n_inputs > 1
                SOMerrors: Errors corresponding to SOMarray values. Must be same shape as SOMarray.
                n_mc: Number of Monte Carlo iterations, default 1000. Must be positive int >=1.
                som: Trained som object, like output from CreateSOM(). Optional. If not provided, previously trained SOMs will be used.
                groups: Required if case=1 and user som provided. Array of ints, one per SOMarray row. 0 for planets, 1 for false positives, 2 for candidates.
                missionflag: 0 for Kepler, 1 for K2. Ignored if user som provided.        
                case: 1 or 2. 1 for Theta1 statistic, 2 for Theta2.
                map_all: previously run output of somtools.MapErrors_MC(). Will be run if not provided
    
            Returns:
                planet_prob: Array of Theta1 or Theta2 statistic, one entry for each row in SOMarray.
    """
    try:
        import somtools
        import selfsom
    except ImportError:
        print 'Accompanying libraries not in PYTHONPATH or current directory'
        return 0
    
    try:
        assert SOMarray.shape==SOMerrors.shape, 'Error array must be same shape as input array.'
        assert n_mc>=1, 'Number of Monte Carlo iterations must be >= 1.'
        assert (missionflag==0) or (missionflag==1) or (som!=None), 'If no user-defined SOM, missionflag must be 0 (Kepler) or 1 (K2).'
        if case ==1:
            if som!=None:
                assert groups!=None, 'For Case = 1 and user-defined SOM, groups array must be provided.'
        assert case==1 or case==2, 'Case must be 1 or 2.'
    except AssertionError as error:
        print error
        print 'Inputs do not meet requirements. See help.'
        
    #if no SOM, load our SOM (kepler or k2 depending on keplerflag)
    if not som:
        selfflag = 1
        if missionflag == 0:    
            som = LoadSOM(os.path.join(flocation,'snrcut_30_lr01_300_20_20_bin50.txt'),20,20,50,0.1)      
        else:
            som = LoadSOM(os.path.join(flocation,'k2all_lr01_500_8_8_bin20.txt'),8,8,20,0.1)
    else:
        selfflag = 0
    
    #check whether we are dealing with one or many transits
    singleflag = 0
    if len(SOMarray.shape)==1:
        singleflag = 1
        #pretend we have two transits - simpler than rewriting PyMVPA's SOM code
        SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray))))
        SOMerrors = np.vstack((SOMerrors,np.ones(len(SOMerrors))))
        
    #apply SOM
    print 'Mapping transit(s) to SOM'
    mapped = som(SOMarray)

    #map_all results
    if (map_all<0).all():
        map_all = somtools.MapErrors_MC(som,SOMarray,SOMerrors,n_mc)
    print 'Transit(s) mapped'

    #classify (depending on case)
    if case==1:
        print 'Case 1: Loading or determining pixel proportions'
        if selfflag:  #load pre calculated proportions
            if missionflag==0:
                prop = somtools.KohonenLoad(os.path.join(flocation,'prop_snrcut_all_kepler.txt'))
                prop_weights = np.genfromtxt(os.path.join(flocation,'prop_snrcut_all_weights_kepler.txt'))
            else:
                prop = somtools.KohonenLoad(os.path.join(flocation,'prop_all_k2.txt'))
                prop_weights = np.genfromtxt(os.path.join(flocation,'prop_all_weights_k2.txt'))
                
        
        else:  #create new proportions
            prop ,prop_weights= somtools.Proportions(som.K,mapped,groups,2,som.K.shape[0],som.K.shape[1])
        
        print 'Case 1: Beginning classification'
        class_probs = somtools.Classify(map_all,prop,2,prop_weights) 
        planet_prob = class_probs[:,0] / np.sum(class_probs,axis=1)
 
    else:
        print 'Case 2: Loading or determining pixel distances'   
        if selfflag:
            if missionflag==0:
                testdistances = somtools.KohonenLoad(os.path.join(flocation,'testdistances_kepler.txt'))
            else:
                testdistances = somtools.KohonenLoad(os.path.join(flocation,'testdistances_k2.txt'))
            pcols = (0,3)
            fpcols = (1,2,4)
        elif missionflag==2:
            testdistances = somtools.KohonenLoad(os.path.join(flocation,'testdistances_NGTS.txt'))        
            pcols = 1
            fpcols = 0
        else:
            SOMarray_PDVM = np.genfromtxt(os.path.join(flocation,'SOMarray_PDVM_perfect_norm.txt'))
            groups_PDVM = np.genfromtxt(os.path.join(flocation,'groups_PDVM_perfect_norm.txt'))
            lowbound = np.floor(SOMarray.shape[1]/3).astype('int')-1
            testdistances = somtools.PixelClassifier(som.K,SOMarray_PDVM,groups_PDVM,6,lowbound=lowbound,highbound=2*lowbound+3)
            pcols = (0,3)
            fpcols = (1,2,4)
            
        print 'Case 2: Beginning classification'
        planet_prob,class_power = somtools.Classify_Distances(map_all,testdistances,pcols=pcols,fpcols=fpcols)

    print 'Classification complete'
    
    #remove faked transit result
    if singleflag:
        planet_prob = planet_prob[0]
            
    return planet_prob
    
    
def CreateSOM(SOMarray,niter=500,learningrate=0.1,learningradius=None,somshape=(20,20),outfile=None):
    """
        Trains a SOM, using an array of pre-prepared lightcurves. Can save the SOM to text file.
        Saved SOM can be reloaded using LoadSOM() function.
    
            Args:
                SOMarray: Array of normalised inputs (e.g. binned transits), of shape [n_inputs,n_bins]. n_inputs > 1
                niter: number of training iterations, default 500. Must be positive integer.
                learningrate: alpha parameter, default 0.1. Must be positive.
                learningradius: sigma parameter, default the largest input SOM dimension. Must be positive.
                somshape: shape of SOM to train, default (20,20). Currently must be 2 dimensional tuple (int, int). Need not be square.
                outfile: File path to save SOM Kohonen Layer to. If None will not save.
    
            Returns:
                The trained SOM object
    """   
    try:
        import somtools
        import selfsom
    except:
        print 'Accompanying libraries not in PYTHONPATH or current directory'
        return 0
        
    try:
        assert niter >= 1, 'niter must be >= 1.'
        assert type(niter) is IntType, 'niter must be integer.'
        assert learningrate > 0, 'learningrate must be positive.'
        if learningradius:
            assert learningradius > 0, 'learningradius must be positive.'
        assert len(somshape)==2, 'SOM must have 2 dimensions.'
        assert type(somshape[0]) is IntType and type(somshape[1]) is IntType, 'somshape must contain integers.'
        assert len(SOMarray.shape)==2, 'Input array must be 2D of shape [ninputs, nbins].'
        assert SOMarray.shape[0]>1, 'ninputs must be greater than 1.'
    except AssertionError as error:
        print error
        print 'Inputs do not meet requirements. See help'
        return 0
        
    nbins = SOMarray.shape[1]
    
    #default learning radius
    if not learningradius:
        learningradius = np.max(somshape)
    
    #define som initialisation function
    def Init(sample):
        return np.random.uniform(0,2,size=(somshape[0],somshape[1],nbins))
    
    #initialise som
    som = selfsom.SimpleSOMMapper(somshape,niter,initialization_func=Init,learning_rate=learningrate,iradius=learningradius)

    #train som
    som.train(SOMarray)

    #save
    if outfile:
        somtools.KohonenSave(som.K,outfile)
    
    #return trained som
    return som

def LoadSOM(filepath,dim_x,dim_y,nbins,lrate=0.1):
    """
        Makes a som object using a saved Kohonen Layer (such as could be saved by CreateSOM().
    
            Args:
                filepath: The path to the saved Kohonen Layer. Must be saved in format created by somtools.KohonenSave().
                dim_x: The size of the first SOM dimension. Int
                dim_y: The size of the second SOM dimension. Int
                nbins: The number of lightcurve bins used (i.e. the 3rd dimension of the Kohonen Layer). Int
                lrate: The learning rate used to train the SOM. Optional, default=0.1. Included for tidiness, 
                       if the SOM is not retrained and only used for classification this parameter does not matter.
            
            Returns:
                The SOM object
    """
    try:
        import somtools
        import selfsom
    except:
        print 'Accompanying libraries not in PYTHONPATH or current directory'
        return 0
    
    def Init(sample):
        return np.random.uniform(0,2,size=(int(dim_x),int(dim_y),int(nbins)))
        
    som = selfsom.SimpleSOMMapper((dim_x,dim_y),1,initialization_func=Init,learning_rate=lrate)
    loadk = somtools.KohonenLoad(filepath)
    som.train(loadk) #tricks the som into thinking it's been trained
    som._K = loadk  #loads the actual Kohonen layer into place.
    return som