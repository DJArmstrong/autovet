import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

class CandidateSet(object):

    def __init__(self,features):  
        """
        Set of candidates
        
        Arguments:
        features   -- np array of feature data, [n_candidates,n_features]
        """        
        self.class_probs = None
        self.features = features
        

  

class TrainingSet_LC(CandidateSet):
    def __init__(self, X, Y):
        """
        """
        self.X_base = X
        self.Y_base = Y
        self.view_index = np.zeros(len(X)).astype('int')
        self.X_train = None
        self.Y_train = None

    def addview(self, newview):
        self.X_base = np.hstack((self.X_base,newview))
        self.view_index = np.hstack((self.view_index,np.ones(newview.shape[1])+max(self.view_index)))
    
    def addmembers(self, newX, newY):
        self.X_base = np.vstack((self.X_base, newX))
        self.Y_base = np.vstack((self.Y_base, newY))
    
    def split_train_test(self, test_size=0.33, random_state=np.random.uniform(0,100)):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_base, self.Y_base, test_size=test_size, random_state=random_state)

    def X(self):
        if self.X_train is not None:
            return self.X_train
        else:
            return self.X_base

    def Y(self):
        if self.Y_train is not None:
            return self.Y_train
        else:
            return self.Y_base

    def scale(self):
        offset = self.X_base.mean()
        offstd = self.X_base.std()
        self.X_base = ( self.X_base - offset ) / offstd
      


class TrainingSet_NGTS(CandidateSet):
    
    def __init__(self,trainingfile,fieldstodrop=None,addrandom=False,dropnoise=False,dropdepth=False,dropdefault=False):
        """
        Set of candidates for training
        
        Arguments:
        trainingfile   -- txt file of training data
        fieldstodrop   -- features to remove before training
        addrandom      -- if True, adds a random feature for calibration
        dropnoise      -- if True, drops default fields from training set file
        dropdepth      -- if True, drops transit depth related fields
        dropdefault    -- if True, drops lightcurve noise related fields
        """
        if fieldstodrop is None: fieldstodrop = []
        if len(fieldstodrop)==0:
            if dropdefault:
                defaultfields = ['tdur_phase','Trapfit_t0','Fit_t0','Even_Fit_aovrstar', 
                				'Even_Fit_chisq','Even_Trapfit_t14phase', 
                				'Even_Trapfit_t23phase','Fit_period','NUM_TRANSITS',
                				'Trapfit_t23phase','Odd_Fit_aovrstar', 'Odd_Fit_chisq', 
                				'Odd_Trapfit_t14phase', 'Odd_Trapfit_t23phase','MaxSecPhase']
                for field in defaultfields:
                    fieldstodrop.append(field)
            if dropnoise:
                noisefields = ['P2P_98perc', 'P2P_mean', 'Peak_to_peak', 'RMS', 
                    			'RMS_TDur', 'Skew','std_ov_error','Kurtosis', 'MAD',
                    			'NZeroCross']
                for field in noisefields:
                    fieldstodrop.append(field)
            if dropdepth:
                depthfields = ['Odd_Fit_depthSNR','Odd_Fit_rprstar', 
                    			'Odd_Trapfit_depth','Fit_rprstar','Even_Fit_depthSNR', 
                    			'Even_Fit_rprstar','Even_Trapfit_depth','Fit_depthSNR',
                    			'Trapfit_depth']
                for field in depthfields:
                    fieldstodrop.append(field)
                                    
        dat = pd.read_csv(trainingfile,index_col=1)
        
        #remove remnant bad columns
        for col in dat.columns:
            if col[:7]=='Unnamed':
                fieldstodrop.append(col)
                
        for field in fieldstodrop:
            if field in np.array(dat.columns):
                dat = dat.drop(field,1)
        dat = dat.replace([np.inf, -np.inf], np.nan) #replace infs with nan
        dat = dat.replace([-10], np.nan) #replace -10 (the standard fill value) with nan
        for col in dat.columns[1:]:  #[1:] is to ignore the #ID column
            dat[col] = dat[col].where(dat[col] < 1e8, np.nan)#blocks stupidly large values that can arise for some bad lightcurves
        dat = dat.dropna()
        
        self.ids = np.array(dat['#ID'])
        dat = dat.drop('#ID',1)
        labels = np.array(dat.index)
        features = dat.values
        self.featurenames = np.array(dat.columns)
        if addrandom:
            randomfeature = np.random.uniform(0,1,features.shape[0])
            features = np.hstack((features,np.column_stack(randomfeature).T)) #all the column_stack bit is to meet the hstack dimension requirements
            self.featurenames = np.append(self.featurenames,'Random')      
        #dat = np.genfromtxt(trainingfile,delimiter=',',dtype=None,names=True)
        #features = dat[list(dat.dtype.names[1:])]
        #features = self.rmfield(dat,'label','tdur_phase','Trapfit_t0','Fit_t0')
        #self.featurenames = features.dtype.names
        #features = features.view(np.float64).reshape(features.shape + (-1,))
        #labels = dat['label']
        
        
        CandidateSet.__init__(self,features)
        self.known_classes = labels
    
    def rmfield(self, a, *fieldnames_to_remove ):
        return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]
