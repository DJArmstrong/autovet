#should act largely independently of loader/feature classes for simplicity. Hence will load in saved file of feature values
import numpy as np
import pandas as pd
import gpflow

class CandidateSet(object):

    def __init__(self,features):  
        """
        Set of candidates
        
        Arguments:
        features   -- np array of feature data, [n_candidates,n_features]
        """        
        self.class_probs = None
        self.features = features


class TrainingSet(CandidateSet):
    
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

class Classifier(object):

    def __init__(self,classifier='RandomForestClassifier',classifier_args=None):  #list of Featureset instances, each with features calculated
        """
        Classifier wrapper for several sklearn classifiers
        
        Arguments:
        classifier		-- classifier to use in {RandomForestClassifier, 
        					ExtraTreesClassifier, AdaBoostClassifier, MLPClassifier}
        classifier_args	-- dict of arguments for classifier {arg:value}
        """    
        if classifier_args is None: classifier_args={}
        #self.classifier_obj = classifier_obj
        self.classifier_args = classifier_args
        self.classifier_type = classifier
        self.cvprobs = None
        
        global classifier_obj
        if classifier=='RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier as classifier_obj
            self.classifier = self.setUpForest()
        elif classifier=='ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier as classifier_obj
            self.classifier = self.setUpForest()        
        elif classifier=='AdaBoostClassifier':
            from sklearn.ensemble import AdaBoostClassifier as classifier_obj
            self.classifier = self.setUpAdaBoost()            
        elif classifier=='MLPClassifier':
            from sklearn.neural_network import MLPClassifier as classifier_obj
            self.classifier = self.setUpMLP()        
            
    def train(self,trainingset):
        self.featurenames = trainingset.featurenames
        self.classifier = self.classifier.fit(trainingset.features,trainingset.known_classes)
    
    def featImportance(self):
        print('Top features:')
        featimportance = self.classifier.feature_importances_
        for i in np.argsort(featimportance)[::-1]:
            print(self.featurenames[i], featimportance[i])
    
    def classify(self,candidateset):
        class_probs = self.classifier.predict_proba(candidateset.features)
        candidateset.class_probs = class_probs
        return class_probs
        
    def crossValidate(self,tset):
        """
        Tests classifier with cross validation. 

        Curently uses KFold validation, with nsamples/500 folds. May take some time.
        """
        from sklearn.model_selection import KFold
        
        if self.classifier_type=='RandomForestClassifier':
            self.classifier.oob_score = False
        shuffleidx = np.random.choice(len(tset.known_classes),len(tset.known_classes),replace=False)
        cvfeatures = tset.features[shuffleidx,:]
        cvgroups = tset.known_classes[shuffleidx]
        split = 500
        #test to make sure all splits contain all classes
        for group in np.unique(tset.known_classes):
            if np.sum(tset.known_classes==group)/2. < split:
                split = np.sum(tset.known_classes==group)/2.
        kf = KFold(n_splits=int(cvfeatures.shape[0]/split))
        probs = []
        print('Generating cross-validated probabilities, may take some time...')
        for train_index,test_index in kf.split(cvfeatures,cvgroups):
            #attempting to avert a memory error
            del self.classifier
            if self.classifier_type=='RandomForestClassifier':
                from sklearn.ensemble import RandomForestClassifier as classifier_obj
                self.classifier = self.setUpForest()
            elif self.classifier_type=='ExtraTreesClassifier':
                from sklearn.ensemble import ExtraTreesClassifier as classifier_obj
                self.classifier = self.setUpForest()        
            elif self.classifier_type=='AdaBoostClassifier':
                from sklearn.ensemble import AdaBoostClassifier as classifier_obj
                self.classifier = self.setUpAdaBoost()            
            elif self.classifier_type=='MLPClassifier':
                from sklearn.neural_network import MLPClassifier as classifier_obj
                self.classifier = self.setUpMLP()        
            self.classifier.fit(cvfeatures[train_index,:],cvgroups[train_index])
            probs.append(self.classifier.predict_proba(cvfeatures[test_index,:]))  
        self.cvprobs = np.vstack(probs)
        unshuffleidx = np.argsort(shuffleidx)
        self.cvprobs = self.cvprobs[unshuffleidx]
        return self.cvprobs

    def optimiseForest(self,trainingset):
        if self.classifier_type != 'RandomForestClassifier':
            return 0
        estimators_set = np.array([100,300,500])
        n_features = len(trainingset.featurenames)
        maxfeat_set = np.array([2,4,5,6,7,9])
        #maxfeat_set = np.linspace(1,n_features,8).astype('int')
        maxdepth_set = np.array([2,5,8,11,n_features])
        #maxdepth_set = np.linspace(1,n_features,5).astype('int')
        minsamples_set = np.arange(4)+2
        output = np.zeros([len(estimators_set),len(maxfeat_set),len(maxdepth_set),len(minsamples_set)])
        
        for i,est in enumerate(estimators_set):

            for j,maxfeat in enumerate(maxfeat_set):

                for k,maxdepth in enumerate(maxdepth_set):

                    print(est,maxfeat,maxdepth)
                    for l,minsamples in enumerate(minsamples_set):
                        classifier_args = {'n_estimators':est,'max_features':maxfeat,'max_depth':maxdepth,'min_samples_split':minsamples}
                        self.classifier_args = classifier_args
                        self.classifier = self.setUpForest()
                        self.classifier = self.classifier.fit(trainingset.features,trainingset.known_classes)
                        output[i,j,k,l] = self.classifier.oob_score_
        return estimators_set,maxfeat_set,maxdepth_set,minsamples_set,output
        
    def setUpForest(self):
        #set up options and defaults   
        inputs = {}
        inputs['n_estimators'] = 300 #higher the better (and slower)
        inputs['max_features'] = 'auto'  #will give SQRT(n_features) as a good typical first guess
        inputs['max_depth'] = None #max depth of tree (needs tuning)
        inputs['min_samples_split'] = 3 #min samples left to split a node (needs tuning)
        inputs['n_jobs'] = -1
        inputs['oob_score'] = True #estimate out-of-bag score
        inputs['random_state'] = 0 #random state initialiser
        inputs['class_weight'] = 'balanced' #uses inverse frequency of classes in training set
        inputs['warm_start'] = False
        #Parse actual user inputs
        for key in self.classifier_args.keys(): 
            if key in inputs.keys():  #ignores unrecognised options
                inputs[key] = self.classifier_args[key]
                
        clf = classifier_obj(n_estimators=inputs['n_estimators'],max_features=inputs['max_features'],max_depth=inputs['max_depth'],min_samples_split=inputs['min_samples_split'],n_jobs=inputs['n_jobs'],oob_score=inputs['oob_score'],random_state=inputs['random_state'],class_weight=inputs['class_weight'])
        return clf
    
    def setUpAdaBoost(self,obj):  #uses decision tree as base estimator, with default values. Could be worked on.
        #set up options and defaults   
        inputs = {}
        inputs['n_estimators'] = 50 #higher the better (and slower)
        inputs['learning_rate'] = 1
        inputs['random_state'] = 0 #random state initialiser
        #Parse actual user inputs
        for key in self.classifier_args.keys(): 
            if key in inputs.keys():  #ignores unrecognised options
                inputs[key] = self.classifier_args[key]
                
        clf = classifier_obj(n_estimators=inputs['n_estimators'],random_state=inputs['random_state'],learning_rate=inputs['learning_rate'])
        return clf
    
    def setUpMLP(self,obj):      
        #set up options and defaults   
        inputs = {}
        inputs['hidden_layer_sizes'] = (100,) #tuple, each entry represents a hidden layer, value is the number of neurons in it
        inputs['alpha'] = 0.0001 #smaller tends to work better. Should be optimised
        inputs['random_state'] = 0 #random state initialiser
        #Parse actual user inputs
        for key in self.classifier_args.keys(): 
            if key in inputs.keys():  #ignores unrecognised options
                inputs[key] = self.classifier_args[key]
                
        clf = classifier_obj(hidden_layer_sizes=inputs['hidden_layer_sizes'],random_state=inputs['random_state'],alpha=inputs['alpha'])
        return clf

class GPFlowClassifier(object):
    
    def __init__(self, Xtrain, Ytrain, num_inducing=16, kernel=None, likelihood=None,
    			num_latent=None):
        '''
        Xtrain is ndarray num_data * input dimension? so each row is lc, col is cadence
        num_inducing is per class
        '''
        import gpflow
    
        if kernel is None:
            self.kernel = gpflow.kernels.RBF(2)
        else:
            self.kernel = kernel
        if likelihood is None:
            self.likelihood = gpflow.likelihoods.Bernoulli()
        else:
            self.likelihood = likelihood
        self.num_inducing = num_inducing
        self.num_latent = num_latent
        
        if self.num_inducing == 0:
            self.model = gpflow.models.VGP(Xtrain, Ytrain,
                      kern=self.kernel,likelihood=self.likelihood)
        else:
            from scipy.cluster.vq import kmeans
            Z = np.zeros([num_inducing*len(set(Ytrain)),Xtrain.shape[1]])
            for c,cl in enumerate(set(Ytrain)):
                Z[c*num_inducing:(c+1)*num_inducing,:] = kmeans(Xtrain[Ytrain==cl],num_inducing)[0]
            
            np.random.shuffle(Z)   
            self.model = gpflow.models.SVGP(Xtrain, Ytrain, kern=self.kernel,
        								likelihood=self.likelihood, Z=Z, num_latent=self.num_latent)

    
    def train(self, method=None):
        if method=='Adam':
            gpflow.training.AdamOptimizer(0.01).minimize(self.model, maxiter=2000)  
        else:
            if self.num_inducing > 0:
                # Initially fix the hyperparameters.
                self.model.feature.set_trainable(False)
                gpflow.train.ScipyOptimizer().minimize(self.model, maxiter=20)

                # Unfix the hyperparameters.
                self.model.feature.set_trainable(True)
            
            gpflow.train.ScipyOptimizer(options=dict(maxiter=10000)).minimize(self.model)

    def predict(self,X):
        return self.model.predict_y(X)[0]
        