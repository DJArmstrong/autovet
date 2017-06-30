#should act largely independently of loader/feature classes for simplicity. Hence will load in saved file of feature values
import numpy as np
import pandas as pd

class CandidateSet(object):

    def __init__(self,features):  
        
        self.class_probs = None
        self.features = features


class TrainingSet(CandidateSet):
    
    def __init__(self,trainingfile):
        dat = pd.read_csv(trainingfile,index_col=0)
        dat = dat.drop('tdur_phase',1)
        dat = dat.drop('Trapfit_t0',1)
        dat = dat.drop('Fit_t0',1)
        dat = dat.replace([np.inf, -np.inf], np.nan) #replace infs with nan
        dat = dat.replace([-10], np.nan) #replace -10 (the standard fill value) with nan
        dat = dat.dropna()
        labels = np.array(dat.index)
        features = dat.values
        self.featurenames = np.array(dat.columns)
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


    def __init__(self,classifier='RandomForestClassifier',classifier_args={}):  #list of Featureset instances, each with features calculated

        #self.classifier_obj = classifier_obj
        self.classifier_args = classifier_args
        
        global classifier_obj
        if classifier=='RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier as classifier_obj
            self.classifier = self.SetUpForest()
        elif classifier=='ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier as classifier_obj
            self.classifier = self.SetUpForest()        
        elif classifier=='AdaBoostClassifier':
            from sklearn.ensemble import AdaBoostClassifier as classifier_obj
            self.classifier = self.SetUpAdaBoost()            
        elif classifier=='MLPClassifier':
            from sklearn.neural_network import MLPClassifier as classifier_obj
            self.classifier = self.SetUpMLP()        
            
              
    def Train(self,trainingset):
        self.featurenames = trainingset.featurenames
        self.classifier = self.classifier.fit(trainingset.features,trainingset.known_classes)
    
    def FeatImportance(self):
        print 'Top 10 features:'
        featimportance = self.classifier.feature_importances_
        for i in np.argsort(featimportance)[-10:]:
            print self.featurenames[i], featimportance[i]
    
    def Classify(self,candidateset):
        class_probs = self.classifier.predict_proba(candidateset.features)
        candidateset.class_probs = class_probs

    def SetUpForest(self):
        #set up options and defaults   
        inputs = {}
        inputs['n_estimators'] = 300 #higher the better (and slower)
        inputs['max_features'] = 'auto'  #will give SQRT(n_features) as a good typical first guess
        inputs['max_depth'] = None #max depth of tree (needs tuning)
        inputs['min_samples_split'] = 2 #min samples left to split a node (needs tuning)
        inputs['n_jobs'] = -1
        inputs['oob_score'] = True #estimate out-of-bag score
        inputs['random_state'] = 0 #random state initialiser
        inputs['class_weight'] = 'balanced' #uses inverse frequency of classes in training set

        #Parse actual user inputs
        for key in self.classifier_args.keys(): 
            if key in inputs.keys():  #ignores unrecognised options
                inputs[key] = self.classifier_args[key]
                
        clf = classifier_obj(n_estimators=inputs['n_estimators'],max_features=inputs['max_features'],max_depth=inputs['max_depth'],min_samples_split=inputs['min_samples_split'],n_jobs=inputs['n_jobs'],oob_score=inputs['oob_score'],random_state=inputs['random_state'],class_weight=inputs['class_weight'])
        return clf
    
    def SetUpAdaBoost(self,obj):  #uses decision tree as base estimator, with default values. Could be worked on.
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
    
    def SetUpMLP(self,obj):      
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
