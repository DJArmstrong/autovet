#should act largely independently of loader/feature classes for simplicity. Hence will load in saved file of feature values
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split, KFold, cross_val_score, PredefinedSplit
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix as confMat
# import classifiers
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class Classifier(object):

    def __init__(self,classifier='RFC',classifier_args=None):  #list of Featureset instances, each with features calculated
        """
        Classifier wrapper for several sklearn classifiers
        
        Arguments:
        classifier		-- classifier to use in {RandomForestClassifier, 
        					ExtraTreesClassifier, AdaBoostClassifier, MLPClassifier}
        classifier_args	-- dict of arguments for classifier {arg:value}
        """    
        
        models = {'QDA': QuadraticDiscriminantAnalysis(),
            	'DecisionTree': DecisionTreeClassifier(),
            	'RFC':RandomForestClassifier(),
            	'ExtraTrees':ExtraTreesClassifier(),
            	'AdaBoost':AdaBoostClassifier()}

        
        if classifier_args is None: classifier_args={}
        #self.classifier_obj = classifier_obj
        self.classifier_args = classifier_args
        self.classifier_type = classifier
        self.cvprobs = None
        
        #global classifier_obj
        self.classifier = models[classifier]                
        self.classifier.set_params(**classifier_args)
                
            
    def train(self,trainingset=None,X=None,Y=None):
        if trainingset is not None:
            X = trainingset.features
            Y = trainingset.known_classes
            self.featurenames = trainingset.featurenames
        self.classifier = self.classifier.fit(X,Y)
    
    def featImportance(self):
        if hasattr(self.classifier, 'feature_importances_'):
            print('Top features:')
            featimportance = self.classifier.feature_importances_
            for i in np.argsort(featimportance)[::-1]:
                print(self.featurenames[i], featimportance[i])
        else:
            print('Feature importances not supported for this classifier')
    
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
        
#     def setUpForest(self):
#         #set up options and defaults   
#         inputs = {}
#         inputs['n_estimators'] = 300 #higher the better (and slower)
#         inputs['max_features'] = 'auto'  #will give SQRT(n_features) as a good typical first guess
#         inputs['max_depth'] = None #max depth of tree (needs tuning)
#         inputs['min_samples_split'] = 3 #min samples left to split a node (needs tuning)
#         inputs['n_jobs'] = -1
#         inputs['oob_score'] = True #estimate out-of-bag score
#         inputs['random_state'] = 0 #random state initialiser
#         inputs['class_weight'] = 'balanced' #uses inverse frequency of classes in training set
#         inputs['warm_start'] = False
#         #Parse actual user inputs
#         for key in self.classifier_args.keys(): 
#             if key in inputs.keys():  #ignores unrecognised options
#                 inputs[key] = self.classifier_args[key]
#                 
#         clf = classifier_obj(n_estimators=inputs['n_estimators'],max_features=inputs['max_features'],max_depth=inputs['max_depth'],min_samples_split=inputs['min_samples_split'],n_jobs=inputs['n_jobs'],oob_score=inputs['oob_score'],random_state=inputs['random_state'],class_weight=inputs['class_weight'])
#         return clf
    
    def Optimise(self, X, Y, params, valtype='CV', val_idx = None, n_jobs=1, verbose=True, scoring='accuracy', refit=True):
        """
        Performs grid (or random?) search CV for each of the models and saves the GridSearchCV trained object
        into a dictionary self.grid_search_objects.
            Inputs:
            	X
            	Y		Should Include Valset
                n_jobs          - # to paralelize
                verbose         - how much to print out into terminal, should be zero if run on cluster
                scoring         - scoring function to use
                refit           - should the best model be fit to a whole training set
        """
        if valtype == 'CV':
            cvsplit = KFold(n_splits=10)        
        elif valtype == 'valset' and val_idx is not None:
            cvsplit = PredefinedSplit(test_fold=val_idx)
        else:
            print('Validation type must be "CV" (cross-validation) or "valset" (separate validation set)')
            

        # Perform grid search cross-validation
        grid_search_object = GridSearchCV(self.classifier, params, cv=cvsplit, n_jobs=n_jobs, verbose = verbose,
                                                 scoring=scoring, refit=refit)
        grid_search_object.fit(X, Y)
        
        #set best parameters for classifier
        self.classifer = grid_search_object.best_estimator_
        
        return grid_search_object
        



class GPClassifier(object):
    
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
            Z = np.zeros([num_inducing*len(set(Ytrain[:,0])),Xtrain.shape[1]])
            for c,cl in enumerate(set(Ytrain[:,0])):
                Z[c*num_inducing:(c+1)*num_inducing,:] = kmeans(Xtrain[Ytrain[:,0]==cl],num_inducing)[0]
            
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
        
