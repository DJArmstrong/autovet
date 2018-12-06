from sklearn.metrics import precision_recall_curve, roc_auc_score
import autovet
from autovet.Classify.Learner import GPClassifier
from autovet.Classify.TrainingSet import TrainingSet_LC
from autovet.Classify.kernels import SpectralMixture, sm_init
import pylab as p
p.ion()
import numpy as np
from gpflow.kernels import Matern32,Matern52
import os

#load data
data_dir = os.path.join(os.getcwd(),'../../Data/Kepinjectv0_run2norm/')

planetX_ph = np.genfromtxt(os.path.join(data_dir,'planetXph_1000bin_kepQ9.txt'))
ebX_ph = np.genfromtxt(os.path.join(data_dir,'ebXph_1000bin_kepQ9.txt'))
bgebX_ph = np.genfromtxt(os.path.join(data_dir,'bgebXph_1000bin_kepQ9.txt'))
hebX_ph = np.genfromtxt(os.path.join(data_dir,'hebXph_1000bin_kepQ9.txt'))
psbX_ph = np.genfromtxt(os.path.join(data_dir,'psbXph_1000bin_kepQ9.txt'))
btpX_ph = np.genfromtxt(os.path.join(data_dir,'btpXph_1000bin_kepQ9.txt'))

planetX = np.genfromtxt(os.path.join(data_dir,'planetX_kepQ9.txt'))
ebX = np.genfromtxt(os.path.join(data_dir,'ebX_kepQ9.txt'))
bgebX = np.genfromtxt(os.path.join(data_dir,'bgebX_kepQ9.txt'))
hebX = np.genfromtxt(os.path.join(data_dir,'hebX_kepQ9.txt'))
psbX = np.genfromtxt(os.path.join(data_dir,'psbX_kepQ9.txt'))
btpX = np.genfromtxt(os.path.join(data_dir,'btpX_kepQ9.txt'))

planetX_loc = np.genfromtxt(os.path.join(data_dir,'planetXloc_100bin_kepQ9.txt'))
ebX_loc = np.genfromtxt(os.path.join(data_dir,'ebXloc_100bin_kepQ9.txt'))
bgebX_loc = np.genfromtxt(os.path.join(data_dir,'bgebXloc_100bin_kepQ9.txt'))
hebX_loc = np.genfromtxt(os.path.join(data_dir,'hebXloc_100bin_kepQ9.txt'))
psbX_loc = np.genfromtxt(os.path.join(data_dir,'psbXloc_100bin_kepQ9.txt'))
btpX_loc = np.genfromtxt(os.path.join(data_dir,'btpXloc_100bin_kepQ9.txt'))


#set up training set

tset = TrainingSet_LC(planetX_ph, np.zeros(planetX_ph.shape[0]))
tset.add_members(bgebX_ph, np.ones(bgebX_ph.shape[0]))
tset.add_view(np.vstack((planetX_loc,bgebX_loc)))

#input_1view: NB 2 views is better for bgEB (0.957 vs 0.949 auc) and marginally worse for eb, although that might be a result of different train/test sets used.
#overall better, although implementation is currently non-ideal.
#tset_1v = TrainingSet_LC(planetX_ph, np.zeros(planetX_ph.shape[0]))
#tset_1v.add_members(bgebX_ph, np.ones(bgebX_ph.shape[0]))

#scale data
tset.scale()

#split train/test
tset.split_train_test(tset_size=0.33, random_state=14)

# Spectral Mixture kernel
#Q = 3 # nnumber of mixtures
#D = np.shape(tset.X)[1]

# first get the sm kernel params set
#weights, means, scales = sm_init(train_x=tset.X, train_y=tset.Y, num_mixtures=Q)
#scales /= 10.
#k_sm = SpectralMixture(num_mixtures=Q,mixture_weights=weights,mixture_scales=scales, mixture_means=means,input_dim=D)

#multi-view kernel (simple Mat32)
kern1 = Matern32(np.sum(tset.view_index==0),active_dims=np.where(tset.view_index==0)[0])
kern2 = Matern32(np.sum(tset.view_index==1),active_dims=np.where(tset.view_index==1)[0])
k_sum = kern1 + kern2

#set up classifier
gpf = GPFlowClassifier(tset.X,tset.Y.reshape(-1,1),num_inducing=16,kernel=k_sum)

#train
gpf.train()

#predict
y_predict = gpf.predict(tset.X_test)

prec,rec,thresh = precision_recall_curve(tset.Y_test,y_predict,0)
auc = roc_auc_score(tset.Y_test,y_predict)
print(auc)

#plot
p.figure()
p.clf()
p.plot(y_predict,tset.Y_test,'rx')
p.title('Num_inducing: '+str(ni))



fig, ax = p.subplots(2,2,figsize=(9,9), 
            subplot_kw={'xticks': [], 'yticks': []})
fig.suptitle('Spectral Mixture Kernel', fontsize=28)
i=1

for max_dist in [0.5,5.0]:
    print(max_dist)
    for min_dist in [0.1,0.00000001]:
        
        means = np.random.rand(Q,D) * 0.5 / min_dist
        scales = (np.abs(np.random.randn(D,Q))*max_dist) **(-1) 

        k_sm = SpectralMixture(num_mixtures=Q,mixture_weights=weights,mixture_scales=scales, mixture_means=means,input_dim=D)
        k_sum = k_sm + White(D)
        ax = p.subplot(2, 2, i)

        K_xx= k_sm.compute_K_symm(X_train) 
        K_xx = (K_xx + K_xx.T)/2.
        ax.imshow(K_xx,interpolation='none')
        ax.set_xlabel(min_dist)
        ax.set_ylabel(max_dist)
        i += 1

fig.text(0.5, -0.04, 'different kernel initializations', ha='center',fontsize=18)
p.subplots_adjust(top=1.35)
fig.tight_layout()
