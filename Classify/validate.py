"""
Validation metrics and plots for a Random Forest Classifier

"""
 
import numpy as np       
import pylab as p
p.ion()

def Precision_Recall_Curve(cvprobs,labels,posclass='synth'): #for binary classifier
    """
    Precision Recall Curve for a classifier (wrapper for sklearn function)
        
    Arguments:
    cvprobs		-- 
    labels		-- 
    posclass	-- 
    """
    from sklearn.metrics import precision_recall_curve
    y_true = np.zeros(len(labels))
    y_true[labels==posclass] = 1
    prec,rec,thresh = precision_recall_curve(y_true,cvprobs[:,1],1)
    return prec, rec, thresh
        
def AUC(cvprobs,labels,posclass='synth'):
    """
    AUC score for a classifier (wrapper for sklearn function)
        
    Arguments:
    cvprobs		-- 
    labels		-- 
    posclass	-- 
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.zeros(len(labels))
    y_true[labels==posclass] = 1
    return roc_auc_score(y_true,cvprobs[:,1])
    
def TrainingSetResponse_recovered(cvprobs, tset, axis1, axis2, xmin, xmax, ymin, ymax, nbinsx=20, nbinsy=20, threshold=0.5, target='synth'):
        
    target_idx = np.where(tset.known_classes==target)[0]
    if type(axis1)==str:
        if (axis1 in tset.featurenames):
            x = tset.features[target_idx,np.where(tset.featurenames==axis1)[0][0]]
        else:
            print 'Axis 1 not in trainingset featurenames'
            return 0
    else:
        x = axis1
    if type(axis2)==str:
        if (axis2 in tset.featurenames):
            y = tset.features[target_idx,np.where(tset.featurenames==axis2)[0][0]]
        else:
            print 'Axis 2 not in trainingset featurenames'
            return 0
    else:
        y = axis2

    xedges,yedges = np.linspace(xmin,xmax,nbinsx), np.linspace(ymin,ymax,nbinsy)

    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))  #gives total in bins

    #assume cvprobs aligned with training set
    recovered = cvprobs>=threshold
    x_rec = x[recovered]
    y_rec = y[recovered]

    histrec, xedges, yedges = np.histogram2d(x_rec, y_rec, (xedges, yedges))  #gives recovered total in bins

    fraction = histrec/hist
    #fraction[hist<30]=np.nan

    palette = p.cm.viridis
    #palette.set_over('r', 1.0)
    #palette.set_under('g', 1.0)
    palette.set_bad('grey', 1.0)
 
    p.figure()
    p.clf()
    p.imshow(fraction,origin='lower',extent=[ymin,ymax,xmin,xmax],aspect='auto',cmap=palette,vmin=0,vmax=1)
    if type(axis2)==str:
        p.xlabel(axis2)
    if type(axis1)==str:
        p.ylabel(axis1)
    cbar = p.colorbar()
    cbar.set_label('Fraction Recovered', rotation=270, labelpad=10)

    #p.savefig(str(xidx)+'.pdf')

def TrainingSetResponse_average(cvprobs, tset, axis1, axis2, xmin, xmax, ymin, ymax, nbinsx=20, nbinsy=20, target='synth'):
    from scipy import stats
    target_idx = np.where(tset.known_classes==target)[0]
    
    if type(axis1)==str:
        if (axis1 in tset.featurenames):
            x = tset.features[target_idx,np.where(tset.featurenames==axis1)[0][0]]
        else:
            print 'Axis 1 not in trainingset featurenames'
            return 0
    else:
        x = axis1
    if type(axis2)==str:
        if (axis2 in tset.featurenames):
            y = tset.features[target_idx,np.where(tset.featurenames==axis2)[0][0]]
        else:
            print 'Axis 2 not in trainingset featurenames'
            return 0
    else:
        y = axis2

    xedges,yedges = np.linspace(xmin,xmax,nbinsx), np.linspace(ymin,ymax,nbinsy)

    ret = stats.binned_statistic_2d(x,y,cvprobs,'median',bins=[xedges,yedges])

    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))  #gives total in bins

    #remove low number bins
    result = ret.statistic
    result[hist<5]=np.nan

    palette = p.cm.viridis
    #palette.set_over('r', 1.0)
    #palette.set_under('g', 1.0)
    palette.set_bad('grey', 1.0)
 
    p.figure()
    p.clf()
    p.imshow(result,origin='lower',extent=[ymin,ymax,xmin,xmax],aspect='auto',cmap=palette,vmin=0,vmax=1)
    if type(axis2)==str:
        p.xlabel(axis2)
    if type(axis1)==str:
        p.ylabel(axis1)
    cbar = p.colorbar()
    cbar.set_label('Median Planet Probability', rotation=270, labelpad=10)

    #p.savefig(str(xidx)+'.pdf')

def FieldDependence(cvprobs,tset,ridx,sidx):
    prop_cycle = p.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fields = []
    for id in tset.ids[ridx]:    
        fields.append(id[:11])
        
    fields = np.array(fields)
    
    unique_fields = np.array(list(set(fields)))

    avg_cvprobs = []
    for fidx,testfield in enumerate(unique_fields):
        field_idx = np.where(fields==testfield)[0]
        avg_cvprobs.append(np.median(cvprobs[ridx,1][field_idx]))
    avg_cvprobs = np.array(avg_cvprobs)
    idx = np.argsort(avg_cvprobs)
    unique_fields = unique_fields[idx]
    avg_cvprobs = avg_cvprobs[idx]
    
    synth_fields = []
    for id in tset.ids[sidx]:
        synth_fields.append(id[:11])
    
    unique_synth_fields = set(synth_fields)
    
    nbins = 50
    bins = np.arange(nbins)/float(nbins)
    #f, ax = p.subplots(1,len(unique_fields),sharey=True)
    p.figure()
    
    for fidx,testfield in enumerate(unique_fields):
        field_idx = np.where(fields==testfield)[0]
        #weights = np.ones_like(cvprobs[ridx,1][field_idx])/float(len(cvprobs[ridx,1][field_idx]))
        #h = np.histogram(cvprobs[ridx,1][field_idx], weights=weights, bins=50)
        h = np.histogram(cvprobs[ridx,1][field_idx], bins=nbins)
        #print h[0]
        vals = h[0].astype('float')
        vals = vals / np.max(h[0])
        #print vals
        #bins=(h[1][:-1]+h[1][1:])/2.
        #bins = h[1][:-1]
        if testfield in unique_synth_fields:
            p.plot(vals+fidx,bins,'-',color=colors[1])
            p.fill_betweenx(bins,fidx,vals+fidx,facecolor=colors[1],alpha=0.7)
        else:
            p.plot(vals+fidx,bins,'-',color=colors[0])
            p.fill_betweenx(bins,fidx,vals+fidx,facecolor=colors[0],alpha=0.7)

        #ax[fidx].hist(cvprobs[ridx,1][field_idx], weights=weights, bins=50)
    #ax[0].set_xlim(0,0.3)
    p.ylim(0,0.3)
    p.ylabel('Planet Probability')
    p.xlabel('Field Index')
    p.pause(1)
    
    #p.figure()
    #p.plot(np.arange(len(unique_fields)),avg_cvprobs,'.-')
    
    #for t,testfield in enumerate(unique_fields):
    #    if testfield in unique_synth_fields:
    #        p.plot(t,avg_cvprobs[t],'rx')
   
def TrainingSetResponse_1d(cvprobs, tset, axis1, xmin, xmax, nbinsx=20, target='synth'):
    from scipy import stats
    target_idx = np.where(tset.known_classes==target)[0]
    
    if type(axis1)==str:
        if (axis1 in tset.featurenames):
            x = tset.features[target_idx,np.where(tset.featurenames==axis1)[0][0]]
        else:
            print 'Axis 1 not in trainingset featurenames'
            return 0
    else:
        x = axis1

    xedges = np.linspace(xmin,xmax,nbinsx)

    ret = stats.binned_statistic(x,cvprobs,'median',bins=xedges)
    bins = ret[1][:-1] + (ret[1][1:]-ret[1][:-1])/2.

    #palette.set_over('r', 1.0)
    #palette.set_under('g', 1.0)
 
    p.figure()
    p.clf()
    p.plot(x,cvprobs,'.')
    p.plot(bins,ret[0],'rx-')
    p.xlim(xmin,xmax)
    if type(axis1)==str:
        p.xlabel(axis1)
    p.ylabel('Planet Probability')
    


#def MLP_plot():
    
#def IsolationOutliers(self):
#    from sklearn.ensemble import IsolationForest as classifier_obj
#    self.classifier_outliers = self.classifier
        
#    self.classifier_outliers.fit(X_data_train)

#    predicted_outliers = clf_outliers.predict(X_data)    
    
    