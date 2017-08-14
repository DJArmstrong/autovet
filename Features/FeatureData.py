import os
import pandas as pd
import numpy as np

class FeatureData():

    def __init__(self):
        '''
    	A holder and unifier for feature sets for multiple input labels
    
    	Reads in datafiles associated with a label, lines up data between labels
		to produce useful arrays to pass to classifier
    
    	Possibly also will have helper functions to remove correlated attributes, 
    	normalise numerical features, etc, eventually
        '''
        self.data = {}
    
    
    def addData(self,filepath,label,addrows=True):
        '''
        Adds feature data to a label.
        
        Inputs
        ------
        filepath:	str
        			filepath of data array, csv format, first row must be column names.
        			First column will be used as index.
        			Cannot contain duplicate indices, or matching will fail.
        			
        			Will add on new columns, new ids, and replaces any previously empty
        			data with values if possible.
        
        label:		str
        			label to add data to. e.g. real_candidate, false...
        			
        addrows:	bool, default True
        			if new ids are in the input file, add them to self.data. If False, 
        			only uses ids which are already in self.data
        '''
        try:
            #dat = np.genfromtxt(filepath,names=True,delimiter=',',dtype=None)
            dat = pd.read_csv(filepath,index_col=0)
            dat.columns = dat.columns.str.strip()
            #if 'ID' not in dat.dtype.names:
            #    print 'File must have ID column'
            #    return 0
            if label in self.data.keys():
                #scan for all new ids
                newids = []
                oldids = []
                
                for id in dat.index:
                    if id not in self.data[label].index:
                        if addrows:
                            newids.append(id)
                    else:
                        oldids.append(id)
                    
                #scan for all new columns
                newcols = []
                oldcols = []
                for col in dat.columns:
                    if col != 'label':
                        if col not in self.data[label].columns:
                            newcols.append(col)
                        else:
                            oldcols.append(col)
                        
                # use .join on the new columns
                joinarray = self.data[label].join(dat[newcols])  #will leave out newids

                #use pd.concat on the new ids
                if len(newids)>0:
                    joinarray = pd.concat([joinarray,dat.loc[newids,:]])
                
                #if len(newids)==0 and len(newcols)==0:
                #    joinarray = self.data[label]
                    
                #for loop over all remaining (old columns combined with old ids)
                for id in oldids:
                    for col in oldcols:
                        #if the corresponding old entry is NaN, update
                        if type(joinarray.loc[id,col]) is not str:
                            if np.isnan(joinarray.loc[id,col]):
                                joinarray.loc[id,col] = dat.loc[id,col]
                self.data[label] = joinarray
            else:
                self.data[label] = dat
        except IOError as e:
            print 'Loading Error, nothing happened. Error copied below.'
            print e
            print 'Loading Error, nothing happened. Error copied above.'
            
    def findCommonCols(self):
        common_cols = []
        excludedcols = []
        labels = self.data.keys()
        common_cols = list(self.data[labels[0]].columns)
        if len(self.data.keys())>=2:
            #remove data columns not present in every label
            for label in labels[1:]:
                checkcols = self.data[label].columns
                for col in common_cols:
                    if col not in checkcols:
                        excludedcols.append(col)
                for removecol in excludedcols:
                    common_cols.remove(removecol)
        if 'ID' in common_cols:
            common_cols.remove('ID')
        if 'label' in common_cols:
            common_cols.remove('label')
        if 'f0' in common_cols:
            common_cols.remove('f0')
        if 'Unnamed' in common_cols:
            common_cols.remove('Unnamed')
        common_cols = np.sort(common_cols) #to give repeatable output order
        return common_cols, excludedcols  
    
    def outputTrainingSet(self,outfile,impute_type='fill'):
        '''
        Writes out one file from self.data, containing only columns
        common to all labels. Suitable for use as immediate training set. First col will
        be integer represented label.
        
        Inputs
        ------
        outfile: str
        		 output filepath. Files will be in csv format.
        
        impute_type: str, options: 'median','fill','None'
        		How to deal with NaNs and infs. median uses median of that column. fill
        		replaces with -10. None leaves them as they were.
        '''
        if not os.path.exists(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])
            
        if len(self.data.keys())>0:
            common_cols, excludedcols = self.findCommonCols()

            if len(excludedcols)>0:
                print 'Warning, some columns excluded:'
                print excludedcols
       
            with open(outfile,'w') as f:
                f.write('#ID,label,')
                for col in common_cols[:-1]:
                    f.write(col.strip(' ')+',')
                f.write(common_cols[-1].strip(' ')+'\n')
                for label in self.data.keys():
                    ids = self.data[label].index
                    output = self.data[label][common_cols]
                    output = self.impute(output,impute_type)
                    for id,row in zip(ids,output.values):
                        f.write(str(id)+','+label+',')
                        for item in row[:-1]:
                            f.write(str(item)+',')
                        f.write(str(row[-1])+'\n')

    def impute(self,output,impute_type):
        '''
        Fill in blanks
        
        Inputs
        ------
        output: Pandas dataset
        		Dataset to impute.
        		
        impute_type: 'median','fill','None'
        		How to deal with NaNs and infs. median uses median of that column. fill
        		replaces with -10. None leaves them as they were.
        '''
        if impute_type != 'None':
            output = output.replace([np.inf, -np.inf], np.nan) #replace infs with nan
            output = output.replace([-10], np.nan) #replace -10 (the standard fill value) with nan
            if impute_type=='fill':
                output = output.fillna(-10) #replace nans with -10
            elif impute_type=='median':
                output = output.fillna(output.median())
        return output

    #def clipOutliers(self):
        #for label in self.data.keys():
            #self.data[label].clip()
    
    def simFeature(self,feat_to_sim,target_class,distribution,dist_params):
        '''
        Simulate a feature for a class, using scipy stats distributions.
        
        Inputs
        ------
        feat_to_sim: 	str
        				Name of feature to simulate
        
        target_class:	str
        				Class to create feature for
        				        			
        distribution:	str, in [truncnorm, expon]
        				Distribution to draw from. Only some supported
        				
        dist_params:	list
        				Distribution parameters. [a, b, mean, std] for truncnorm,
        				[loc, scale] for expon. See scipy.stats.
        '''
        if distribution=='truncnorm':
            from scipy.stats import truncnorm
            dist = truncnorm(dist_params[0], dist_params[1], loc=dist_params[2], scale=dist_params[3])
            simdata = dist.rvs(len(self.data[target_class].index))
        elif distribution=='expon':
            from scipy.stats import expon
            dist = expon(loc=dist_params[0], scale=dist_params[1])
            simdata = dist.rvs(len(self.data[target_class].index))
        elif distribution=='binom':
            from scipy.stats import binom,truncnorm
            dist = binom(1,dist_params[0])
            simdata = dist.rvs(len(self.data[target_class].index)).astype('float')
            npositive = np.sum(simdata>0.5)
            nsample = int(npositive*0.15)
            sample = np.random.choice(np.arange(npositive),size=nsample,replace=False)
            simdata[sample] = truncnorm(-2.5,0.,loc=1.0,scale=0.2).rvs(nsample)
        else:
            print 'Distribution not supported'
            return 0
        if target_class in self.data.keys():
            self.data[target_class][feat_to_sim] = pd.Series(simdata,index=self.data[target_class].index)

    def joinCentroids(self):
        self.avgCols('Binom_X','Binom_Y','Binom')
        self.quadCols('CENTDX_fda_PHASE_RMSE','CENTDY_fda_PHASE_RMSE','CENT_fda_PHASE_RMSE')
        self.quadCols('CrossCorrSNR_X','CrossCorrSNR_Y','CrossCorrSNR')

    def quadCols(self,col1,col2,target):
        for label in self.data.keys():
            if col1 in self.data[label].columns and col2 in self.data[label].columns:
                #add quadrature column
                quads = np.sqrt(np.power(self.data[label][col1],2) + np.power(self.data[label][col2],2))
                self.data[label][target] = pd.Series(quads,index=self.data[label].index)
                #remove old columns
                self.data[label] = self.data[label].drop(col1,1)
                self.data[label] = self.data[label].drop(col2,1)
                
    def avgCols(self,col1,col2,target):
        for label in self.data.keys():
            if col1 in self.data[label].columns and col2 in self.data[label].columns:
                #add quadrature column
                avg = (self.data[label][col1] + self.data[label][col2])/2.
                self.data[label][target] = pd.Series(avg,index=self.data[label].index)
                #remove old columns
                self.data[label] = self.data[label].drop(col1,1)
                self.data[label] = self.data[label].drop(col2,1)
        
    