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
    
    
    def add_data(self,filepath,label):
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
        '''
        try:
            #dat = np.genfromtxt(filepath,names=True,delimiter=',',dtype=None)
            dat = pd.read_csv(filepath,index_col=0)
            #if 'ID' not in dat.dtype.names:
            #    print 'File must have ID column'
            #    return 0
            if label in self.data.keys():
                #scan for all new ids
                newids = []
                oldids = []
                for id in dat.index:
                    if id not in self.data[label].index:
                        newids.append(id)
                    else:
                        oldids.append(id)
                    
                #scan for all new columns
                newcols = []
                oldcols = []
                for col in dat.columns:
                    if col not in self.data[label].columns:
                        newcols.append(col)
                    else:
                        oldcols.append(col)
                        
                # use .join on the new columns
                joinarray = self.data[label].join(dat[newcols])  #will leave out newids

                #use pd.concat on the new ids
                joinarray = pd.concat([joinarray,dat.loc[newids,:]])
                                        
                #for loop over all remaining (old columns combined with old ids)
                    #if the corresponding old entry is NaN, update
                for id in oldids:
                    for col in oldcols:
                        if np.isnan(joinarray.loc[id,col]):
                            joinarray.loc[id,col] = dat.loc[id,col]
                
                self.data[label] = joinarray
            else:
                self.data[label] = dat
        except IOError as e:
            print 'Loading Error, nothing happened. Error copied below.'
            print e
            print 'Loading Error, nothing happened. Error copied above.'
            
    
    def output_sets(self,outdir):
        '''
        Writes out a file for each label in self.data, containing only columns
        common to all labels.
        
        Inputs
        ------
        outdir: str, output directory for files. Files will have format 'label_features.txt',
        		and will be in csv format.
        '''
        if len(self.data.keys())>0:
            common_cols = []
            labels = self.data.keys()
            common_cols = list(self.data[labels[0]].columns)
            if len(self.data.keys())>=2:
                #remove data columns not present in every label
                for label in labels[1:]:
                    newcols = self.data[label].columns
                    for col in common_cols:
                        if col not in newcols:
                            common_cols.remove(col)
            if 'ID' in common_cols:
                common_cols.remove('ID')
            if 'label' in common_cols:
                common_cols.remove('label')
            if 'f0' in common_cols:
                common_cols.remove('f0')
            if 'Unnamed' in common_cols:
                common_cols.remove('Unnamed')
            common_cols = np.sort(common_cols) #to give repeatable output order
            for label in labels:
                output = self.data[label][common_cols]
                with open(os.path.join(outdir,label+'_features.txt'),'w') as f:
                    f.write('#')
                    for col in common_cols:
                        f.write(col+',')
                    f.write('\n')
                    for row in output.values:
                        for item in row:
                            f.write(str(item)+',')
                        f.write('\n')

                with open(os.path.join(outdir,label+'_ids.txt'),'w') as f:
                    output = self.data[label].index
                    for id in output:
                        f.write(str(id)+'\n')               
        else:
            print 'No data to output'

#    def sim_feature()
    
        
    