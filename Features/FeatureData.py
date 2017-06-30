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
    
    def outputTrainingSet(self,outfile,impute_type='median'):
        '''
        Writes out one file from self.data, containing only columns
        common to all labels. Suitable for use as immediate training set. First col will
        be integer represented label.
        
        Inputs
        ------
        outfile: str, output filepath. Files will be in csv format.
        
        impute_type: 'median','fill','None'
        		How to deal with NaNs and infs. median uses median of that column. fill
        		replaces with -10. None leaves them as they were.
        '''
        if len(self.data.keys())>0:
            common_cols, excludedcols = self.findCommonCols()

            if len(excludedcols)>0:
                print 'Warning, some columns excluded:'
                print excludedcols
       
            with open(outfile,'w') as f:
                f.write('# label,')
                for col in common_cols[:-1]:
                    f.write(col.strip(' ')+',')
                f.write(common_cols[-1].strip(' ')+'\n')
                for label in self.data.keys():
                    output = self.data[label][common_cols]
                    output = self.impute(output,impute_type)
                    for row in output.values:
                        f.write(label+',')
                        for item in row[:-1]:
                            f.write(str(item)+',')
                        f.write(str(row[-1])+'\n')
                
                       
    def outputSets(self,outdir,impute_type='median'):
        '''
        Writes out a file for each label in self.data, containing only columns
        common to all labels.
        
        Inputs
        ------
        outdir: str, output directory for files. Files will have format 'label_features.txt',
        		and will be in csv format.
        		
        impute_type: 'median','fill','None'
        		How to deal with NaNs and infs. median uses median of that column. fill
        		replaces with -10. None leaves them as they were.
        '''
        if len(self.data.keys())>0:
            common_cols, excludedcols = self.findCommonCols()
            
            if len(excludedcols)>0:
                print 'Warning, some columns excluded:'
                print excludedcols
            
            for label in self.data.keys():
                output = self.data[label][common_cols]
                output = self.impute(output,impute_type)
                with open(os.path.join(outdir,label+'_features.txt'),'w') as f:
                    f.write('#')
                    for col in common_cols[:-1]:
                        f.write(col.strip(' ')+',')
                    f.write(common_cols[-1].strip(' ')+'\n')
                    for row in output.values:
                        for item in row[:-1]:
                            f.write(str(item)+',')
                        f.write(str(row[-1])+'\n')

                with open(os.path.join(outdir,label+'_ids.txt'),'w') as f:
                    output = self.data[label].index
                    for id in output:
                        f.write(str(id)+'\n') 
                print 'N objects in '+label+': ',len(self.data[label].index)           
        else:
            print 'No data to output'

    def impute(self,output,impute_type):
        if impute_type != 'None':
            output = output.replace([np.inf, -np.inf], np.nan) #replace infs with nan
            output = output.replace([-10], np.nan) #replace -10 (the standard fill value) with nan
            if impute_type=='fill':
                output = output.fillna(-10) #replace nans with -10
            elif impute_type=='median':
                output = output.fillna(output.median())
        return output

#    def sim_feature()
    
        
    