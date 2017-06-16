



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
        			filepath of data array, csv format, first row must be column names
        
        label:	str
        		label to add data to. e.g. real_candidate, false...
        '''
        try:
            dat = np.genfromtxt(filepath,names=True,delimiter=',',dtype=None)
            if label in self.data.keys():
                for col in dat.dtype.names:
                    if (col != 'f0') and (col != 'label'):
                        if col not in self.data['label'].keys():
                            self.data['label'][col] = dat['col']
                        else:
                            print col+' already loaded for label '+label+'. Skipping.'
            else:
                self.data[label] = {}
                for col in dat.dtype.names:
                    if (col != 'f0') and (col != 'label'):
                        self.data['label'][col] = dat['col']
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
            common_cols = self.data[labels[0]].keys()
            if len(self.data.keys())>=2:
                #remove data columns not present in every label
                for label in labels[1:]:
                    newcols = self.data[label].keys()
                    for col in common_cols:
                        if col not in newcols:
                            common_cols.remove(col)
            if 'ID' in common_cols:
                common_cols.remove('ID')
            common_cols = np.sort(common_cols) #to give repeatable output order
            for label in labels:
                output = self.data[label][common_cols]
                with open(os.path.join(outdir,label+'_features.txt'),'w') as f:
                    f.write('#')
                    for col in common_cols:
                        f.write(col+',')
                    f.write('\n')
                    for row in output:
                        for item in row:
                            f.write(str(item)+',')
                        f.write('\n')
                if 'ID' in self.data[label].keys():
                    with open(os.path.join(outdir,label+'_ids.txt'),'w') as f:
                        output = self.data[label]['ID']
                        for id in output:
                            f.write(str(id)+'\n')               
        else:
            print 'No data to output'

#    def sim_feature()
    
        
    