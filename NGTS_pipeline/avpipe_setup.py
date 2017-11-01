#NGTS pipeline

#setup pipeline - reads metadata, reads orion features, does pmatch cut, saves one input file.

import fitsio
import numpy as np
import sys
import glob

def Run(outfile, orioncall, version, label):
    """
    Scans a list of ORION output files and produces a file suitable for input into 
    autovetter. Cuts candidates failing pmatch test.
    
    Inputs
    ----------
    outfile (str)				Filepath for output file
    
    orioncall (str)				A call to all ORION files to use (e.g. '.../ORION*').
    							Can be one file.
    
    version (str)				NGTS run version, e.g. CYCLE1706. Used by ngtsio
    
    label (str)					Type of candidate. Typically 'real_candidate' or 'synth'
    
    Usage
    ----------
    " python avpipe_setup.py outfile orioncall version='' label='' "
    
    """
    fields, pers, diags = [], [], []
    orionfilelist = glob.glob(orioncall)      
    for infile in orionfilelist:
        dat = fitsio.FITS(infile)
        
        for cand in dat[4]:
        
            obj_id = cand['OBJ_ID'].strip(' ')
        
            candidates_on_same_id = np.where(dat[4]['obj_id'].read()==cand['OBJ_ID'])[0]  #will include same candidate

            cut = False
            if len(candidates_on_same_id)>1:  #there's a match aside from this candidate itself
                for matchcand in dat[4][candidates_on_same_id]:
                    if (matchcand['PERIOD']/cand['PERIOD']>0.998) & (matchcand['PERIOD']/cand['PERIOD']<1.002):
                        if matchcand['RANK'] < cand['RANK']:
                            cut = True
            if not cut:
                field = dat[2]['FIELD'].read()[0].strip(' ')
                fields.append(field)
                pers.append(cand['PERIOD']/86400.)
                #data to be saved if candidate passes pmatch cut:
                diags.append([version, obj_id, label, str(cand['PERIOD']/86400.), str(cand['EPOCH']/86400.), str(cand['WIDTH']/86400.),
                             str(int(cand['RANK'])), str(cand['DEPTH']), str(cand['DELTA_CHISQ']), str(cand['NPTS_TRANSIT']), str(cand['NUM_TRANSITS']),
                             str(cand['NBOUND_IN_TRANS']), str(cand['AMP_ELLIPSE']), str(cand['SN_ELLIPSE']), str(cand['GAP_RATIO']), str(cand['SN_ANTI']), str(cand['SDE'])])

    pers = np.array(pers)
    unique_field_ids = set(fields)
    passing_indices = []
    
    for field_id in unique_field_ids:
        ind = np.where( np.array(fields) == field_id )[0]
    
        #get the full list of periods in this field
        field_periods = pers[ ind ]
        
        #::: loop over all candidates in this field
        for candidate_index in ind:
            #apply pmatch cut. This is being put in early to reduce the numbers we have to deal with
            pmatch = np.sum((field_periods/pers[candidate_index]>0.998) & (field_periods/pers[candidate_index]<1.002))
            if pmatch <= 5:#cuts to ~27115 total in TEST18 (down from 96716 after cutting same object same per peaks)
                passing_indices.append(candidate_index)
    
    #set up headers
    with open(outfile,'w') as f:
        f.write('#fieldname,ngts_version,obj_id,label,per,t0,tdur,rank,DEPTH,DELTA_CHISQ,NPTS_TRANSIT,NUM_TRANSITS,NBOUND_IN_TRANS,AMP_ELLIPSE,SN_ELLIPSE,GAP_RATIO,SN_ANTI,SDE\n')

    #save candidates passing cut, with necessary data
    for s in passing_indices:

        with open(outfile,'a') as f:
            f.write(fields[s]+',')
            for entry in diags[s][:-1]:
                f.write(str(entry)+',')
            f.write(diags[s][-1]+'\n')
            

if __name__=='__main__':
    if len(sys.argv)<3:
        print "Usage: python avpipe_setup.py outfile orioncall version='' label=''"
    else:
        optionalinputs = {}	
        # Set up default values
        optionalinputs['version'] = 'CYCLE1706'
        optionalinputs['label'] = 'real_candidate'
        try:
            for inputval in sys.argv[3:]:
                key,val = inputval.split('=')
                optionalinputs[key] = val
            Run(sys.argv[1],sys.argv[2],optionalinputs['version'],optionalinputs['label'])
        except ValueError:
            print "Usage: python avpipe_setup.py outfile orioncall version='' label=''"
        