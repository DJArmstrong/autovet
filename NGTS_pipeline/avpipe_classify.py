#classifier pipeline - takes in featurerunner output, same for synth, runs classification, saves cvprobs and features in meta fits file
import fitsio
import numpy as np

def Run(featfile, outfile, synthfile):
    """
    Runs a Random Forest classifier on previously calculated features. Outputs FITS
    file of classifier probabilities and features used.
    
    Inputs
    ----------
    featfile (str)		Filepath to real candidate features file.
    					(the output of avpipe_featurerun.py)
    
    synthfile (str)		Filepath to synthetic candidate features file.
    
    outfile  (str)		Filepath to save output to.
        
    Usage
    ----------
    " python avpipe_classify.py featfile outfile synthfile "
    
    """
    from Classify import Learner
    from Features.FeatureData import FeatureData

    #merge candidate and synthetic features into training set
    fd = FeatureData()
    fd.addData(featfile,'real_candidate')
    fd.addData(synthfile,'synth')
    fd.outputTrainingSet('temp_trainset.txt')

    #set up training set and classifier
    tset = Learner.TrainingSet('temp_trainset.txt')
    cl = Learner.Classifier(classifier_args={'n_estimators':300,'max_depth':8})

    #perform initial training 
    #cl.train(tset)
    #verify oobscore>?

    #cross-validated class probabilities
    cvprobs_md8=cl.crossValidate(tset)

    #output
    ridx=np.where(tset.known_classes=='real_candidate')[0]
    sidx=np.where(tset.known_classes=='synth')[0]
    
    real_data = [tset.ids[ridx],cvprobs_md8[ridx,1]]
    names=['ID','P_PLANET']
    real_featuredata = [tset.features[ridx,i] for i in range(len(tset.featurenames))]
    featurenames = list(tset.featurenames)
    
    fitsout = FITS(outfile,'rw')
    fits.write(real_data, names=names)
    fits.write(real_featuredata, names=featurenames)

    synth_data = [tset.ids[sidx],cvprobs_md8[sidx,1]]
    synth_featuredata = [tset.features[sidx,i] for i in range(len(tset.featurenames))]
    
    fitsout = FITS(outfile[:-4]+'_synth.fits','rw')
    fits.write(synth_data, names=names)
    fits.write(synth_featuredata, names=featurenames)


if __name__=='__main__':
    if len(sys.argv)<4:
        print "Usage: python avpipe_classify.py featfile outfile synthfile "
    else:
        Run(sys.argv[1],sys.argv[2],sys.argv[3])