#featurerunner pipeline - take in input file, poss overview (such as TEST18), first row last row. Saves one output file.

#separate featurerunner code for synthetics? Only rerun of big change happened.
#external cat to link feature outputs

import sys

def Run(infile, outfile, firstrow, lastrow):
    """
    Calculates features from lightcurves of input candidates. Saves them combined with
    some ORION features.
    
    Inputs
    ----------
    infile (str)		Filepath for input file (the output of avpipe_setup.py)
    
    outfile (str)		Filepath for output file containing features
    
    firstrow  (int)		First row of input file to use (inclusive)
    
    lastrow (int)		Last row of input file to use (exclusive)
    
    Usage
    ----------
    "python avpipe_featurerun.py infile outfile firstrow lastrow"
    
    """
    from Loader.NGTS_MultiLoader_avpipe import NGTS_MultiLoader_avpipe

    featurestocalc = {'tdur_phase':[],'pmatch':[],'ntransits':[],'missingDataFlag':[],'SOM_Theta1':[],'SOM_Distance':[],
            		'Skew':[],'Kurtosis':[],'NZeroCross':[],'P2P_mean':[],'P2P_98perc':[],
            		'Peak_to_peak':[],'std_ov_error':[],'MAD':[],'RMS':[],'RMS_TDur':[],'MaxSecDepth':[],
            		'MaxSecPhase':[],'MaxSecSig':[],'MaxSecSelfSig':[],'Even_Odd_depthratio':[],'Even_Odd_depthdiff_fractional':[],
            		'TransitSNR':[],'PointDensity_ingress':[],'PointDensity_transit':[],'Scatter_transit':[],
            		'Fit_period':[],'Fit_chisq':[],'Fit_depthSNR':[],'Fit_t0':[],'Fit_aovrstar':[],'Fit_rprstar':[],
            		'Even_Fit_chisq':[],'Even_Fit_depthSNR':[],'Even_Fit_aovrstar':[],'Even_Fit_rprstar':[],
            		'Odd_Fit_chisq':[],'Odd_Fit_depthSNR':[],'Odd_Fit_aovrstar':[],'Odd_Fit_rprstar':[],
            		'Trapfit_t0':[],'Trapfit_t23phase':[],'Trapfit_t14phase':[],'Trapfit_depth':[],
            		'Even_Trapfit_t23phase':[],'Even_Trapfit_t14phase':[],'Even_Trapfit_depth':[],
            		'Odd_Trapfit_t23phase':[],'Odd_Trapfit_t14phase':[],'Odd_Trapfit_depth':[],
            		'Even_Odd_trapdurratio':[],'Even_Odd_trapdepthratio':[],'Full_partial_tdurratio':[],
            		'Even_Full_partial_tdurratio':[],'Odd_Full_partial_tdurratio':[]}
            		
    NGTS_MultiLoader_avpipe(infile, firstrow=firstrow, lastrow=lastrow, dofeatures=outfile, featoutfile=featoutfile, overwrite=True)

if __name__ == '__main__':
    if len(sys.argv)<5:
        print " Usage: python avpipe_featurerun.py infile outfile firstrow lastrow "
    else:
        Run(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    