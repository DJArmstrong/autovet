import fitsio
import os

#'Candidate' Class
class Candidate(object):

    """ Obtain meta and lightcurve information for a specific candidate. """
    
    def __init__(self,id,filepath,observatory='NGTS',label=None):
        """
        Take candidate and load lightcurve, dependent on observatory.
        
        Arguments:
        id          -- object identifier. Observatory dependent.
        filepath    -- location of object file. Observatory dependent.
        observatory -- source of candidate. Accepted values are: [NGTS,Kepler,K2]
        label       -- known classification, if candidate in training set. None if not known.
        """
        self.id = id   
        self.filepath = filepath
        self.obs = observatory 
        self.lightcurve = LoadLightcurve(self.id,self.filepath,self.obs)
        self.label = label    

    def LoadLightcurve(self.id,self.filepath,self.obs):
        """
        Load lightcurve from set observatory.
        
        Returns:
        lc -- lightcurve as dict. Minimum keys are [time, flux, error].
        """
        if self.obs=='NGTS':
            #self.field = os.path.split(filepath)[1][:11]
            lc = NGTSload(self.id,self.filepath)
        elif self.obs=='Kepler' or self.obs=='K2':
            lc = KepK2load(self.filepath)
        else:
            print 'Observatory not supported'
            
        return lc
        
    
    
def NGTSload(id,filepath):



def KepK2Load(infile,inputcol='PDCSAP_FLUX',inputerr='PDCSAP_FLUX_ERR'):
    """
    Loads a Kepler or K2 lightcurve, normalised and with NaNs removed.
    
    Inputs:
    infile -- input filepath
    
    Returns:
    lc -- lightcurve as dict, with keys time, flux, error
    """
    dat = fitsio.FITS(infile)
    time = dat[1].read('TIME')
    flux = dat[1].read(inputcol)
    err = dat[1].read(inputerr)
    nancut = np.isnan(time) | np.isnan(flux)
    norm = np.median(flux[~nancut])
    lc = {}
    lc['time'] = time[~nancut]
    lc['flux'] = flux[~nancut]/norm
    lc['error'] = err[~nancut]/norm
    return lc
