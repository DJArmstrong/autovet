
import numpy as np
import fitsio
#import os


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
        self.lightcurve = self.LoadLightcurve()
        self.label = label    



    def LoadLightcurve(self):
        """
        Load lightcurve from set observatory.
        
        Returns:
        lc -- lightcurve as dict. Minimum keys are [time, flux, error].
        """
        if self.obs=='NGTS':
            #self.field = os.path.split(filepath)[1][:11]
            lc = self.NGTSload()
        elif self.obs=='Kepler' or self.obs=='K2':
            lc = self.KepK2load()
        else:
            print 'Observatory not supported'
            
        return lc
        
    
    
    def NGTSload(self):
        '''
        filepath = <root> + <ngts_version> + <fieldname_camera_year_ngts_version.fits>
        '''
        import ngtsio_autovet
        fnames = {'nights':self.filepath, 'sysrem':None, 'bls':None, 'canvas':None}
        dic = ngtsio_autovet.get( None, ['HJD','FLUX','FLUX_ERR'], obj_id=self.id, fnames=fnames )
        
        nan_zero_cut = np.isnan(dic['HJD']) | np.isnan(dic['FLUX']) | np.equal(dic['FLUX'], 0)    
        norm = np.median(dic['FLUX'][~nan_zero_cut])
        lc = {}
        lc['time'] = dic['HJD'][~nan_zero_cut]
        lc['flux'] = dic['FLUX'][~nan_zero_cut]/norm
        lc['error'] = dic['FLUX_ERR'][~nan_zero_cut]/norm
        del dic
        return lc
        
    
    
    def KepK2Load(self,inputcol='PDCSAP_FLUX',inputerr='PDCSAP_FLUX_ERR'):
        """
        Loads a Kepler or K2 lightcurve, normalised and with NaNs removed.
        
        Inputs:
        infile -- input filepath
        
        Returns:
        lc -- lightcurve as dict, with keys time, flux, error
        """
        dat = fitsio.FITS(self.filepath)
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




def test():
    can = Candidate('020057', '/wasp/scratch/TEST18/NG0409-1941_812_2016_TEST18.fits')
    print can.lightcurve
    
    
    
if __name__ == '__main__':
    test()