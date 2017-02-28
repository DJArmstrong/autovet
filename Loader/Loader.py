
import numpy as np
import fitsio
#import os
import kepselfflatten


#'Candidate' Class
class Candidate(object):

    """ Obtain meta and lightcurve information for a specific candidate. """
    
    def __init__(self,id,filepath,observatory='NGTS',label=-10,hasplanet={'per':0.}):
        """
        Take candidate and load lightcurve, dependent on observatory.
        
        Arguments:
        id          -- object identifier. Observatory dependent.
        filepath    -- location of object file. Observatory dependent.
        observatory -- source of candidate. Accepted values are: [NGTS,Kepler,K2]
        label       -- known classification, if known. 0 = false positive, 1 = planet. -10 if not known.
        hasplanet   -- if candidate has a known planet. If so, hasplanet should be a dict containing the keys 'per', 't0' and 'tdur' (planet period, epoch and transit duration, all in days)
        """
        self.id = id
        self.filepath = filepath
        self.obs = observatory 
        self.planet = hasplanet
        self.lightcurve = self.LoadLightcurve()
        self.label = label
        self.lightcurve_f = self.Flatten()



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
        
    
    
    def KepK2load(self,inputcol='PDCSAP_FLUX',inputerr='PDCSAP_FLUX_ERR'):
        """
        Loads a Kepler or K2 lightcurve, normalised and with NaNs removed.
        
        Returns:
        lc -- lightcurve as dict, with keys time, flux, error
        """
        if self.filepath[-4:]=='.txt':
            dat = np.genfromtxt(self.filepath)
            time = dat[:,0]
            flux = dat[:,1]
            err = dat[:,2]
        else:
            dat = fitsio.FITS(self.filepath)
            time = dat[1]['TIME'][:]
            flux = dat[1][inputcol][:]
            err = dat[1][inputerr][:]
        nancut = np.isnan(time) | np.isnan(flux) | np.isnan(err)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        if self.obs=='K2':
            linfit = np.polyfit(lc['time'],lc['flux'],1)
            lc['flux'] = lc['flux'] - np.polyval(linfit,lc['time']) + 1
        del dat
        return lc

    def Flatten(self,winsize=6.,stepsize=0.3,polydegree=3,niter=10,sigmaclip=8.,gapthreshold=1.):
        """
        Flattens loaded lightcurve using a running polynomial
        
        Returns:
        lc_flatten -- flattened lightcurve as dict, with keys time, flux, error
        """
        lc = self.lightcurve
        if self.planet['per']>0:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],lc['error'],np.zeros(len(lc['time'])),winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,lc['time'][0],False,True,self.planet['per'],self.planet['t0'],self.planet['tdur'])        
        else:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],lc['error'],np.zeros(len(lc['time'])),winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,lc['time'][0],False,False,0.,0.,0.)
        lc_flatten = {}
        lc_flatten['time'] = lcf[:,0]
        lc_flatten['flux'] = lcf[:,1]
        lc_flatten['error'] = lcf[:,2]
        return lc_flatten


def test():
    can = Candidate('020057', '/wasp/scratch/TEST18/NG0409-1941_812_2016_TEST18.fits')
    print can.lightcurve
    
    
    
if __name__ == '__main__':
    test()