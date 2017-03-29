import numpy as np
import fitsio
#import os
import kepselfflatten
#import itertools
        
try:
    from ngtsio import ngtsio
except ImportError:
    from Features.Centroiding.scripts import ngtsio_v1_1_1_autovet as ngtsio
            



#'Candidate' Class
class Candidate(object):

    """ Obtain meta and lightcurve information for a specific candidate. """
    

    def __init__(self,id,filepath,observatory='NGTS',field_dic=None,label=-10,candidate_data={'per':0.,'t0':0.,'tdur':0.}):
        """
        Take candidate and load lightcurve, dependent on observatory.
        
        Arguments:
        id          -- object identifier. Observatory dependent.
                       if NGTS, id should be either a single obj_id (int / string) or array_like containing multiple obj_ids (int / string)
        filepath    -- location of object file. Observatory dependent.
                       if NGTS, filepath should be array_like containing ['fieldname', 'ngts_version'] 
        observatory -- source of candidate. Accepted values are: [NGTS,Kepler,K2]
        label       -- known classification, if known. 0 = false positive, 1 = planet. -10 if not known.
        candidate_data   -- details of candidate transit. Should be a dict containing the keys 'per', 't0' and 'tdur' (planet period, epoch and transit duration, all in days). If not filled, certain features may not work. Candidate will be ignored in flattening procedure
        """
        self.id = id
        self.filepath = filepath
        self.obs = observatory 
        self.field_dic = field_dic
        self.lightcurve, self.info = self.LoadLightcurve()
        self.label = label
        self.candidate_data = candidate_data
        if observatory == 'Kepler' or observatory == 'K2':
            self.lightcurve_f = self.Flatten()
        else:
            self.lightcurve_f = self.lightcurve



    def LoadLightcurve(self):
        """
        Load lightcurve from set observatory.
        
        Returns:
        lc -- lightcurve as dict. Minimum keys are [time, flux, error].
        """
        if self.obs=='NGTS':
            #self.field = os.path.split(filepath)[1][:11]
            lc, info = self.NGTSload()
        elif self.obs=='Kepler' or self.obs=='K2':
            lc = self.KepK2load()
            info = None
        elif self.obs=='TESS':
            lc = self.TESSload()
            info = None
        else:
            print 'Observatory not supported'
            
        return lc, info
        
    
    
    def NGTSload(self):
        '''
        filepath = ['fieldname', 'ngts_version']
        obj_id = 1 or '000001' or [1,2,3] or ['000001','000002','000003']
        '''

        lc_keys = ['HJD', 'FLUX', 'FLUX_ERR']
        info_keys = ['OBJ_ID','FIELDNAME','NGTS_VERSION','FLUX_MEAN','RA','DEC','NIGHT','AIRMASS','CCDX','CCDY','CENTDX','CENTDY']
        
        #if there is no field_dic passed, use ngtsio to read out the info for a single object from the fits files
        if self.field_dic is None:
            fieldname, ngts_version = self.filepath
            dic = ngtsio.get( fieldname, lc_keys + info_keys, obj_id=str(self.id).zfill(6), ngts_version=ngts_version, silent=True, set_nan=True )
            
        #if a field_dic is passed (in memory), then select the specific object
        else:
            ind_obj = np.where( self.field_dic['OBJ_ID'] == self.id )[0]
            dic = {}
            for key in self.field_dic:
                try:
                    dic[key] = self.field_dic[key][ind_obj].flatten()
                except:
                    dic[key] = self.field_dic[key]
        
        norm = np.nanmedian(dic['FLUX'])
        lc = {}
        lc['time'] = dic['HJD']
        lc['flux'] = 1.*dic['FLUX']/norm
        lc['error'] = 1.*dic['FLUX_ERR']/norm
        
        info = {}
        for info_key in info_keys: 
            info[info_key] = dic[info_key]
            
        del dic
        return lc, info
        
        
    
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

    def TESSload(self):
        """
        Loads a TESS lightcurve (currently just the TASC WG0 simulated ones), normalised and with NaNs removed.
        
        Returns:
        lc -- lightcurve as dict, with keys time, flux, error. error is populated with zeros.
        """
        dat = np.genfromtxt(self.filepath)
        time = dat[:,0]
        flux = dat[:,1]
        err = np.zeros(len(time))
        nancut = np.isnan(time) | np.isnan(flux)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        del dat
        return lc

    def Flatten(self,winsize=6.,stepsize=0.3,polydegree=3,niter=10,sigmaclip=8.,gapthreshold=1.):
        """
        Flattens loaded lightcurve using a running polynomial
        
        Returns:
        lc_flatten -- flattened lightcurve as dict, with keys time, flux, error
        """
        lc = self.lightcurve
        if self.candidate_data['per']>0:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],lc['error'],np.zeros(len(lc['time'])),winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,lc['time'][0],False,True,self.candidate_data['per'],self.candidate_data['t0'],self.candidate_data['tdur'])        
        else:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],lc['error'],np.zeros(len(lc['time'])),winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,lc['time'][0],False,False,0.,0.,0.)
        lc_flatten = {}
        lc_flatten['time'] = lcf[:,0]
        lc_flatten['flux'] = lcf[:,1]
        lc_flatten['error'] = lcf[:,2]
        return lc_flatten

    