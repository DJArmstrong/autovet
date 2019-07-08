import numpy as np
#import fitsio #for super-big NGTS field files
import math
import os
from . import kepselfflatten
#import itertools
import pylab as p
p.ion()
from astropy.io import fits #general fits reading
import astropy.units as u
import astropy.constants as const
RSUN = const.R_sun.cgs.value
REARTH = const.R_earth.cgs.value

from ..FPPcalc import priorutils, utils #fix path to priorutils

class Source(object):

    def __init__(self,ra,dec,mags=[],type=None,undilutedflux=None,lclass=None,star_rad=None,star_mass=None, intic=False,plx=None):
        """
        """
        self.coords = [ra,dec]
        self.mags = {}
        for mag in mags:
            self.mags[mag[0]] = (float(mag[1][0]),float(mag[1][1]))
        self.type = type
        self.lumclass = lclass
        self.ud_flux = undilutedflux
        self.trilegal_density_beb = None
        self.trilegal_density_btp = None
        self.plradius = None
        self.star_rad = star_rad
        self.star_mass = star_mass
        self.intic = intic
        self.plx = plx
        
    def set_lightcurve(self, flux, contam):
        """
        """
        if contam == 1:
            self.ud_flux = np.nan
        else:
            self.ud_flux = (flux - contam) / (1. - contam)
    
    def set_pl_radius(self,depth,contam):
        """
        """
        if contam == 1:
            self.plradius = np.nan
        else:
            truedepth = depth / (1. - contam)
            self.plradius = (truedepth * self.star_rad**2)**(0.5) * RSUN/REARTH
        
       
#'Candidate' Class
class Candidate(object):
    '''
    Obtain meta and lightcurve information for a specific candidate.
    '''
    def __init__(	self,id,filepath,observatory='TESS',field_dic=None,label=-10,
    				candidate_data=None,stellar_radius=1.,field_periods=None, 
    				field_epochs=None, centroid=None):
        """
        Take candidate and load lightcurve, dependent on observatory.
        
        Arguments:
        id          -- 	Object identifier. Observatory dependent.
                       	If NGTS, id should be either a single obj_id (int / string) 
                       	or array_like containing multiple obj_ids (int / string)
        filepath    -- 	Location of object file. Observatory dependent.
                       	If NGTS, filepath should be array_like containing 
                       	['fieldname', 'ngts_version'] 
        observatory -- 	Source of candidate. Accepted values are: [NGTS,Kepler,K2,'TESS','ETE6']
        label       -- 	Known classification, if known. 0 = false positive, 1 = planet. 
        				-10 if not known.
        candidate_data   -- Details of candidate transit. Should be a dict containing 
        					the keys 'per', 't0', 'tdur' and 'depth' (planet period, epoch and 
        					transit duration, all in days, depth as fraction). 
        					If not filled, certain features may not work. 
        					Candidate will be ignored in flattening procedure.
        stellar_radius   -- Estimate of the host star radius, in solar radii. 
        					Used to estimate the planetary radius.
        field_periods    -- Array of all candidate periods from that field.
        field_epochs     -- Array of all candidate epochs from that field.
        centroid		--	Centroid information, in format [pos_x,pos_y,sig_x,sig_y]
        					of signal
        """
        dirname = os.path.dirname(__file__)
        if candidate_data == None : candidate_data = {'per':0.,'t0':0.,'tdur':0.,'depth':0.}
        self.id = id
        self.filepath = filepath
        self.obs = observatory 
        self.field_dic = field_dic
        self.label = label
        self.candidate_data = candidate_data
        self.stellar_radius = stellar_radius  #in solar, will default to 1 if not given
        self.mapfile = os.path.join(dirname,'../FPPcalc/trilegal_density_21.75_nside16.fits')
        if not os.path.exists(self.mapfile):
            self.mapfile = None

        self.lightcurve, self.info = self.LoadLightcurve()
        self.exp_time = np.median(np.diff(self.lightcurve['time']))
        self.field_periods = field_periods
        self.field_epochs = field_epochs
        if observatory == 'Kepler' or observatory == 'K2':
            self.lightcurve_f = self.Flatten()
        else:
            self.lightcurve_f = self.lightcurve
        if observatory=='TESS' or observatory=='ETE6':
            source = utils.TIC_byID(id)
            mags=[['TESS',(source['Tmag'],source['e_Tmag'])],
            	['G',(source['GAIAmag'],source['e_GAIAmag'])],
            	['B',(source['Bmag'],source['e_Bmag'])],
            	['V',(source['Vmag'],source['e_Vmag'])],
            	['u',(source['umag'],source['e_umag'])],
            	['g',(source['gmag'],source['e_gmag'])],
            	['r',(source['rmag'],source['e_rmag'])],
            	['i',(source['imag'],source['e_imag'])],
            	['z',(source['zmag'],source['e_zmag'])],
            	['J',(source['Jmag'],source['e_Jmag'])],
            	['H',(source['Hmag'],source['e_Hmag'])],
            	['K',(source['Kmag'],source['e_Kmag'])],
            	['W1',(source['w1mag'],source['e_w1mag'])],
            	['W2',(source['w2mag'],source['e_w2mag'])],
        		['W3',(source['w3mag'],source['e_w3mag'])],
        		['W4',(source['w4mag'],source['e_w4mag'])]]
            s = Source(source['ra'][0],source['dec'][0],mags=mags,type=source['objType'][0],lclass=source['lumclass'][0],
            	star_rad=source['rad'][0],star_mass=source['mass'][0],plx=(float(source['plx']),float(source['e_plx'])),intic=True)
            self.target_source = s
        else:
            self.target_source = None
        self.priors = {}
        self.centroid = centroid
        self.nearby_sources = []
       

    def LoadLightcurve(self):
        """
        Load lightcurve from set observatory.
        
        Returns:
        lc -- lightcurve as dict. Minimum keys are [time, flux, error].
        """
        if self.obs=='NGTS':
            lc, info = self.NGTSload()
        elif self.obs=='NGTS_synth':
            lc = self.NGTS_synthload()
            info = None
        elif self.obs=='Kepler' or self.obs=='K2':
            lc = self.KepK2load()
            info = None
        elif self.obs=='TASCsim':
            lc = self.TASCsimload()
            info = None
        elif self.obs=='ETE6':
            lc = self.TESS_ETE6load()
            info = None
        elif self.obs=='TESS':
            lc = self.TESSload()
            info = None
        else:
            print('Observatory not supported')
        return lc, info
        
    def NGTS_synthload(self):
        '''
        Loads a synthetic NGTS lightcurve, using self.filepath.
        
        Returns:
        lc -- 	dict
        		Lightcurve with keys time, flux, error. Error is populated with zeros.
        '''
        dat = np.genfromtxt(self.filepath)
        time = dat[:,0]
        flux = dat[:,1]
        err = dat[:,2]
        nancut = np.isnan(time) | np.isnan(flux) | np.isnan(err) | (flux==0)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]/86400.
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        return lc
 
    def NGTSload(self):
        '''
        Loads an NGTS lightcurve using destination defined in init. 
        Will only work on ngtshead.
        
        self.filepath = ['fieldname', 'ngts_version']
        self.id = 1 or '000001' or [1,2,3] or ['000001','000002','000003']
        '''
        from ngtsio import ngtsio

        lc_keys = ['HJD', 'SYSREM_FLUX3', 'FLUX3_ERR']
        info_keys = [	'OBJ_ID', 'FIELDNAME', 'NGTS_VERSION', 'FLUX_MEAN', 'RA', 'DEC',
        				'NIGHT', 'AIRMASS', 'CCDX', 'CCDY', 'CENTDX', 'CENTDY']
        passflag = False
        
        #if there is no field_dic passed, use ngtsio to read out the info for 
        #a single object from the fits files
        if self.field_dic is None:
            if self.filepath is not None:
                fieldname, ngts_version = self.filepath
                dic = ngtsio.get( 	fieldname, ngts_version, lc_keys + info_keys, 
                					obj_id=str(self.id).zfill(6), silent=True, 
                					set_nan=True )
            else:
                passflag = True
        #if a field_dic was passed (in memory), then select the specific object 
        #and store it into dic
        else:
            ind_obj = np.where( self.field_dic['OBJ_ID'] == self.id )[0]
            if (len(ind_obj)>0) and ('FLUX3_ERR' in self.field_dic.keys()):
                dic = {}
                for key in ['OBJ_ID','FLUX_MEAN','RA','DEC']:
                    dic[key] = self.field_dic[key][ind_obj][0]
                for key in ['HJD','SYSREM_FLUX3','FLUX3_ERR','CCDX','CCDY',
                			'CENTDX','CENTDY']:
                    dic[key] = self.field_dic[key][ind_obj].flatten()
                for key in ['FIELDNAME','NGTS_VERSION','NIGHT','AIRMASS']:
                    dic[key] = self.field_dic[key]
            else:  #this is a candidate that wasn't in the field_dic
                passflag = True
        if passflag:
            lc = {}
            lc['time'] = np.zeros(10)-10
            lc['flux'] = np.zeros(10)-10
            lc['error'] = np.zeros(10)-10
            info = 0
            return lc, info
                
        nancut = np.isnan(dic['HJD']) | np.isnan(dic['SYSREM_FLUX3']) | np.isnan(dic['FLUX3_ERR']) | np.isinf(dic['HJD']) | np.isinf(dic['SYSREM_FLUX3']) | np.isinf(dic['FLUX3_ERR']) | (dic['SYSREM_FLUX3']==0) 
        norm = np.median(dic['SYSREM_FLUX3'][~nancut])
        lc = {}
        lc['time'] = dic['HJD'][~nancut]/86400.
        lc['flux'] = 1.*dic['SYSREM_FLUX3'][~nancut]/norm
        lc['error'] = 1.*dic['FLUX3_ERR'][~nancut]/norm
        
        info = {}
        for info_key in info_keys: 
            if isinstance(dic[info_key], np.ndarray):
                info[info_key] = dic[info_key][~nancut]
            else:
                info[info_key] = dic[info_key]
        info['nancut'] = nancut
        
        del dic
        return lc, info
        
        
    
    def KepK2load(self,inputcol='PDCSAP_FLUX',inputerr='PDCSAP_FLUX_ERR'):
        """
        Loads a Kepler or K2 lightcurve, normalised and with NaNs removed.
        
        Returns:
        lc -- 	dict
        		Lightcurve, with keys time, flux, error
        """
        if self.filepath[-4:]=='.txt':
            dat = np.genfromtxt(self.filepath)
            time = dat[:,0]
            flux = dat[:,1]
            err = dat[:,2]
        else:
            dat = fits.open(self.filepath)
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

    def TASCsimload(self):
        """
        Loads a TESS lightcurve (currently just TASC WG0 simulated ones, 
        i.e. time,flux,err txt files), normalised and with NaNs removed.
        
        Returns:
        lc -- 	dict
        		Lightcurve with keys time, flux, error. Error is populated with zeros.
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

    def TESS_ETE6load(self):
        """
        Loads an ETE6 TESS simulated extracted lightcurve, normalised wity NaNs removed.
        
        Returns:
        lc -- 	dict
        		Lightcurve with keys time, flux, error. Error is populated with zeros.
        """
        dat = fits.open(self.filepath)
        time = dat[1].data['TIME']
        flux = dat[1].data['PDCSAP_FLUX']
        err = dat[1].data['PDCSAP_FLUX_ERR']
        nancut = np.isnan(time) | np.isnan(flux)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        del dat
        return lc
        
    def TESSload(self):
        """
        Loads a TESS lightcurve, normalised wity NaNs removed.
        
        Returns:
        lc -- 	dict
        		Lightcurve with keys time, flux, error. Error is populated with zeros.
        """
        dat = fits.open(self.filepath)
        time = dat[1].data['TIME']
        if 'PDCSAP_FLUX' in dat[1].columns.names:
            flux = dat[1].data['PDCSAP_FLUX']
            err = dat[1].data['PDCSAP_FLUX_ERR']
        else:
            flux = dat[1].data['SAP_FLUX']
            err = np.zeros(len(flux))
        nancut = np.isnan(time) | np.isnan(flux) | (flux==0)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        del dat
        return lc        

    def Flatten(self,winsize=6.,stepsize=0.3,polydegree=3,niter=10,sigmaclip=8.,
    			gapthreshold=1.):
        """
        Flattens loaded lightcurve using a running polynomial
        
        Returns:
        lc_flatten -- 	dict
        				Flattened lightcurve as dict, with keys time, flux, error
        """
        lc = self.lightcurve
        if self.candidate_data['per']>0:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],
            								lc['error'],np.zeros(len(lc['time'])),
            								winsize,stepsize,polydegree,niter,sigmaclip,
            								gapthreshold,lc['time'][0],False,True,
            								self.candidate_data['per'],
            								self.candidate_data['t0'],
            								self.candidate_data['tdur'])        
        else:
            lcf = kepselfflatten.Kepflatten(lc['time']-lc['time'][0],lc['flux'],
            								lc['error'],np.zeros(len(lc['time'])),
            								winsize,stepsize,polydegree,niter,sigmaclip,
            								gapthreshold,lc['time'][0],False,False,
            								0.,0.,0.)
        lc_flatten = {}
        lc_flatten['time'] = lcf[:,0]
        lc_flatten['flux'] = lcf[:,1]
        lc_flatten['error'] = lcf[:,2]
        return lc_flatten

   
    def find_nearby(self):
        '''
        Cross matches to TIC (via MAST) and GAIA. Identifies all sources within 200 arcsec.
        Starts with TIC and appends GAIA sources which aren't found. Gaia to TESS mags
        done oversimply ( T = G-0.5 ). Matching is also oversimple - idea is that TIC will 
        incorporate GAIA DR2 eventually and make the GAIA step redundant.
        
        200 arcsec can be trimmed down on knowledge of TESS PSF.
        '''
        s_rad = 200.*u.arcsec.to('degree') #~10 TESS pixels
                
        #scan TIC for nearby sources - at some point, the TIC will contain all of GAIA DR2
        #with standardised magnitude conversions, etc
        ticsources = utils.TIC_lookup(self.target_source.coords,search_radius=s_rad)
        
        #scan GAIA for nearby sources
        gaiasources = utils.GAIA_lookup(self.target_source.coords,search_radius=s_rad) #check units, limits
        
        self.targetmatched = False
        target = np.where(ticsources['ID']==str(self.id))[0][0]
        print(target)
        
        nearsources = []
        for source in ticsources:
            if int(source['ID']) != self.id:
                mags=[['TESS',(source['Tmag'],source['e_Tmag'])],
            	    ['G',(source['GAIAmag'],source['e_GAIAmag'])],
            	    ['B',(source['Bmag'],source['e_Bmag'])],
            	    ['V',(source['Vmag'],source['e_Vmag'])],
            	    ['u',(source['umag'],source['e_umag'])],
            	    ['g',(source['gmag'],source['e_gmag'])],
            	    ['r',(source['rmag'],source['e_rmag'])],
            	    ['i',(source['imag'],source['e_imag'])],
            	    ['z',(source['zmag'],source['e_zmag'])],
            	    ['J',(source['Jmag'],source['e_Jmag'])],
            	    ['H',(source['Hmag'],source['e_Hmag'])],
            	    ['K',(source['Kmag'],source['e_Kmag'])],
            	    ['W1',(source['w1mag'],source['e_w1mag'])],
            	    ['W2',(source['w2mag'],source['e_w2mag'])],
        		    ['W3',(source['w3mag'],source['e_w3mag'])],
        		    ['W4',(source['w4mag'],source['e_w4mag'])]]
                s = Source(source['ra'],source['dec'],mags=mags,lclass=source['lumclass'],star_rad=source['rad'],star_mass=source['mass'],intic=True,plx=(float(source['plx']),float(source['e_plx'])))
                nearsources.append(s)
            
        #match GAIA to TIC IDs
        for source in ticsources:
            separations = (np.power(source['ra']-gaiasources['ra'],2) + np.power(source['dec']-gaiasources['dec'],2))**(1/2.)
            closest = np.argmin(separations)
            if separations[closest] <= 0.001:# 3.6 arcsec, bit arbitrary
                if np.abs(gaiasources[closest]['phot_g_mean_mag']-0.5 - source['Tmag']) <= 1: #because the G-T calibration is very rough
                    #matched!
                    if int(source['ID']) == self.id:
                        self.targetmatched = True
                    gaiasources.remove_row(closest)
             
        if not self.targetmatched:            
            for row,source in enumerate(gaiasources):
                separation = (np.power(source['ra']-ticsources[target]['ra'],2) + np.power(source['dec']-ticsources[target]['dec'],2))**(1/2.)
                magdiff = source['phot_g_mean_mag']-0.5 - ticsources[target]['Tmag']
                if separation < 0.003 and np.abs(magdiff) < 0.5:
                    gaiasources.remove_row(row)
                    print('WARNING - GAIA Matched to TIC with separation '+str(separation))
                    self.targetmatched = True
                    
        for row,source in enumerate(gaiasources):            
            #stopgap: T = G-0.5
            #could improve by looking up other catalogues but gets time consuming and awkward
            #TIC will replace this GAIA search anyway, likely before first data release
            mags = [['G',(source['phot_g_mean_mag'],0.01)],['TESS',(source['phot_g_mean_mag']-0.5,0.5)]] #guessed errors..
            s = Source(source['ra'],source['dec'],mags=mags,star_rad=source['radius_val'],star_mass=1.0)
            nearsources.append(s)
        
        if not self.targetmatched:
            print('PROBLEM - Could not match GAIA with TIC target')
        self.nearby_sources = nearsources

    def tabulate_nearby(self):
        """
        """
        if len(self.nearby_sources) == 0:
            self.find_nearby()
        
        for source in self.nearby_sources:
            separation = (np.power(self.target_source.coords[0]-source.coords[0],2) + np.power(self.target_source.coords[1]-source.coords[1],2))**(1/2.)
            print(source.mags['TESS'],separation)
    
    def plot_nearby(self):
        """
        """
        if len(self.nearby_sources) == 0:
            self.find_nearby()
            
        import matplotlib.cm as cm
        from matplotlib.patches import Rectangle
        fig,ax = p.subplots(1,1)
        for source in self.nearby_sources:
            if source.mags['TESS'][0] - self.target_source.mags['TESS'][0] <=7.5:
                if source.intic:
                    ax.scatter(source.coords[0],source.coords[1],c=source.mags['TESS'][0],cmap=cm.viridis,marker='D',vmin=self.target_source.mags['TESS'][0],vmax=self.target_source.mags['TESS'][0]+7.5)
                    ax.text(source.coords[0],source.coords[1],str(source.mags['TESS'][0])[:4])
                else:
                    ax.scatter(source.coords[0],source.coords[1],c=source.mags['TESS'][0],cmap=cm.viridis,vmin=self.target_source.mags['TESS'][0],vmax=self.target_source.mags['TESS'][0]+7.5)
                    ax.text(source.coords[0],source.coords[1],str(source.mags['TESS'][0])[:4])

        ax.plot(self.target_source.coords[0],self.target_source.coords[1],'mD')
        ax.text(self.target_source.coords[0],self.target_source.coords[1],str(self.target_source.mags['TESS'][0])[:4])
        
        #TESS pixel
        pix = 20*u.arcsec.to('deg')
        
        ax.add_patch(Rectangle((self.target_source.coords[0]-5*pix,self.target_source.coords[1]+5*pix),pix,pix,fill=None))
        p.xlim(self.target_source.coords[0]-10*pix,self.target_source.coords[0]+10*pix)
        p.ylim(self.target_source.coords[1]-10*pix,self.target_source.coords[1]+10*pix)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title('Red=TIC. Blue=GAIA DR2. Magenta=Target')
                
        
    def undilute_lightcurves(self):
        '''
        
        '''
        if not self.nearby_sources:
            self.find_nearby()
        
        if len(self.nearby_sources)>0:
            contam = utils.calc_contam(self.target_source,self.nearby_sources)
            self.target_source.set_lightcurve(self.lightcurve['flux'],contam)
            self.target_source.set_pl_radius(self.candidate_data['depth'],contam)
            
            for sidx in range(len(self.nearby_sources)):
                #calc contamination
                #will be done by TIC? Very hard without apertures etc.
                #What did sims do? Any contam at all? 
                contamination = utils.calc_contam(self.nearby_sources[sidx],
            					[self.target_source]+self.nearby_sources[:sidx]+self.nearby_sources[sidx+1:])
                self.nearby_sources[sidx].set_lightcurve(self.lightcurve['flux'],contamination)
                self.nearby_sources[sidx].set_pl_radius(self.candidate_data['depth'],contamination)
        else:
            contam = 0.
            self.target_source.set_lightcurve(self.lightcurve['flux'],contam)
            self.target_source.set_pl_radius(self.candidate_data['depth'],contam)
        
    
    def get_priors(self, include_neighbours=False, trilegal_interp=True):
        '''
        '''
        if self.centroid is None:
            self.centroid = [self.target_source.coords[0],self.target_source.coords[1],
            				20*u.arcsec.to('degree'),20*u.arcsec.to('degree')]
        
        if self.candidate_data['per'] == 0 or self.candidate_data['depth']==0:
            print('Candidate period and transit depth must be defined')
            return 0
        
        #maglims are based off a 50% max EB eclipse, and a 2% max planet transit, requiring
        #a 0.1% eclipse on the target to be detected.
        if self.target_source.trilegal_density_beb is None:
            if trilegal_interp:
                self.target_source.trilegal_density_beb = priorutils.trilegal_density(self.target_source.coords[0],
            										self.target_source.coords[1],kind='interp',
            										mapfile=self.mapfile)
            else:
                self.target_source.trilegal_density_beb = priorutils.trilegal_density(self.target_source.coords[0],
            										self.target_source.coords[1],kind='target',
            										maglim=self.target_source.mags['T']+6.75)

        if self.target_source.trilegal_density_btp is None:
            if trilegal_interp:
                self.target_source.trilegal_density_btp = priorutils.trilegal_density(self.target_source.coords[0],
            										self.target_source.coords[1],kind='interp',
            										mapfile=self.mapfile)
            else:
                self.target_source.trilegal_density_btp = priorutils.trilegal_density(self.target_source.coords[0],
            										self.target_source.coords[1],kind='target',
            										maglim=self.target_source.mags['T']+3.25)

        if include_neighbours:
            self.find_nearby()
            self.undilute_lightcurves()        
        else:
            self.target_source.set_lightcurve(self.lightcurve['flux'],0.)
            self.target_source.set_pl_radius(self.candidate_data['depth'],0.)

            
        targetcentroid = priorutils.centroid_PDF_source(self.target_source.coords,self.centroid) #check units
        
        #need to add radii of host and eclipsing body (estimated from undiluted curve?)
        self.priors['pl'] = priorutils.planet_prior(targetcentroid,self.candidate_data['per'],
        					r1=self.target_source.star_rad,rp=self.target_source.plradius, m1=self.target_source.star_mass)
        self.priors['eb'] = priorutils.eb_prior(targetcentroid,self.candidate_data['per'],r1=self.target_source.star_rad,m1=self.target_source.star_mass,r2=self.target_source.star_rad,m2=self.target_source.star_mass)
        self.priors['heb'] = priorutils.heb_prior(targetcentroid,self.candidate_data['per'],r1=self.target_source.star_rad,m1=self.target_source.star_mass,r2=self.target_source.star_rad,m2=self.target_source.star_mass)

        self.priors['seb'] = []
        self.priors['stp'] = []
        
        #take smallest of centroid 3sigma and GAIA worst resolution
        blind_area = math.pi * np.min([2.2*u.arcsec.to('degree'),3*(np.sqrt(self.centroid[2]**2+self.centroid[3]**2))])**2
        
        #linear blind area: 2.2arcsec for mag diff of 5 or GAIA faint limit, to mag diff of 3 at 1.2 arcsec
        #source Ziegler et al 2018 (Robo-AO--GAIA test)
        #not enough larger mag diffs tested (due to faint Kepler sample?)
        #effect is this line below 2.2 arcsec, then from there to GAIA faint limit
        
        #TODO - ARE STARS FAINTER THAN GAIA FAINT LIMIT POSSIBLE FP SOURCES FOR A GIVEN TARGET?
        #IF SO: NEED ADDITION TO BEB PRIOR COVERING WHOLE CENTROID AREA
        
        #ETE6 implies 1 arcsec centroid errors direct from TESS. So at 3sigma, GAIA more constraining
        #but centroid should often be able to select a GAIA source.
        
        sourcecentroids = [targetcentroid]
        
        #what about overlapping background blind areas? We are potentially overestimating the blind area
        #especially as 2.2 arcsec is very conservative
        
        #turned off for ETE6
        if include_neighbours:
            for source in self.nearby_sources:
                sourcecentroid.append(priorutils.centroid_PDF_source(source.coords,self.centroid))  #check units   
                self.priors['seb'].append(priorutils.eb_prior(sourcecentroids[-1], self.candidate_data['per'],r1=source.star_rad,m1=source.star_mass,r2=source.star_rad,m2=source.star_mass))
                self.priors['stp'].append(priorutils.planet_prior(sourcecentroids[-1], self.candidate_data['per'],r1=source.star_rad,rp=source.plradius, m1=source.star_mass))
            
        #below only considers blind area from target star (still with conservative blind radius)
        
        self.priors['beb'] = priorutils.bgeb_prior(sourcecentroids[0], self.target_source.trilegal_density_beb, 
        											blind_area, self.candidate_data['per'])
        self.priors['btp'] = priorutils.bgtp_prior(sourcecentroids[0], self.target_source.trilegal_density_btp, 
        											blind_area, self.candidate_data['per'])
        
    #def get_isochrones(self):
    #    """
    #    """
    #    if self.target_source is None:
    #        print('No target source identified')
    #        return 0
            
        
        