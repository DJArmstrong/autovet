import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

def TIC_lookup(coords,search_radius=0.05555):
    """
    """
    scoord = SkyCoord(ra=coords[0], dec=coords[1], unit=(u.degree, u.degree),
        				 frame='icrs')
    radius = u.Quantity(search_radius, u.deg)
        
    catTable = Catalogs.query_region(scoord, catalog="Tic", radius=radius)

    return catTable['ID','ra','dec','Tmag','e_Tmag','objType','lumclass','rad','mass','Teff','GAIAmag',
    				'umag','gmag','rmag','imag','zmag','Jmag','Hmag','Kmag',
    				'w1mag','w2mag','w3mag','w4mag','Bmag','Vmag','e_Bmag','e_Vmag','e_GAIAmag',
    				'e_umag','e_gmag','e_rmag','e_imag','e_zmag','e_Jmag','e_Hmag','e_Kmag',
    				'e_w1mag','e_w2mag','e_w3mag','e_w4mag',
    				'plx','e_plx']

def TIC_byID(ID):
    """
    """
    catTable = Catalogs.query_criteria(ID=ID, catalog="Tic")

    return catTable['ID','ra','dec','Tmag','e_Tmag','objType','lumclass','rad','mass','Teff','GAIAmag',
    				'umag','gmag','rmag','imag','zmag','Jmag','Hmag','Kmag',
    				'w1mag','w2mag','w3mag','w4mag','Bmag','Vmag','e_Bmag','e_Vmag','e_GAIAmag',
    				'e_umag','e_gmag','e_rmag','e_imag','e_zmag','e_Jmag','e_Hmag','e_Kmag',
    				'e_w1mag','e_w2mag','e_w3mag','e_w4mag',
    				'plx','e_plx']
    
        
def GAIA_lookup(coords,search_radius=0.05555, maglim=24.):
    """
    Cone searches GAIA data centred on candidate.
        
    Inputs:
    search_radius	--	cone search radius in degrees.
    maglim			--	maximum (gaia) magnitude to return
    """

    from astroquery.gaia import Gaia

    scoord = SkyCoord(ra=coords[0], dec=coords[1], unit=(u.degree, u.degree),
        				 frame='icrs')
    radius = u.Quantity(search_radius, u.deg)
    j = Gaia.cone_search(scoord, radius)
    r = j.get_results()
    r.pprint()   
    #cut to some limit of separation and magnitude
    #extract: sourceid, ra, dec, parallax, gmag, duplicated source?
    #append separation
    return r
    
def isochrone_fit(objid,mags,plx):
    from isochrones.mist import MIST_Isochrone
    from isochrones import StarModel
 
    isomags = parse_mags(mags)    

#    get A_V? do we have this anywhere?
#    why are the u etc mags nan so often
    mist = MIST_Isochrone()
    mod = StarModel(mist, parallax=plx, **isomags)
    mod.fit(basename=str(objid))


def parse_mags(mags):
    mist_bands = ['B','BP','G','H','J','K','Kepler','RP','TESS','V','W1','W2','W3','g','i','r','z']
    isomags = {}
    for key in mags.keys():
        if key in mist_bands and np.isfinite(mags[key][0]):
            isomags[key] = (mags[key][0],mags[key][1])
    return isomags

   
    #calculate separation
    #calculate blending effect in TESS prf.



 #def calc_contam(targetsource,blendsources,aperture,prf):
 #   """
 #   calculate contamination in aperture
 #   """
 #   pass
    #need TESS PSF, APERTURE, SOURCE LOCATIONS AND MAGNITUDES
    
    #create simulated TESS PRF
    #2d gaussian
    
    #for s in blendsources:
    
        #convert its magnitude to flux (in units of the target star flux)
    #    flux = 10**((targetsource.mag['T'] - s.mag['T'])/2.5)
        
        #calculate fraction of its flux in aperture
        
            #translate ra and dec to pixel coordinates?
            #for pixel in aperture:
                #integrate prf centred on source over pixel
            #sum contributions
        
        
    #sum non-target flux contributions
    
    #compare against target flux contribution
    
    
    #assume centroid data is 2d gaussian - offset, x error, y error
    
    #centroid has a distribution of likelihoods, from a measured offset and error
    #this overlaps with source positions etc
    #get prior for each background source (normalise over all)
    #including regions GAIA is incomplete under (behind stars, more for bright stars)
