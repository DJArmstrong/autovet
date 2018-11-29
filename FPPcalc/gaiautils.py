

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord


    
def GAIA_lookup(coords,search_radius=0.1, maglim=20.):
    """
    Cone searches GAIA data centred on candidate.
        
    Inputs:
    search_radius	--	cone search radius in degrees.
    maglim			--	maximum (gaia) magnitude to return
    """

    from astroquery.gaia import Gaia

    coord = SkyCoord(ra=coords[0], dec=coords[1], unit=(u.degree, u.degree),
        				 frame='icrs')
    radius = u.Quantity(searchradius, u.deg)
    j = Gaia.cone_search(coord, radius)
    r = j.get_results()
    r.pprint()   
    #cut to some limit of separation and magnitude
    #extract: sourceid, ra, dec, parallax, gmag, duplicated source?
    #append separation
    return r
    
    
    #calculate separation
    #calculate blending effect in TESS prf.

 def Contamination():
    """
    calculate contamination in aperture
    """
    pass
    #assume centroid data is 2d gaussian - offset, x error, y error
    
    #centroid has a distribution of likelihoods, from a measured offset and error
    #this overlaps with source positions etc
    #get prior for each background source (normalise over all)
    #including regions GAIA is incomplete under (behind stars, more for bright stars)
