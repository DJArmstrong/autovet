
        coords		--	RA and DEC of target in degrees. Needed for GAIA querying.
        				Degrees, 0-360 and -90 to +90. List format [RA,DEC].



class GAIApriors(object):

    def __init__(self, coords, centroiddata )
    
    
    
    def GAIA_lookup(self,search_radius=0.1, maglim=20.):
        """
        Cone searches GAIA data centred on candidate.
        
        Inputs:
        search_radius	--	cone search radius in degrees.
        maglim			--	maximum (gaia) magnitude to return
        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia

        coord = SkyCoord(ra=self.coords[0], dec=self.coords[1], unit=(u.degree, u.degree),
        				 frame='icrs')
        radius = u.Quantity(searchradius, u.deg)
        j = Gaia.cone_search(coord, radius)
        r = j.get_results()
        r.pprint()   
        #extract: sourceid, ra, dec, parallax, gmag, duplicated source?
        #calculate separation
        #calculate blending effect in TESS prf.

    def Contamination()
        """
        calculate contamination in aperture
        """

    #assume centroid data is 2d gaussian - offset, x error, y error
    
    #centroid has a distribution of likelihoods, from a measured offset and error
    #this overlaps with source positions etc
    #get prior for each background source (normalise over all)
    #including regions GAIA is incomplete under (behind stars, more for bright stars)
    
    def Centroid_PDF_source(self,pos,centre,sig):
        return stats.multivariate_normal.pdf([pos[0],pos[1]],mean=[centre[0],centre[1]],
        									 cov=[[sig[0]**(1/2.),0],[0,sig[1]**(1/2.)]])
        									 
            
    def BgEB_prior()
    
    sky area = GAIA blanks  #centroid only comes in if this is offset from target.
    
    (star density) * (sky area) * (binary fraction) * (eclipse probability)
    
    #def prior(self):
    #    return (super(BEBPopulation, self).prior *
    #            self.density.to('arcsec^-2').value * #sky density
    #            np.pi*(self.maxrad.to('arcsec').value)**2) # sky area
                
    def EB_prior()
        centroid pdf at source location
        f_binary = 0.4
        eclipse prob
       
    def boundEB_prior()
        centroid pdf at source location
        f_triple = 0.12
        eclipse prob
        
    def planet_prior()
        centroid pdf at source location
        planet occurrence
        eclipse prob
                
    def sBEB_prior()
        centroid pdf at source location
        f_binary = 0.4
        eclipse prob
        
    def sBgplanet_prior()
        centroid pdf at source location
        planet occurrence
        eclipse prob