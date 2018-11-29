#significant input and copied functions from T. Morton's VESPA code (all mistakes are my own)

#coords		--	RA and DEC of target in degrees. Needed for GAIA querying.
#        		Degrees, 0-360 and -90 to +90. List format [RA,DEC].

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy import stats
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
import subprocess as sp
import os, re
import time
AU = const.au.cgs.value
RSUN = const.R_sun.cgs.value
REARTH = const.R_earth.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400 #seconds
G = const.G.cgs.value
import logging


    
def semimajor(P,mtotal=1.):
    """
    Returns semimajor axis in AU given P in days, total mass in solar masses.
    """
    return ((P*DAY/2/np.pi)**2*G*mtotal*MSUN)**(1./3)/AU
    
def eclipse_probability(R1, R2, P, M1, M2):
    return (R1 + R2) *RSUN / (semimajor(P , M1 + M2)*AU)
                
def centroid_PDF_source(pos,centroiddat):
    cent_x, cent_y = centroiddat[0], centroiddat[1]
    sig_x, sig_y = centroiddat[2], centroiddat[3]
    return stats.multivariate_normal.pdf([pos[0],pos[1]],mean=[cent_x,cent_y],
        									 cov=[[sig_x**(1/2.),0],[0,sig_y**(1/2.)]])
        									 
            
def bgeb_prior(centroid_val, star_density, skyarea, P, r1=1.0, r2=1.0, m1=1.0, m2=1.0, f_binary=0.3, f_close=0.12):
    '''
    Centroid val is value at source (no integration over area). This allows comparison
    to planet_prior without having two planet_prior functions.
    '''    
    return centroid_val * skyarea  * star_density * f_binary * f_close * eclipse_probability(r1, r2, P, m1, m2)

def bgtp_prior(centroid_val, star_density, skyarea, P, r1=1.0, rp=1.0, m1=1.0, mp=0.0, f_planet=0.2):
    '''
    Centroid val is value at source (no integration over area). This allows comparison
    to planet_prior without having two planet_prior functions.
    '''
    return centroid_val * skyarea  * star_density * f_planet * eclipse_probability(r1, rp*REARTH/RSUN, P, m1, mp)
        
def eb_prior(centroid_val, P, r1=1.0, r2=1.0, m1=1.0, m2=1.0, f_binary=0.3, f_close=0.027):
    '''
    centroid pdf at source location
    f_binary = 0.3 (moe + di stefano 2017) - valid for 0.8-1.2 Msun! 
    				could improve to be average over all types?
    f_close  = 0.027 (moe + di stefano 2017) fraction of binaries with P between 3.2-32d
    
    eclipse prob
    works for defined source EBs too, just use appropriate centroid pdf value.
    '''
    return centroid_val * f_binary * f_close * eclipse_probability(r1, r2, P, m1, m2)

       
def heb_prior(centroid_val, P, r1=1.0, r2=1.0, m1=1.0, m2=1.0, f_triple=0.1, f_close=1.0):
    '''
    centroid pdf at source location
    f_triple = 0.1 (moe + di stefano 2017) - valid for 0.8-1.2 Msun! 
    				could improve to be average over all types?
    f_close  = 1.0 implies all triples have a close binary. May be over-generous	
    eclipse prob
    '''
    return centroid_val * f_triple * f_close * eclipse_probability(r1, r2, P, m1, m2)
        
def planet_prior(centroid_val, P, r1=1.0, rp=1.0, m1=1.0, mp=0.0, f_planet=0.2957):
    '''
    centroid pdf at source location
    planet occurrence (fressin, any planet<29d)
    eclipse prob
    works for defined source planets too, just use appropriate centroid pdf value.
    possibly needs a more general f_planet - as classifier will be using a range of planets.
    should prior then be the prior of being in the whole training set, rather than the specific depth seen?
    if so, need to change to 'fraction of ALL stars with planets' (i.e. including EBs etc).
    Also look into default radii and masses. Precalculate mean eclipse probability for training set?
    '''
    return centroid_val * f_planet * eclipse_probability(r1, rp*REARTH/RSUN, P, m1, mp)


def fp_fressin(rp,dr=None):
    if dr is None:
        dr = rp*0.3
    fp = quad(fressin_occurrence,rp-dr,rp+dr)[0]
    return max(fp, 0.001) #to avoid zero

def fressin_occurrence(rp):
    """
    Occurrence rates per bin from Fressin+ (2013)
    """
    rp = np.atleast_1d(rp)

    sq2 = np.sqrt(2)
    bins = np.array([1/sq2,1,sq2,2,2*sq2,
                    4,4*sq2,8,8*sq2,
                    16,16*sq2])
    rates = np.array([0,0.155,0.155,0.165,0.17,0.065,0.02,0.01,0.012,0.01,0.002,0])
    return rates[np.digitize(rp,bins)]


def trilegal_density(ra,dec,kind='target',maglim=21.75,area=1.0,mapfile=None):
    if kind=='interp' and mapfile is None:
        print('HEALPIX map file must be passed')
        return 0
    if kind not in ['target','interp']:
        print('kind not recognised. Setting kind=target')
        kind = 'target'
        
    if kind=='target':
        
        basefilename = 'trilegal_'+str(ra)+'_'+str(dec)
        h5filename = basefilename + '.h5'
        if not os.path.exists(h5filename):
            get_trilegal(basefilename,ra,dec,maglim=maglim,area=area)
        else:
            print('Using cached trilegal file. Sky area may be different.')

        if os.path.exists(h5filename):
        
            stars = pd.read_hdf(h5filename,'df')
            with pd.HDFStore(h5filename) as store:
                trilegal_args = store.get_storer('df').attrs.trilegal_args

            if trilegal_args['maglim'] < maglim:
                print('Re-calling trilegal with extended magnitude range')
                get_trilegal(basefilename,ra,dec,maglim=maglim,area=area)
                stars = pd.read_hdf(h5filename,'df')
            
            stars = stars[stars['TESS_mag'] < maglim]  #in case reading from file
        
            #c = SkyCoord(trilegal_args['l'],trilegal_args['b'],
             #               unit='deg',frame='galactic')

        #self.coords = c.icrs

            area = trilegal_args['area']*(u.deg)**2
            density = len(stars)/area
            return density.value
        else:
            return 0
    else:
        import healpy as hp
        #interpolate pre-calculated densities
        coord = SkyCoord(ra,dec,unit='deg')
        if np.abs(coord.galactic.b.value)<5:
            print('Near galactic plane, Trilegal density may be inaccurate.')
            
        #Density map will set mag limits
        densitymap = hp.read_map(mapfile)
        density = hp.get_interp_val(densitymap,ra,dec,lonlat=True)
        return density
    
#maglim of 21 used following sullivan 2015

def get_trilegal(filename,ra,dec,folder='.', galactic=False,
                 filterset='TESS_2mass_kepler',area=1,maglim=21,binaries=False,
                 trilegal_version='1.6',sigma_AV=0.1,convert_h5=True):
    """Runs get_trilegal perl script; optionally saves output into .h5 file
    Depends on a perl script provided by L. Girardi; calls the
    web form simulation, downloads the file, and (optionally) converts
    to HDF format.
    Uses A_V at infinity from :func:`utils.get_AV_infinity`.
    .. note::
        Would be desirable to re-write the get_trilegal script
        all in python.
    :param filename:
        Desired output filename.  If extension not provided, it will
        be added.
    :param ra,dec:
        Coordinates (ecliptic) for line-of-sight simulation.
    :param folder: (optional)
        Folder to which to save file.  *Acknowledged, file control
        in this function is a bit wonky.*
    :param filterset: (optional)
        Filter set for which to call TRILEGAL.
    :param area: (optional)
        Area of TRILEGAL simulation [sq. deg]
    :param maglim: (optional)
        Limiting magnitude in first mag (by default will be Kepler band)
        If want to limit in different band, then you have to
        got directly to the ``get_trilegal`` perl script.
    :param binaries: (optional)
        Whether to have TRILEGAL include binary stars.  Default ``False``.
    :param trilegal_version: (optional)
        Default ``'1.6'``.
    :param sigma_AV: (optional)
        Fractional spread in A_V along the line of sight.
    :param convert_h5: (optional)
        If true, text file downloaded from TRILEGAL will be converted
        into a ``pandas.DataFrame`` stored in an HDF file, with ``'df'``
        path.
    """
    if galactic:
        l, b = ra, dec
    else:
        try:
            c = SkyCoord(ra,dec)
        except:
            c = SkyCoord(ra,dec,unit='deg')
        l,b = (c.galactic.l.value,c.galactic.b.value)

    if os.path.isabs(filename):
        folder = ''

    if not re.search('\.dat$',filename):
        outfile = '{}/{}.dat'.format(folder,filename)
    else:
        outfile = '{}/{}'.format(folder,filename)
        
    NONMAG_COLS = ['Gc','logAge', '[M/H]', 'm_ini', 'logL', 'logTe', 'logg',
               'm-M0', 'Av', 'm2/m1', 'mbol', 'Mact'] #all the rest are mags

    AV = get_AV_infinity(l,b,frame='galactic')
    print(AV)
    if AV is not None:
      if AV<=1.5:
        trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,outfile)
        #cmd = './get_trilegal %s %f %f %f %i %.3f %.2f %s 1 %.1f %s' % (trilegal_version,l,b,
        #                                                          area,binaries,AV,sigma_AV,
        #                                                          filterset,maglim,outfile)
        #sp.Popen(cmd,shell=True).wait()
        if convert_h5:
            df = pd.read_table(outfile, sep='\s+', skipfooter=1, engine='python')
            df = df.rename(columns={'#Gc':'Gc'})
            for col in df.columns:
                if col not in NONMAG_COLS:
                    df.rename(columns={col:'{}_mag'.format(col)},inplace=True)
            if not re.search('\.h5$', filename):
                h5file = '{}/{}.h5'.format(folder,filename)
            else:
                h5file = '{}/{}'.format(folder,filename)
            df.to_hdf(h5file,'df')
            with pd.HDFStore(h5file) as store:
                attrs = store.get_storer('df').attrs
                attrs.trilegal_args = {'version':trilegal_version,
                                   'ra':ra, 'dec':dec,
                                   'l':l,'b':b,'area':area,
                                   'AV':AV, 'sigma_AV':sigma_AV,
                                   'filterset':filterset,
                                   'maglim':maglim,
                                   'binaries':binaries}
            os.remove(outfile)
    else:
        print('Skipping, AV > 10 or not found')

def trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,
					 outfile):
    """Calls TRILEGAL webserver and downloads results file.
    :param trilegal_version:
        Version of trilegal (only tested on 1.6).
    :param l,b:
        Coordinates (galactic) for line-of-sight simulation.
    :param area:
        Area of TRILEGAL simulation [sq. deg]
    :param binaries:
        Whether to have TRILEGAL include binary stars.  Default ``False``.
    :param AV:
    	Extinction along the line of sight.
    :param sigma_AV:
        Fractional spread in A_V along the line of sight.
    :param filterset: (optional)
        Filter set for which to call TRILEGAL.
    :param maglim:
        Limiting magnitude in mag (by default will be 1st band of filterset)
        If want to limit in different band, then you have to
        change function directly.
    :param outfile:
        Desired output filename.
    """
    webserver = 'http://stev.oapd.inaf.it'
    args = [l,b,area,AV,sigma_AV,filterset,maglim,1,binaries]
    mainparams = ('imf_file=tab_imf%2Fimf_chabrier_lognormal.dat&binary_frac=0.3&'
    			  'binary_mrinf=0.7&binary_mrsup=1&extinction_h_r=100000&extinction_h_z='
    			  '110&extinction_kind=2&extinction_rho_sun=0.00015&extinction_infty={}&'
    			  'extinction_sigma={}&r_sun=8700&z_sun=24.2&thindisk_h_r=2800&'
    			  'thindisk_r_min=0&thindisk_r_max=15000&thindisk_kind=3&thindisk_h_z0='
    			  '95&thindisk_hz_tau0=4400000000&thindisk_hz_alpha=1.6666&'
    			  'thindisk_rho_sun=59&thindisk_file=tab_sfr%2Ffile_sfr_thindisk_mod.dat&'
    			  'thindisk_a=0.8&thindisk_b=0&thickdisk_kind=0&thickdisk_h_r=2800&'
    			  'thickdisk_r_min=0&thickdisk_r_max=15000&thickdisk_h_z=800&'
    			  'thickdisk_rho_sun=0.0015&thickdisk_file=tab_sfr%2Ffile_sfr_thickdisk.dat&'
    			  'thickdisk_a=1&thickdisk_b=0&halo_kind=2&halo_r_eff=2800&halo_q=0.65&'
    			  'halo_rho_sun=0.00015&halo_file=tab_sfr%2Ffile_sfr_halo.dat&halo_a=1&'
    			  'halo_b=0&bulge_kind=2&bulge_am=2500&bulge_a0=95&bulge_eta=0.68&'
    			  'bulge_csi=0.31&bulge_phi0=15&bulge_rho_central=406.0&'
    			  'bulge_cutoffmass=0.01&bulge_file=tab_sfr%2Ffile_sfr_bulge_zoccali_p03.dat&'
    			  'bulge_a=1&bulge_b=-2.0e9&object_kind=0&object_mass=1280&object_dist=1658&'
    			  'object_av=1.504&object_avkind=1&object_cutoffmass=0.8&'
    			  'object_file=tab_sfr%2Ffile_sfr_m4.dat&object_a=1&object_b=0&'
    			  'output_kind=1').format(AV,sigma_AV)
    cmdargs = [trilegal_version,l,b,area,filterset,1,maglim,binaries,mainparams,
    		   webserver,trilegal_version]
    cmd = ("wget -o lixo -Otmpfile --post-data='submit_form=Submit&trilegal_version={}"
    	   "&gal_coord=1&gc_l={}&gc_b={}&eq_alpha=0&eq_delta=0&field={}&photsys_file="
    	   "tab_mag_odfnew%2Ftab_mag_{}.dat&icm_lim={}&mag_lim={}&mag_res=0.1&"
    	   "binary_kind={}&{}' {}/cgi-bin/trilegal_{}").format(*cmdargs)
    complete = False
    while not complete:
        notconnected = True
        busy = True
        print("TRILEGAL is being called with \n l={} deg, b={} deg, area={} sqrdeg\n "
        "Av={} with {} fractional r.m.s. spread \n in the {} system, complete down to "
        "mag={} in its {}th filter, use_binaries set to {}.".format(*args))
        sp.Popen(cmd,shell=True).wait()
        if os.path.exists('tmpfile') and os.path.getsize('tmpfile')>0:
            notconnected = False
        else:
            print("No communication with {}, will retry in 2 min".format(webserver))
            time.sleep(120)
        if not notconnected:
            with open('tmpfile','r') as f:
                lines = f.readlines()
            for line in lines:
                if 'The results will be available after about 2 minutes' in line:
                    busy = False
                    break
            sp.Popen('rm -f lixo tmpfile',shell=True)
            if not busy:
                filenameidx = line.find('<a href=../tmp/') +15
                fileendidx = line[filenameidx:].find('.dat')
                filename = line[filenameidx:filenameidx+fileendidx+4]
                print("retrieving data from {} ...".format(filename))
                while not complete:
                    time.sleep(120)
                    modcmd = 'wget -o lixo -O{} {}/tmp/{}'.format(filename,webserver,filename)
                    modcall = sp.Popen(modcmd,shell=True).wait()
                    if os.path.getsize(filename)>0:
                        with open(filename,'r') as f:
                            lastline = f.readlines()[-1]
                        if 'normally' in lastline:
                            complete = True
                            print('model downloaded!..')
                    if not complete:
                        print('still running...')        
            else:
                print('Server busy, trying again in 2 minutes')
                time.sleep(120)
    sp.Popen('mv {} {}'.format(filename,outfile),shell=True).wait()
    print('results copied to {}'.format(outfile))
       
   


        
def get_AV_infinity(ra,dec,frame='icrs'):
    """
    Gets the A_V exctinction at infinity for a given line of sight.
    Queries the NED database using ``curl``.
    .. note::
        It would be desirable to rewrite this to avoid dependence
        on ``curl``.
    :param ra,dec:
        Desired coordinates, in degrees.
    :param frame: (optional)
        Frame of input coordinates (e.g., ``'icrs', 'galactic'``)
    """
    coords = SkyCoord(ra,dec,unit='deg',frame=frame).transform_to('icrs')

    rah,ram,ras = coords.ra.hms
    decd,decm,decs = coords.dec.dms
    if decd > 0:
        decsign = '%2B'
    else:
        decsign = '%2D'
    url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon='+'%i' % rah + \
        '%3A'+'%i' % ram + '%3A' + '%05.2f' % ras + '&lat=%s' % decsign + '%i' % abs(decd) + '%3A' + '%i' % abs(decm) + '%3A' + '%05.2f' % abs(decs) + \
        '&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0'

    tmpfile = '/tmp/nedsearch%s%s.html' % (ra,dec)
    cmd = 'curl -s \'%s\' -o %s' % (url,tmpfile)
    sp.Popen(cmd,shell=True).wait()
    AV = None
    try:
        with open(tmpfile, 'r') as f:
            for line in f:
                m = re.search('V \(0.54\)\s+(\S+)',line)
                if m:
                    AV = float(m.group(1))
        os.remove(tmpfile)
    except:
        logging.warning('Error accessing NED, url={}'.format(url))

    return AV