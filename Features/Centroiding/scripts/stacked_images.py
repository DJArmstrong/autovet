# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:27:46 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

#import warnings
import numpy as np
import matplotlib.pyplot as plt
#from astropy.modeling import models, fitting
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import os, sys, socket
import glob
import fitsio
#from ngtsio import ngtsio
try:
    from photutils.background import Background
except ImportError:
    from photutils.background import Background2D as Background
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from photutils import daofind
from astropy.stats import sigma_clipped_stats
#from photutils import CircularAperture


import ngtsio_v1_2_0_centroiding as ngtsio
    

def standard_fnames(fieldname, ngts_version, root=None):
    
    if (root is None):
        
        #::: on laptop (OS X)
        if sys.platform == "darwin":
            root = '/Users/mx/Big_Data/BIG_DATA_NGTS/2016/'+ngts_version+'/STACKED_IMAGES/'
                
        #::: on Cambridge servers
        elif 'ra.phy.cam.ac.uk' in socket.gethostname():
            root = '/appch/data/mg719/ngts_pipeline_output/'+ngts_version+'/STACKED_IMAGES/'
            
        #::: on Warwick servers (ngtshead)
        elif 'ngts' in socket.gethostname():
            root = '/home/maxg/ngts_pipeline_output/'+ngts_version+'/STACKED_IMAGES/'
   
    filename = glob.glob( os.path.join( root, 'DITHERED_STACK_'+fieldname+'*.fits' ) )
    
    if len(filename)>0: 
        filename = filename[-1]
    else:
        filename = None
                        
    return filename
        
        
        
def load(fieldname, ngts_version):
    
    filename = standard_fnames(fieldname, ngts_version)
    
    if filename is not None:
        data = fitsio.read( filename )
    else:
        data = None
        
    return data
    
    

def plot(fieldname, ngts_version, x, y, r=15, scale = 4, ax=None, show_apt=True, show_cbar=True, markersize=6):
    '''
        Note: it is important to transform the coordinates!
        
        resolution of stacked images: (8192, 8192)
        resolution of normal CCD images: (2048, 2048)
        -> scaling factor of 4
        
        x and y must additionally be shifted by - 0.5 + 0.5/scale 
        due to miss-match of different image origins in fits vs python vs C
        
        y axis is originally flipped if image is laoded via fitsio.read
    '''
    #::: settings
    x = x - 0.5 + 0.5/scale
    y = y - 0.5 + 0.5/scale
    
    
    #::: load field image
    stacked_image = load(fieldname, ngts_version)
    
    if stacked_image is not None:
        
        #::: cut out stamp/thumbnail
        x0 = np.int( x - r )*scale #*4 as stacked images have 4 x higher resolution than normal images
        x1 = np.int( x + r )*scale
        y0 = np.int( y - r )*scale
        y1 = np.int( y + r )*scale
        stamp = stacked_image[ y0:y1, x0:x1 ] #x and y are switched in indexing
        
        
        #::: plot
        extent=1.*np.array([x0,x1,y0,y1])/scale
        
        if ax is None:
            fig = remove_medianbkg_plot(stamp, x, y, r, scale, extent, show_apt)
    #        fig = remove_bkg_plot(stamp, x, y, r, scale, extent, show_apt)
    
        else:
            simple_plot(stamp, x, y, extent, show_apt, ax, show_cbar, markersize)
            fig = None
    
    else:
        fig = None
        
    return fig
    
    

def simple_plot(stamp, x, y, extent, show_apt, ax, show_cbar, markersize):    
    #::: remove bkg
    try:
        bkg = Background(stamp, (15,15), filter_shape=(3, 3), method='median')
        stamp -= bkg.background
    except:
        pass
    
    #::: plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        
    im = ax.imshow( stamp, norm=LogNorm(vmin=1, vmax=1000), interpolation='nearest', origin='lower', extent=extent )
    ax.set(xlabel='CCDX', ylabel='CCDY')
        
    if show_cbar==True:    
#        cbar = plt.colorbar(im)
#        cbar.set_label('FLUX')
        '''
        to properly show cbar in subplot:
        http://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
        '''
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax3, with 20% width of ax3
        cax = divider.append_axes("right", size="20%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('FLUX')
    
    if show_apt == True:
        ax.plot( x, y, 'ro', markersize=markersize )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
    
    
    
    
def remove_medianbkg_plot(z, x, y, r, scale, extent, show_apt=True, vmin=1, vmax=10000):
    mean, median, std = sigma_clipped_stats(z, sigma=3.0, iters=5)   
    bkg = median
    
    
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    
    ax = axes[0]
    im = ax.imshow(z, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', ylabel='CCDY', title='raw')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    ax = axes[1]
    im = ax.imshow(z - bkg, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', origin='lower', extent=extent, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', title='raw - bkg')
    plt.setp(ax.get_xticklabels(), rotation=45)
   
    if show_apt == True:
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
    
    ax = axes[2]
    rr = scale*r
    cc = scale*5
    image = ( z - bkg )[rr-cc:rr+cc,rr-cc:rr+cc]
    im = ax.imshow(image, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', extent=extent+np.array([r-5,-r+5,r-5,-r+5]), origin='lower', cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', title='raw - bkg (zoom)')

    plt.setp(ax.get_xticklabels(), rotation=45)
           
    if show_apt == True:
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
        
    plt.tight_layout()
    
    return fig

    
    
    
def remove_bkg_plot(z, x, y, r, scale, extent, show_apt=True, vmin=1, vmax=10000):
    bkg = Background(z, (30,30), filter_shape=(3,3), method='median')
    
    fig, axes = plt.subplots(1,4,figsize=(16,6))
    
    ax = axes[0]
    im = ax.imshow(z, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', extent=extent, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', ylabel='CCDY', title='raw')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    ax = axes[1]
    im = ax.imshow(bkg.background, origin='lower', extent=extent, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, format="%.1f")
    ax.set(xlabel='CCDX', title='bkg')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    ax = axes[2]
    im = ax.imshow(z - bkg.background, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', origin='lower', extent=extent, cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', title='raw - bkg')
    plt.setp(ax.get_xticklabels(), rotation=45)
   
    if show_apt == True:
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
    
    ax = axes[3]
    rr = scale*r
    cc = scale*5
    image = ( z - bkg.background )[rr-cc:rr+cc,rr-cc:rr+cc]
    im = ax.imshow(image, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest', extent=extent+np.array([r-5,-r+5,r-5,-r+5]), origin='lower', cmap='Greys')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set(xlabel='CCDX', title='raw - bkg (zoom)')
    plt.setp(ax.get_xticklabels(), rotation=45)
           
    if show_apt == True:
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
        
    plt.tight_layout()
    
    return fig




#def fit_Gaussian2D(z):
#    
##    xx, yy = np.meshgrid( np.arange(x0, x1), np.arange(y0, y1) )
#
#    x = np.arange(0,z.shape[0])
#    y = np.arange(0,z.shape[1])
#    xx, yy = np.meshgrid( x,y )
#
#    plt.figure()
#    im = plt.imshow(z, norm=LogNorm(vmin=1, vmax=1000), origin='lower', cmap='Greys')
#    plt.colorbar(im)
#    
#    
#    #::: PHOTUTILS SOURCE DETECTION
#    data = z
#    mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)     
#    sources = daofind(data - median, fwhm=4.0, threshold=4.*std)
#    positions = (sources['xcentroid'], sources['ycentroid'])
#    apertures = CircularAperture(positions, r=3.*4)
##    plt.figure()
##    plt.imshow(data, cmap='Greys', norm=LogNorm(vmin=1, vmax=1000))
#    apertures.plot(color='blue', lw=1.5, alpha=0.5)
#    #:::
#    
#    
#    amplitude_A = np.max( z )
#    x_A_mean = x[-1]/2
#    y_A_mean = y[-1]/2
#    x_A_stddev = 4. #assume FWHM of 1 pixel
#    y_A_stddev = 4.
#    
#    amplitude_B = 0. #assume no second peak
#    x_B_mean = x[-1]/2
#    y_B_mean = y[-1]/2
#    x_B_stddev = 4. #assume FWHM of 1 pixel
#    y_B_stddev = 4.
#    
#    
#    # Now to fit the data create a new superposition with initial
#    # guesses for the parameters:
#    gg_init = models.Gaussian2D(amplitude_A, x_A_mean, y_A_mean, x_A_stddev, y_A_stddev) \
#                + models.Gaussian2D(amplitude_B, x_B_mean, y_B_mean, x_B_stddev, y_B_stddev)
#    fitter = fitting.SLSQPLSQFitter()
#    gg_fit = fitter(gg_init, xx, yy, z)
#    
#    print gg_fit
#    print gg_fit.parameters
#    print gg_fit(xx, yy)
#    print gg_fit(0,0)
#    print gg_fit(60,60)
#    
#    
#    plt.figure()
#    plt.imshow(gg_fit(xx,yy), label='Gaussian')


    
        
        
#def test_plot(fieldname, x, y, r=15, ax=None):
#    '''
#        Note: it is important to transform the coordinates!
#        
#        resolution of stacked images: (8192, 8192)
#        resolution of normal CCD images: (2048, 2048)
#        
#        y axis is originally flipped if image is laoded via fitsio.read
#    '''
#    scale = 4
#    
#    stacked_image = load(fieldname)
#    stacked_image -= 0.95*np.nanmedian(stacked_image) #200
#    
##    fig, ax1 = plt.subplots()
##    ax1.hist( stacked_image.flatten(), bins=np.linspace(200, 240, 50))
#    
#    x0 = np.int( x - r )*scale #*4 as stacked images have 4 x higher resolution than normal images
#    x1 = np.int( x + r )*scale
#    y0 = np.int( y - r )*scale
#    y1 = np.int( y + r )*scale
#    
#    im = stacked_image[ y0:y1, x0:x1 ]
#    
#    if ax is None:
#        fig, axes = plt.subplots(1,3,figsize=(18,6))
#    
#    for i in range(3):
#        if i == 0:
#            x = x
#            y = y
#            title = 'no shift'
#        elif i == 1:
#            x = x - 0.5 + 0.5/scale
#            y = y - 0.5 + 0.5/scale
#            title = 'shift - 3/4 * 0.5'
#        elif i == 2:
#            x = x -0.5
#            y = y -0.5
#            title = 'shift - 0.5'
#        ax = axes[i]
#        
#        ax.imshow( im, norm=LogNorm(vmin=1, vmax=1000), interpolation='nearest', origin='lower', extent=1.*np.array([x0,x1,y0,y1])/scale )
#        ax.scatter( x, y, c='r' )
#        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
#        ax.add_artist(circle1)
#        ax.set_title(title)
        
        
        
if __name__ == '__main__':
    fieldname = 'NG0304-1115'
#    obj_id = 9861
#    obj_id = 992
    obj_id = 1703
#    obj_id = 6190
#    obj_id = 24503
#    obj_id = 7505
#    obj_id = 16335
    
    data = ngtsio.get(fieldname, ['CCDX','CCDY'], obj_id=obj_id)
    
    plot(fieldname, np.mean(data['CCDX']), np.mean(data['CCDY']), r=15) #obj 9861
#    plot('NG0304-1115', 1112, 75, r=15)
    
    
