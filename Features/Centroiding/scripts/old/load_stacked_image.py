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

import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from matplotlib.colors import LogNorm
import os, sys, socket
import glob
import fitsio
from ngtsio import ngtsio
from photutils.background import Background
from photutils import daofind
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture



def standard_fnames(fieldname, root=None):
    
    if (root is None):
        
        #::: on laptop (OS X)
        if sys.platform == "darwin":
            root = '/Users/mx/Big_Data/BIG_DATA_NGTS/2016/STACKED_IMAGES/'
                
        #::: on Cambridge servers
        elif 'ra.phy.cam.ac.uk' in socket.gethostname():
            root = '/appch/data/mg719/ngts_pipeline_output/STACKED_IMAGES/'
                              
    filename = glob.glob( os.path.join( root, 'DITHERED_STACK_'+fieldname+'*.fits' ) )[0]
                        
    return filename
        
        
        
def load(fieldname):
    
    filename = standard_fnames(fieldname)
    data = fitsio.read( filename )
    
    return data
    
    

def plot(fieldname, x, y, r=15, ax=None, show_apt=True):
    '''
        Note: it is important to transform the coordinates!
        
        resolution of stacked images: (8192, 8192)
        resolution of normal CCD images: (2048, 2048)
        -> scaling factor of 4
        
        x and y must additionally be shifted by - 0.5 + 0.5/scale 
        due to miss-match of different image origins in fits vs python vs C
        
        y axis is originally flipped if image is laoded via fitsio.read
    '''
    scale = 4
    x = x - 0.5 + 0.5/scale
    y = y - 0.5 + 0.5/scale
    
    stacked_image = load(fieldname)
#    bkg = np.nanmedian(stacked_image)
#    stacked_image -= bkg
    
    x0 = np.int( x - r )*scale #*4 as stacked images have 4 x higher resolution than normal images
    x1 = np.int( x + r )*scale
    y0 = np.int( y - r )*scale
    y1 = np.int( y + r )*scale
    
    stamp = stacked_image[ y0:y1, x0:x1 ] #x and y are switched in indexing
    bkg = Background(stamp, (50,50), filter_shape=(3, 3), method='median')
    stamp -= bkg.background
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        
    im = ax.imshow( stamp, norm=LogNorm(vmin=1, vmax=1000), interpolation='nearest', origin='lower', extent=1.*np.array([x0,x1,y0,y1])/scale )
    plt.colorbar(im)
    
    if show_apt == True:
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)

    fit_Gaussian2D(stamp)#, x, y, x0, x1, y0, y1)
    
    
    

def remove_bkg(z):
    bkg = Background(z, (50,50), filter_shape=(3, 3), method='median')
    
    plt.figure()
    im = plt.imshow(z, norm=LogNorm(vmin=1, vmax=1000), origin='lower', cmap='Greys')
    plt.colorbar(im)
    
    plt.figure()
    im = plt.imshow(bkg.background, origin='lower', cmap='Greys')
    plt.colorbar(im)
    
    plt.figure()
    im = plt.imshow(z - bkg.background, norm=LogNorm(vmin=1, vmax=1000), interpolation='nearest', origin='lower')
    plt.colorbar(im)
           
           
           

def fit_Gaussian2D(z):
    
#    xx, yy = np.meshgrid( np.arange(x0, x1), np.arange(y0, y1) )

    x = np.arange(0,z.shape[0])
    y = np.arange(0,z.shape[1])
    xx, yy = np.meshgrid( x,y )

    plt.figure()
    im = plt.imshow(z, norm=LogNorm(vmin=1, vmax=1000), origin='lower', cmap='Greys')
    plt.colorbar(im)
    
    
    #::: PHOTUTILS SOURCE DETECTION
    data = z
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)     
    sources = daofind(data - median, fwhm=4.0, threshold=4.*std)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=3.*4)
#    plt.figure()
#    plt.imshow(data, cmap='Greys', norm=LogNorm(vmin=1, vmax=1000))
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    #:::
    
    
    amplitude_A = np.max( z )
    x_A_mean = x[-1]/2
    y_A_mean = y[-1]/2
    x_A_stddev = 4. #assume FWHM of 1 pixel
    y_A_stddev = 4.
    
    amplitude_B = 0. #assume no second peak
    x_B_mean = x[-1]/2
    y_B_mean = y[-1]/2
    x_B_stddev = 4. #assume FWHM of 1 pixel
    y_B_stddev = 4.
    
    
    # Now to fit the data create a new superposition with initial
    # guesses for the parameters:
    gg_init = models.Gaussian2D(amplitude_A, x_A_mean, y_A_mean, x_A_stddev, y_A_stddev) \
                + models.Gaussian2D(amplitude_B, x_B_mean, y_B_mean, x_B_stddev, y_B_stddev)
    fitter = fitting.SLSQPLSQFitter()
    gg_fit = fitter(gg_init, xx, yy, z)
    
    print gg_fit
    print gg_fit.parameters
    print gg_fit(xx, yy)
    print gg_fit(0,0)
    print gg_fit(60,60)
    
    
    plt.figure()
    plt.imshow(gg_fit(xx,yy), label='Gaussian')


    
        
        
def test_plot(fieldname, x, y, r=15, ax=None):
    '''
        Note: it is important to transform the coordinates!
        
        resolution of stacked images: (8192, 8192)
        resolution of normal CCD images: (2048, 2048)
        
        y axis is originally flipped if image is laoded via fitsio.read
    '''
    scale = 4
    
    stacked_image = load(fieldname)
    stacked_image -= 0.95*np.nanmedian(stacked_image) #200
    
#    fig, ax1 = plt.subplots()
#    ax1.hist( stacked_image.flatten(), bins=np.linspace(200, 240, 50))
    
    x0 = np.int( x - r )*scale #*4 as stacked images have 4 x higher resolution than normal images
    x1 = np.int( x + r )*scale
    y0 = np.int( y - r )*scale
    y1 = np.int( y + r )*scale
    
    im = stacked_image[ y0:y1, x0:x1 ]
    
    if ax is None:
        fig, axes = plt.subplots(1,3,figsize=(18,6))
    
    for i in range(3):
        if i == 0:
            x = x
            y = y
            title = 'no shift'
        elif i == 1:
            x = x - 0.5 + 0.5/scale
            y = y - 0.5 + 0.5/scale
            title = 'shift - 3/4 * 0.5'
        elif i == 2:
            x = x -0.5
            y = y -0.5
            title = 'shift - 0.5'
        ax = axes[i]
        
        ax.imshow( im, norm=LogNorm(vmin=1, vmax=1000), interpolation='nearest', origin='lower', extent=1.*np.array([x0,x1,y0,y1])/scale )
        ax.scatter( x, y, c='r' )
        circle1 = plt.Circle((x, y), 3, color='r', fill=False, linewidth=3)
        ax.add_artist(circle1)
        ax.set_title(title)
        
        
        
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
    
    