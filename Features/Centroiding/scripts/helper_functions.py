# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:42:43 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u



###########################################################################
#::: plotting helper
###########################################################################    
def norm_scatter(ax, x, y, c, rasterized=True, label='', cmap='jet', norm_x=True, norm_y=True, vmin=None, vmax=None, lw=None ):
    if norm_x==True: x = x - np.nanmean(x)
    if norm_y==True: y = y - np.nanmean(y)
    sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, label=label, lw=lw)
    return ax, sc   
    


def mystr(x,digits=0):
    if np.isnan(x): return '.'
    elif digits==0: return str(int(round(x,digits)))
    else: return str(round(x,digits))
    
    
 
def deg2hmsdms(ra, dec):
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    return c.to_string('hmsdms', precision=2, sep=':')