# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:44:37 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

'''
Convert Field RA (hms) and DEC (dms) (taken from OPIS)
into RA (h, float) and DEC (d, float)
'''
c = SkyCoord('04:10:47.8 -20:31:57.47', unit=(u.hourangle, u.deg))
print c.ra.hms, c.dec
print c.ra.hour, c.dec.deg