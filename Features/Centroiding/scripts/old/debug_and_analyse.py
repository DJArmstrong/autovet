# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:35:45 2016

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


    ind_bp, exposures_per_night, obstime_per_night = breakpoints(dic)
    
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    axes[0].hist( exposures_per_night )
    axes[1].hist( obstime_per_night / 3600. )