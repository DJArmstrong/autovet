# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:56:17 2017

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
import pandas as pd

def fct( dic, keys ):
    dates = dic['HJD']
    pandic = pd.Series( [dic[key] for key in keys], index=dates )
    print pandic
    

dic = ngtsio.get('NG0409-1941', ['HJD','FLUX'])