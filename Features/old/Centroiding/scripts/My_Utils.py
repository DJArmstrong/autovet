# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:07:00 2016

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
#from scipy import stats
import time
import os, glob
#from running_median import RunningMedian
#from mpl_toolkits.axes_grid1 import AxesGrid
try:
    import pandas as pd
except ImportError:
    'Warning: module "pandas" not installed.'
        
        
def medsig(a):
    '''Compute median and MAD-estimated scatter of array a'''
#    try:
    med = np.nanmedian(a)
    sig = 1.48 * np.nanmedian(abs(a-med))
#    except:
#        AttributeError
#    else:
#        med = stats.nanmedian(a)
#        sig = 1.48 * stats.nanmedian(abs(a-med))
    return med, sig   
    
  
  
def running_mean(x, N):
    x[np.isnan(x)] = 0. #reset NAN to 0 to calculate the cumulative sum; mimics the 'pandas' behavior
    cumsum = np.cumsum(np.insert(x, 0., 0.)) 
    return 1.*(cumsum[N:] - cumsum[:-N]) / N 
    
    
# 'running_median' DOES NOT AGREE WITH THE PANDAS IMPLEMENTATION 'running_median_pandas'
#def running_median(x, N):
#    return np.array(list(RunningMedian(N, x)))
    
    
def running_mean_pandas(x, N):
    ts = pd.Series(x).rolling(window=N, center=False).mean()
    return ts[~np.isnan(ts)].as_matrix()



def running_median_pandas(x, N):
    ts = pd.Series(x).rolling(window=N, center=False).median()
    return ts[~np.isnan(ts)].as_matrix()


    
def mask_ranges(x, x_min, x_max):
    """"
    Crop out values and indices out of an array x for multiple given ranges x_min to x_max.
    
    Input:
    x: array, 
    x_min: lower limits of the ranges
    x_max: upper limits of the ranges
    
    Output:
    
    
    Example:
    x = np.arange(200)    
    x_min = [5, 25, 90]
    x_max = [10, 35, 110]
    """

    mask = np.zeros(len(x), dtype=bool)
    for i in range(len(x_min)): 
        mask = mask | ((x >= x_min[i]) & (x <= x_max[i]))
    ind_mask = np.arange(len(mask))[mask]
    
    return x[mask], ind_mask, mask 
    
    

def mystr(x,digits=0):
    if np.isnan(x): return '.'
    elif digits==0: return str(int(round(x,digits)))
    else: return str(round(x,digits))
    
    

def version_control(files='*.py', printing=True):
    #    last_created_file = max(glob.iglob(files), key=os.path.getctime)
    last_updated_file = max(glob.iglob(files), key=os.path.getmtime)
    if printing == True: 
    #    print "# Last created script: %s, %s" % ( last_created_file, time.ctime(os.path.getmtime(last_created_file)) )
    #    print "# Last updated script: %s, %s" % ( last_updated_file, time.ctime(os.path.getmtime(last_updated_file)) )
        print "# Last update: %s" % time.ctime(os.path.getmtime(last_updated_file))
        

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z
        
        
def table_view(dic):
    from astropy.table import Table 
    dic_table = {}
    subkeys = ['OBJ_ID', 'SYSREM_FLUX3_median', 'PERIOD', 'DEPTH', 'WIDTH', 'NUM_TRANSITS']
    for key in subkeys:
            dic_table[key] = dic[key]
    dic_table = Table(dic_table)
    dic_table = dic_table[subkeys]
    print dic_table