# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np
import matplotlib.pyplot as plt



def plot(dic):
    
    #::: detrended curves
    fig_phasefold, axes = plt.subplots( 3, 1, sharex=True, figsize=(12,6) )
    print '******'
    print dic['SYSREM_FLUX3_PHASE']
    axes[0].errorbar( dic['HJD_PHASE'], dic['SYSREM_FLUX3_PHASE'], yerr=dic['SYSREM_FLUX3_PHASE_ERR'], fmt='o', color='r', ms=10, rasterized=True )
    axes[0].set( ylabel='FLUX', ylim=[ np.nanmin(dic['SYSREM_FLUX3_PHASE']-dic['SYSREM_FLUX3_PHASE_ERR']), np.nanmax(dic['SYSREM_FLUX3_PHASE']+dic['SYSREM_FLUX3_PHASE_ERR']) ])
    
    axes[1].errorbar( dic['HJD_PHASE'], dic['CENTDX_fda_PHASE'], yerr=dic['CENTDX_fda_PHASE_ERR'], fmt='o', ms=10, rasterized=True ) #, color='darkgrey')
    axes[1].set( ylabel='CENTDX (in pixel)', ylim=[ np.nanmin(dic['CENTDX_fda_PHASE']-dic['CENTDX_fda_PHASE_ERR']), np.nanmax(dic['CENTDX_fda_PHASE']+dic['CENTDX_fda_PHASE_ERR']) ])
    
    axes[2].errorbar( dic['HJD_PHASE'], dic['CENTDY_fda_PHASE'], yerr=dic['CENTDY_fda_PHASE_ERR'], fmt='o', ms=10, rasterized=True ) #, color='darkgrey')
    axes[2].set( ylabel='CENTDY (in pixel)', xlabel='Phase', ylim=[ np.nanmin(dic['CENTDY_fda_PHASE']-dic['CENTDY_fda_PHASE_ERR']), np.nanmax(dic['CENTDY_fda_PHASE']+dic['CENTDY_fda_PHASE_ERR']) ], xlim=[-0.25,0.75])
    
    plt.tight_layout()
        
        

#fname = 'output/TEST18/NG1421+0000/200.0_500.0_10000.0_transit_0.0_20_300.0_1800.0_0.001/NG1421+0000_006328_TEST18_centroid_data_ALL.txt'
#fname = 'output/TEST18/NG1421+0000/200.0_500.0_10000.0_transit_0.0_20_300.0_1800.0_0.001/NG1421+0000_006328_TEST18_centroid_data_BIN.txt'
#fname = 'output/TEST18/NG1421+0000/200.0_500.0_10000.0_transit_0.0_20_300.0_1800.0_0.001/NG1421+0000_006328_TEST18_centroid_data_PHASE.txt'
fname = '/appch/data/mg719/CENTROIDING/Centroiding/output/candidate_shortlists/candidates_TEST18_20170720/dt=0.0025/NG0509-3345_000350_TEST18_centroid_data_PHASE.txt'
dic = np.genfromtxt(fname, names=True)

plot(dic)


#plt.figure()
#plt.plot(dic['HJD_PHASE'], dic['SYSREM_FLUX3_PHASE'], 'k.', rasterized=True)



#plt.figure()
#plt.errorbar( dic['HJD_PHASE'], dic['SYSREM_FLUX3_PHASE'], yerr=dic['SYSREM_FLUX3_PHASE_ERR'], fmt='o', color='r', ms=10, rasterized=True )
#plt.ylim( [ np.nanmin(dic['SYSREM_FLUX3_PHASE']-dic['SYSREM_FLUX3_PHASE_ERR']), np.nanmax(dic['SYSREM_FLUX3_PHASE']+dic['SYSREM_FLUX3_PHASE_ERR']) ] )