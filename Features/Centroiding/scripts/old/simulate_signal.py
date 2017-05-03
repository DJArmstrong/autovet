# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:05:11 2016

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
import batman
import eb


def simulate( dic, tipo='planet', period=1.2, epoch=2., depth=0.01, dil=0.9, centdx_shift=0.01, centdy_shift=-0.004, plot=False ):
    '''
    tipo: 'planet', 'EB_FGKM', 'EB'
    
    period (days)
    epoch (days from start of measurement)
    depth (relative fraction)
    centdx_shift (pixel)
    centdy_shift (pixel)
    '''
    
    #::: convert period and epoch into seconds
    period = period * (3600. * 24.) #from days to seconds
    epoch = epoch * (3600. * 24.) #from days to seconds
    
    
    #:::: choose which model to apply
    if tipo == 'planet':
        print 'Simulating planet'
        simulated_flux = planet_lightcurve(dic, period, epoch, depth, dil)
    
    elif tipo == 'EB_FGKM':
        print 'Simulating EB of type FGKM'
        simulated_flux = EB_FGKM_lightcurve(dic, period, epoch, None, dil)
    
    elif tipo == 'EB':
        print 'Simulating EB'
        simulated_flux = EB_lightcurve(dic, period, epoch, None, dil)
    else:
        print 'No Simulation'
        simulated_flux = np.ones(len(dic['HJD']))
        
        
    #::: add transit signal to CENTDX/Y
    depth_max = 1. - np.min(simulated_flux)
    simulated_centdx = (simulated_flux - 1.) / (- depth_max) * centdx_shift #curve from 0 to centdx
    simulated_centdy = (simulated_flux - 1.) / (- depth_max) * centdy_shift #curve from 0 to centdx
    print 'DEPTH MAX', depth_max
    print 'CENTDX MAX', np.max(np.abs(simulated_centdx))
    print 'CENTDY MAX', np.max(np.abs(simulated_centdy)) 
      
    #::: save the transit signal in the dic
    dic['SYSREM_FLUX3'] *= simulated_flux #multiplicative
    dic['PERIOD'] = period
    dic['EPOCH'] = dic['HJD'][0] + epoch    
    dic['DEPTH'] = - depth_max
    dic['CENTDX'] += np.int64( simulated_centdx * 1024. ) / 1024. #additive
    dic['CENTDY'] += np.int64( simulated_centdy * 1024. ) / 1024.  #additive
    
    
    
    #::: plot (if requested)
    if plot == True:
        fig, axes = plt.subplots(3,1, figsize=(100,12))
        
        ax = axes[0]
    #    ax.plot( dic['HJD'], dic['FLUX'], 'g.', rasterized=True )
        ax.plot( dic['HJD'], dic['SYSREM_FLUX3'], 'b.', rasterized=True )
        ax.plot( dic['HJD'], np.nanmedian(dic['SYSREM_FLUX3']) * simulated_flux, 'r-', rasterized=True )
        ax.set( xlim=[dic['HJD'][0],dic['HJD'][100000]] )
        
        ax = axes[1]
    #    ax.plot( dic['HJD'], dic['CENTDX'], 'g.', rasterized=True )
        ax.plot( dic['HJD'], dic['CENTDX'], 'b.', rasterized=True )
        ax.plot( dic['HJD'], simulated_centdx, 'r-', rasterized=True )
        ax.set( xlim=[dic['HJD'][0],dic['HJD'][100000]], ylim=[-0.2,0.2] )
        
        ax = axes[2]
    #    ax.plot( dic['HJD'], dic['CENTDY'], 'g.', rasterized=True )
        ax.plot( dic['HJD'], dic['CENTDY'], 'b.', rasterized=True )
        ax.plot( dic['HJD'], simulated_centdy, 'r-', rasterized=True )
        ax.set( xlim=[dic['HJD'][0],dic['HJD'][100000]], ylim=[-0.2,0.2] )
        
        plt.show()    
        
        
    
    return dic
    




def planet_lightcurve(dic, period, epoch, depth, dil):
    
    #::: add the transit signal to FLUX
    params = batman.TransitParams()
    params.t0 = dic['HJD'][0] + epoch         #time of inferior conjunction
    params.per = period                     #orbital period
    params.rp = np.sqrt( depth )            #planet radius (in units of stellar radii)
    params.a = 1/(4*np.pi**2) * (period/(3600. * 24. * 365.))**2 / (1./215.)**3         #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients
    params.limb_dark = "quadratic"       #limb darkening model
    
    m = batman.TransitModel(params, dic['HJD'])    #initializes model
    batman_flux = m.light_curve(params)          #calculates light curve
    
    
    #::: include dilution
    batman_flux = 1. - (1. - batman_flux) * (1. - dil)


    #::: return
    return batman_flux    



def EB_FGKM_lightcurve(dic, period, epoch, depth, dil):
    '''
    period in s
    epoch in s
    '''
    params = {
                'M1': 1.5,\
                'M2': 0.7,\
                'P':period,\
                'COSI':0.,\
                'DIL':dil
              }
    
    params['R1'] = M2R(params['M1'])
    params['R2'] = M2R(params['M2'])
    params['L1'] = M2L(params['M1'])
    params['L2'] = M2L(params['M2'])
        
    params['a'] = ( ( params['P'] / 365. / 24. / 3600. )**2 * ( params['M1'] + params['M2'] ) ) ** (1./3.) * 215. #in Rsun
    
    # Allocate main parameter vector, init to zero.
    parm = np.zeros(eb.NPAR, dtype=np.double)
    
#    if params is None:
    # These are the basic parameters of the model.
    parm[eb.PAR_J]      =  params['L2']/params['L1']                # J surface brightness ratio, scales roughly with temperature (Rayleigh-Jeans limit)
    parm[eb.PAR_RASUM]  =  (params['R1']+params['R2'])/params['a']  # (R_1+R_2)/a
    parm[eb.PAR_RR]     =  params['R2']/params['R1']                # R_2/R_1
    parm[eb.PAR_COSI]   =  params['COSI']                           # cos i
    
    # Orbital parameters
    parm[eb.PAR_P]      = period                  # period
    parm[eb.PAR_T0]     = dic['HJD'][0] + epoch   # T0 (epoch of primary eclipse)
        
    # Radiative properties of star 1.
    parm[eb.PAR_LDLIN1] =  0.2094    # u1 star 1
    parm[eb.PAR_LDNON1] =  0.6043    # u2 star 1
    parm[eb.PAR_GD1]    =  0.32      # gravity darkening, std. value
    parm[eb.PAR_REFL1]  =  0.4       # albedo, std. value
    
    # Assume star 2 is the same as star 1
    parm[eb.PAR_LDLIN2] = parm[eb.PAR_LDLIN1]
    parm[eb.PAR_LDNON2] = parm[eb.PAR_LDNON1]
    parm[eb.PAR_GD2]    = parm[eb.PAR_GD1]
    parm[eb.PAR_REFL2]  = parm[eb.PAR_REFL1]
    
    # All magnitudes.
    typ = np.empty_like(dic['HJD'], dtype=np.uint8)
    typ.fill(1.)
    
    # Calculate Flux, scaled to 1.
    eb_fgkm_flux =  1. - (1. - eb.model(parm, dic['HJD'], typ)) * (1. - params['DIL'])

    return eb_fgkm_flux
 
 
 

def M2R(M):
    if M < 1.66:
        return 1.06*M**0.945
    else:
        return 1.33*M*0.555
        
        
        
def M2L(M):
    '''
    https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
    '''
#    if M < 0.7:
#        return 0.35*M**2.62
#    else:
#        return 1.02*M*3.92
    if M < 0.43:
        return 0.23*(M**2.3)
    elif M < 2.:
        return M**4.
    else:
        return 1.5*(M**3.5)
         
 
 
def EB_lightcurve(dic, period, epoch, depth, dil):
    
    params = {
            'J': 0.7,\
            'RASUM': 0.2,\
            'RR':np.sqrt(depth),\
            'COSI':0.,\
            'P':period,\
            'T0':dic['HJD'][0] + epoch,\
            'LDLIN':0.2,\
            'LDNON':0.6,\
            'GD':0.3,\
            'REFL1':0.4,\
            'REFL2':0.4,\
            'DIL':dil
          }
        
    # Allocate main parameter vector, init to zero.
    parm = np.zeros(eb.NPAR, dtype=np.double)
    
    # These are the basic parameters of the model.
    parm[eb.PAR_J]      =  params['J']          # J surface brightness ratio
    parm[eb.PAR_RASUM]  =  params['RASUM']      # (R_1+R_2)/a
    parm[eb.PAR_RR]     =  params['RR']         # R_2/R_1
    parm[eb.PAR_COSI]   =  params['COSI']       # cos i
    
    # Orbital parameters
    parm[eb.PAR_P]      = params['P']     # period
    parm[eb.PAR_T0]     = params['T0']    # T0 (epoch of primary eclipse)
    
    # Radiative properties of star 1.
    parm[eb.PAR_LDLIN1] =  params['LDLIN']   # u1 star 1
    parm[eb.PAR_LDNON1] =  params['LDNON']    # u2 star 1
    parm[eb.PAR_GD1]    =  params['GD']      # gravity darkening, std. value
    parm[eb.PAR_REFL1]  =  params['REFL1']       # albedo, std. value
    
    # Assume star 2 is the same as star 1
    parm[eb.PAR_LDLIN2] = parm[eb.PAR_LDLIN1]
    parm[eb.PAR_LDNON2] = parm[eb.PAR_LDNON1]
    parm[eb.PAR_GD2]    = parm[eb.PAR_GD1]
    parm[eb.PAR_REFL2]  = params['REFL2']
    
    
    # All magnitudes.
    typ = np.empty_like(dic['HJD'], dtype=np.uint8)
    typ.fill(eb.OBS_MAG)
    print eb.OBS_MAG
    
    # Calculate Flux, scaled to 1.
    eb_flux = 1. - (1. - params['DIL']) * eb.model(parm, dic['HJD'], typ)

    return eb_flux

    
    