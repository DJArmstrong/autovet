import numpy as np
from scipy import optimize
import batman


def Transitfitfunc(fitparams,y_data,y_err,x_data,m,bparams,init_per,init_t0):
    per = fitparams[0]
    t0 = fitparams[1]
    arstar = fitparams[2]
    rprstar = fitparams[3]
#    inc = fitparams[4]
#    inc_rad = inc*np.pi/180. 
#    if (rprstar < 0) or (arstar < 1.5) or (np.cos(inc_rad) > 1./arstar) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
    if (rprstar < 0) or (arstar < 1.5) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
        return np.ones(len(x_data))*1e8
    bparams.t0 = t0                        #time of inferior conjunction
    bparams.per = per                       #orbital period
    bparams.rp = rprstar                       #planet radius (in units of stellar radii)
    bparams.a = arstar                        #semi-major axis (in units of stellar radii)
    flux = m.light_curve(bparams)
    return (y_data - flux)/y_err

class TransitFit(object):

    def __init__(self,lc,initialguess,exp_time,sfactor):
        self.lc = lc
        self.init = initialguess
        self.exp_time = exp_time
        self.sfactor = sfactor
        self.params,self.cov = self.FitTransitModel()
        self.errors,self.chisq = self.GetErrors()

    def FitTransitModel(self):
        #initialguess = np.array([init_per,init_t0,init_arstar,init_rprstar,init_inc])
        fix_e = 0.
        fix_w = 90.
        ldlaw = 'quadratic'
        fix_ld = [0.1,0.3]
        bparams = batman.TransitParams()       #object to store transit parameters
        bparams.t0 = self.init[1]                        #time of inferior conjunction
        bparams.per = self.init[0]                      #orbital period
        bparams.rp = self.init[3]                       #planet radius (in units of stellar radii)
        bparams.a = self.init[2]                        #semi-major axis (in units of stellar radii)
        bparams.inc = 90.
        bparams.ecc = fix_e                      #eccentricity
        bparams.w = fix_w                        #longitude of periastron (in degrees)
        bparams.limb_dark = ldlaw        #limb darkening model
        bparams.u = fix_ld      #limb darkening coefficients
        m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,supersample_factor=self.sfactor)    #initializes model
        fit = optimize.leastsq(Transitfitfunc, self.init.copy(), args=(self.lc['flux'],self.lc['error'],self.lc['time'],m,bparams,self.init[0],self.init[1]),full_output=True)
        return fit[0],fit[1]   

    def GetErrors(self):
        bparams = batman.TransitParams()       #object to store transit parameters
        bparams.t0 = self.params[1]                        #time of inferior conjunction
        bparams.per = self.params[0]                       #orbital period
        bparams.rp = self.params[3]                      #planet radius (in units of stellar radii)
        bparams.a = self.params[2]                       #semi-major axis (in units of stellar radii)
        bparams.inc = 90.
        bparams.ecc = 0.                      #eccentricity
        bparams.w = 90.                        #longitude of periastron (in degrees)
        bparams.limb_dark = 'quadratic'        #limb darkening model
        bparams.u = [0.1,0.3]      #limb darkening coefficients
        m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,supersample_factor=self.sfactor)          
        flux = m.light_curve(bparams)
        if self.cov is not None:
            s_sq = np.sum(np.power((self.lc['flux'] - flux),2))/(len(self.lc['flux'])-4)
            err = (np.diag(self.cov*s_sq))**0.5
        else:
            print 'Fit did not give covariance, error based features will not be meaningful'
            err = np.ones(4)*-10
        chisq = 1./len(self.lc['flux']-4) * np.sum(np.power((self.lc['flux'] - flux)/self.lc['error'],2))
        return err,chisq
