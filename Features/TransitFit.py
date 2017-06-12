import numpy as np
from scipy import optimize
import batman
import utils

def Trapezoidmodel(t0_phase,t23,t14,depth,phase_data):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    return model
    
def Trapezoidfitfunc(fitparams,y_data,y_err,x_phase_data):

    t0 = fitparams[0]
    t23 = fitparams[1]
    t14 = fitparams[2]
    depth = fitparams[3]

    if (t0<0.45) or (t0>0.55) or (t23 < 0) or (t14 < 0) or (t14 < t23) or (depth < 0):
        return np.ones(len(x_phase_data))*1e8
    
    model = Trapezoidmodel(t0,t23,t14,depth,x_phase_data)
    return (y_data - model)/y_err

def Trapezoidfitfunc_fixephem(fitparams,y_data,y_err,x_phase_data):
    t0 = 0.5   
    t23 = fitparams[0]
    t14 = fitparams[1]
    depth = fitparams[2]
    
    if (t23 < 0) or (t14 < 0) or (t14 < t23) or (depth < 0):
        return np.ones(len(x_phase_data))*1e8
    
    model = Trapezoidmodel(t0,t23,t14,depth,x_phase_data)
    return (y_data - model)/y_err

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

def Transitfitfunc_fixephem(fitparams,y_data,y_err,x_data,m,bparams):
    arstar = fitparams[0]
    rprstar = fitparams[1]
#    inc = fitparams[4]
#    inc_rad = inc*np.pi/180. 
#    if (rprstar < 0) or (arstar < 1.5) or (np.cos(inc_rad) > 1./arstar) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
    if (rprstar < 0) or (arstar < 1.5):
        return np.ones(len(x_data))*1e8
    bparams.rp = rprstar                       #planet radius (in units of stellar radii)
    bparams.a = arstar                        #semi-major axis (in units of stellar radii)
    flux = m.light_curve(bparams)
    return (y_data - flux)/y_err


class TransitFit(object):

    def __init__(self,lc,initialguess,exp_time,sfactor,fittype='model',fixper=None,fixt0=None):
        self.lc = lc
        self.init = initialguess
        self.exp_time = exp_time
        self.sfactor = sfactor
        self.fixper = fixper
        self.fixt0 = fixt0        
        if fittype == 'model':
            self.params,self.cov = self.FitTransitModel()
            self.errors,self.chisq = self.GetErrors()
        elif fittype == 'trap':
            self.params,self.cov = self.FitTrapezoid()
            if self.fixt0 is None:
                self.params[0] = (self.params[0]-0.5)*self.fixper + self.initial_t0   #convert t0 back to time
    
    def FitTrapezoid(self):
        if self.fixt0 is None:
            self.initial_t0 = self.init[0]
            phase = utils.phasefold(self.lc['time'],self.fixper,self.initial_t0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            initialguess[0] = 0.5
            fit = optimize.leastsq(Trapezoidfitfunc, initialguess, args=(self.lc['flux'],self.lc['error'],phase),full_output=True)
        else:
            self.initial_t0 = self.fixt0
            phase = utils.phasefold(self.lc['time'],self.fixper,self.initial_t0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            fit = optimize.leastsq(Trapezoidfitfunc_fixephem, initialguess[1:], args=(self.lc['flux'],self.lc['error'],phase),full_output=True)            
        return fit[0],fit[1]

    def FitTransitModel(self):
        #initialguess = np.array([init_per,init_t0,init_arstar,init_rprstar,init_inc])
        fix_e = 0.
        fix_w = 90.
        ldlaw = 'quadratic'
        fix_ld = [0.1,0.3]
        bparams = batman.TransitParams()       #object to store transit parameters
        bparams.rp = self.init[3]                       #planet radius (in units of stellar radii)
        bparams.a = self.init[2]                        #semi-major axis (in units of stellar radii)
        bparams.inc = 90.
        bparams.ecc = fix_e                      #eccentricity
        bparams.w = fix_w                        #longitude of periastron (in degrees)
        bparams.limb_dark = ldlaw        #limb darkening model
        bparams.u = fix_ld      #limb darkening coefficients
        if self.fixper is None:
            bparams.t0 = self.init[1]                        #time of inferior conjunction
            bparams.per = self.init[0]                      #orbital period
            m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,supersample_factor=self.sfactor)    #initializes model
            fit = optimize.leastsq(Transitfitfunc, self.init.copy(), args=(self.lc['flux'],self.lc['error'],self.lc['time'],m,bparams,self.init[0],self.init[1]),full_output=True)
        else:
            bparams.t0 = self.fixt0
            bparams.per = self.fixper
            m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,supersample_factor=self.sfactor)    #initializes model
            fit = optimize.leastsq(Transitfitfunc_fixephem, self.init.copy()[2:], args=(self.lc['flux'],self.lc['error'],self.lc['time'],m,bparams),full_output=True)            
        return fit[0],fit[1]   

    def GetErrors(self):
        bparams = batman.TransitParams()       #object to store transit parameters
        bparams.inc = 90.
        bparams.ecc = 0.                      #eccentricity
        bparams.w = 90.                        #longitude of periastron (in degrees)
        bparams.limb_dark = 'quadratic'        #limb darkening model
        bparams.u = [0.1,0.3]      #limb darkening coefficients
        if self.fixper is None:
            bparams.t0 = self.params[1]                        #time of inferior conjunction
            bparams.per = self.params[0]                       #orbital period
            bparams.rp = self.params[3]                      #planet radius (in units of stellar radii)
            bparams.a = self.params[2]                       #semi-major axis (in units of stellar radii)

        else:
            bparams.t0 = self.fixt0
            bparams.per = self.fixper
            bparams.rp = self.params[1]                      #planet radius (in units of stellar radii)
            bparams.a = self.params[0]                       #semi-major axis (in units of stellar radii)

        m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,supersample_factor=self.sfactor)          
        flux = m.light_curve(bparams)
        if self.fixper is None:
            if self.cov is not None:
                s_sq = np.sum(np.power((self.lc['flux'] - flux),2))/(len(self.lc['flux'])-4)
                err = (np.diag(self.cov*s_sq))**0.5
            else:
                print 'Fit did not give covariance, error based features will not be meaningful'
                err = np.ones(4)*-10
            chisq = 1./len(self.lc['flux']-4) * np.sum(np.power((self.lc['flux'] - flux)/self.lc['error'],2))
        else:
            if self.cov is not None:
                s_sq = np.sum(np.power((self.lc['flux'] - flux),2))/(len(self.lc['flux'])-2)
                err = (np.diag(self.cov*s_sq))**0.5
            else:
                print 'Fit did not give covariance, error based features will not be meaningful'
                err = np.ones(2)*-10
            chisq = 1./len(self.lc['flux']-2) * np.sum(np.power((self.lc['flux'] - flux)/self.lc['error'],2))   
        return err,chisq
