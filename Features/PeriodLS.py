
import matplotlib.pyplot as p
p.ion()
import imp
lomb = imp.load_source('lomb','/Users/davidarmstrong/Software/Python/Periodfinding/lomb.py')

import numpy as np
from scipy.optimize import curve_fit


class PeriodLS():

    def __init__(self, lc, ofac=20., observatory=None, removethruster=True, removecadence=True):

        self.data = lc
        self.ofac = ofac
        self.removethruster = removethruster
        self.removecadence = removecadence
        self.obs = observatory

    def fit(self):
        
        flux = self.data['flux']
        
        time = self.data['time']


        #fx, fy, nout, jmax, prob = lomb.fasper(time, magnitude, self.ofac, 1.)
        #import pylab as p
        #p.ion()
        #p.clf()
        #p.plot(fx,fy,'b.-')
        #raw_input()


        def model(x, a, b, c, Freq):
            return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c
            
        def yfunc(Freq):
            def func(x, a, b, c):
                return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c
            return func
        
        def removeharmonics(time,flux,Freq):
            popts = []
            for j in range(4):
                popt, pcov = curve_fit(yfunc((j+1)*fundamental_Freq), time, flux)
                popts.append(popt)

            for j in range(4):
                flux = np.array(flux) - model(time, popts[j][0], popts[j][1], popts[j][2], (j+1)*fundamental_Freq)
            popt, pcov = curve_fit(yfunc((0.5)*fundamental_Freq), time, flux)
            flux = np.array(flux) - model(time, popt[0], popt[1], popt[2], (0.5)*fundamental_Freq)
            return flux
        
        def cutthrusterfreqs(fx,fy):
            for i in range(4):
                thrustercut = (fx > 1./((i+1)*0.2448)*(1-0.01)) & (fx < 1./((i+1)*0.2448)*(1+0.01)) #just in case...   
                fx = fx[~thrustercut]
                fy = fy[~thrustercut]
            return fx,fy
                
        def cutcadencefreqs(fx,fy):
            for i in range(4):
                cadencecut = (fx > 1./((i+1)*0.020431698)*(1-0.01)) & (fx < 1./((i+1)*0.020431698)*(1+0.01)) #just in case...   
                fx = fx[~cadencecut]
                fy = fy[~cadencecut]
            return fx,fy
                     
        if self.removethruster:        
            thrusterfreqs = 1./np.array([0.23995485,0.24058965,0.24122445,0.24185925,0.24249405,0.24312886,0.24376366,0.24439846,0.24503326,0.24566806,0.24630286,0.24693766,0.24757246,0.24820727,0.24884207,0.24947687])
            #thrusterfreqs = (fx > 0.2448*(1-0.05)) & (fx < 0.2448*(1+0.05))
            #thrusterfreqs = np.array([0.23297204,0.23360684,0.23424164,0.23487644,0.23551124,0.23614604,0.23678084,0.23741564,0.23805045,0.23868525,0.23932005,0.23995485,0.24058965,0.24122445,0.24185925,0.24249405,0.24312886,0.24376366,0.24439846,0.24503326,0.24566806,0.24630286,0.24693766,0.24757246,0.24820727,0.24884207,0.24947687,0.25011167,0.25074647,0.25138127,0.25201607,0.25265087,0.25328567,0.25392048,0.25455528,0.25519008,0.25582488,0.25645968])
            #print fx[thrusterfreqs]
            for fundamental_Freq in thrusterfreqs:

                flux = removeharmonics(time,flux,fundamental_Freq)
                
        if self.removecadence:
            #remove cadence frequency
            cadencefreqs = 1./np.array([0.020431698,0.049032325])
            #thrusterfreqs = (fx > 0.020431698*(1-0.05)) & (fx < 0.020431698*(1+0.05))
            #thrusterfreqs = np.array([0.23297204,0.23360684,0.23424164,0.23487644,0.23551124,0.23614604,0.23678084,0.23741564,0.23805045,0.23868525,0.23932005,0.23995485,0.24058965,0.24122445,0.24185925,0.24249405,0.24312886,0.24376366,0.24439846,0.24503326,0.24566806,0.24630286,0.24693766,0.24757246,0.24820727,0.24884207,0.24947687,0.25011167,0.25074647,0.25138127,0.25201607,0.25265087,0.25328567,0.25392048,0.25455528,0.25519008,0.25582488,0.25645968])
            #print fx[thrusterfreqs]
            for fundamental_Freq in cadencefreqs:
     
                for j in [0.5,1]: #only half and the exact cadence frequency fit and removed
                    popt, pcov = curve_fit(yfunc((j)*fundamental_Freq), time, flux)
                    flux = np.array(flux) - model(time, popt[0], popt[1], popt[2], (j)*fundamental_Freq)


        fx, fy, nout, jmax, prob = lomb.fasper(time, flux, self.ofac, 1.)

        if self.obs == 'K2':
            if time[-1]-time[0] >= 50.:
                lowcut = 1./fx<=20.
            else:
                lowcut = 1./fx<=10.
            fx = fx[lowcut]
            fy = fy[lowcut]
        
        if self.removethruster:
            fx,fy = cutthrusterfreqs(fx,fy)
             
        if self.removecadence:
            fx,fy = cutcadencefreqs(fx,fy)
 
        #p.figure(1)
        #p.clf()
        #p.plot(fx,fy,'b.-')
        #max_y = np.max(fy)
        
        #period = fx[jmax]
        freqs = [fx[np.argmax(fy)]]
        popt, pcov = curve_fit(yfunc(freqs[-1]), time, flux)
        ampmax = np.sqrt(popt[0]**2+popt[1]**2)
        ampratios = [1.]
        
        for ni in range(3):
            fundamental_Freq = freqs[-1]
            flux = removeharmonics(time,flux,fundamental_Freq)
            fx, fy, nout, jmax, prob = lomb.fasper(time, flux, self.ofac, 1.)
                    
            if self.obs == 'K2':
                if time[-1]-time[0] >= 50.:
                    lowcut = 1./fx<=20.
                else:
                    lowcut = 1./fx<=10.
                fx = fx[lowcut]
                fy = fy[lowcut]
            if self.removethruster:
                fx,fy = cutthrusterfreqs(fx,fy)
            if self.removecadence:
                fx,fy = cutcadencefreqs(fx,fy)
            freq = fx[np.argmax(fy)]
            popt, pcov = curve_fit(yfunc(freq), time, flux)
            amp = np.sqrt(popt[0]**2+popt[1]**2)
                
            freqs.append(fx[np.argmax(fy)])
            ampratios.append(amp/ampmax)
                        
        #print freqs
        #print ampratios
        #for freq in freqs[:3]:
        #    p.plot([freq,freq],[0,max_y],'r--')
        #raw_input()
        
        T = 1.0 / np.array(freqs)
        #new_time = np.mod(time, 2 * T) / (2 * T)
        self.periods = T
        self.ampratios = np.array(ampratios)
