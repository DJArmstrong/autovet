
import matplotlib.pyplot as p
p.ion()
import lomb
import numpy as np
from scipy.optimize import curve_fit


class PeriodLS():

    def __init__(self, lc, ofac=20., observatory=None, removethruster=False, removecadence=False):

        self.data = lc
        self.ofac = ofac
        if observatory=='K2':
            self.removethruster = True
            self.removecadence = True
        else:
            self.removethruster = removethruster
            self.removecadence = removecadence
        self.obs = observatory
        self.periods = np.empty(0)
        self.ampratios = np.empty(0)
        
    def fit(self,peak_number):
        if len(self.periods) <= peak_number: #then calculate
        
            flux = self.data['flux']
            time = self.data['time']

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
                thrusterfreq = 0.2448
                for i in range(4):
                    thrustercut = (fx > 1./((i+1)*thrusterfreq)*(1-0.01)) & (fx < 1./((i+1)*thrusterfreq)*(1+0.01)) #just in case...   
                    fx = fx[~thrustercut]
                    fy = fy[~thrustercut]
                return fx,fy
                
            def cutcadencefreqs(fx,fy):
                cadencefreq = 0.020431698
                for i in range(4):  #cuts 4 harmonics of frequency
                    cadencecut = (fx > 1./((i+1)*cadencefreq)*(1-0.01)) & (fx < 1./((i+1)*cadencefreq)*(1+0.01)) #just in case...   
                    fx = fx[~cadencecut]
                    fy = fy[~cadencecut]
                return fx,fy
        
            def cutTESSfocusfreqs(fx,fy):
                focusfreq = 13.49/4.  #so cuts the main one and higher frequencies
                for i in range(4):  #cuts 4 harmonics of frequency, within +-3% of given frequency
                    focuscut = (fx > 1./((i+1)*focusfreq)*(1-0.03)) & (fx < 1./((i+1)*focusfreq)*(1+0.03)) #just in case...   
                    fx = fx[~focuscut]
                    fy = fy[~focuscut]
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

            #limit allowed periods, depending on mission
            if self.obs == 'K2':
                if time[-1]-time[0] >= 50.:
                    lowcut = 1./fx<=20.
                else:
                    lowcut = 1./fx<=10.
            elif self.obs == 'TESS':
                trange = time[-1]-time[0]
                lowcut = 1./fx <= trange/2.*0.9  #less 10%, to cleanly avoid the focuser period for the 27day targets at least.
            else:
                lowcut = 1./fx<= 1000000.
            fx = fx[lowcut]
            fy = fy[lowcut]
        
            if self.removethruster:
                fx,fy = cutthrusterfreqs(fx,fy)
             
            if self.removecadence:
                fx,fy = cutcadencefreqs(fx,fy)
            
            if self.obs == 'TESS':
                #print 'DIAG HERE'
                #p.figure(1)
                #p.clf()
                #p.plot(fx,fy,'b.-')
                fx,fy = cutTESSfocusfreqs(fx,fy)
                #p.plot(fx,fy,'r.-')
                #p.figure(2)
                #p.clf()
                #p.plot(time,flux,'b.')
                #p.figure(3)
                #p.clf()
                #p.plot(np.mod(time,fx[np.argmax(fy)])/fx[np.argmax(fy)],flux,'r.')
                #print fx[np.argmax(fy)]
                #raw_input()
            #max_y = np.max(fy)
        
            #period = fx[jmax]
            freqs = [fx[np.argmax(fy)]]
            popt, pcov = curve_fit(yfunc(freqs[-1]), time, flux)
            ampmax = np.sqrt(popt[0]**2+popt[1]**2)
            ampratios = [1.]
        
            for ni in range(peak_number+1):
                fundamental_Freq = freqs[-1]
                flux = removeharmonics(time,flux,fundamental_Freq)
                fx, fy, nout, jmax, prob = lomb.fasper(time, flux, self.ofac, 1.)
                    
                if self.obs == 'K2':
                    if time[-1]-time[0] >= 50.:
                        lowcut = 1./fx<=20.
                    else:
                        lowcut = 1./fx<=10.
                elif self.obs == 'TESS':
                    trange = time[-1]-time[0]
                    lowcut = 1./fx <= trange/2.
                else:
                    lowcut = 1./fx<= 1000000.
                fx = fx[lowcut]
                fy = fy[lowcut]
                if self.removethruster:
                    fx,fy = cutthrusterfreqs(fx,fy)
                if self.removecadence:
                    fx,fy = cutcadencefreqs(fx,fy)
                if self.obs == 'TESS':
                    fx,fy = cutTESSfocusfreqs(fx,fy)
                
                freq = fx[np.argmax(fy)]
                popt, pcov = curve_fit(yfunc(freq), time, flux)
                amp = np.sqrt(popt[0]**2+popt[1]**2)
               
                freqs.append(freq)
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
            