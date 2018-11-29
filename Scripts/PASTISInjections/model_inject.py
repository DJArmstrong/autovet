
import numpy as np
from lightkurve import KeplerLightCurveFile
import kepselfflatten as ksf
from transit_periodogram import transit_periodogram
from astropy.io import fits
#import pyfits
import glob, os
from scipy import interpolate


def MovingAverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def phasefold(time,period,epoch):
    return np.mod(time-epoch,period)/period

#load Berger list
kicstars_gaia = np.genfromtxt('/Users/davidarmstrong/Data/Kepler/Berger_KepGAIA/DR2PapTable1.txt',delimiter='&',names=True)

cut = (kicstars_gaia['evState'] == 0) & (kicstars_gaia['teff']<9500.) #main sequence, A and cooler
kicstars_gaia = kicstars_gaia[cut]

#load model list
modlist = glob.glob('/Users/davidarmstrong/Data/Santerne/models_corrected/PASTIS*.fits')
modlist = np.sort(modlist)
exptime = 1764./86400.

stepsize = 0.1
polydegree = 3
niter = 10
sigmaclip = 5
gapthreshold = 0.4
plot = False
winsize = 3.

if os.path.exists('Injections_withstellar/metadata.txt'):
    oldmeta = np.genfromtxt('Injections_withstellar/metadata.txt',delimiter=',')
    prerunmods = oldmeta[:,0] 
else:
    prerunmods = []

for infile in modlist:
    modelno = os.path.split(infile)[1][13:18] 
    if float(modelno) not in prerunmods:
        print(modelno)

        dat = fits.open(infile)
        modtime = dat[1].data['TIME']
        modflux = dat[1].data['FLUX']
        #headerdata = np.array(dat[1].header[12:-4].values())
        headerdata = np.array([dat[1].header['HIERARCH ORBIT PERIOD'],dat[1].header['HIERARCH ORBIT T0'],dat[1].header['HIERARCH ORBIT ECC'],dat[1].header['HIERARCH ORBIT OMEGA'],dat[1].header['HIERARCH ORBIT INCL'],dat[1].header['HIERARCH TARGET LD1'],dat[1].header['HIERARCH TARGET LD2'],dat[1].header['HIERARCH TRANSIT AR'],dat[1].header['HIERARCH TRANSIT KR'],dat[1].header['HIERARCH TRANSIT B'],dat[1].header['HIERARCH TRANSIT T14'],dat[1].header['HIERARCH TRANSIT DEPTH'],dat[1].header['HIERARCH OCCULTATION B'],dat[1].header['HIERARCH OCCULTATION DEPTH']])
        stellarmeta = np.array([dat[1].header['HIERARCH TARGET TEFF'],dat[1].header['HIERARCH TARGET LOGG'],dat[1].header['HIERARCH TARGET FEH']])
        dat.close()
        per = headerdata[0]
        
        if per <= 30:
            headerdata[1] += np.random.uniform(0,per)
            t0 = headerdata[1]
            depth = headerdata[11]
            dur = headerdata[10]/24.
    
            #interp object (smearing over Kepler exptime)
    
            #interp_PASTISlc_old = interpolate.interp1d(dat[1].data['Time']/per,dat[1].data['Flux']-1,kind='linear') #baseline at 0, goes negative for transit
    
            nphasepoints = int( ( exptime /  per ) * len(modflux) )
            exptimesmear = np.roll(MovingAverage(np.roll(modflux-1,int(len(modflux)/2)),nphasepoints),int(-len(modflux)/2))
            interp_PASTISlc = interpolate.interp1d(modtime/per,exptimesmear,kind='linear') #baseline at 0, goes negative for transit

            detection = False
            count = 0   
            while not detection and count<5:
                lcunloaded = True
                print(count)
                while lcunloaded:
                    temp = 50000.
                    while np.abs(stellarmeta[0]-temp) > 600:
                        target = int(np.random.choice(np.arange(len(kicstars_gaia['KIC']))))
                        kicid = kicstars_gaia['KIC'][target]
                        temp = kicstars_gaia['teff'][target] #really should be similar logg and FEH as well, ideally. Might be impractical though.
                    print(kicid)
                    try:
                        lc = KeplerLightCurveFile.from_archive(kicid, quarter=9).PDCSAP_FLUX.normalize()
                        lcunloaded = False
                    except:
                        print('No lightcurve found')
                        lcunloaded = True
                    
                lc = lc.remove_nans()
        
                lc.time -= lc.time[0]

                phasedkeplc = phasefold(lc.time,per,t0)
                lc.flux += interp_PASTISlc(phasedkeplc)


                #flatten
                transitcut = 0 #number of periodic events to cut
                t_per = 0 
                t_t0 = 0
                t_dur = 0

                lcf =   ksf.Kepflatten(lc.time-lc.time[0],lc.flux,lc.flux_err,np.zeros(len(lc.time)),
                       winsize,stepsize,polydegree,niter,sigmaclip,
                       gapthreshold,lc.time[0],plot,transitcut,
                       t_per,t_t0,t_dur)        

                #run lk boxsearch, or our own?
                freqs = np.arange(1./30.,1./0.3,0.0001)
                periods = 1./freqs
                durations = np.arange(0.005, 0.15, 0.01)
                power, _, _, _, _, _, _ = transit_periodogram(time=lcf[:,0],
                                              flux=lcf[:,1],
                                              flux_err=lcf[:,2],
                                              periods=periods,
                                              durations=durations)
                best_fit = periods[np.argmax(power)]

                if np.abs((best_fit - per) / per) <= 0.05 or np.abs((best_fit*2 - per) / per) <= 0.05: #5%
                    #reflatten
                    transitcut = 1 #number of periodic events to cut
                    t_per = per 
                    t_t0 = t0
                    t_dur = dur
                    lcf =  ksf.Kepflatten(lc.time-lc.time[0],lc.flux,lc.flux_err,np.zeros(len(lc.time)),
                       winsize,stepsize,polydegree,niter,sigmaclip,
                       gapthreshold,lc.time[0],plot,transitcut,
                       t_per,t_t0,t_dur)        
                
                    #save lc
                    np.savetxt('Injections_withstellar/'+modelno+'_'+str(kicid)+'.txt',lcf)
                    #save count
                    with open('Injections_withstellar/metadata.txt','a') as f:
                        f.write(modelno+','+str(kicid)+','+str(count)+','+str(per)+','+str(t0)+','+str(dur)+',')
                        f.write(str(stellarmeta[0])+','+str(stellarmeta[1])+','+str(stellarmeta[2])+'\n')
                    detection = True
                else:
                    count += 1
                #count non-detection
                    if count >=5:
                        with open('Injections_withstellar/metadata.txt','a') as f:
                            f.write(modelno+','+str(kicid)+','+str(count)+',0,0,0,0,0,0\n')
