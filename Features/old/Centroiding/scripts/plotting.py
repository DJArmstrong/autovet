# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:02:38 2016

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






def plot_detrending(self):
    ind = slice( None )
    
#            ind = np.where( self.dic['NIGHT'] == date )    
#            
#            #::: plot neighbours
#            N = len( self.dic_nb['OBJ_ID'] )
#            N_cols = 4
#            N_rows = int(np.ceil(1.*N/N_cols))                
#            fig, axes = plt.subplots( N_rows, N_cols, sharex=True, sharey=True, figsize=(4*N_cols,4*N_rows))
#            for i,_ in enumerate( self.dic_nb['OBJ_ID'] ):
#                ind_ax = np.unravel_index(i, (N_rows, N_cols))
#                axes[ ind_ax ].scatter( self.dic_nb['CENTDX'][i][ind], self.dic_nb['CENTDY'][i][ind], c=ind, rasterized=True)#, label='OBJID='+str(self.dic_nb['OBJ_ID'][i])+'\nFLUX='+str(self.dic_nb['FLUX_MEAN'][i]) + '\nCCDX=' + str(self.dic_nb['CCDX'][i,0]) + '\nCCDY=' + str(self.dic_nb['CCDY'][i,0]) )
#                axes[ ind_ax ].legend(loc='upper left', framealpha=0.5, scatterpoints=1)
#                axes[ ind_ax ].axis('equal')
#            plt.suptitle( date, fontsize=16 )
#            plt.tight_layout()
#            plt.subplots_adjust(top=0.95)
#    #        pdf.savefig()
#    #        plt.close()
    

    
    #::: plot object CENTDXY results
    fig, axes = plt.subplots( 1, 3, sharex=True, sharey=True, figsize=(4*3,4*1))
    
    xkeys = ['CENTDX', 'CENTDX_detrended_mean', 'CENTDX_detrended_median']
    ykeys = ['CENTDY', 'CENTDY_detrended_mean', 'CENTDY_detrended_median']
    titles = ['before local detrending','after local detrending (mean)','after local detrending (median)']
    for i, (xkey, ykey, title) in enumerate( zip( xkeys, ykeys, titles ) ):
        ax = axes[i]
        ax, sc = helper_functions.norm_scatter(ax, self.dic[xkey][ind], self.dic[ykey][ind], self.dic['tr_flag'][ind] )
        ax.set( xlim=[-0.1,0.1], ylim=[-0.1,0.1], title=title )
        ax.axis('equal')
    
    
#        axes[1,0].plot( self.df_gb_date['CENTDX'].std() )
#        
#        axes[1,1].plot( self.df_gb_date['CENTDY'].std() )
#        
#        axes[2,0].plot( self.df_gb_date['CENTDX_dmedc'].std() )
#        
#        axes[2,1].plot( self.df_gb_date['CENTDY_dmedc'].std() )
    
    plt.tight_layout()
    
    self.fig_detrending = fig



def plot_rainplot_summary(self):
    
    for chosendf, bin_style in izip( [self.df_gb_tr_flag, self.bindf_gb_tr_flag], ['unbinned', 'binned' ]):
    
        fig, ax = plt.subplots( 1, 1 )
        fig.suptitle(bin_style)
        colors = ['grey','k','r','goldenrod']
        i = 0
        for tr_flag, gdata in chosendf:
            ax.plot( gdata['CENTDX_dmedc'], gdata['CENTDY_dmedc'], 'k.', c=colors[i%4], alpha=0.33, rasterized=True, zorder=-1 )
            ax.errorbar( gdata['CENTDX_dmedc'].mean(), gdata['CENTDY_dmedc'].mean(), xerr=gdata['CENTDX_dmedc'].std(), yerr=gdata['CENTDY_dmedc'].std(), color=colors[i%4], lw=3, zorder=1 )             
            i += 1
        plt.axis('square')        
        plt.xticks(rotation=45)
            
        if bin_style == 'binned': self.fig1a = fig
        if bin_style == 'unbinned': self.fig2a = fig    
    


def plot_rainplot_per_night(self):
    
    for chosendf, bin_style in izip( [self.df_gb_date_tr_flag, self.bindf_gb_date_tr_flag], ['unbinned', 'binned' ]):

        i = 0
        N_dates = len( chosendf )/4
        
        fig, axes = plt.subplots( np.int( N_dates/4. )+1, 4, sharex=True, sharey=True, figsize=(4*3,N_dates))
        fig.suptitle(bin_style)
        colors = ['grey','k','r','goldenrod']
        for (date, tr_flag), gdata in chosendf:
            if i%4 == 0:        
                axind = np.unravel_index( i/4, (axes.shape[0],axes.shape[1]) )
                ax = axes[axind]
                ax.set_title(date)
                plt.xticks(rotation=45)
#                    ax.set_xlim([-0.1, 0.1])
#                    ax.set_ylim([-0.1, 0.1])
#                    plt.axis('square')                
#                    plt.axis('equal') 
            ax.plot( gdata['CENTDX_dmedc'], gdata['CENTDY_dmedc'], 'k.', c=colors[i%4], alpha=0.33, rasterized=True, zorder=-1 )
            ax.errorbar( gdata['CENTDX_dmedc'].mean(), gdata['CENTDY_dmedc'].mean(), xerr=gdata['CENTDX_dmedc'].std(), yerr=gdata['CENTDY_dmedc'].std(), color=colors[i%4], lw=3, zorder=1 )    
            i += 1
            
        if bin_style == 'binned': self.fig1 = fig
        if bin_style == 'unbinned': self.fig2 = fig



def plot_detrending_over_time(self):
    
    fig, axes = plt.subplots( 5, 1, sharex=True, figsize=(8,16) )
    self.bindf.plot( kind='scatter', x='HJD', y='SYSREM_FLUX3', c='tr_flag', ax=axes[0] )
    self.bindf.plot( kind='scatter', x='HJD', y='CENTDX_dmedc', c='tr_flag', ax=axes[1] )
    self.bindf.plot( kind='scatter', x='HJD', y='CENTDY_dmedc',  c='tr_flag', ax=axes[2] )
    self.bindf.plot( kind='scatter', x='HJD', y='CENTDX_dmedc_ERR',  c='tr_flag', ax=axes[3] )
    self.bindf.plot( kind='scatter', x='HJD', y='CENTDY_dmedc_ERR', c='tr_flag', ax=axes[4] )
    self.fig_detrending_over_time = fig
    

 
###########################################################################
#::: look at phase-folded lightcurve and centroid curve
###########################################################################        
def plot_phase_folded_curves(self):
    
    #::: detrended curves
    fig, axes = plt.subplots( 4, 1, sharex=True, figsize=(16,16) )
    
    axes[0].plot( self.dic['PHI'], self.dic['SYSREM_FLUX3']/np.nanmedian(self.dic['SYSREM_FLUX3']), 'k.', alpha=0.1, rasterized=True )
    axes[0].errorbar( self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASE'], yerr=self.dic['SYSREM_FLUX3_PHASE_ERR'], fmt='o', color='r', rasterized=True )
#        axes[0].errorbar( self.dic['HJD_PHASE'], self.dic['SYSREM_FLUX3_PHASEmedian'], yerr=self.dic['SYSREM_FLUX3_PHASEmedian_ERR'], fmt='o', color='grey', alpha=0.5, rasterized=True )
    axes[0].set_ylabel('FLUX')
    axes[0].set_ylim([ np.min(self.dic['SYSREM_FLUX3_PHASE']-self.dic['SYSREM_FLUX3_PHASE_ERR']), np.max(self.dic['SYSREM_FLUX3_PHASE']+self.dic['SYSREM_FLUX3_PHASE_ERR']) ])
    
    axes[1].plot( self.dic['PHI'], self.dic['CENTDX'], 'k.', alpha=0.1, rasterized=True )
    axes[1].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDX_dmedc_PHASE'], yerr=self.dic['CENTDX_dmedc_PHASE_ERR'], fmt='o', rasterized=True ) #, color='darkgrey')
#        axes[1].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDX_dmedc_PHASEmedian'], yerr=self.dic['CENTDX_dmedc_PHASEmedian_ERR'], fmt='o', color='grey', alpha=0.5, rasterized=True ) #, color='darkgrey')
    axes[1].set_ylabel('CENTDX (in pixel)')
    axes[1].set_ylim([ np.min(self.dic['CENTDX_dmedc_PHASE']-self.dic['CENTDX_dmedc_PHASE_ERR']), np.max(self.dic['CENTDX_dmedc_PHASE']+self.dic['CENTDX_dmedc_PHASE_ERR']) ])
    
    axes[2].plot( self.dic['PHI'], self.dic['CENTDY'], 'k.', alpha=0.1, rasterized=True )
    axes[2].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDY_dmedc_PHASE'], yerr=self.dic['CENTDY_dmedc_PHASE_ERR'], fmt='o', rasterized=True ) #, color='darkgrey')
#        axes[2].errorbar( self.dic['HJD_PHASE'], self.dic['CENTDY_dmedc_PHASEmedian'], yerr=self.dic['CENTDY_dmedc_PHASEmedian_ERR'], fmt='o', color='grey', alpha=0.5, rasterized=True ) #, color='darkgrey')
    axes[2].set_ylabel('CENTDY (in pixel)')
    axes[2].set_xlabel('Phase')
    axes[2].set_ylim([ np.min(self.dic['CENTDY_dmedc_PHASE']-self.dic['CENTDY_dmedc_PHASE_ERR']), np.max(self.dic['CENTDY_dmedc_PHASE']+self.dic['CENTDY_dmedc_PHASE_ERR']) ])
    axes[2].set_xlim([-0.25,0.75])
    
    axes[3].plot( self.dic['HJD_PHASE'], self.dic['N_PHASE'], 'go', rasterized=True ) #, color='darkgrey')
    axes[3].set_ylabel('Nr of exposures')
    axes[3].set_xlabel('Phase')
    axes[3].set_xlim([-0.25,0.75])
    
    plt.tight_layout()
    self.fig_phasefold = fig      
    
    


###########################################################################
#::: plot info page
###########################################################################  
def plot_info_page(self):
    #::: plot object
    fig = plt.figure(figsize=(16,4))
    gs = gridspec.GridSpec(1, 4)
    
    #::: plot locations on CCD    
    ax = plt.subplot(gs[0, 0])
    label = 'Flux ' + str(self.flux_min) + '-' + str(self.flux_max) + ', ' + str(self.pixel_radius) + ' px'
    ax.plot( self.dic_nb['CCDX_0'], self.dic_nb['CCDY_0'], 'k.', rasterized=True, label=label )
    ax.plot( self.dic['CCDX'][0], self.dic['CCDY'][0], 'r+', rasterized=True ) 
    ax.add_patch( patches.Rectangle(
                    (self.dic['CCDX'][0]-self.pixel_radius, self.dic['CCDY'][0]-self.pixel_radius),
                    2*self.pixel_radius,
                    2*self.pixel_radius,
                    fill=False, color='r') )
#    ax.axis('equal')
    ax.set_xlim([0,2048])  
    ax.set_ylim([0,2048]) 
    ax.legend(loc='best', numpoints=1)
    
    
    #::: plot lightcurve
    ax = plt.subplot(gs[0, 1:3])
    ax.plot( self.dic['HJD'], self.dic['SYSREM_FLUX3'], 'k.', rasterized=True )
    ax.set(xlim=[np.min(self.dic['HJD']), np.max(self.dic['HJD'])])
    
    
    
    #::: plot info text
    ax = plt.subplot(gs[0, 3])
    self.plot_info_text(ax)    
    
    
    plt.tight_layout()
    
    self.fig_info_page = fig



def plot_info_text(self, ax):
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.axis('off')
    ra, dec = deg2HMS.deg2HMS(ra=self.dic['RA'], dec=self.dic['DEC'])
    ax.text(0,1.0,'OBJ_ID: '+self.dic['OBJ_ID']+'\t'+self.source)
    ax.text(0,0.9,'FLUX: '+str(self.dic['FLUX_MEAN']))
    ax.text(0,0.8,'RA (deg): '+ra)
    ax.text(0,0.7,'DEC (deg): '+dec)
    ax.text(0,0.6,'PERIOD (s): '+helper_functions.mystr(self.dic['PERIOD'],2))
    ax.text(0,0.5,'PERIOD (d): '+helper_functions.mystr(self.dic['PERIOD']/3600./24.,2))
    ax.text(0,0.4,'Width (s): '+helper_functions.mystr(self.dic['WIDTH'],2))
    ax.text(0,0.3,'Width (h): '+helper_functions.mystr(self.dic['WIDTH']/3600.,2))
    ax.text(0,0.2,'EPOCH (s): '+helper_functions.mystr(self.dic['EPOCH'],2))
    ax.text(0,0.1,'Depth (mmag): '+helper_functions.mystr(np.abs(self.dic['DEPTH'])*1000.,2))
    ax.text(0,0.0,'Num Transits: '+helper_functions.mystr(self.dic['NUM_TRANSITS'],0))
    


def plot_stacked_image(self):
    
    self.fig_stacked_image = stacked_images.plot(self.fieldname, np.nanmean(self.dic['CCDX']), np.nanmean(self.dic['CCDY']), r=15) 
    
    
    


###########################################################################
#::: save all plots in one pdf per target object
###########################################################################   
def save_pdf(self):
    outfilename = os.path.join( self.outdir, self.fieldname + '_' + self.obj_id + '_centroid_analysis.pdf' )
    with PdfPages( outfilename ) as pdf:
#            pdf.savefig( self.fig3 ) #self.fig_phasefold
#            pdf.savefig( self.fig1a )
#            pdf.savefig( self.fig2a )            
#            pdf.savefig( self.fig1 )
#            pdf.savefig( self.fig2 )
        pdf.savefig( self.fig_corrfx  )
        pdf.savefig( self.fig_corrfy  )
        pdf.savefig( self.fig_corrxy  )
        pdf.savefig( self.fig_autocorr  )
        pdf.savefig( self.fig_matrix  )
        pdf.savefig( self.fig_phasefold  )
#            pdf.savefig( self.fig_phasefold_arma  )
        pdf.savefig( self.fig_info_page  )
        pdf.savefig( self.fig_stacked_image  )
        pdf.savefig( self.fig_detrending  )
#            pdf.savefig( self.fig_detrending_over_time )
        print 'Plots saved as ' + outfilename
        
    if self.show_plot == False: plt.close('all'):
        
        
    

     