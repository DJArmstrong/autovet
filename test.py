# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:37:37 2017

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

#import numpy as np
#import matplotlib.pyplot as plt

from ngtsio import ngtsio
from autovet.Loader.Loader import Candidate
from autovet.Loader.NGTS_MultiLoader import NGTS_MultiLoader
#from autovet.Features.Centroiding.Centroiding_autovet_wrapper import centroid_autovet



def test(case):
       
    # a) load candidate for a specific NGTS object
    if case=='single':

        obj_id = '012109'
        fieldname = 'NG0304-1115'
        ngts_version = 'CYCLE1706'

#        obj_id = '003294'
#        fieldname = 'NG0409-1941'
#        ngts_version = 'TEST18'
#
#        obj_id = '003811'
#        fieldname = 'NG1318-4500'
#        ngts_version = 'TEST18'
#
#        obj_id = '058138'
#        fieldname = 'NG1318-4500'
#        ngts_version = 'TEST18'
#
#        obj_id = '001519'
#        fieldname = 'NG0522-2518'
#        ngts_version = 'TEST18'
#
#        obj_id = '020057'
#        fieldname = 'NG0409-1941'
#        ngts_version = 'TEST18'
#
#        obj_id = '000401'
#        fieldname = 'NG0537-3056'
#        ngts_version = 'TEST18'
#
#        obj_id = '006328'
#        fieldname = 'NG1421+0000'
#        ngts_version = 'TEST18'
#
#        obj_id = '019164'
#        fieldname = 'NG0524-3056'
#        ngts_version = 'CYCLE1706'

#        obj_id = '022551'
#        fieldname = 'NG0524-3056'
#        ngts_version = 'CYCLE1706'
        
        dic = ngtsio.get(fieldname, ngts_version, ['PERIOD','EPOCH','WIDTH'], obj_id=obj_id, silent=False)
        period = dic['PERIOD'] / 3600. / 24. #in days
        epoch = dic['EPOCH'] / 3600. / 24. #in days
        width = dic['WIDTH'] / 3600. / 24. #in days
        
        filepath = [fieldname, ngts_version]
        can = Candidate( obj_id, filepath, observatory='NGTS', field_dic=None, label=None, candidate_data={'per':period, 't0':epoch, 'tdur':width} )

        #centroid_autovet( can, do_plot=True )
    
    

    # b) load candidates with the NGTS_MultiLoader function
    elif case=='multi':
        
        infile = 'test_infile.txt'
        NGTS_MultiLoader(infile, 'output/', docentroid=False)



if __name__ == '__main__':
    test('single')
#    test('multi')
