
#special version for ETE6 data:
	#no resolved background sources (hence no contamination calc etc)
	#blind area appropriate to bgEB injections


#load candidate
    #needs to load lightcurve, metadata inc position, magnitude, colours, ...
cand = Candidate()

#calculate priors
    #GAIA call, ...
    #training set parameters will be an input, as these define the range for trilegal
    #and potentially for mean eclipse probabilities
    #precalc trilegal densities - once per training set option anyway.
        #can do grid on sky
    
cand.get_priors(search_nearby=False)

#create 'undiluted' lightcurves, one for each tested source


#apply classifiers

classifier_pl_beb = Classifier()
classifier_pl_heb = Classifier() 
classifier_pl_eb = Classifier()   
classifier_pl_btp = Classifier()
classifier_pl_nonastro = Classifier()

#specific check against known blended sources
for resolved_source in ...
    classifier_pl_seb = Classifier()
    classifier_pl_spl = Classifier()  #use planet training set, 'correct' input lightcurves as appropriate in each case
    									#i.e. pass the classifier the lightcurve as it would be, undiluted, if the source
    									# was the star under consideration. Train on undiluted lightcurves.

#calculate FPP (priors * classifier posteriors)?



