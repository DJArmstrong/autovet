#quick code to split a multiloader input file

infile = '/home/dja/Autovetting/Dataprep/multiloader_input_TEST18_v2.txt'
indata = np.genfromtxt(infile, names=True, dtype=None)

nsplits = 24

splitarrays = np.array_split(indata,nsplits)


for s,savearray in enumerate(splitarrays):
    outfile = infile[:-4]+'_'+str(s)+'.txt'
    with open(outfile,'w') as f:
        f.write('#')
        for name in savearray.dtype.names:
            f.write(name + ' ') 
        for row in savearray:
            for el in row:
                f.write(str(el)+' ')
            
