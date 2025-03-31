#!/usr/bin/env python
import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
from   LOC_aux import *

if (len(sys.argv)<2):
    print("Usage:")
    print("    ConvolveSpectra1D.py filename fwhm  [-angle angle_as] [-samples samples] [-cpu]")
    print("")
    print("Input:")
    print("    filename         = name of a spectrum file produced by LOC1D.py")
    print("    fwhm             = FWHM of the convolving beam given in arcsec units")
    print("    -angle angle_as  = angle_as is the model cloud [arcsec], usually already included in the spectrum file")
    print("    -samples samples = samples is the number of samples per dimension used in the convolution (default 201)")
    print("    -cpu             = use CPU instead, instead of the default that is to use a GPU")
    print("Place options only after the <filename> and <fwhm> arguments.")
    sys.exit()
    
filename  =  sys.argv[1]
fwhm_as   =  float(sys.argv[2])
angle_as  =  -1
samples   =  201
GPU       =  1

V = sys.argv
n = len(V)
i = 3
while(i<len(sys.argv)):
    if ((V[i]=="-angle")&(i<(n-1))):
        i += 1
        angle_as   = float(sys.argv[i])
    if ((V[i]=="-samples")&(i<(n-1))):
        i += 1
        samples    = int(sys.argv[i])
    if (V[i].lower()=="-cpu"):
        GPU = 0
    i += 1
    
## print(fwhm_as, angle_as, samples, GPU)
## sys.exit()
    
if  (angle_as<0):
    print("Convolve %s, FWHM=%.2f arcsec, %dx%d samples" % (filename, fwhm_as, samples, samples))
else:
    print("Convolve %s, FWHM=%.2f arcsec, model %.2f arcsec, %d x %d samples" % (filename, fwhm_as, angle_as, samples, samples))
    
ConvolveSpectra1D(filename, fwhm_as, GPU=GPU, platforms=[0,1,2,3,4], angle_as=angle_as, samples=samples)
