#!/usr/bin/env python

"""
LOC.py copied to LOC_OT.py  2020-06-15
because Paths and Update started to have several additional parameters not needed 
in case of regular cartesian grid. LOC.py was already using separate kernel_update_py_OT.c for OT.
And OT uses work group per ray while in LOC.py it was still one work item per ray.
"""


import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)   
from   LOC_aux import *
t000 = time.time()


if (0):
    nside = 12
    tmp = zeros((12*nside*nside, 4), float32)
    for idir in range(12*nside*nside):
        theta, phi     =  Pixel2AnglesRing(nside, idir)
        tmp[idir, 0:2] = [theta, phi]
        theta, phi     =  healpy.pix2ang(nside, idir)
        tmp[idir, 2:4] = [theta, phi]
    clf()
    plot(tmp[:,0], tmp[:,1], 'r+')
    plot(tmp[:,2], tmp[:,3], 'bx')
    show()
    sys.exit()
    
"""
x = cl.cltypes.make_float2() 
x['x'], x['y']
x = np.zeros((3,3), cl.cltypes.float2) 
x[0,0]['x']
"""

if (len(sys.argv)<2):
    print("Usage:  LOC.py  <ini-file>")
    sys.exit()
    
INI         =  ReadIni(sys.argv[1])

MOL         =  ReadMolecule(INI['molecule'])
HFS         =  len(INI['hfsfile'])>0               # HFS with LTE components
LEVELS      =  INI['levels']                       # energy levels in the species
TRANSITIONS =  MOL.Transitions(LEVELS)             # how many transitions among LEVELS levels
CHANNELS    =  INI['channels']
OCTREE      =  INI['octree']                       # 0 = regular Cartesian, 1 = octree, 2 = octree with ray splitting
WITH_ALI    =  (INI['WITH_ALI']>0)
DOUBLE_POS  =  ((OCTREE==4)|(OCTREE==40))          # 2020-08-01 this was the working combination (not very fast though)
MAX_NBUF    =  40
MAX_NBUF    =  INI['maxbuf']
PLWEIGHT    =  INI['plweight']
WITH_HALF   =  INI['WITH_HALF']>0

REAL, REAL3 = None, None
if (DOUBLE_POS>0):
    REAL, REAL3 = np.float64, cl.cltypes.double3
else:
    REAL, REAL3 = np.float32, cl.cltypes.float3
FLOAT3 = cl.cltypes.float3
print("LOC_OT.py, OCTREE=%d, WITH_ALI=%d, DOUBLE_POS=%d, WITH_HALF=%d, PLEIGHT=%d, MAX_NBUF=%d" % \
(OCTREE, WITH_ALI, DOUBLE_POS, WITH_HALF, PLWEIGHT, MAX_NBUF))



FIX_PATHS   =  0
if (OCTREE>0):
    FIX_PATHS = 0     # does not work for octrees, especially OT4
# FIX_PATHS does not work for OCTREE>0 cases (yet)
#   Should work for OCTREE=0 => dispersion on PL is decreased. While this effect is very small
#   and dispersion is already in the 5th significant digit and the mean PL remains the same to
#   the fifth significant digit, in test case the minimum Tex increases from 2.69K to 2.81K !!
# 2020-01-10 ... FIX_PATHS *not* used


if (OCTREE):
    # RHO[CELLS], TKIN[CELLS], CLOUD[CELLS]{vx,vy,vz,sigma}, ABU[CELLS], OFF[OTL], OTL = octree levels
    RHO, TKIN, CLOUD, ABU, CELLS, OTL, LCELLS, OFF,  NX, NY, NZ =  ReadCloudOT(INI, MOL)
else:
    print("*** Cartesian grid cloud ***")
    RHO, TKIN, CLOUD, ABU, NX, NY, NZ  =  ReadCloud3D(INI, MOL)
    CELLS  =  NX*NY*NZ
    OTL    =  -1

ONESHOT    =  INI['oneshot']    # no loop over ray offsets, kernel lops over all the rays
NSIDE      =  int(INI['nside'])
NDIR       =  max([6, 12*NSIDE*NSIDE])           # NSIDE=0 => 6 directions, cardinal directions only
WIDTH      =  INI['bandwidth']/INI['channels']   # channel width [km/s], even for HFS calculations
AREA       =  2.0*(NX*NY+NY*NZ+NZ*NX) 
# NRAY should be enough for the largest side of the cloud
if ((NX<=NY)&(NX<=NZ)):  NRAY =  ((NY+1)//2) * ((NZ+1)//2) 
if ((NY<=NX)&(NY<=NZ)):  NRAY =  ((NX+1)//2) * ((NZ+1)//2) 
if ((NZ<=NX)&(NZ<=NY)):  NRAY =  ((NX+1)//2) * ((NY+1)//2) 
if (ONESHOT):
    if ((NX<=NY)&(NX<=NZ)):  NRAY =  NY * NZ 
    if ((NY<=NX)&(NY<=NZ)):  NRAY =  NX * NZ 
    if ((NZ<=NX)&(NZ<=NY)):  NRAY =  NX * NY 
    if (not(OCTREE in [0, 4, 40])):
        print("Option ONESHOT applies to method OCTREE=0, OCTREE=4 and OCTREE=40 only!!")
        sys.exit()
    
VOLUME      =  1.0/(NX*NY*NZ)         #   Vcell / Vcloud ... for octree it is volume of root-grid cell
GL          =  INI['angle'] * ARCSEC_TO_RADIAN * INI['distance'] * PARSEC
APL         =  0.0
# APL_WEIGHT  =  1.0

print("================================================================================")
if (OCTREE>0):
    print("CLOUD %s, ROOT TGRID %d %d %d, OTL %d, LCELLS " % (INI['cloud'], NX, NY, NZ, OTL), LCELLS)
m = nonzero(RHO>0.0)  # only leaf nodes
print("    CELLS    %d" % CELLS)
if (WITH_HALF==0):
    print("    density  %10.3e  %10.3e" % (np.min(RHO[m]),  np.max(RHO[m])))
    print("    Tkin     %10.3e  %10.3e" % (np.min(TKIN[m]), np.max(TKIN[m])))
    print("    Sigma    %10.3e  %10.3e" % (np.min(CLOUD['w'][m]), np.max(CLOUD['w'][m])))
    print("    vx       %10.3e  %10.3e" % (np.min(CLOUD['x'][m]), np.max(CLOUD['x'][m])))
    print("    vy       %10.3e  %10.3e" % (np.min(CLOUD['y'][m]), np.max(CLOUD['y'][m])))
    print("    vz       %10.3e  %10.3e" % (np.min(CLOUD['z'][m]), np.max(CLOUD['z'][m])))
    print("    chi      %10.3e  %10.3e" % (np.min(ABU[m]),  np.max(ABU[m])))
    if ((np.min(TKIN[m])<0.0)|(np.min(ABU[m])<0.0)|(np.min(CLOUD['w'][m])<0.0)):
        print("*** Check the cloud parameters: Tkin, abundance, sigma must all be non-negative")
        sys.exit()
else:
    print("    density  %10.3e  %10.3e" % (np.min(RHO[m]),        np.max(RHO[m])))
    print("    Tkin     %10.3e  %10.3e" % (np.min(TKIN[m]),       np.max(TKIN[m])))
    print("    Sigma    %10.3e  %10.3e" % (np.min(CLOUD[:,3][m]), np.max(CLOUD[:,3][m])))
    print("    vx       %10.3e  %10.3e" % (np.min(CLOUD[:,0][m]), np.max(CLOUD[:,0][m])))
    print("    vy       %10.3e  %10.3e" % (np.min(CLOUD[:,1][m]), np.max(CLOUD[:,1][m])))
    print("    vz       %10.3e  %10.3e" % (np.min(CLOUD[:,2][m]), np.max(CLOUD[:,2][m])))
    print("    chi      %10.3e  %10.3e" % (np.min(ABU[m]),        np.max(ABU[m])))
    if ((np.min(TKIN[m])<0.0)|(np.min(ABU[m])<0.0)|(np.min(CLOUD[:,3][m])<0.0)):
        print("*** Check the cloud parameters: Tkin, abundance, sigma must all be non-negative")
        sys.exit()
print("GL %.3e, NSIDE %d, NDIR %d, NRAY %d" % (GL, NSIDE, NDIR, NRAY))
print("================================================================================")


if (0):
    for i in [ 837288, 837233 ]:
        print()
        print("CELL %d" % i)
        print("    density  %10.4e" % RHO[i])
        print("    Tkin     %10.4e" % TKIN[i])
        print("    Sigma    %10.4e" % CLOUD['w'][i])
        print("    vx       %10.4e" % CLOUD['x'][i])
        print("    vy       %10.4e" % CLOUD['y'][i])
        print("    vz       %10.4e" % CLOUD['z'][i])
        print("    chi      %10.4e" % ABU[i])
        # peak optical depth for sigma(V) = 1 km/s line
        s          =  250.0/64.0  * PARSEC * 0.25
        GN         =  C_LIGHT/(1.0e5*WIDTH*MOL.F[0])
        taumax     =  (RHO[i]*ABU[i]  *   MOL.A[0]  *   C_LIGHT**2 / (8.0*pi*MOL.F[0]**2)) * s * GN        
        print("    taumax   %10.3e" % taumax)
        print("    GN = %10.3e" % GN)
        print()
    sys.exit()


LOWMEM      =  INI['lowmem']
COOLING     =  INI['cooling']
MAXCHN      =  INI['channels']
MAXCMP      =  1

if (COOLING & HFS):
    print("*** Cooling not implemented for HFS => cooling will not be calculated!")
    COOLING =  0
if (HFS):
    BAND, MAXCHN, MAXCMP = ReadHFS(INI, MOL)     # MAXCHN becomes the maximum over all transitions
    print("HFS revised =>  CHANNELS %d,  MAXCMP = %d" % (CHANNELS, MAXCMP))
    HF      =  zeros(MAXCMP, cl.cltypes.float2)

    
WITH_OVERLAP  = (len(INI['overlap'])>0)
if (WITH_OVERLAP):
    OLBAND, OLTRAN, OLOFF, MAXCMP = ReadOverlap(INI['overlap'], MOL, WIDTH, TRANSITIONS, CHANNELS)
    ##
    SINGLE = ones(TRANSITIONS, int32)  # by default all transitions single
    MAXCHN = 0
    for iband in range(OLBAND.Bands()):
        for icmp in range(OLBAND.Components(iband)):
            SINGLE[OLBAND.GetTransition(iband, icmp)] = 0  # transition part of a wider band (overlap)
        MAXCHN = max([MAXCHN, OLBAND.Channels(iband)])     # Channels() == extra channels!
        MAXCMP = max([MAXCMP, OLBAND.NCMP[iband]])
        if (INI['verbose']>1): 
            print("OVERLAP => CHANNELS %d, MAXCHN %d, MAXCMP %d" % (CHANNELS, MAXCHN, MAXCMP))
            for i in range(TRANSITIONS):
                print("  SINGLE[%d] = %d" % (i, SINGLE[i]))
    MAXCHN += CHANNELS
    
print("TRANSITIONS %d, CELLS %d = %d x %d x %d" % (TRANSITIONS, CELLS, NX, NY, NZ))
SIJ_ARRAY, ESC_ARRAY = None, None
if (LOWMEM>1): #  NI_ARRAY, SIJ_ARRAY and (for ALI) ESC_ARRAY are mmap files
    SIJ_ARRAY = np.memmap('LOC_SIJ.mmap', dtype='float32', mode='w+', offset=0, shape=(CELLS, TRANSITIONS))
    if (WITH_ALI==0):  
        ESC_ARRAY =  zeros(1, float32)  # dummy array
    else:              
        ESC_ARRAY =  np.memmap('LOC_ESC.mmap', dtype='float32', mode='w+', offset=0, shape=(CELLS, TRANSITIONS))
else:  # SIJ_ARRAY and ESC_ARRAY normal in-memory arrays
    SIJ_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)
    if (WITH_ALI==0):  ESC_ARRAY   =  zeros(1, float32)  # dummy array
    else:              ESC_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)

WITH_CRT    =  INI['with_crt']
CRT_TAU     =  []
CRT_EMI     =  []
TMP         =  []
if (WITH_CRT):
    # Also in case of octree, CRT_TAU[CELLS] and CRT_EMI[CELLS] are simple contiguous vectors
    TMP     =  zeros(CELLS, float32)
    print(INI['crttau'])
    CRT_TAU =  ReadDustTau(INI['crttau'], GL, CELLS, TRANSITIONS)                # [CELLS, TRANSITIONS]
    CRT_EMI =  ReadDustEmission(INI['crtemit'], CELLS, TRANSITIONS, WIDTH, MOL)  # [CELLS, TRANSITIONS]
    # conversion from photons / s / channel / H    -->   photons / s / channel / cm3
    for t in range(TRANSITIONS):
        CRT_EMI[:,t] *=  RHO

    
# unlike for LOC1D.py which has GAU[TRANSITIONS, CELLS, CHANNELS], LOC.py still has
# GAU[GNO, CHANNELS] ... we probably should have this in global memory, not to restrict GNO.
# overlap uses the same LIM
GAUSTORE = '__global'
GNO      =  100      # number of precalculated Gaussians
G0, GX, GAU, LIM =  GaussianProfiles(INI['min_sigma'], INI['max_sigma'], GNO, CHANNELS, WIDTH)
print("CHANNELS %d, GAU" % CHANNELS, GAU.shape)

if (INI['sdevice']==''):
    # We use ini['GPU'], INI['platforms'], INI['idevice'] to select the platform and device
    if (INI['GPU']):  LOCAL = 32
    else:             LOCAL =  1
    if (INI['LOCAL']>0): LOCAL = INI['LOCAL']
    platform, device, context, queue,  mf = InitCL(INI['GPU'], INI['platforms'], INI['idevice'], sub=INI['cores'])
else:
    # we use INI['GPU'] and optionally INI['sdevice'] to select the device
    platform, device, context, queue, mf = InitCL_string(INI) # "sub" read directly from INI
    if (INI['GPU']):  LOCAL = 32
    else:             LOCAL =  1
    if (INI['LOCAL']>0): LOCAL = INI['LOCAL']
    

if (OCTREE<2):
    NWG    =  -1
    GLOBAL =  IRound(NRAY, 32)
elif (OCTREE in [2,3,4,5]):   # one work group per ray
    NWG    =  NRAY
    # for memory reasons, we need to limit NWG
    # BUFFER requires possibly 20 kB per work group, limit that to ~100MB => maximum of ~10000 work groups
    #  each level can add one fron-ray and one side-ray entry to the buffer,
    #  each entry is for OCTREE4  equal to 26+CHANNELS floats
    #  MAXL=7 ==   7*2*(26+CHANNELS) =  4 kB if CHANNELS=256
    #  NWG=16384  =>  buffer allocation 61.7 MB
    ########################
    #  BUFFER  ~  8*NWG*(26+CHANNELS)*MAX_NBUF
    NWG   = min([NRAY,    int(1+200.0e6/(8.0*(26.0+CHANNELS)*MAX_NBUF))   ])
    if (NWG>16384):
        NWG = 16384
    
    ## NWG = 1024
    
    GLOBAL =  IRound(NWG*LOCAL, 32)
    print("*** NWG SET TO %d   -->   GLOBAL %d --- NWG=%d == GLOBAL/LOCAL = %.3f" % \
    (NWG, GLOBAL, NWG, GLOBAL/float(LOCAL)))
elif (OCTREE in [40,]):  # one work item per ray
    NWG    =  -1
    GLOBAL =  IRound(NRAY, 32)  # for ONESHOT, this is ~NX*NY
    
    
# TAUSAVE = (OCTREE>0)&(INI['tausave']>0)
TAUSAVE = (INI['tausave']>0)

# note -- CHANNELS is compile-time parameter, not changed by HFS or OVERLAP
OPT = " -D NX=%d -D NY=%d -D NZ=%d -D NRAY=%d -D CHANNELS=%d -D WIDTH=%.5ff -D ONESHOT=%d \
-D VOLUME=%.5ef -D CELLS=%d -D LOCAL=%d -D GLOBAL=%d -D GNO=%d -D SIGMA0=%.5ff -D SIGMAX=%.4ff \
-D GL=%.4ef -D MAXCHN=%d -D WITH_HALF=%d -D PLWEIGHT=%d -D LOC_LOWMEM=%d -D CLIP=%.3ef -D BRUTE_COOLING=%d \
-D LEVELS=%d -D TRANSITIONS=%d -D WITH_HFS=%d -D WITH_CRT=%d -D DOUBLE_POS=%d -D MAX_NBUF=%d \
-I%s -D OTLEVELS=%d -D NWG=%d -D WITH_OCTREE=%d -D GAUSTORE=%s -D WITH_ALI=%d -D FIX_PATHS=%d \
-D MINMAPLEVEL=%d -D MAP_INTERPOLATION=%d -D TAUSAVE=%d -D WITH_OVERLAP=%d -D MAXCMP=%d" % \
(NX, NY, NZ, NRAY, CHANNELS, WIDTH, ONESHOT, 
VOLUME, CELLS, LOCAL, GLOBAL, GNO, G0, GX,
GL, MAXCHN, WITH_HALF, PLWEIGHT, LOWMEM, INI['clip'], (COOLING==2),
LEVELS, TRANSITIONS, HFS, WITH_CRT, DOUBLE_POS, MAX_NBUF, 
INSTALL_DIR, OTL, NWG, OCTREE, GAUSTORE, WITH_ALI, FIX_PATHS, 
int(INI['minmaplevel']), INI['MAP_INTERPOLATION'], TAUSAVE, WITH_OVERLAP, MAXCMP)

if (0):
    # -cl-fast-relaxed-math == -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only
    # -cl-unsafe-math-optimizations is ***DANGEROUS*** ON NVIDIA, cell links rounded to zero !!!
    OPT += "-cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only"  # this seems ok on NVidia !!
    
if (0):
    OPT  += " -cl-std=CL1.1"
if (INI['GPU']==99):  # Intel fails .... does optimisation work for POCL? Yes - with no visible effect...
    OPT  += " -cl-opt-disable"  # opt-disable faster??? at least 3D up to 64^3
if (1):
    print("Kernel options:")
    print(OPT)
    
# Note: unlike in 1d versions, both SIJ and ESC are divided by VOLUME only in the solver
#       3d and octree use kernel_LOC_aux.c, where the solver is incompatible with the 1d routines !!!
source    =  open(INSTALL_DIR+"/kernel_update_py_OT.c").read()
#  MAXL3_P20.0 WITH_HALF   10.7/9.6   s1   65/7.5  68645100 7834372
#  gid   54746 ------  NBUF = 50 !
#  during run 68646124 7834372
#  MAXL3_P20.0 WITH_HALF   10.7/9.6    65.5/7.5
print("--- Create program -------------------------------------------------------------")
print("")
program       =  cl.Program(context, source).build(OPT, cache_dir=None)
print("--------------------------------------------------------------------------------")

# Set up kernels
kernel_clear   =  program.Clear
kernel_paths   =  None   # only if PLWEIGHT in use... no we need this also to get correct PACKETS !!
kernel_sim_multitran = None
if (OCTREE>0):
    if (OCTREE==1): 
        print("USING  program.UpdateOT1: offs %d" % INI['offsets'])
        kernel_sim   =  program.UpdateOT1
        kernel_paths =  program.PathsOT1
    elif (OCTREE==2):           
        print("USING  program.UpdateOT2: offs %d, ALI %d" % (INI['offsets'], WITH_ALI))
        kernel_sim   =  program.UpdateOT2
        kernel_paths =  program.PathsOT2
    elif (OCTREE==3):
        print("USING  program.UpdateOT3:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT3
        kernel_paths =  program.PathsOT3
    elif (OCTREE==4):
        print("USING  program.UpdateOT4:  ALI %d" % ( WITH_ALI))
        kernel_sim    =  program.UpdateOT4
        kernel_sim_multitran = program.UpdateOT4Multitran
        kernel_paths  =  program.PathsOT4
    elif (OCTREE==5):
        print("USING  program.UpdateOT4:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT5
        kernel_paths =  program.PathsOT5
    elif (OCTREE==40):
        print("USING  program.UpdateOT40:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT40
        kernel_paths =  program.PathsOT40
else:
    print("USING  program.Update")
    kernel_sim                           =  program.Update
    if (WITH_OVERLAP):  kernel_sim_ol    =  program.UpdateOL
    if (PLWEIGHT):      kernel_paths     =  program.Paths

kernel_solve         =  program.SolveCL
kernel_parents       =  program.Parents


# 2020-05-31:  WITH_CRT implemented only for the default case (no HFS)
if (WITH_CRT):
    kernel_sim.set_scalar_arg_dtypes([
    # 0     1     2     3           4           5           6           7      8           9           10        
    # id0   CLOUD GAU   LIM         Aul         A_b         GN          PL     APL         BG          DIRWEI    
    None,   None, None, np.float32, np.float32, np.float32, np.float32, None,  np.float32, np.float32, np.float32,
    #  11       12       13     14      15     16     17     18       19      
    #  EWEI     LEADING  POS0   DIR     NI     RES    NTRUE  CRT_TAU  CRT_EMIT
    np.float32, np.int32,  REAL3, FLOAT3, None,  None,  None,  None,    None  ])
else:
    if (OCTREE<1):
        kernel_sim.set_scalar_arg_dtypes([
        # 0        1      2     3     4           5           6           7     8           9           
        # id0      CLOUD  GAU   LIM   Aul         A_b         GN          PL    APL         BG          
        np.int32,  None,  None, None, np.float32, np.float32, np.float32, None, np.float32, np.float32,
        # 10        11          12         13      14      15      16     17    
        # DIRWEI    EWEI        LEADING    POS0    DIR     NI      RES    NTRUE 
        np.float32, np.float32, np.int32,  REAL3,  FLOAT3, None,  None,   None  ])
        if (WITH_OVERLAP):
            kernel_sim_ol.set_scalar_arg_dtypes([
            # 0        1         2          3           4            5       6                  7                           
            # NCMP     NCHN,     Aul[NCMP]  A_b[NCMP]   GN           COFF,   NI[2,NCMP,CELLS]   NTRUES[GLOBAL, NCMP, MAXCHN]
            np.int32,  np.int32, None,      None,       np.float32,  None,    None,             None,
            # 8       9       10                 11        12      13          14    
            # id0     CLOUD   GAU[GNO,CHANNELS]  LIM[GNO]  PL      APL         BG    
            np.int32, None,   None,              None,     None,   np.float32, np.float32,
            # 15        16           17         18         19       20                 
            # DIRWEI    EWEI         LEADING    POS0       DIR      RES[2, NCMP, CELLS]
            np.float32, np.float32,  np.int32,  REAL3,     FLOAT3,  None,
            # 21       22        23    
            # TAU      EMIT      TT    
            None,      None,     None   ])
            
    elif (OCTREE==1):
        if (PLWEIGHT>0):
            kernel_sim.set_scalar_arg_dtypes([
            # 0   1     2     3           4           5           6           7           8     9           10
            # PL  CLOUD GAU   LIM         Aul         A_b         GN          APL         BG    DIRWEI      EWEI
            None, None, None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
            #   11     12     13       14     15     16   
            # LEADING  POS0   DIR      NI     RES    NTRUE
            np.int32,  REAL3, FLOAT3,  None,  None,  None ,
            #  17      18     19     20 
            # LCELLS   OFF    PAR    RHO
            None,      None,  None,  None])
        else:
            kernel_sim.set_scalar_arg_dtypes([
            # 0     1     2     3           4           5           6           7           8           9    
            # CLOUD GAU   LIM   Aul         A_b         GN          APL         BG          DIRWEI      EWEI,
            None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
            #  10      11     12       13     14     15   
            # LEADING  POS0   DIR      NI     RES    NTRUE
            np.int32,  REAL3, FLOAT3,  None,  None,  None ,
            #  16      17     18     19    
            # LCELLS   OFF    PAR    RHO   
            None,      None,  None,  None])
    elif (OCTREE in [2,3]): # OCTREE=2,3
        kernel_sim.set_scalar_arg_dtypes([
        # 0    1      2     3     4           5           6           7           8           9           10        
        # PL   CLOUD  GAU   LIM   Aul         A_b         GN          APL         BG          DIRWEI      EWEI      
        None,  None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
        #  11      12      13       14     15     16    
        # LEADING  POS0    DIR      NI     RES    NTRUE 
        np.int32,  REAL3,  FLOAT3,  None,  None,  None,
        #  17     18     19     20     21     
        # LCELLS, OFF,   PAR,   RHO    BUFFER 
        None,     None,  None,  None,  None ])
    elif (OCTREE>=4):
        if (PLWEIGHT>0):
            kernel_sim.set_scalar_arg_dtypes([
            # 0       1      2      3     4     5           6           7           8           9           
            # gid0    PL     CLOUD  GAU   LIM   Aul         A_b         GN          APL         BG          
            np.int32, None,  None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, 
            #  10       11          12         13      14       15     16     17    
            #  DIRWEI   EWEI        LEADING    POS0    DIR      NI     RES    NTRUE 
            np.float32, np.float32, np.int32,  REAL3,  FLOAT3,  None,  None,  None,
            #  18     19     20     21     22     
            # LCELLS, OFF,   PAR,   RHO    BUFFER 
            None,     None,  None,  None,  None ])
            ######
            if (kernel_sim_multitran):
                kernel_sim_multitran.set_scalar_arg_dtypes([
                    # 0       1      2      3     4     5           6           7            8           9           
                    # gid0    PL     CLOUD  GAU   LIM   Aul         A_b         GN           APL         BG          
                    np.int32, None,  None,  None, None, None,       None,       None,        np.float32, None, 
                    #  10       11          12         13      14       15     16     17     18     19
                    #  DIRWEI   EWEI        LEADING    POS0    DIR      NI     NBNB   RES    ESC    NTRUES 
                    np.float32, np.float32, np.int32,  REAL3,  FLOAT3,  None,  None,  None,  None,  None,
                    #  20     21     22     23     24      25        26
                    # LCELLS, OFF,   PAR,   RHO    BUFFER  ntran     btran
                    None,     None,  None,  None,  None,   np.int32, np.int32 ])
            
        else:
            kernel_sim.set_scalar_arg_dtypes([
            # 0       1      2     3     4      5           6           7           8              
            # gid0    CLOUD  GAU   LIM   Aul    A_b         GN          APL         BG             
            np.int32, None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, 
            #  9        10          11         12      13       14     15     16    
            #  DIRWEI   EWEI        LEADING    POS0    DIR      NI     RES    NTRUE 
            np.float32, np.float32, np.int32,  REAL3,  FLOAT3,  None,  None,  None,
            #  17     18     19     20     21    
            # LCELLS, OFF,   PAR,   RHO    BUFFER 
            None,     None,  None,  None,  None ])
   
#                                   RES[CELLS].xy
kernel_clear.set_scalar_arg_dtypes([None, ])

if (OCTREE==0):  ##  Cartesian
    if (PLWEIGHT>0):
        kernel_paths.set_scalar_arg_dtypes([np.int32, None, None, None,  np.int32, REAL3, FLOAT3])
    else:
        kernel_paths.set_scalar_arg_dtypes([np.int32, None, None,  np.int32, REAL3, FLOAT3])
elif (OCTREE==1):
    kernel_paths.set_scalar_arg_dtypes([None, None, None, np.int32, REAL3, FLOAT3,
    None, None, None, None])
elif (OCTREE in [2, 3]):
    kernel_paths.set_scalar_arg_dtypes([None, None, None,  np.int32, REAL3, FLOAT3, 
    None, None, None, None, None])
elif (OCTREE>=4):
    kernel_paths.set_scalar_arg_dtypes([np.int32, None, None, None,  np.int32, REAL3, FLOAT3,
    None, None, None, None, None])    


if (HFS):   
    if (OCTREE==0):
        kernel_hf    =  program.UpdateHF
        kernel_hf.set_scalar_arg_dtypes([
        # 0     1     2     3           4           5           6           7           8           9         
        # CLOUD GAU   LIM   Aul         A_b,        GN          APL         BG          DIRWEI      EWEI      
        None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
        # 10       11       12     13    14    15        16        17     18      19 
        # LEADING  POS      DIR    NI    RES   NCHN      NCOMP     HF     NTRUES  PROFILE
        np.int32,  REAL3,  FLOAT3, None, None, np.int32, np.int32, None,  None,   None])
    else:
        if (OCTREE==4):
            kernel_hf    =  program.UpdateHF4
            kernel_hf.set_scalar_arg_dtypes([
            # 0       1     2      3     4     5           6           7           8           9         
            # gid0    PL,   CLOUD  GAU   LIM   Aul         A_b,        GN          APL         BG        
            np.int32, None, None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32,
            # 10          11        12        13     14      15    16        17        17    
            # DIRWEI      EWEI      LEADING   POS    DIR     NI    NCHN      NCOMP     HF    
            np.float32, np.float32, np.int32, REAL3, FLOAT3, None, np.int32, np.int32, None,
            # 18   19      20      21    22    23    24    
            # RES  NTRUES  LCELLS  OFF   PAR   RHO   BUFFER
            None,  None,   None,   None, None, None, None  ])
        else:
            print("HFS not implemented for OCTREE=%d" % OCTREE), sys.exit()
else:       
    kernel_hf    =  None


if (OCTREE>0):
    kernel_solve.set_scalar_arg_dtypes(
    # 0        1         2     3     4     5     6         7         8         9         10    11    12   
    # OTL      BATCH     A     UL    E     G     PARTNERS  NTKIN     NCUL      MOL_TKIN  CUL   C     CABU 
    [np.int32, np.int32, None, None, None, None, np.int32, np.int32, np.int32, None,     None, None, None,
    # 13  14    15    16    17    18    19    20   
    # RHO TKIN  ABU   NI    SIJ   ESC   RES   WRK   
    None, None, None, None, None, None, None, None, np.int32 ])
else:
    kernel_solve.set_scalar_arg_dtypes(
    # 0        1     2     3     4     5         6         7         8         9     10    11   
    # BATCH    A     UL    E     G     PARTNERS  NTKIN     NCUL      MOL_TKIN  CUL   C     CABU 
    [np.int32, None, None, None, None, np.int32, np.int32, np.int32, None,     None, None, None,
    # 12  13    14    15    16    17    18    19    
    # RHO TKIN  ABU   NI    SIJ   ESC   RES   WRK   
    None, None, None, None, None, None, None, None, np.int32 ])


# Set up input and output arrays
# print("Set up input arrays")
if (PLWEIGHT):
    PL_buf =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)   # could be half??
else:
    PL_buf =  cl.Buffer(context, mf.READ_WRITE, 4)         # dummy
GAU_buf    =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=GAU)  # CHANNELS channels
LIM_buf    =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=LIM)
TPL_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY)
COUNT_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY)
if (0):
    CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CLOUD)   # vx, vy, vz, sigma
else:
    # On the host side, CLOUD is the second largest array after NI_ARRAY.
    # Here we drop CLOUD before the NI_ARRAY is allocated.
    if (WITH_HALF==0):
        CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY, 4*4*np.int64(CELLS))   # vx, vy, vz, sigma
    else:
        CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY, 2*4*np.int64(CELLS))   # vx, vy, vz, sigma
    cl.enqueue_copy(queue, CLOUD_buf, CLOUD)
    CLOUD = None

    
NI_buf    =  cl.Buffer(context, mf.READ_ONLY,   8*CELLS)                          # nupper, nb_nb
if (WITH_ALI>0):
    RES_buf   =  cl.Buffer(context, mf.READ_WRITE,  8*CELLS)                      # SIJ, ESC  RES[CELLS,2]
else:
    RES_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS)                      # SIJ only
HF_buf    =  None
if (HFS):
    HF_buf  =  cl.Buffer(context, mf.READ_ONLY,   8*MAXCMP)

if (COOLING==2):
    COOL_buf = cl.Buffer(context,mf.WRITE_ONLY, 4*CELLS)

# 2021-07-14 -- INI['points'] is now the maximum over all map views
print("NTRUE_buf %d spectra x %d channels" % (max([INI['points'][0], NRAY]), MAXCHN))
NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
STAU_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
WRK       =  np.zeros((CELLS,2), np.float32)     #  NI=(nu, nb_nb)  and  RES=(SIJ, ESC)

# Buffers for SolveCL
NTKIN        =  len(MOL.TKIN[0])
PARTNERS     =  MOL.PARTNERS
NCUL         =  MOL.CUL[0].shape[0] 
MOL_A_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.A)
MOL_UL_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.TRANSITION)
MOL_E_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.E)
MOL_G_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.G)
MOL_TKIN_buf =  cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NTKIN) 
# -- note -- All partners must have collisional coefficients for the *same transitions*
#            and in the *same order* (as in MOL.CUL[0]).
#            They also must have the *same number of Tkin values* but the actual 
#            TKIN values in the temperature grid can be different for different partners.
#            The above is ensured when reading the molecule file but one could also
#            already convert the input molecule files to fulfill these conditions,
#            before running LOC. This might be needed, for example, to deal with 
#            partially missing collisional coefficient (some collisional partner,
#            some transitions).
MOL_TKIN     =  zeros((PARTNERS, NTKIN), float32)
for i in range(PARTNERS):
    MOL_TKIN[i, :]  =  MOL.TKIN[i][:]
cl.enqueue_copy(queue, MOL_TKIN_buf, MOL_TKIN)        
# CUL  -- levels are included separately for each partner... must have the same number of rows!
#         KERNEL ASSUMES IT IS THE SAME TRANSITIONS, IN THE SAME ORDER
if (MOL.CUL[i].shape[0]!=NCUL):
    print("SolveCL assumes the same number of C rows for each collisional partner!!"),  sys.exit()
CUL   =  zeros((PARTNERS,NCUL,2), int32)
for i in range(PARTNERS):
    CUL[i, :, :]  =  MOL.CUL[i][:,:]
    # KERNEL USES ONLY THE CUL ARRAY FOR THE FIRST PARTNER -- CHECK THAT TRANSITIONS ARE IN THE SAME ORDER
    delta =   max(ravel(MOL.CUL[i]-MOL.CUL[0]))
    if (delta>0):
        print("*** ERROR: SolveCL assumes all partners have C in the same order of transitions!!"), sys.exit()
MOL_CUL_buf  = cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NCUL*2)
cl.enqueue_copy(queue, MOL_CUL_buf, CUL)        
# C
C    =  zeros((PARTNERS, NCUL, NTKIN), float32)
for i in range(PARTNERS):   
    C[i, :, :]  =  MOL.CC[i][:, :]
MOL_C_buf  = cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NCUL*NTKIN)
cl.enqueue_copy(queue, MOL_C_buf, C)        
# abundance of collisional partners,  MOL.CABU[PARTNERS]
# new buffer for matrices and the right side of the equilibrium equations
BATCH        =  max([1,CELLS//max([LEVELS, TRANSITIONS])]) # now ESC, SIJ fit in NI_buf
BATCH        =  min([BATCH, CELLS, 16384])        #  16384*100**2 = 0.6 GB
##  BATCH    =  16384  # @@@


SOL_WRK_buf  = cl.Buffer(context, mf.READ_WRITE, 4*BATCH*LEVELS*(LEVELS+1))
SOL_RHO_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_TKIN_buf = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_ABU_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_SIJ_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
SOL_ESC_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
SOL_NI_buf   = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*LEVELS)
SOL_CABU_buf = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*PARTNERS)   #  2024-06-02


if (WITH_CRT):
    # unlike in LOC1D.py, buffers are for single transition at a time
    CRT_TAU_buf = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
    CRT_EMI_buf = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
                

    
if (OCTREE>0):
    # Octree-hierarchy-specific buffers
    LCELLS_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    OFF_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    PAR_buf     = cl.Buffer(context, mf.READ_WRITE, 4*max([1, (CELLS-NX*NY*NZ)]))  # no space for root cells!
    RHO_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=RHO)
    BUFFER_buf  = None
    if (OCTREE==2):
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(14+CHANNELS)*512)    # 1024 ???
    elif (OCTREE==3):  #  OTL, OTI,  {x, y, z, OTLRAY} [CHANNELS]  = 2+12*4+CHANNELS
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(50+CHANNELS)*512)
    elif (OCTREE in [4,5]):  #  OTL, OTI,  {x, y, z, OTLRAY} [CHANNELS]  = 2+12*4+CHANNELS
        #  for each additionl level of hierarchy, original ray + 2 new rays to buffer, one new continues
        #  => BUFFER must be allocated to for just 3*MAXL rays... actually there are also siderays
        #  leading-edge rays go to one slot (26+CHANNELS) includes space for 4 rays),
        #  siderays go to another slot with (26+CHANNELS) numbers
        #  => each increase of refinement requires two slots (slot = 26+CHANNELS)
        #  for MAXL=7 that is 14 slots
        #  if root grid is 512^2, NWG~256^2 and the memory requirement for BUFFER is
        #  256^2*14*(26+CHANNELS~256)*4B  ~ 1 GB, 2 GB for DOUBLE_POS ....        
        #  => we limit NWG (OCTREE==4) and loop over the rays using several kernel calls
        #  Also, if we are computing hyperfine structure lines (HFS), we need storage for 
        #  not CHANNELS but for MAXCHN channels!!
        #  Combining HFS and ONESHOT options then means that 512^3 root grid and MAXCHN=512 lead to
        #     NWG=65536  x   MAXCHN=1024  x  4B  = 320 MB  ... still ok
        if (HFS):
            if (DOUBLE_POS>0): BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 8*NWG*(26+MAXCHN)*MAX_NBUF)
            else:              BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(26+MAXCHN)*MAX_NBUF)
        else:
            if (DOUBLE_POS>0): BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 8*NWG*(26+CHANNELS)*MAX_NBUF)
            else:              BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(26+CHANNELS)*MAX_NBUF)
            print("BUFFER ALLOCATION  %.3e MB" % (float(8.0*NWG*(26+MAXCHN)*MAX_NBUF)/(1024.0*1024.0)))
            ## sys.exit()
            
    elif (OCTREE in [40,]):
        # We allocate in BUFFER also room for GLOBAL*CHANNELS, the NTRUE vectors in the update kernel
        print("OCTREE=40, ALLOCATE BUFFER = %.3f MB" % ( (GLOBAL*(26+CHANNELS)*MAX_NBUF + 4*GLOBAL*CHANNELS)/(1024.0*1024.0)))
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL*(26+CHANNELS)*MAX_NBUF + 4*GLOBAL*CHANNELS)
        
    # update links from cells to parents:  DENS  LCELLS  OFF   PAR
    kernel_parents.set_scalar_arg_dtypes([ None, None,   None, None])
    program.Parents(queue, [GLOBAL,], [LOCAL,], RHO_buf, LCELLS_buf, OFF_buf, PAR_buf)


PROFILE_buf = None
if ((LOWMEM>0) & HFS):
    # the writing of spectra needs GLOBAL*MAXCHN for the profile vectors
    # lowmem>1 does not work with writing of HFS spectra ??? 2023-12-02
    # => allocation must be max(GLOBAL,NRA), where NRA>GLOBAL possible for large maps
    theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][0]
    tmp = max(NRA, GLOBAL)
    PROFILE_buf = cl.Buffer(context, mf.READ_WRITE, 4*tmp*MAXCHN)
    print("PROFILE BUFFER ALLOCATED FOR %d x %d = %.3e FLOATS" % (tmp, MAXCHN, tmp*MAXCHN))
else:
    PROFILE_buf = cl.Buffer(context, mf.READ_WRITE, 4)  # dummy
    
    

# 2020-06-26 using  DIRWEI ==  normalisation factor <cos theta>, calculated for the NDIR directions
#                   DIRWEI may still be different for the six sides, depending on how the directions are
PACKETS     =   0
TPL, COUNT  =  [], []
DIRWEI      =  zeros(6, float32)  #  <cos(theta)> = <DIR.normal_component> for each six sides
EWEI        =  0.0 
RCOUNT      =  zeros(6, int32)
TRUE_APL    =  0.0
if (PLWEIGHT):
    PL      =  1.0e-30*ones(CELLS, float32)
else:
    PL      =  None
if (INI['iterations']>0):
    TPL     =  zeros(NRAY, float32)
    COUNT   =  zeros(NRAY, int32)
    offs    =  INI['offsets']          # default = 1, one ray per root grid surface element
    print("Paths...  for NRAY=%d, CELLS=%d, offs=%d" % (NRAY, CELLS, offs))
    if (PLWEIGHT):
        cl.enqueue_copy(queue, PL_buf,  PL)
        cl.enqueue_copy(queue, TPL_buf, TPL)
        queue.finish()
    t00     =  time.time()  
    inner_loop = 4*offs*offs

    for idir in range(NDIR):
        # offsets=1  =>   for ioff in range(4),    offsets=2 => for ioff in range(16)
        
        
        if (ONESHOT): inner_loop = 1
        
        for ioff in range(inner_loop):

            if (0):
                theta0, phi0  =   30.0, 10.0    # +Z
                theta0, phi0  =   170.0, 10.0   # -Z
                theta0, phi0  =   60.0,  60.0   # +Y
                theta0, phi0  =   60.0, -50.0   # -Y
                theta0, phi0  =   55.0,  25.0   # +X
                theta0, phi0  =   55.0,  155.0  # -X  ???
                theta0, phi0  =   60.0,  170.0  # -X  ???
                theta0, phi0  =   60.0,  160.0  # -X  ???
                theta0, phi0  =   theta0*pi/180.0, phi0*pi/180.0
                
            # if (idir!=13): continue
            
            queue.finish()
            POS, DIR, LEADING = GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) ## , theta0, phi0)

            if (0):
                print("IDIR %2d / %2d     %8.5f %8.5f %8.5f  %7.4f %7.4f %7.4f   %d" % 
                (idir, NDIR, POS['x'], POS['y'], POS['z'], DIR['x'], DIR['y'], DIR['z'], LEADING))
            
            
            # Calculate DIRWEI part
            #    DIRWEI is calculated taking into account all rays that enter a given side
            #    However, kernel will use DIRWEI[i] only for rays with leading edge i.
            #    Does that mean that this direction weighting is less useful?
            #    or DIRWEI should be calculated based on leading edge cos(theta) only???
            #    .... because rays entering the sides are further apart in surface elements.
            if    (LEADING==0):  
                DIRWEI[0] += DIR['x']        # summing cos(theta) of the rays hitting given side
                RCOUNT[0] += 1
            elif  (LEADING==1):                
                DIRWEI[1] -= DIR['x'] 
                RCOUNT[1] += 1
            elif  (LEADING==2):
                DIRWEI[2] += DIR['y'] 
                RCOUNT[2] += 1                
            elif  (LEADING==3):
                DIRWEI[3] -= DIR['y'] 
                RCOUNT[3] += 1
            elif  (LEADING==4):
                DIRWEI[4] += DIR['z'] 
                RCOUNT[4] += 1
            elif  (LEADING==5):
                DIRWEI[5] -= DIR['z'] 
                RCOUNT[5] += 1
            ####
            if (LEADING   in [0,1]):
                EWEI +=  1.0/abs(DIR['x'])
            elif (LEADING in [2,3]):
                EWEI +=  1.0/abs(DIR['y'])
            else:
                EWEI +=  1.0/abs(DIR['z'])

            # we have to process NRAY with NWG workgroups, possibly NWG<NRAY
            if (OCTREE==0):
                if (NRAY%GLOBAL==0): niter = NRAY//GLOBAL
                else:                niter = NRAY//GLOBAL+1
                for ibatch in range(niter):
                    if (PLWEIGHT>0):
                        kernel_paths(queue, [GLOBAL,], [LOCAL], 
                        ibatch*GLOBAL, PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR)
                    else:
                        kernel_paths(queue, [GLOBAL,], [LOCAL], 
                        ibatch*GLOBAL,         TPL_buf, COUNT_buf, LEADING, POS, DIR)                
            elif (OCTREE==1):
                kernel_paths(queue, [GLOBAL,], [LOCAL], PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR, 
                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)
            elif ((OCTREE==2)|(OCTREE==3)):
                kernel_paths(queue, [GLOBAL,], [LOCAL], PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR, 
                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf)            
            elif (OCTREE in [4,5]): # one work group per ray
                if (NRAY%NWG==0): niter =  NRAY//NWG     # ONESHOT => NRAY ~ NX*NY, not ~(NX/2)*(NY/2)
                else:             niter = (NRAY//NWG)+1
                if (PLWEIGHT<1):  niter = 0   # just skip the path calculation
                
                for ibatch in range(niter):
                    # print("paths, idir %3d, ioff %3d, batch %d/%d, dir %7.3f %7.3f %7.3f -- PLWEIGHT=%d, NWG=%d" % (idir, ioff, ibatch, niter, DIR['x'], DIR['y'], DIR['z'], PLWEIGHT, NWG))                        
                    kernel_paths(queue, [GLOBAL,], [LOCAL],
                    #gid0       [CELLS]  [NRAY]    [NRAY]      np.int32
                    ibatch*NWG, PL_buf,   TPL_buf, COUNT_buf,  LEADING,
                    # REAL3  float3  [LEVELS]    [LEVELS]   [CELLS-NX*NY*NZ]  CELLS     [4|8]*NWG*(26+CHANNELS)*MAX_NBUF
                    POS,     DIR,    LCELLS_buf, OFF_buf,   PAR_buf,          RHO_buf,  BUFFER_buf)
                    queue.finish()
                    
            elif (OCTREE in [40,]):  # one work item per ray
                # with option ONESHOT, kernel will loop over all ray offsets, not only every other
                if (NRAY%GLOBAL==0): niter = NRAY//GLOBAL
                else:                niter = NRAY//GLOBAL+1
                if (PLWEIGHT<1): niter = 0   # just skip the path calculation
                for ibatch in range(niter):
                    kernel_paths(queue, [GLOBAL,], [LOCAL], ibatch*GLOBAL, PL_buf,  TPL_buf,  COUNT_buf,  LEADING,
                    POS,     DIR,    LCELLS_buf, OFF_buf,   PAR_buf,       RHO_buf,  BUFFER_buf)            
                
            queue.finish()
            cl.enqueue_copy(queue, COUNT, COUNT_buf)
            queue.finish()
            # PACKETS = total number of packets from the background, including rays continued to model sides
            for i in range(NRAY):  PACKETS  += COUNT[i]      

            if (LEADING in [0,1]):
                TRUE_APL          +=  1.0/fabs(DIR['x'])     # expected path length per root grid cell
            elif (LEADING in [2,3]):
                TRUE_APL          +=  1.0/fabs(DIR['y'])
            else:
                TRUE_APL          +=  1.0/fabs(DIR['z'])            
    

            if (PLWEIGHT):
                cl.enqueue_copy(queue, PL, PL_buf)
                queue.finish()
                m = nonzero(RHO[0:NX*NY*NZ]>0.0)
                if (0):
                    if (len(m[0])>0):
                        print("  <PL> root grid: %.3e +- %.3e" % (mean(PL[0:NX*NY*NZ][m[0]]), std(PL[0:NX*NY*NZ][m[0]])))
                        
    ## end of for idir
    
                    
        ### break
    ### end for idir
    
    if (PLWEIGHT):
        cl.enqueue_copy(queue, TPL,   TPL_buf)
        # print("PL[20444] = %8.4f" % PL[20444])
        if (np.min(TPL)<0.0):
            print("PATH KERNEL RAN OUT OF BUFFER SPACE !!!")
            sys.exit()
        
    #   BG is calculated as the total number of photons entering the clouds divided by the number of
    #      rays == AVERAGE NUMBER OF PHOTONS PER PACKAGE
    #   Individual rays are weighted ~   cos(theta) / <cos(theta)>, where the denominator should be 0.5
    #   We provide kernel the weight factors  =  cos(theta) / <cos(theta)>
    if (1):
        # 2020-01-13 -- revised weighting, BGPAC * cos(theta) / DIRWEI, where DIRWEI
        #               is the sum of cos(theta) for each side separately
        #               *and* the weighting of photon packages does not depend on PACKETS variable
        print("NEW DIRWEI ", DIRWEI)    # nside=0  => DIRWEI[:] ~ 1.0
        # sys.exit()
    else:
        DIRWEI  /=  RCOUNT         #  now DIRWEI is <cos(theta)> for the rays entering each of the six sides
    EWEI    /=  inner_loop ;       #  <1/cosT>
    EWEI     =  1.0/(EWEI*NDIR)    #  emission from a cell ~  (1/cosT) * EWEI, larger fraction when LOS longer 
    if (ONESHOT<1):
        TRUE_APL =  TRUE_APL/4.0   # no division by offs*offs !!   -- fixed 2020-07-20
    APL      =  TRUE_APL

    # average path length APL/(inner_loop*NDIR) through a cell
    # random rays + cubic shape => average should be 1.222
    # possible additional weighting   1.222*inner_loop*NDIR/APL ?
    # APL_WEIGHT  =  1.222*inner_loop*NDIR/APL
    # APL_WEIGHT *=  0.8
    # print("APL_WEIGHT = %8.5f" % APL_WEIGHT)
        
    if (PLWEIGHT):
        cl.enqueue_copy(queue, PL, PL_buf)
        m = nonzero(RHO>0.0)
        print('SUM OF PL = %.3e, APL %.3e,  <PL> %.3e, TRUE_APL %.3e, PACKETS=%d' %
        (sum(PL), APL, mean(PL[m[0]]), TRUE_APL, PACKETS))
        m = nonzero(RHO[0:NX*NY*NZ]>0.0)
        if (len(m[0])>0):
            print("<PL> for root grid cells: %.3e +- %.3e" % (mean(PL[0:NX*NY*NZ][m[0]]), std(PL[0:NX*NY*NZ][m[0]])))
        # print("PL[20444] = %8.4f" % PL[20444])
    print("Paths kernel: %.3f seconds" % (time.time()-t00))
    #    WITH_HALF=1   68646124 7840992   65.5/7.7 GB
    # print('DIRWEI     ', DIRWEI)
    
    if (0):
        print("SAVING PL%d.dat" % OCTREE)
        PL.tofile('PL%d.dat' % OCTREE)
        sys.exit()
        
    if (OCTREE>999):
        tmp  = PL[0:NX*NY*NZ].copy()
        tmp2 = tmp[nonzero(RHO[0:NX*NY*NZ]>0.0)].copy()
        print("FIRST OCTREE LEVEL PL", percentile(tmp2, (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
        tmp.shape = (NX,NY,NZ)
        clf()
        for ii in range(4):
            subplot(2,2,1+ii)
            title("i=%d" % (NX//2-2+ii))
            imshow(tmp[NX//2-2+ii,:,:])
            colorbar()
        show()
        sys.exit()
    ## sys.exit()    
    if (0): # cartesian
        PL.shape = (NX, NY, NZ)
        subplot(221)
        imshow(PL[NX//2-2,:,:])
        colorbar()
        subplot(222)
        imshow(PL[NX//2-1,:,:])
        colorbar()
        subplot(223)
        imshow(PL[NX//2-0,:,:])
        colorbar()
        subplot(224)
        imshow(sum(PL, axis=0))
        colorbar()
        show()
        sys.exit()
    
    if (PLWEIGHT):
        m = nonzero(RHO>0.0)
        print('PL  ', percentile(PL[m], (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
    # sys.exit()
    if (0):
        print("APL = %.3e    ---    <PL> = %.3e" % (APL, mean(PL))) # yes, they are ~ the same
        clf()
        for i in range(OTL):
            m = nonzero(RHO[OFF[i]:(OFF[i]+LCELLS[i])]>0.0)
            plot(PL[OFF[i]:(OFF[i]+LCELLS[i])][m], '.', label='LEVEL %d' % i)
        legend()
        show()
        sys.exit()
        
    if (0):
        x =   PL[OFF[1]:(OFF[1]+LCELLS[1])]
        print("PL, LEVEL1", percentile(x, (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
        # PL[OFF[1]:(OFF[1]+LCELLS[1])]  = APL/2.0
        # sys.exit()
    
    
# Read or generate NI_ARRAY
if (LOWMEM>1): # not only is kernel using more global arrays, also NI_ARRAY, SIJ_ARRAY, ESC_ARRAY are memmap
    # do we keep this separate from load/save... in case one does not want to load/save the populations....
    fp = open('LOC_NI.mmap', 'wb')
    asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
    fp.close()
    NI_ARRAY = np.memmap('LOC_NI.mmap', dtype='float32', mode='r+', offset=16, shape=(CELLS, LEVELS))
    NI_ARRAY[:,:] = 1.0
else:
    NI_ARRAY = ones((CELLS, LEVELS), float32)
ok = False
if (len(INI['load'])>0):  # load saved level populations
    try:
        fp = open(INI['load'], 'rb')
        nx, ny, nz, lev = fromfile(fp, int32, 4)
        #print(nx, ny, nz, lev)
        #print(NX, NY, NZ, LEVELS)
        if ((nx!=NX)|(ny!=NY)|(nz!=NZ)|(lev!=LEVELS)):
            print("Reading %s => %d x %d x %d cells, %d levels" % (nx, ny, nz, lev))
            print("but we have now %d x %d x %d cells, %d levels ?? "  % (NX, NY, NZ, LEVELS))
            sys.exit()
        NI_ARRAY[:,:] = fromfile(fp, float32).reshape(CELLS, LEVELS)
        fp.close()
        ok = True
        print("Level populations read from: %s" % INI['load'])
    except:
        print("Failed to load level populations from: %s" % INI['load'])
        pass
if (not(ok)): # reset LTE populations
    print("***** Resetting level populations to LTE values !!! *****")
    J   =  asarray(arange(LEVELS), int32)
    m   =  nonzero(RHO>0.0)
    t0  =  time.time()
    if (0):  # this one took  9    seconds
        for icell in m[0]:
            NI_ARRAY[icell,:] = RHO[icell] * ABU[icell] * MOL.Partition(J, TKIN[icell])
    else:    # this one took  0.01 seconds
        LTE = program.LTE
        #                          BATCH     E      G     TKIN   RHO    NI   
        LTE.set_scalar_arg_dtypes([np.int32, None,  None, None,  None,  None])
        for i in range(CELLS//BATCH+1):
            a     = i*BATCH                    # first index included
            b     = min([CELLS, a+BATCH])      # last index included + 1
            cells = b-a
            if (cells<1): break
            cl.enqueue_copy(queue, SOL_TKIN_buf,  TKIN[a:b].copy())
            cl.enqueue_copy(queue, SOL_RHO_buf,   ABU[a:b]*RHO[a:b])                        
            LTE(queue, [GLOBAL,], [LOCAL,], BATCH, MOL_E_buf, MOL_G_buf, SOL_TKIN_buf, SOL_RHO_buf, SOL_NI_buf)
            cl.enqueue_copy(queue, NI_ARRAY[a:b,:], SOL_NI_buf)
            # print("%7d - %7d    %10.3e %10.3e" % (a, b, NI_ARRAY[a,0], NI_ARRAY[b-1,0])) ;
    print("LTE populations set in %.3f seconds" % (time.time()-t0))
    # write also directly to disk
    if (len(INI['save'])>0):
        fp = open(INI['save'], 'wb')
        asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
        print("Level populations (LTE) saved to: %s" % INI['save'])
        if (0):
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  <ni[%02d]*gg-ni[%02d]> = %11.3e" % (l, u, mean(NI_ARRAY[:,l]*MOL.GG[tr]-NI_ARRAY[:,u])))
            Tkin = 10.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            Tkin = 20.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            sys.exit()
    if (0):
        print(NI_ARRAY[0,:])
        print(NI_ARRAY[1000,:])
        print(NI_ARRAY[10000,:])
        sys.exit()

            
LTE_10_pop = zeros(LEVELS, float32)
for i in range(LEVELS):
    # print(MOL.G[i], MOL.E[i])
    LTE_10_pop[i] = MOL.G[i] * exp(-MOL.E[i]*PLANCK/(BOLTZMANN*10.0))
LTE_10_pop /= sum(LTE_10_pop)


if (1):
    m = nonzero(~isfinite(sum(NI_ARRAY, axis=1)))
    if (len(m[0])>0):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("NI_ARRAY READ ---- %d WITH DATA NOT FINITE !" % (len(m[0])))
        for i in m[0]:
            NI_ARRAY[i,:] = LTE_10_pop * RHO[i]*ABU[i]
        print("???? FIXED ????")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        

        
#================================================================================
#================================================================================
#================================================================================
#================================================================================


#  65.5/7.7 GB   68646124 7840992
# [  +0.000000] amdgpu 0000:3e:00.0: VM_L2_PROTECTION_FAULT_STATUS:0x00301031



def Simulate():
    global INI, MOL, queue, LOCAL, GLOBAL, WIDTH, VOLUME, GL, COOLING, NSIDE, HFS
    global RES_buf, GAU_buf, CLOUD_buf, NI_buf, LIM_buf, PL, EWEI, PL_buf
    global ESC_ARRAY, SIJ_ARRAY, PACKETS, OLBAND

    niter   =  0
    ncmp    =  1
    tmp_1   =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg     =  INI['Tbg']
    SUM_COOL, LEV_COOL, hf = [], [], []
    if (COOLING==2):
        SUM_COOL = zeros(CELLS, float32)
        LEV_COOL = zeros(CELLS, float32)
        hf       = MOL.F*PLANCK/VOLUME
    if (INI['verbose']<2):  sys.stdout.write('      ')
    
    for tran in range(MOL.TRANSITIONS): # ------>
        t_tran        =  time.time()
        upper, lower  =  MOL.T2L(tran)
        Ab            =  MOL.BB[tran]
        Aul           =  MOL.A[tran]
        freq          =  MOL.F[tran]
        gg            =  MOL.GG[tran]
        BG            =  1.0
        
        if (WITH_OVERLAP): # in Simulate() skip transitions that are part of OVERLAP lines
            skip = False
            for icmp in range(OLBAND.BANDS):
                if (tran in OLBAND.TRAN[icmp]): skip = True
            if (skip): continue

        # print("\n**** Simulate()  ==>  %2d -> %2d" % (upper, lower))
            
        # 2021-01-13 -- weighting directly by cos(theta)/DIRWEI where DIRWEI = sum(cos(theta)) for each side
        #  BGPHOT = number of photons per a single surface element, not the whole cloud
        #  ***AND*** only photons into the solid angle where the largest vector component is 
        #            perpendicular to the surface element !
        #            instead of pi, integral is 1.74080 +-   0.00016  => total number of photons per LEADING
        #  BG =  BGPHOT * cos(theta) / sum(cos(theta))
        BGPHOT   =  Planck(freq, Tbg)* 1.74080     / (PLANCK*C_LIGHT) * (1.0e5*WIDTH)*VOLUME/GL

        if (ONESHOT==0):
            # 2025-07-08  => now ONESHOT=0 gives same results as ONESHOT=1,
            # but probably only if offs==1
            assert(INI['offsets']==1)
            BGPHOT *= 4.0

        if (HFS):
            nchn = BAND[tran].Channels()
            ncmp = BAND[tran].N
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
        if (WITH_CRT):
            cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))
            cl.enqueue_copy(queue, CRT_EMI_buf, asarray(CRT_EMI[:,tran].copy(), float32))
        
            
        
        GNORM         = (C_LIGHT/(1.0e5*WIDTH*freq)) * GL  # GRID_LENGTH multiplied to gauss norm
        if (INI['verbose']<2):
             sys.stdout.write(' %2d' % tran)
             sys.stdout.flush()
        kernel_clear(queue, [GLOBAL,], [LOCAL,], RES_buf)  # RES[CELLS,2] for ALI

        # Upload NI[upper] and NB_NB[tran] values
        tmp  =  tmp_1 * Aul * (NI_ARRAY[:,lower]*gg-NI_ARRAY[:,upper]) / (freq*freq)  # [CELLS]

        if (0):
            tmp  = np.clip(tmp, -1.0e-12, 1.0e32)      # kernel still may have clamp on tau
            tmp[nonzero(abs(tmp)<1.0e-32)] = 1.0e-32   # nb_nb -> one must not divide by zero  $$$
        else:
            tmp = np.clip(tmp, 1.0e-25, 1.0e32)        # KILL ALL MASERS  $$$
            
        WRK[:,0]  = NI_ARRAY[:, upper]   # ni          WRK[CELLS,2]
        WRK[:,1]  = tmp                  # nb_nb
        cl.enqueue_copy(queue, NI_buf, WRK)
        # the next loop is 99% of the Simulate() routine run time
        offs  =  INI['offsets']  # default was 1, one ray per cell
        t000  =  time.time()

        if (0):
            print("  A_b     %12.4e" % Ab)
            print("  GL      %12.4e" % GL)
            print("  GN      %12.4e" % GNORM)
            print("  Aul     %12.4e" % Aul)
            print("  freq    %12.4e" % freq)
            print("  gg      %12.4e" % gg)
            print("  BGPHOT  %12.4e" % BGPHOT)
            print("  PACKETS %d"     % PACKETS)
            print("  BG      %12.4e" % Planck(freq, Tbg))
            print("  NI      %12.4e" % (np.mean(WRK[:,0])))
            print("  NBNB    %12.4e" % (np.mean(WRK[:,1])))


        SUM_DIRWEI = 0.0   # weight ~ cos(theta)/sum(cos(theta)) ... should sum to 1.0 for each side, 6.0 total
        
        for idir in range(NDIR):            
            inner_loop = 4*offs*offs
            if (ONESHOT): inner_loop = 1    # for  OCTREE=4, OCTREE=40
            for ioff in range(inner_loop):  # 4 staggered initial positions over 2x2 cells -- if ONESHOT==0
                
                ## if ((idir!=0)|(ioff!=0)): continue 
                
                
                POS, DIR, LEADING  =  GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) # < 0.001 seconds !
                dirwei, ewei = 1.0, 1.0                
                if (1):
                    # 2021-01-13 --- BGPHOT * cos(theta)/DIRWEI, DIRWEI = sum(cos(theta)) for each side separately
                    # there is no change at this point, except that DIRWEI is sum, not the average cos(theta)
                    # *AND* BG is computed without dependence on the PACKETS variable
                    if (LEADING in   [0, 1]):    
                        dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                        ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                    elif (LEADING in [2, 3]):  
                        dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['y']))
                    else:                     
                        dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['z']))                    
                    BG     = BGPHOT * dirwei  # this is per surface element => ok for ONESHOT==1
                    
                    # print("idir = %3d   ioff = %3d  =>   BG = %.3e" % (idir, ioff, BG))
                    SUM_DIRWEI += dirwei
                    dirwei = 1.0        # this is the weight factor that goes to kernel... now not used
                else:
                    # kernel gets  WEI = dirwei/DIRWEI ==  cos(theta)/<cos(theta>)>, to weight BG
                    #   One needs a weighting according to the angle between the ray and the surface.
                    #   That could be taken care by the density of rays hitting a surface, if the rays 
                    #   were equidistant in 3d and therefore density of hits on the surface would get 
                    #   lower by ~1/cos(theta) for more obliques angles.
                    # *However*, rays are created equidistant on the leading edge, irrespective of cos(theta).
                    #    Therefore, we do need to include the weighting 1/cos(theta) because we have rays too densely for
                    #    oblique angles. On the other hand, simulated rays correspond exactly to the true number
                    #    of photons that should enter the model volume  => it is only a relative weighting
                    #    and <dirwei/DIRWEI> == 1.0   =>  PHOTONS ~ sum{   (BGPHOT/PACKETS) * (dirwei/DIRWEI)  }
                    # DIRWEI was computed for each of the six sides separately
                    if (LEADING in [0, 1]):    
                        dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                        ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                    elif (LEADING in [2, 3]):  
                        dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['y']))
                    else:                     
                        dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['z']))
                        
                        
                # print("         idir %3d  =>  dirwei %8.5f" % (idir, dirwei))
                               
                if (ncmp==1):
                    if (WITH_CRT):
                        niter = NRAY//GLOBAL
                        if (NRAY%GLOBAL!=0): niter += 1
                        for ibatch in range(niter):                                
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  
                            # 0             1          2        3        4    5   6      7     8  
                            ibatch*GLOBAL,  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, None, APL,
                            # 9   10      11    12       13   14   15      16       17        
                            BG,   dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                            # 18         19         
                            CRT_TAU_buf, CRT_EMI_buf)
                    else:
                        if (OCTREE<1):
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            # print("kernel_sim, niter=%d, NRAY=%d, GLOBAL=%d" % (niter, NRAY, GLOBAL))
                            for ibatch in range(niter):
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],
                                # 0            1          2        3        4    5 
                                ibatch*GLOBAL, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 6    7       8    9   10      11    12       13   14   15      16       17      
                                GNORM, PL_buf, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf)
                        elif (OCTREE==1):
                            # OCTREE==1 does always calculate paths but we choose not to use those??
                            if (PLWEIGHT<1):
                                #                                       0          1        2        3    4  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19      
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)
                            else:
                                #                                      0       1          2        3        4    5  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,], PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19    
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)                                
                        elif (OCTREE in [2,3]): # OCTREE=2, OCTREE=3
                            #                                       0       1          2        3        4    5   
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                            # 6    7    8   9       10    11       12   13   14      15       16        
                            GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                            #  17       18       19       20        21        
                            LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                        elif (OCTREE in [4,5]):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):                                
                                # print("idir=%2d, kernel_sim %d/%d" %(idir, 1+ibatch, niter))
                                if (PLWEIGHT>0):
                                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, PL_buf, CLOUD_buf, 
                                               GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                               LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                               LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                                else:
                                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, CLOUD_buf,
                                               GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                               LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                               LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                                
                            # sys.exit()
                                
                        elif (OCTREE in [40,]):
                            # with option ONESHOT, kernel should loop over all ray position, not only every other
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            for ibatch in range(niter):
                                # print("kernel_sim batch %d/%d" %(1+ibatch, niter))
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*GLOBAL, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                            
                else:
                    if (OCTREE<1): #  GLOBAL >= NRAY
                        #                                       0          1        2        3    4   5      6   
                        kernel_hf(queue, [GLOBAL,], [LOCAL,],   CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, 
                        # 7  8       9     10       11   12   13      14       15    16    17      18      
                        BG,  dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, nchn, ncmp, HF_buf, NTRUE_buf,
                        PROFILE_buf)
                    else:
                        if (OCTREE==4):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):
                                kernel_hf(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf,  nchn, ncmp, HF_buf, 
                                RES_buf, NTRUE_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf)
                        else:
                            print("HFS + OCTREE%d not yet implemented" % OCTREE), sys.exit()
                        
                queue.finish()

                
                if (0):
                    print("NOL HOST: Aul %.3e Ab %.3e GN %.3e BG %.3e DIRWEI %.3e EWEI %.3e\n" %
                    (Aul, Ab, GNORM, BG, dirwei, ewei))
                    sys.exit()

                
                
        # end of loop over NDIR
        # print("--- tran=%2d   ncmp=%2d  %.3f seconds" % (tran, ncmp, time.time()-t000))
        # print("--- SUM_DIRWEI %8.5f" % SUM_DIRWEI)
        
        # print("**** DIRWEI AVERAGE VALUE %8.4f ***" % (dwsum/dwcount))
        # average weight was == 1.000 but this reduced Tex ... the weighting is not ok?
        
        # for plot_pl.py
        if (0):  # debugging ... see that Update has the same PL as Path
            print("SAVING PL.dat")
            PL.tofile('PL.dat')
            cl.enqueue_copy(queue, PL, PL_buf)
            PL.tofile('PL_zero.dat')
            sys.exit()

            
        if (0):
            cl.enqueue_copy(queue, WRK, RES_buf)   #  WRK[CELLS, 2],    RES[CELLS, 2] for ALI
            print()
            asarray(WRK[:,0],float32).tofile('nol.sij')
            print("RAW nol.sij WRITTED !!")
            ####
            print('NRAY %d, GLOBAL %d, niter %d' % (NRAY, GLOBAL, niter))
            print("PLWEIGHT %d, APL %.3e PL %.3e %.3e" % (PLWEIGHT, APL, np.mean(PL), np.std(PL)))
            print("# NOL SIJ   =====>  ", np.mean(WRK[:,0]))
            print("# NOL ESC   =====>  ", np.mean(WRK[:,1]))
            clf()
            plot(WRK[:,0], 'k.')
            plot(WRK[:,1], 'c.')
            # show(block=True)
            sys.exit()
            

        # PLWEIGHT
        # - OCTREE=0 -- PLWEIGHT can be 1 (for testing). Update kernel can take PL_buf but only for testing
        #   and we do not use PL here.
        # - OCTREE>0, we assume PL is calculated and it is not used only for OCTREE=4, if INI['plweight']==0
        # post weighting
        if (WITH_ALI>0):
            cl.enqueue_copy(queue, WRK, RES_buf)       # pull both SIJ and ESC, RES[CELLS,2] => WRK[CELLS,2]
            if (PLWEIGHT==0):    # assume PLWEIGHT implies OCTREE==0
                #  Cartesian grid -- no PL weighting, no cell-volume weighting
                SIJ_ARRAY[:, tran]                    =  WRK[:,0]  
                ESC_ARRAY[:, tran]                    =  WRK[:,1]
            else:
                if (OCTREE==0):   # regular Cartesian but PLWEIGHT=1 => use PL weighting
                    SIJ_ARRAY[:, tran]                =  WRK[:,0] #* APL/PL[:]
                    ESC_ARRAY[:, tran]                =  WRK[:,1] #* APL/PL[:]
                elif (OCTREE in [1,2]):   #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    a, b  =  0, LCELLS[0]
                    SIJ_ARRAY[a:b, tran]              =  WRK[a:b, 0] * (APL/PL[a:b])
                    ESC_ARRAY[a:b, tran]              =  WRK[a:b, 1] * (APL/PL[a:b])
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):             k =  8.0**l  # OCTREE1   PL should be  APL/8^l
                        else:                       k =  2.0**l  # OCTREE2   PL should be  APL/2^l
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0] * (1.0/k)  * (APL/PL[a:b])
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1] * (1.0/k)  * (APL/PL[a:b])
                else:  # OCTREE 3,4,5,40   --- assuming  it should be  PL[] == APL/2^l ;
                    if (PLWEIGHT):
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0] * APL/PL[a:b] 
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1] * APL/PL[a:b]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, 0] * (APL/2.0**l) /  PL[a:b]
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, 1] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct ~ 0.5^level
                        # above scaling  APL/2**l/PL translates to 1.0 
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0]
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, 0] #### / (2.0**l)
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, 1] #### / (2.0**l)
        else:  # no ALI, SIJ only
            WRK.shape = (2, CELLS)         # trickery... we use only first CELLS elements of WRK
            cl.enqueue_copy(queue, WRK[0,:], RES_buf)    # SIJ only .... RES is only RES[CELLS]
            if (PLWEIGHT==0):              # PLWEIGHT==0 implies OCTREE=0, Cartesian grid without PL weighting
                SIJ_ARRAY[:, tran]                    =  WRK[0,:]
            else:
                if (OCTREE==0):            # Cartesian grid with PL weighting
                    SIJ_ARRAY[:, tran]                =  WRK[0,:]  * APL/PL[:]
                elif (OCTREE in [1,2]):    #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    a, b                              =  0, LCELLS[0]
                    SIJ_ARRAY[a:b, tran]              =  WRK[0, a:b] * APL/PL[a:b] 
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):            k  =  8.0**l    # OCTREE1   PL should be  APL/8^l
                        else:                      k  =  2.0**l    # OCTREE2   PL should be  APL/2^l
                        SIJ_ARRAY[a:b, tran]          =  WRK[0,a:b] * (APL/k) / PL[a:b]
                else:  # OCTREE 3,4,5,40   weight 2**-l  --- except OCTREE=3 is not exact
                    if (PLWEIGHT):  # include APL/PL weighting
                        # print("*** PLWEIGHT***")
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[0, a:b] * APL/PL[a:b] 
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[0, a:b] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct  - only volume weighting
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[0, a:b]
                        m = nonzero(RHO[a:b]>0.0)
                        yyy  =  APL/PL[a:b]
                        print("*** L=0, OMIT  APL/PL = %.4f +- %.4f" % (mean(yyy[m]), std(yyy[m])))
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]      
                            m  =  nonzero(RHO[a:b]>0.0)
                            yyy  =  APL / (2.0**l) / PL[a:b]
                            print("*** L=%d, OMIT  APL/PL = %.4f +- %.4f" % (l, mean(yyy[m]), std(yyy[m])))
                            SIJ_ARRAY[a:b, tran]      =  WRK[0, a:b] ### / (2.0**l)
            WRK.shape = (CELLS, 2)
                        
        m = nonzero(RHO>0.0)
        if (WITH_ALI>0):
            if (INI['verbose']>1):
                print("      TRANSITION %2d  <SIJ> %12.5e   <ESC> %12.5e  T/TRAN  %6.1f" %
                (tran, mean(SIJ_ARRAY[m[0],tran]), mean(ESC_ARRAY[m[0],tran]), time.time()-t_tran))

            if (0): sys.exit()
            
            
        if (0):
            m1  =  nonzero(~isfinite(SIJ_ARRAY[m[0],0]))
            # m1  =  ([6083,],)
            # m1  =  ([5083,],)
            if (len(m1[0])>0):
                print("\nSIJ NOT FINITE:", len(m1[0]))
                print('SIJ  ', SIJ_ARRAY[m[0],0][m1[0]])
                print('RHO  ',       RHO[m[0]][m1[0]])
                print('TKIN ',      TKIN[m[0]][m1[0]])
                print('ABU  ',       ABU[m[0]][m1[0]])
                if (len(m1[0])>0): sys.exit()
            
        if (0):
            clf()
            ax = axes()
            m = nonzero(RHO>0.0)
            print("============ tran=0 ==========")
            print("<APL/PL> = %.4f" % (APL/np.mean(PL)))
            print("CELLS %d" % len(m[0]))
            print("SIJ_ARRAY[%d] " % tran,  percentile(SIJ_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
            plot(SIJ_ARRAY[m[0],tran], 'r.')
            text(0.2, 0.5, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(SIJ_ARRAY[m[0],tran]), std(SIJ_ARRAY[m[0],tran])/mean(SIJ_ARRAY[m[0],tran])), transform=ax.transAxes, color='m', size=15)
            if (WITH_ALI>0):
                print("ESC_ARRAY[%d] " % tran,  percentile(ESC_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
                plot(ESC_ARRAY[m[0],tran], 'b.')
                text(0.2, 0.4, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(ESC_ARRAY[m[0],tran]), std(ESC_ARRAY[m[0],tran])/mean(ESC_ARRAY[m[0],tran])), transform=ax.transAxes, color='c', size=15)
            title("SIJ and ESC directly from kernel")
            # show(block=True)
            sys.exit()

        if (0):
            subplot(221)
            x = PL[0:NX*NY*NZ].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99.9))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("PL, root level")
            colorbar()
            subplot(222)
            x = SIJ_ARRAY[0:NX*NY*NZ,tran].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99.9))
            imshow(x[NX//2,:,:], vmin=2.0e-12, vmax=2.4e-12)
            title("SIJ, root level")
            colorbar()
            subplot(223)
            x = ESC_ARRAY[0:NX*NY*NZ,tran].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("ESC, root level")
            colorbar()
            subplot(224)
            x = RHO[0:NX*NY*NZ].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("RHO, root level")
            colorbar()
            show()
            sys.exit()
            
        if (0):
            print("")
            print("SIJ_ARRAY[%d] " % tran,  percentile(SIJ_ARRAY[:,tran], (1.0, 50.0, 99.0)))
            print("ESC_ARRAY[%d] " % tran,  percentile(ESC_ARRAY[:,tran], (1.0, 50.0, 99.0)))
        if (COOLING==2):
            cl.enqueue_copy(queue, LEV_COOL, COOL_buf)
            SUM_COOL[:]   +=   LEV_COOL[:] * hf[tran]    # per cm3
        if (0):
            print("       tran = %3d  = %2d - %2d  => <SIJ> = %.3e   <ESC> = %.3e" % 
            (tran, upper, lower, mean(WRK[:,0]), mean(WRK[:,1])))


    # --- end of loop over transitions ---

            
    if (INI['verbose']<2):
        sys.stdout.write('\n')



    if (0): # check SIJ vs cell
        close(1)
        figure("NoFocus")
        a  =      SIJ_ARRAY[0:LCELLS[0],               0]
        b  =  8.0*SIJ_ARRAY[OFF[1]:(OFF[1]+LCELLS[1]), 0]
        plot(a, 'b+')
        plot(b, 'r+')
        title("2/1 = %.3f" % (mean(b[nonzero(b>0.0)])/mean(a)))
        plot(    SIJ_ARRAY[0:LCELLS[0]              , 1], 'bx')
        plot(8.0*SIJ_ARRAY[OFF[1]:(OFF[1]+LCELLS[1]), 1], 'rx')
        show()
        sys.exit()
                    
    
    if (0):
        print("------------------------------------------------------------------------------------------")
        for i in range(6):
            for j in range(7):
                sys.stdout.write(" %12.5e" % SIJ_ARRAY[i,j])
            sys.stdout.write('\n')
        print("------------------------------------------------------------------------------------------")
        for i in range(6):
            for j in range(7):
                sys.stdout.write(" %12.5e" % ESC_ARRAY[i,j])
            sys.stdout.write('\n')
        print("------------------------------------------------------------------------------------------")
        sys.exit()
        

                        
    # <--- for tran ---
    
    
    if (COOLING==2):
        print("BRUTE COOLING: %10.3e" % (sum(SUM_COOL)/CELLS))
        fpb = open('brute.cooling', 'wb')
        asarray(SUM_COOL, float32).tofile(fpb)
        fpb.close()
        SUM_COOL = []
        LEV_COOL = []
        




def SimulateMultitran(ntran=2):
    """
    A version of the simulation routine where the kernel calculates several transitions at a time.
    ntran = number of transitions processed with a single kernel call
    """
    global INI, MOL, queue, LOCAL, GLOBAL, WIDTH, VOLUME, GL, COOLING, NSIDE, HFS
    global GAU_buf, CLOUD_buf, LIM_buf, PL, EWEI, PL_buf
    global ESC_ARRAY, SIJ_ARRAY    #  [CELLS, TRANSITIONS]
    global PACKETS, OLBAND
    global RES_buf, NI_buf         #  RES_buf[CELLS, ntran],  NI_buf[CELLS, ntran]
    global NTRUE_buf, BUFFER_buf
    print("*** SimulateMultitran")
    
    del    RES_buf, NI_buf, NTRUE_buf, BUFFER_buf

    btran      =  ntran     # can change for the last batch, if TRANSITIONS % ntran != 0
    Aul_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*ntran)
    Ab_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*ntran)
    GN_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*ntran)
    BG_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*ntran)
    # NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
    NTRUE_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*max([INI['points'][0], NRAY])*MAXCHN*ntran)
    NI_buf     =  cl.Buffer(context, mf.READ_ONLY,   4*CELLS*ntran)      # NI_buf[CELLS, TRANSITIONS]
    NBNB_buf   =  cl.Buffer(context, mf.READ_ONLY,   4*CELLS*ntran)      # NBNB_buf[CELLS, TRANSITIONS]
    RES_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS*ntran)      # SIJ   RES[CELLS, btran]    
    if (WITH_ALI>0):
        ESC_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS*ntran)    # ESC   ESC[CELLS, btran]
    else:
        ESC_buf  =  cl.Buffer(context, mf.READ_WRITE,  4)
    ###
    BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(26+ntran*CHANNELS)*MAX_NBUF)
    
    niter   =  0
    ncmp    =  1
    tmp_1   =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg     =  INI['Tbg']
    SUM_COOL, LEV_COOL, hf = [], [], []
    if (COOLING==2):
        assert(1==0)
        SUM_COOL = zeros(CELLS, float32)
        LEV_COOL = zeros(CELLS, float32)
        hf       = MOL.F*PLANCK/VOLUME
    if (INI['verbose']<2):  sys.stdout.write('      ')


    Aul    = zeros(ntran, float32)
    Ab     = zeros(ntran, float32)
    GNORM  = zeros(ntran, float32)
    BG     = zeros(ntran, float32)
    gg     = zeros(ntran, float32)
    freq   = zeros(ntran, float32)
    BGPHOT = zeros(ntran, float32)
    BG     = zeros(ntran, float32)
    WRK    = zeros((CELLS, ntran), float32)
    
    tranO = 0
    while(tranO<MOL.TRANSITIONS):
        tranA = tranO
        tranO = min([tranA+btran, MOL.TRANSITIONS])   # [tranA, tranB[
        btran = tranO - tranA                         # actual number of transitions
        print("\nTransitions %d - %d " % (tranA, tranO))
        
        t_tran        =  time.time()
        
        for tran in range(tranA, tranO):
            j             =  tran-tranA
            upper, lower  =  MOL.T2L(tran)
            Aul[j]        =  MOL.A[tran]
            Ab[j]         =  MOL.BB[tran]
            freq[j]       =  MOL.F[tran]
            BG[j]         =  1.0            
            BGPHOT[j]     =  Planck(freq[j], Tbg)* 1.74080         /(PLANCK*C_LIGHT) * (1.0e5*WIDTH)*VOLUME/GL
            GNORM[j]      = (C_LIGHT/(1.0e5*WIDTH*freq[j])) * GL  # GRID_LENGTH multiplied to gauss norm
            WRK[:,j]      =  NI_ARRAY[:, upper]    # NI_ARRAY[CELLS, LEVELS],  WRK[CELLS, ntran]

        if (ONESHOT==0):
            # 2025-07-08  => now ONESHOT=0 gives same results as ONESHOT=1,
            # but probably only if offs==1
            assert(INI['offsets']==1)
            BGPHOT *= 4.0
            
        cl.enqueue_copy(queue, NI_buf,  WRK)       #  WRK[CELLS, ntran] ->  NI_buf[CELLS, ntran]
        cl.enqueue_copy(queue, Aul_buf, Aul)       #  Aul[ntran]
        cl.enqueue_copy(queue, Ab_buf,  Ab)        #  Ab[ntran]
        cl.enqueue_copy(queue, GN_buf,  GNORM)     #  GNORM[ntran]
        
        # Upload NB_NB[tran] values
        for tran in range(tranA, tranO):
            upper, lower  =  MOL.T2L(tran)
            j             =  tran-tranA
            print("  tran = %d  .... %02d -> %02d   BG %10.3e" % (tran, upper, lower, BGPHOT[j]))
            tmp           =  tmp_1 * Aul[j] * (NI_ARRAY[:,lower]*MOL.GG[tran]-NI_ARRAY[:,upper]) / (freq[j]*freq[j])
            if (0):
                tmp  = np.clip(tmp, -1.0e-12, 1.0e32)      # kernel still may have clamp on tau
                tmp[nonzero(abs(tmp)<1.0e-32)] = 1.0e-32   # nb_nb -> one must not divide by zero  $$$
            else:
                tmp = np.clip(tmp, 1.0e-25, 1.0e32)        # KILL ALL MASERS  $$$            
            WRK[:, j]     =  tmp                           # WRK[CELLS, ntran]
        cl.enqueue_copy(queue, NBNB_buf, WRK)              # NB_NB_buf[CELLS, ntran]

        
        if (WITH_OVERLAP): # in Simulate() skip transitions that are part of OVERLAP lines
            assert(1==0)
            skip = False
            for icmp in range(OLBAND.BANDS):
                if (tran in OLBAND.TRAN[icmp]): skip = True
            if (skip): continue
                            
        if (HFS):
            assert(1==0)
            nchn = BAND[tran].Channels()
            ncmp = BAND[tran].N
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
            
        if (WITH_CRT):
            assert(1==0)
            cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))
            cl.enqueue_copy(queue, CRT_EMI_buf, asarray(CRT_EMI[:,tran].copy(), float32))
                            
        if (INI['verbose']<2):
             sys.stdout.write(' %2d' % tran)
             sys.stdout.flush()
        
        WRK[:,:] = 0.0
        cl.enqueue_copy(queue, RES_buf, WRK)   #   RES[CELLS, btran]  ~  SIJ
        cl.enqueue_copy(queue, ESC_buf, WRK)   #   ESC[CELLS, btran]  ~  ESC
    
        
        offs  =  INI['offsets']  # default was 1, one ray per cell
        t000  =  time.time()


        SUM_DIRWEI = 0.0   # weight ~ cos(theta)/sum(cos(theta)) ... should sum to 1.0 for each side, 6.0 total

        
        for idir in range(NDIR):            
            inner_loop = 4*offs*offs
            if (ONESHOT): inner_loop = 1    # for  OCTREE=4, OCTREE=40
            for ioff in range(inner_loop):  # 4 staggered initial positions over 2x2 cells -- if ONESHOT==0
                
                POS, DIR, LEADING  =  GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) # < 0.001 seconds !
                dirwei, ewei = 1.0, 1.0                

                # 2021-01-13 --- BGPHOT * cos(theta)/DIRWEI, DIRWEI = sum(cos(theta)) for each side separately
                # there is no change at this point, except that DIRWEI is sum, not the average cos(theta)
                # *AND* BG is computed without dependence on the PACKETS variable
                if (LEADING in   [0, 1]):    
                    dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                    ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                elif (LEADING in [2, 3]):  
                    dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                    ewei     =  float32(EWEI / abs(DIR['y']))
                else:                     
                    dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                    ewei     =  float32(EWEI / abs(DIR['z']))                    
                BG[:]   =  BGPHOT[:] * dirwei
                # print("idir = %3d   ioff = %3d  =>   BG = %.3e" % (idir, ioff, BG))
                SUM_DIRWEI += dirwei
                dirwei = 1.0        # this is the weight factor that goes to kernel... now not used
                        
                cl.enqueue_copy(queue, BG_buf, BG)
                
                # print("         idir %3d  =>  dirwei %8.5f" % (idir, dirwei))
                               
                if (ncmp==1):
                    if (WITH_CRT):
                        assert(1==0)
                        niter = NRAY//GLOBAL
                        if (NRAY%GLOBAL!=0): niter += 1
                        for ibatch in range(niter):                                
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  
                            # 0             1          2        3        4    5   6      7     8  
                            ibatch*GLOBAL,  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, None, APL,
                            # 9   10      11    12       13   14   15      16       17        
                            BG,   dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                            # 18         19         
                            CRT_TAU_buf, CRT_EMI_buf)
                    else:
                        if (OCTREE<1):
                            assert(1==0)
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            for ibatch in range(niter):
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],
                                # 0            1          2        3        4    5 
                                ibatch*GLOBAL, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 6    7       8    9   10      11    12       13   14   15      16       17      
                                GNORM, PL_buf, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf)
                        elif (OCTREE==1):
                            # OCTREE==1 does always calculate paths but we choose not to use those??
                            assert(1==0)
                            if (PLWEIGHT<1):
                                #                                       0          1        2        3    4  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19      
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)
                            else:
                                #                                      0       1          2        3        4    5  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,], PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19    
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)                                
                        elif (OCTREE in [2,3]): # OCTREE=2, OCTREE=3
                            assert(1==0)
                            #                                       0       1          2        3        4    5   
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                            # 6    7    8   9       10    11       12   13   14      15       16        
                            GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                            #  17       18       19       20        21        
                            LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                        elif (OCTREE in [4,5]):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):                                
                                # print("idir=%2d, kernel_sim %d/%d" %(idir, 1+ibatch, niter))
                                # print("\n\n\nOCTREE = ", OCTREE)
                                if (PLWEIGHT>0):
                                    kernel_sim_multitran(queue, [GLOBAL,], [LOCAL,],
                                                         ibatch*NWG, PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul_buf, Ab_buf,
                                                         GN_buf, APL, BG_buf, dirwei, ewei, LEADING, POS, DIR, NI_buf,
                                                         NBNB_buf, RES_buf, ESC_buf, NTRUE_buf,
                                                         LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf, ntran, btran)
                                else:
                                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, CLOUD_buf,
                                    GAU_buf, LIM_buf, Aul_buf, Ab_buf, GNORM_buf, APL, BG_buf, dirwei, ewei,
                                    LEADING, POS, DIR, NI_buf, NBNB_buf, RES_buf, ESC_buf, NTRUE_buf,
                                    LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)                                
                            # sys.exit()                                
                        elif (OCTREE in [40,]):
                            assert(1==0)
                            # with option ONESHOT, kernel should loop over all ray position, not only every other
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            for ibatch in range(niter):
                                # print("kernel_sim batch %d/%d" %(1+ibatch, niter))
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*GLOBAL, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                            
                else:
                    assert(1==0)
                    if (OCTREE<1): #  GLOBAL >= NRAY
                        #                                       0          1        2        3    4   5      6   
                        kernel_hf(queue, [GLOBAL,], [LOCAL,],   CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, 
                        # 7  8       9     10       11   12   13      14       15    16    17      18      
                        BG,  dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, nchn, ncmp, HF_buf, NTRUE_buf,
                        PROFILE_buf)
                    else:
                        if (OCTREE==4):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):
                                kernel_hf(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf,  nchn, ncmp, HF_buf, 
                                RES_buf, NTRUE_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf)
                        else:
                            print("HFS + OCTREE%d not yet implemented" % OCTREE), sys.exit()
                        
                queue.finish()
                
        # end of loop over NDIR


        # for plot_pl.py
        if (0):  # debugging ... see that Update has the same PL as Path
            print("SAVING PL.dat")
            PL.tofile('PL.dat')
            cl.enqueue_copy(queue, PL, PL_buf)
            PL.tofile('PL_zero.dat')
            sys.exit()



        if (0):
            # !!! at this point intentical to the normal calculations
            m = nonzero(RHO>0.0)
            print("================================================================================")
            print("AFTER KERNEL SIJ")
            cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]
            for tran in range(btran):
                sys.stdout.write("TRAN=%d   " % tran)
                for icell in range(5):
                    sys.stdout.write("  %10.3e " % (WRK[CELLS-1-icell, tran]))
                sys.stdout.write('  ==> MEAN %10.3e\n' % mean(WRK[m[0], tran]))
            print("AFTER KERNEL ESC")
            cl.enqueue_copy(queue, WRK, ESC_buf)       # ESC[CELLS, btran]
            for tran in range(btran):
                sys.stdout.write("TRAN=%d   " % tran)
                for icell in range(5):
                    sys.stdout.write("  %10.3e " % (WRK[CELLS-1-icell, tran]))
                sys.stdout.write('  ==> MEAN %10.3e\n' % mean(WRK[m[0], tran]))
            print("================================================================================")
            
            
        # PLWEIGHT
        # - OCTREE=0 -- PLWEIGHT can be 1 (for testing). Update kernel can take PL_buf but only for testing
        #   and we do not use PL here.
        # - OCTREE>0, we assume PL is calculated and it is not used only for OCTREE=4, if INI['plweight']==0
        # post weighting
        if (WITH_ALI>0):
            # ... pull RES_buf and ESC_buf
            if (PLWEIGHT==0):    # assume PLWEIGHT implies OCTREE==0
                #  Cartesian grid -- no PL weighting, no cell-volume weighting
                cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]  
                for tran in range(tranA, tranO):
                    SIJ_ARRAY[:, tran]                =  WRK[:, tran-tranA]
                cl.enqueue_copy(queue, WRK, ESC_buf)       # ESC[CELLS, btran]
                for tran in range(tranA,tranO):
                    ESC_ARRAY[:, tran]                =  WRK[:, tran-tranA]
            else:
                if (OCTREE==0):   # regular Cartesian but PLWEIGHT=1 => use PL weighting
                    # SIJ_ARRAY[:, tran]                =  WRK[:,0] #* APL/PL[:]
                    # ESC_ARRAY[:, tran]                =  WRK[:,1] #* APL/PL[:]
                    cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]
                    for tran in range(tranA, tranO):
                        SIJ_ARRAY[:, tran]            =  WRK[:, tran-tranA]
                    cl.enqueue_copy(queue, WRK, ESC_buf)       # RES[CELLS, btran]
                    for tran in range(tranA,tranO):
                        ESC_ARRAY[:, tran]            =  WRK[:, tran-tranA] #* APL/PL[:]                    
                elif (OCTREE in [1,2]):   #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    # root level
                    a, b  =  0, LCELLS[0]
                    cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]
                    for tran in range(tranA, tranO):
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, tran-tranA] * (APL/PL[a:b])
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):             k =  8.0**l  # OCTREE1   PL should be  APL/8^l
                        else:                       k =  2.0**l  # OCTREE2   PL should be  APL/2^l
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * (1.0/k)  * (APL/PL[a:b])
                    cl.enqueue_copy(queue, WRK, ESC_buf)       # RES[CELLS, btran]
                    for tran in range(tranA, tranO):
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, tran-tranA] * (APL/PL[a:b])
                        # loop over other levels                        
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):             k =  8.0**l  # OCTREE1   PL should be  APL/8^l
                        else:                       k =  2.0**l  # OCTREE2   PL should be  APL/2^l
                        for tran in range(tranA, tranO):
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * (1.0/k)  * (APL/PL[a:b])
                else:  # OCTREE 3,4,5,40   --- assuming  it should be  PL[] == APL/2^l ;
                    if (PLWEIGHT):
                        #================================================================================
                        a, b                          =  0, LCELLS[0]
                        cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, ntran]
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * APL/PL[a:b]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            for tran in range(tranA, tranO):
                                SIJ_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] * (APL/2.0**l) /  PL[a:b]
                        cl.enqueue_copy(queue, WRK, ESC_buf)       # ESC[CELLS, ntran]
                        a, b                          =  0, LCELLS[0]
                        for tran in range(tranA, tranO):
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * APL/PL[a:b]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            for tran in range(tranA, tranO):
                                ESC_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] * (APL/2.0**l) /  PL[a:b]
                        #================================================================================
                    else:  # if we rely on PL being correct ~ 0.5^level
                        # above scaling  APL/2**l/PL translates to 1.0 
                        a, b                          =  0, LCELLS[0]
                        cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            cl.enqueue_copy(queue, WRK, RES_buf)       # RES[CELLS, btran]
                            for tran in range(tranA, tranO):
                                SIJ_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] #### / (2.0**l)
                        cl.enqueue_copy(queue, WRK, ESC_buf)       # RES[CELLS, btran]
                        for tran in range(tranA, tranO):
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            for tran in range(tranA, tranO):
                                ESC_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] #### / (2.0**l)
        else:  # no ALI, SIJ only
            ## WRK.shape = (2, CELLS)         # trickery... we use only first CELLS elements of WRK
            ## no: WRK[CELLS, btran]
            cl.enqueue_copy(queue, WRK, RES_buf)      # SIJ[CELLS, btran] only
            if (PLWEIGHT==0):              # PLWEIGHT==0 implies OCTREE=0, Cartesian grid without PL weighting
                for tran in range(tranA, tranO):
                    SIJ_ARRAY[:, tran]                =  WRK[:, tran-tranA]
            else:
                if (OCTREE==0):            # Cartesian grid with PL weighting
                    for tran in range(tranA, tranO):
                        SIJ_ARRAY[:, tran]            =  WRK[:, tran-tranA]  * APL/PL[:]
                elif (OCTREE in [1,2]):    #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    a, b                              =  0, LCELLS[0]
                    for tran in range(tranA, tranO):
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, tran-tranA] * APL/PL[a:b]
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):            k  =  8.0**l    # OCTREE1   PL should be  APL/8^l
                        else:                      k  =  2.0**l    # OCTREE2   PL should be  APL/2^l
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * (APL/k) / PL[a:b]
                else:  # OCTREE 3,4,5,40   weight 2**-l  --- except OCTREE=3 is not exact
                    if (PLWEIGHT):  # include APL/PL weighting
                        # print("*** PLWEIGHT***")
                        a, b                          =  0, LCELLS[0]
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA] * APL/PL[a:b]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            for tran in range(tranA, tranO):
                                SIJ_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct  - only volume weighting
                        a, b                          =  0, LCELLS[0]
                        for tran in range(tranA, tranO):
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, tran-tranA]
                        m = nonzero(RHO[a:b]>0.0)
                        yyy  =  APL/PL[a:b]
                        print("*** L=0, OMIT  APL/PL = %.4f +- %.4f" % (mean(yyy[m]), std(yyy[m])))
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]      
                            m  =  nonzero(RHO[a:b]>0.0)
                            yyy  =  APL / (2.0**l) / PL[a:b]
                            print("*** L=%d, OMIT  APL/PL = %.4f +- %.4f" % (l, mean(yyy[m]), std(yyy[m])))
                            for tran in range(tranA, tranO):
                                SIJ_ARRAY[a:b, tran]  =  WRK[a:b, tran-tranA] ### / (2.0**l)

            
        m = nonzero(RHO>0.0)
        if (WITH_ALI>0):
            if (INI['verbose']>1):
                print("      TRANSITION %2d  <SIJ> %12.5e   <ESC> %12.5e  T/TRAN  %6.1f" %
                (tran, mean(SIJ_ARRAY[m[0],tran]), mean(ESC_ARRAY[m[0],tran]), time.time()-t_tran))

            if (0): sys.exit()
            
            
        if (COOLING==2):
            cl.enqueue_copy(queue, LEV_COOL, COOL_buf)
            SUM_COOL[:]   +=   LEV_COOL[:] * hf[tran]    # per cm3


    #  ---- end of loop over transition batches ----

    if (INI['verbose']<2):
        sys.stdout.write('\n')

            
    # <--- for tran ---
    
    
    if (COOLING==2):
        assert(1==0)
        print("BRUTE COOLING: %10.3e" % (sum(SUM_COOL)/CELLS))
        fpb = open('brute.cooling', 'wb')
        asarray(SUM_COOL, float32).tofile(fpb)
        fpb.close()
        SUM_COOL = []
        LEV_COOL = []
        

        
            
def Cooling():
    """
    cell emits             n_u*Aul  photons / cm3
    escaping photons       ESC
    all absorbed in        SIJ
       => enough information to calculate net cooling
    COOL  =  2*ESC - SIJ*NI[lower] - n[upper]*Aul
    """
    global CELLS, TRANSITIONS, MOL, INI, SIJ_ARRAY, ESC_ARRAY, NI_ARRAY
    COOL  =  zeros(CELLS, float32)
    Aul   =  MOL.A
    hf    =  MOL.F*PLANCK
    AVE   =  0.0
    U, L  =  zeros(TRANSITIONS, int32), zeros(TRANSITIONS, int32)
    for tr in range(TRANSITIONS):
        u, l         =  MOL.T2L(tr)
        U[tr], L[tr] =  u, l   
    for icell in range(CELLS):
        # COOL[icell] +=  sum(hf[:] * (  NI_ARRAY[icell,[U[:]]*Aul[:] - SIJ_ARRAY[icell,:]*NI[L[:]] ))
        COOL[icell] +=  sum(hf[:] * (  ESC_ARRAY[icell,:]/VOLUME - SIJ_ARRAY[icell,:]*NI_ARRAY[icell, L[:]] ))
    print("COOLING: AVERAGE %12.4e" % (mean(COOL)))
    fp = open('cooling.bin', 'wb')
    asarray(COOL, float32).tofile(fp)
    fp.close()

    

    
    
def SimulateOL():
    # General line overlap, simulate only the bands of overlapping lines
    global INI, MOL, queue, context, LOCAL, GLOBAL, WIDTH, VOLUME, GL, NSIDE
    global GAU_buf, CLOUD_buf, LIM_buf, PL, EWEI, PL_buf
    global PACKETS, OLBAND, MAXCMP, MAXCHN
    print("***** SimulateOL *****")
    
    tmp_1   =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg     =  INI['Tbg']
    if (INI['verbose']<2):  sys.stdout.write('      ')
    niter   =  0
    
    Aul_buf      =  cl.Buffer(context, mf.READ_ONLY,  4 * MAXCMP)
    Ab_buf       =  cl.Buffer(context, mf.READ_ONLY,  4 * MAXCMP)
    OL_OFF_buf   =  cl.Buffer(context, mf.READ_ONLY,  4 * MAXCMP)
    OL_NI_buf    =  cl.Buffer(context, mf.READ_ONLY, 4 * 2*MAXCMP*CELLS)  #   OL_NI[2, NCMP,CELLS] = { nu, nbnb }
    OL_RES_buf   =  cl.Buffer(context, mf.READ_WRITE, 4 * 2*MAXCMP*CELLS) #  OL_RES[2, NCMP,CELLS] = { sij, esc }

    # print("GLOBAL %d, MAXCHN %d" % (GLOBAL, MAXCHN))
    OL_NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4 * GLOBAL*MAXCHN)  #  NTRUE[GLOBAL, MAXCHN]
    OL_TAU_buf   =  cl.Buffer(context, mf.READ_WRITE, 4 * GLOBAL*MAXCHN)  #    TAU[GLOBAL, MAXCHN]
    OL_EMIT_buf  =  cl.Buffer(context, mf.READ_WRITE, 4 * GLOBAL*MAXCHN)  #   EMIT[GLOBAL, MAXCHN]
    OL_TT_buf    =  cl.Buffer(context, mf.READ_WRITE, 4 * GLOBAL*MAXCHN)  #     TT[GLOBAL, MAXCHN]
    
    
    for iband in range(OLBAND.BANDS):
        
        t_tran        =  time.time()
        NCMP          =  OLBAND.NCMP[iband]  # number of transitions within this band
        NCHN          =  OLBAND.NCHN[iband]  # number of channels in the band
        TRAN          =  OLBAND.TRAN[iband]  # transitions in this band
        COFF          =  OLBAND.COFF[iband]  # offsets in channels for each transition

        # print("\n**** SimulateOL  band = %d  -- " % iband, "   TRAN = ", TRAN)
        assert(len(COFF)<=MAXCMP)
        Aul           =  zeros(NCMP, float32)
        Ab            =  zeros(NCMP, float32)
        GG            =  zeros(NCMP, float32)
        BG            =  1.0
        
        for icmp in range(NCMP):
            tran          =  TRAN[icmp]
            # upper, lower  =  MOL.T2L(tran)
            Aul[icmp]     =  MOL.A[tran]
            Ab[ icmp]     =  MOL.BB[tran]
            GG[ icmp]     =  MOL.GG[tran]
            # print("icmp=%d    Aul=%10.3e  Ab=%10.3e  GG=%10.3e  COFF=%d" %
            # (icmp, Aul[icmp], Ab[icmp], GG[icmp], COFF[icmp])) ;
            
        cl.enqueue_copy(queue, Aul_buf,    Aul)
        cl.enqueue_copy(queue, Ab_buf,     Ab)
        cl.enqueue_copy(queue, OL_OFF_buf, COFF)
            
        freq      =  0.5*(OLBAND.FMIN[iband]+OLBAND.FMAX[iband])
        freq      =  MOL.F[TRAN[0]]
        
        BGPHOT    =  Planck(freq, Tbg)* 1.74080   /  (PLANCK*C_LIGHT) * (1.0e5*WIDTH)*VOLUME/GL
        GN        = (C_LIGHT/(1.0e5*WIDTH*freq)) * GL  # GRID_LENGTH multiplied to gauss norm

        if (ONESHOT==0):
            # 2025-07-08  => now ONESHOT=0 gives same results as ONESHOT=1,
            # but probably only if offs==1
            assert(INI['offsets']==1)
            BGPHOT *= 4.0
        
        if (INI['verbose']<2):
             sys.stdout.write(' %2d' % tran)
             sys.stdout.flush()
             
        # kernel_clear(queue, [GLOBAL,], [LOCAL,], RES_buf)
        OL_WRK = zeros((2, NCMP, CELLS), float32)
        cl.enqueue_copy(queue, OL_RES_buf, OL_WRK)  # RES[2,NCMP,CELLS]
        assert(CELLS==(NI_ARRAY.shape[0]))
        
        # Upload NI[upper] and NB_NB[tran] values  NI_buf ~ [2, NCMP, CELLS]
        for icmp in range(NCMP):
            tran          =  TRAN[icmp]
            upper, lower  =  MOL.T2L(tran)
            freq          =  MOL.F[tran]
            tmp           =  tmp_1 * Aul[icmp] * (NI_ARRAY[:,lower]*GG[icmp]-NI_ARRAY[:,upper]) / (freq*freq)  # [CELLS]
            tmp           =  np.clip(tmp, 1.0e-25, 1.0e32)        # KILL ALL MASERS  $$$
            OL_WRK[0,icmp,:] =  NI_ARRAY[:, upper]   # ni
            OL_WRK[1,icmp,:] =  tmp                  # nb_nb
        cl.enqueue_copy(queue, OL_NI_buf, OL_WRK)    # OL_NI_buf [2, NCMP, CELLS]
        offs  =  INI['offsets']  # default was 1, one ray per cell
        t000  =  time.time()

        if (0):
            print("  A_b     %12.4e" % Ab[0])
            print("  GL      %12.4e" % GL)
            print("  GN      %12.4e" % GN)
            print("  Aul     %12.4e" % Aul[0])
            print("  freq    %12.4e" % freq)
            print("  gg      %12.4e" % GG[0])
            print("  BGPHOT  %12.4e" % BGPHOT)
            print("  PACKETS %d"     % PACKETS)
            print("  BG      %12.4e" % Planck(freq, Tbg))
            print("  NI      %12.4e" % (np.mean(OL_WRK[0,0,:])))
            print("  NBNB    %12.4e" % (np.mean(OL_WRK[1,0,:])))
        
        
        
        SUM_DIRWEI = 0.0   # weight ~ cos(theta)/sum(cos(theta)) ... should sum to 1.0 for each side, 6.0 total
        
        for idir in range(NDIR):            
            inner_loop = 4*offs*offs
            if (ONESHOT): inner_loop = 1    # for  OCTREE=4, OCTREE=40
            for ioff in range(inner_loop):  # 4 staggered initial positions over 2x2 cells -- if ONESHOT==0
                                
                POS, DIR, LEADING  =  GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) # < 0.001 seconds !
                dirwei, ewei = 1.0, 1.0
                # 2021-01-13 --- BGPHOT * cos(theta)/DIRWEI, DIRWEI = sum(cos(theta)) for each side separately
                # there is no change at this point, except that DIRWEI is sum, not the average cos(theta)
                # *AND* BG is computed without dependence on the PACKETS variable
                if (LEADING in   [0, 1]):    
                    dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                    ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                elif (LEADING in [2, 3]):  
                    dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                    ewei     =  float32(EWEI / abs(DIR['y']))
                else:                     
                    dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                    ewei     =  float32(EWEI / abs(DIR['z']))                    
                BG           =  BGPHOT * dirwei
                SUM_DIRWEI  +=  dirwei
                dirwei = 1.0        # this is the weight factor that goes to kernel... now not used
                niter  = NRAY//GLOBAL
                if (NRAY%GLOBAL!=0):  niter += 1                
                for ibatch in range(niter):
                    # print("    ibatch %d" % ibatch)
                    kernel_sim_ol(queue, [GLOBAL,], [LOCAL,],
                    # 0     1     2         3        4   5           6                7                   
                    # NCMP  NCHN  Aul[NCMP] Ab[NCMP] GN  COFF[NCMP]  NI[2,NCMP,CELLS] NTRUE[GLOBAL,MAXCHN]
                    NCMP,   NCHN, Aul_buf,  Ab_buf,  GN, OL_OFF_buf, OL_NI_buf,       OL_NTRUE_buf,
                    # 8            9           10                 11        12        13        14
                    # id0          CLOUD       GAU[GNO,CHANNELS]  LIM[GNO]  PL        APL       BG
                    ibatch*GLOBAL, CLOUD_buf,  GAU_buf,           LIM_buf,  PL_buf,   APL,      BG,
                    # 15        16      17         18      19     20                 
                    # DIRWEI    EWEI    LEADING    POS0    DIR    RES[2, NCMP, CELLS]
                    dirwei,     ewei,   LEADING,   POS,    DIR,   OL_RES_buf,
                    # 21                    22                    23                
                    # TAU[GLOBAL, MAXCHN],  EMIT[GLOBAL, MAXCHN]  TT[GLOBAL, MAXCHN]
                    OL_TAU_buf,             OL_EMIT_buf,          OL_TT_buf)
                    ###
                queue.finish()

                if (0):
                    print()
                    print("OL  HOST: Aul %.3e Ab %.3e GN %.3e BG %.3e DIRWEI %.3e EWEI %.3e\n" %
                    (Aul, Ab, GN, BG, dirwei, ewei))
                    sys.exit()
                
                    
                    
        if (0):  # debugging ... see that Update has the same PL as Path
            print("SAVING PL.dat")
            PL.tofile('PL.dat')
            cl.enqueue_copy(queue, PL, PL_buf)
            PL.tofile('PL_zero.dat')
            sys.exit()
                    
                
        if (0):
            print()
            cl.enqueue_copy(queue, OL_WRK, OL_RES_buf)   # WRK[2, NCMP, CELLS]
            asarray(OL_WRK[0,0,:], float32).tofile('ol.sij')
            print("RAW nol.sij WRITTED !!")
            print('NRAY %d, GLOBAL %d, niter %d' % (NRAY, GLOBAL, niter))            
            print("PLWEIGHT %d, APL %.3e PL %.3e %.3e" % (PLWEIGHT, APL, np.mean(PL), np.std(PL)))
            print("#  OL SIJ   =====>  ", np.mean(OL_WRK[0,0,:]))
            print("#  OL ESC   =====>  ", np.mean(OL_WRK[1,0,:]))
            clf()
            plot(OL_WRK[0,0,:], 'k.')
            plot(WRK[:,1], 'c.')
            # show(block=True)
            sys.exit()
            
                
        # PLWEIGHT
        # - OCTREE=0 -- PLWEIGHT can be 1 (for testing). Update kernel can take PL_buf but only for testing
        #   and we do not use PL here.
        # - OCTREE>0, we assume PL is calculated and it is not used only for OCTREE=4, if INI['plweight']==0
        # post weighting
        if (WITH_ALI>0):
            cl.enqueue_copy(queue, OL_WRK, OL_RES_buf)       # SIJ and ESC, RES[NCMP,CELLS,2] => OL_WRK[2,NCMP,CELLS]
            if (0):
                m0 = nonzero(~isfinite(OL_WRK[0,:,:]))       # WRK[2, NCMP, CELLS]
                m1 = nonzero(~isfinite(OL_WRK[1,:,:]))
                tmp = zeros((NCMP, CELLS), int32)
                tmp[m0] = 1
                print("NOT FINITE:  SIJ %d,  ESC %d" %(len(m0[0]), len(m1[0])))
                if (0):
                    clf()
                    for i in range(NCMP):
                        subplot(3,3,1+i)
                        plot(tmp[i,:])
                    show(block=True)
                    sys.exit()
            if (PLWEIGHT==0):    # assume PLWEIGHT implies OCTREE==0
                #  Cartesian grid -- no PL weighting, no cell-volume weighting
                # assert(1==0)
                for icmp in range(NCMP):
                    tran = TRAN[icmp]
                    SIJ_ARRAY[:, tran]          =  OL_WRK[0,icmp,:]  
                    ESC_ARRAY[:, tran]          =  OL_WRK[1,icmp,:]
            else:
                if (OCTREE==0):   # regular Cartesian but PLWEIGHT=1 ==> use PL weighting
                    for icmp in range(NCMP):
                        tran = TRAN[icmp]
                        # print("icmp %d, tran %d" % (icmp, tran))
                        SIJ_ARRAY[:, tran]            =  OL_WRK[0,icmp,:] #* APL/PL[:]
                        ESC_ARRAY[:, tran]            =  OL_WRK[1,icmp,:] #* APL/PL[:]
                elif (OCTREE in [1,2]):   #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    for icmp in range(NCMP):
                        tran = TRAN[icmp]
                        a, b  =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  OL_WRK[0,icmp,a:b] * (APL/PL[a:b])
                        ESC_ARRAY[a:b, tran]          =  OL_WRK[1,icmp,a:b] * (APL/PL[a:b])
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            if (OCTREE==1):         k =  8.0**l  # OCTREE1   PL should be  APL/8^l
                            else:                   k =  2.0**l  # OCTREE2   PL should be  APL/2^l
                            SIJ_ARRAY[a:b, tran]      =  OL_WRK[0,icmp,a:b] * (1.0/k)  * (APL/PL[a:b])
                            ESC_ARRAY[a:b, tran]      =  OL_WRK[1,icmp,a:b] * (1.0/k)  * (APL/PL[a:b])
                else:  # OCTREE 3,4,5,40   --- assuming  it should be  PL[] == APL/2^l ;
                    if (PLWEIGHT):
                        a, b                          =  0, LCELLS[0]
                        for icmp in range(NCMP):
                            tran = TRAN[icmp]
                            SIJ_ARRAY[a:b, tran]      =  OL_WRK[0,icmp,a:b] * APL/PL[a:b] 
                            ESC_ARRAY[a:b, tran]      =  OL_WRK[1,icmp,a:b] * APL/PL[a:b]
                            for l in range(1, OTL):
                                a, b                  =  OFF[l], OFF[l]+LCELLS[l]                    
                                SIJ_ARRAY[a:b, tran]  =  OL_WRK[0,icmp,a:b] * (APL/2.0**l) /  PL[a:b]
                                ESC_ARRAY[a:b, tran]  =  OL_WRK[1,icmp,a:b] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct ~ 0.5^level
                        # above scaling  APL/2**l/PL translates to 1.0 
                        a, b                          =  0, LCELLS[0]
                        for icmp in range(NCMP):
                            tran = TRAN[icmp]
                            SIJ_ARRAY[a:b, tran]      =  OL_WRK[0,icmp,a:b]
                            ESC_ARRAY[a:b, tran]      =  OL_WRK[1,icmp,a:b]
                            for l in range(1, OTL):
                                a, b                  =  OFF[l], OFF[l]+LCELLS[l]                    
                                SIJ_ARRAY[a:b, tran]  =  OL_WRK[0,icmp,a:b]
                                ESC_ARRAY[a:b, tran]  =  OL_WRK[1,icmp,a:b]
                                
                            
            if (0):
                tran = 0
                print("============ tran=0 =============")
                print("<APL/PL> = %.4f" % (APL/np.mean(PL)))
                clf()
                ax = axes()
                m = nonzero(RHO>0.0)
                print("CELLS %d" % len(m[0]))
                print("SIJ_ARRAY[%d] " % tran,  percentile(SIJ_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
                plot(SIJ_ARRAY[m[0],tran], 'r.')
                text(0.2, 0.5, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(SIJ_ARRAY[m[0],tran]), std(SIJ_ARRAY[m[0],tran])/mean(SIJ_ARRAY[m[0],tran])), transform=ax.transAxes, color='m', size=15)
                if (WITH_ALI>0):
                    print("ESC_ARRAY[%d] " % tran,  percentile(ESC_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
                    plot(ESC_ARRAY[m[0],tran], 'b.')
                    text(0.2, 0.4, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(ESC_ARRAY[m[0],tran]), std(ESC_ARRAY[m[0],tran])/mean(ESC_ARRAY[m[0],tran])), transform=ax.transAxes, color='c', size=15)
                title("SIJ and ESC directly from kernel")
                # show(block=True)
                sys.exit()
                                
        else:  # no ALI, SIJ only
            assert(0==1)  # it seems that this option does not exist... must use ALI
            # pass
                                                                            
        if (WITH_ALI>0):
            if (INI['verbose']>1):
                m = nonzero(RHO>0.0)
                print("      TRANSITION %2d  <SIJ> %12.5e   <ESC> %12.5e  T/TRAN  %6.1f" %
                (tran, mean(SIJ_ARRAY[m[0],tran]), mean(ESC_ARRAY[m[0],tran]), time.time()-t_tran))
            
        if (0):
            m1  =  nonzero(~isfinite(SIJ_ARRAY[m[0],0]))
            if (len(m1[0])>0):
                print("\nSIJ NOT FINITE:", len(m1[0]))
                print('SIJ  ', SIJ_ARRAY[m[0],0][m1[0]])
                print('RHO  ',       RHO[m[0]][m1[0]])
                print('TKIN ',      TKIN[m[0]][m1[0]])
                print('ABU  ',       ABU[m[0]][m1[0]])
                if (len(m1[0])>0): sys.exit()
            
    if (INI['verbose']<2):
        sys.stdout.write('\n')

        
    if (0):
        asarray(SIJ_ARRAY[:,0], float32).tofile('ol.sij')
        # sys.exit()
          
    del Aul_buf
    del Ab_buf
    del OL_NI_buf
    del OL_NTRUE_buf
    del OL_RES_buf
    del OL_TAU_buf
    del OL_TT_buf
    del OL_WRK

        
        
        
        
    

def Solve(CELLS, MOL, INI, LEVELS, TKIN, RHO, ABU, ESC_ARRAY):
    NI_LIMIT  =  1.0e-28
    CHECK     =  min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    cab       =  ones(10, float32)                  # scalar abundances of different collision partners
    for i in range(PARTNERS):
        cab[i] = MOL.CABU[i]                        # default values == 1.0
    # possible abundance file for abundances of all collisional partners
    CABFP = None
    if (len(INI['cabfile'])>0): # we have a file with abundances of each collisional partner
        CABFP = open(INI['cabfile'], 'rb')
        tmp   = np.fromfile(CABFP, int32, 4)
        if ((tmp[0]!=NX)|(tmp[1]!=NY)|(tmp[2]!=NZ)|(tmp[3]!=PARTNERS)):
            print("*** ERROR: CABFILE has dimensions %d x %d x %d, for %d partners" % (tmp[0], tmp[1], tmp[2], tmp[3]))
            sys.exit()            
    MATRIX    =  np.zeros((LEVELS, LEVELS), float32)
    VECTOR    =  np.zeros(LEVELS, float32)
    COMATRIX  =  []
    ave_max_change, global_max_change = 0.0, 0.0
    
    if (INI['constant_tkin']): # Tkin same for all cells => precalculate collisional part
        print("Tkin assumed to be constant !")
        constant_tkin = True
    else:
        constant_tkin = False
    if (constant_tkin):
        if (CABFP):
            print("Cannot have variable CAB if Tkin is assumed to be constant")
        COMATRIX = zeros((LEVELS, LEVELS), float32)
        tkin     = TKIN[1]
        for iii in range(LEVELS):
            for jjj in range(LEVELS):
                if (iii==jjj):
                    COMATRIX[iii,jjj] = 0.0
                else:
                    if (PARTNERS==1):
                        gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                    else:
                        gamma = 0.0
                        for ip in range(PARTNERS):
                            gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                    COMATRIX[jjj, iii] = gamma

    uu, ll  = zeros(TRANSITIONS, int32), zeros(TRANSITIONS, int32)
    for t in range(TRANSITIONS):
        u, l  = MOL.T2L(t)
        uu[t], ll[t] = u, l
    # all_cab = fromfile(CABFP, float32, PARTNERS*CELLS).reshape(CELLS, PARTNERS)
    for icell in range(CELLS):
        if (icell%(CELLS//20)==0):
            print("  solve   %7d / %7d  .... %3.0f%%" % (icell, CELLS, 100.0*icell/float(CELLS)))
        tkin, rho, chi  =  TKIN[icell], RHO[icell], ABU[icell]
        if (rho<1.0e-2):  continue
        if (chi<1.0e-20): continue
        
        if (constant_tkin):
            MATRIX[:,:] = COMATRIX[:,:] * rho 
        else:
            if (CABFP):
                cab = np.fromfile(CABFP, float32, PARTNERS)   # abundances for current cell, cab[PARTNERS]
            if (PARTNERS==1):
                for iii in range(LEVELS):
                    for jjj in range(LEVELS):
                        if (iii==jjj):
                            MATRIX[iii,jjj] = 0.0
                        else:
                            gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                            MATRIX[jjj, iii] = gamma*rho
            else:
                for iii in range(LEVELS):
                    for jjj in range(LEVELS):
                        if (iii==jjj):
                            MATRIX[iii,jjj] = 0.0
                        else:
                            gamma = 0.0
                            for ip in range(PARTNERS):
                                gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                            MATRIX[jjj, iii] = gamma*rho
                            
        if (len(ESC_ARRAY)>1):
            MATRIX[ll,uu]    +=  ESC_ARRAY[icell, :] / (VOLUME*NI_ARRAY[icell, uu])
        else:
            for t in range(TRANSITIONS):
                u,l           =  MOL.T2L(t)
                MATRIX[l,u]  +=  MOL.A[t]

                                
        #   X[l,u] = Aul + Bul*I       =  Aul + Blu*gl/gu*I = Aul + Slu/GG
        #   X[u,l] = Blu = Bul*gu/gl
        
        MATRIX[uu, ll]    +=  SIJ_ARRAY[icell, :] /  VOLUME
        MATRIX[ll, uu]    +=  SIJ_ARRAY[icell, :] / (VOLUME * MOL.GG[:])

                    
        for u in range(LEVELS-1): # diagonal = -sum of the column
            tmp            = sum(MATRIX[:,u])    # MATRIX[i,i] was still == 0
            MATRIX[u,u]    = -tmp
            
        MATRIX[LEVELS-1, :]   =  -MATRIX[0,0]    # replace last equation = last row
        
                
        VECTOR[:]             =   0.0
        VECTOR[LEVELS-1]      =  -(rho*chi) * MATRIX[0,0]  # ???

        
        VECTOR  =  np.linalg.solve(MATRIX, VECTOR)        
        VECTOR  =  np.clip(VECTOR, NI_LIMIT, 1e99)
        VECTOR *=  rho*chi / sum(VECTOR)        
        
        if (0): #@@
            if (icell==9310):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("MATRIX\n")
                print(MATRIX)
                asarray(MATRIX,float32).tofile('matrix.bin')
                print("VECTOR\n")
                print(VECTOR)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                            
        
        
        if (0):
            print("F= %12.4e  Gu = %.3f  Gl = %3f" % (MOL.F[0], MOL.G[1], MOL.G[0]))
            tex  =  -H_K*MOL.F[0] / log(MOL.G[0]*VECTOR[1]/(MOL.G[1]*VECTOR[0]))
            print("Tex(1-0) = %.4f" % tex)
            print("CO_ARRAY")
            for j in range(LEVELS):
                for i in range(LEVELS):
                    sys.stdout.write('%10.4e ' % (MATRIX[j,i]))
                sys.stdout.write('   %10.4e\n' % (VECTOR[j]))
            print('')
            print("VECTOR")
            for j in range(LEVELS):
                sys.stdout.write(' %10.2e' % VECTOR[j])
            sys.stdout.write('\n')
            print("")
            print("SIJ")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % SIJ_ARRAY[icell,j])
            sys.stdout.write('\n')
            if (WITH_ALI):
                print("ESC")
                for j in range(TRANSITIONS):
                    sys.stdout.write(' %10.2e' % ESC_ARRAY[icell,j])
                sys.stdout.write('\n')
            sys.exit()
            
        max_relative_change =  np.max(abs((NI_ARRAY[icell, 0:CHECK]-VECTOR[0:CHECK])/(NI_ARRAY[icell, 0:CHECK])))
        NI_ARRAY[icell,:]   =  VECTOR        
        ave_max_change     +=  max_relative_change
        global_max_change   =  max([global_max_change, max_relative_change])    
    # <--- for icell
    ave_max_change /= CELLS
    print("     AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    return ave_max_change
    



def SolveCL():
    """
    Solve equilibrium equations on the device. We do this is batches, perhaps 10000 cells
    at a time => could be up to GB of device memory. 
    Note:
        In case of octree, SIJ and ESC values must be scaled by 2^level
    """
    global NI_buf, CELLS, queue, kernel_solve, RES_buf, VOLUME, CELLS, LEVELS, TRANSITIONS, MOL
    global RHO, TKIN, ABU
    global MOL_A_buf, MOL_UL_buf, MOL_E_buf, MOL_G_buf
    global MOL_TKIN_buf, MOL_CUL_buf, MOL_C_buf
    global SOL_WRK_buf, SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf, SOL_NI_buf, SOL_CABU_buf
    global SOL_SIJ_buf, SOL_ESC_buf, OCTREE, BATCH
    # print("SolveCL")
    # TKIN
    for i in range(1, PARTNERS):
        if (len(MOL.TKIN[i])!=NTKIN):
            print("SolveCL assumes the same number of Tkin for each collisional partner!!"), sys.exit()
        
    CHECK = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    GLOBAL_SOLVE = IRound(BATCH, LOCAL)
    # print("GLOBAL_SOLVE %d, BATCH %d, LOCAL %d" % (GLOBAL_SOLVE, BATCH, LOCAL))
    tmp   = zeros((BATCH, 2, TRANSITIONS), float32)
    res   = zeros((BATCH, LEVELS), float32)
    ave_max_change     = 0.0
    global_max_change  = 0.0

    CABU  = zeros((BATCH, PARTNERS), float32)
    CABFP = None
    if (len(INI['cabfile'])>0): # we have a file with abundances of each collisional partner
        CABFP = open(INI['cabfile'], 'rb')   #  [CELLS, PARTNERS]
        tmp   = np.fromfile(CABFP, int32, 4)
        if ((tmp[0]!=NX)|(tmp[1]!=NY)|(tmp[2]!=NZ)|(tmp[3]!=PARTNERS)):
            print("*** ERROR: CABFILE has dimensions %d x %d x %d, for %d partners" % (tmp[0], tmp[1], tmp[2], tmp[3]))
            sys.exit()            
    else:                      # same abundances of collisional partners in every cell, copy from MOL.CABU to CABU
        for p in range(PARTNERS):
            CABU[:,p] = MOL.CABU[p]
        cl.enqueue_copy(queue, SOL_CABU_buf, CABU)
            
    if (OCTREE>0):
        # follow_ind = 13965
        follow_ind =   -1
        for ilevel in range(OTL):
            # print("OCTREE LEVEL %d" % ilevel)
            debug_i   = -1
            for ibatch in range(LCELLS[ilevel]//BATCH+1):
                a     = OFF[ilevel] + ibatch*BATCH
                b     = min([a+BATCH, OFF[ilevel]+LCELLS[ilevel]])
                batch = b-a
                if (batch<1): break  #   division CELLS//BATCH went even...
                if (ibatch%5==-1):
                    print(" SolveCL, batch %5d,  [%7d, %7d[, batch %4d, BATCH %4d - LCELLS %d" % 
                    (ibatch, a, b, batch, BATCH, LCELLS[ilevel]))
                debug_i = -1
                if (0):
                    if ((follow_ind>=a)&(follow_ind<b)):
                        debug_i = follow_ind-a  # cell of interest the debug-i:th in the current batch
                        print("--------------------------------------------------------------------------------") 
                        print("*** BATCH with COI %d  ... entry %d in batch=%d***" % (follow_ind, follow_ind-a, batch))
                        print("    RHO  %12.4e" %  RHO[follow_ind])
                        print("    TKIN %12.4e" % TKIN[follow_ind])
                        print("    ABU  %12.4e" %  ABU[follow_ind])
                        print("    NI   ",    NI_ARRAY[follow_ind,:])
                        print("    SIJ  ",   SIJ_ARRAY[follow_ind,:])
                        if (WITH_ALI>0):
                            print("    ESC  ",   ESC_ARRAY[follow_ind,:])
                        tex  =  -H_K*MOL.F[0] / log(MOL.G[0]*NI_ARRAY[follow_ind,1]/(MOL.G[1]*NI_ARRAY[follow_ind,0]))
                        tmp_nbnb  =  NI_ARRAY[follow_ind, 0]*MOL.GG[0]-NI_ARRAY[follow_ind, 1]
                        tmp_nbnb *=  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[0]  /  (MOL.F[0]**2.0)
                        print("    TEX10= %+8.4f,   nb_nb= %.3e,   ni= %10.3e %10.3e %10.3e" % \
                        (tex, tmp_nbnb, NI_ARRAY[follow_ind,0], NI_ARRAY[follow_ind,1], NI_ARRAY[follow_ind,2]))
                        print("--------------------------------------------------------------------------------")
                    else:
                        debug_i = -1
                # copy RHO, TKIN, ABU
                cl.enqueue_copy(queue, SOL_RHO_buf,  RHO[a:b].copy())  # without copy() "ndarray is not contiguous"
                cl.enqueue_copy(queue, SOL_TKIN_buf, TKIN[a:b].copy())
                cl.enqueue_copy(queue, SOL_ABU_buf,  ABU[a:b].copy())
                cl.enqueue_copy(queue, SOL_NI_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
                cl.enqueue_copy(queue, SOL_SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
                if (CABFP!=None):  # read abundances of collisional partners, b-a cells
                    CABFP.seek(4*4+a*PARTNERS)   # 4 ints + CELLS*PARTNERS
                    CABU[0:(b-a),:] = fromfile(CABFP, float32, (b-a)*PARTNERS).reshape(b-a, PARTNERS)
                    cl.enqueue_copy(queue, SOL_CABU_buf, CABU)
                if (WITH_ALI>0):
                    cl.enqueue_copy(queue, SOL_ESC_buf,  ESC_ARRAY[a:b,:].copy())
                # solve
                kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], ilevel, batch, 
                MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf, PARTNERS, NTKIN, NCUL,   
                MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, SOL_CABU_buf,
                SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf,  SOL_NI_buf, SOL_SIJ_buf, SOL_ESC_buf,  
                RES_buf, SOL_WRK_buf, debug_i)   # was follow_ind, should be debug_i ???
                cl.enqueue_copy(queue, res, RES_buf)
                if (INI['dnlimit']>0.0): # Limit maximum change in NI
                    # if change is above dnlimit, use average of old and new level populations (plus damping below)
                    m            =  nonzero(abs((res[0:batch,0]-NI_ARRAY[a:b,0])/NI_ARRAY[a:b,0])>INI['dnlimit'])
                    if (len(m[0])>0):
                        print("*** DNLIMIT APPLIED TO %d CELLS, dnlimit=%.3f" % (len(m[0]), INI['dnlimit']))
                        res[m[0],:]  =  0.5*res[m[0],:] + 0.5*NI_ARRAY[a+m[0],:]
                if (INI['damping']>0.0): #  Dampen change
                    # print("*** DAMPING APPLIED WITH damping=%.3f" % (INI['damping']))
                    res[0:batch,:] =  INI['damping']*NI_ARRAY[a:b,:] + (1.0-INI['damping'])*res[0:batch,:]
                # delta = for each cell, the maximum level populations change among levels 0:CHECK
                delta             =  np.max((res[0:batch,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / (1.0e-22+NI_ARRAY[a:b,0:CHECK]), axis=1)
                global_max_change =  max([global_max_change, max(delta)])
                ave_max_change   +=  sum(delta)
                NI_ARRAY[a:b,:]   =  res[0:batch,:]
                ####
                if (0):
                    if (debug_i>=0):
                        print("================================================================================")
                        print("*** BATCH with COI %d ***" % follow_ind)
                        print("    RHO  %12.4e" %  RHO[follow_ind])
                        print("    TKIN %12.4e" % TKIN[follow_ind])
                        print("    ABU  %12.4e" %  ABU[follow_ind])
                        print("    NI   ",    NI_ARRAY[follow_ind,:])
                        print("    SIJ  ",   SIJ_ARRAY[follow_ind,:])
                        print("    CLOUD ", CLOUD[follow_ind])
                        if (WITH_ALI>0):
                            print("    ESC  ",   ESC_ARRAY[follow_ind,:])
                        tex   =  -H_K*MOL.F[0] / log(MOL.G[0]*NI_ARRAY[follow_ind,1]/(MOL.G[1]*NI_ARRAY[follow_ind,0]))
                        tmp_nbnb  =  NI_ARRAY[follow_ind, 0]*MOL.GG[0]-NI_ARRAY[follow_ind, 1]
                        tmp_nbnb *=  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[0]  /  (MOL.F[0]**2.0)
                        print("    TEX10= %+8.4f,   nb_nb= %.3e,   ni= %10.3e %10.3e %10.3e" % \
                        (tex, tmp_nbnb, NI_ARRAY[follow_ind,0], NI_ARRAY[follow_ind,1], NI_ARRAY[follow_ind,2]))
                        print("================================================================================")
                    else:
                        debug_i = -1
                
            if (0):   # SIJ LOOK OK ACROSS LEVELS !!!
                a, b =  OFF[ilevel], OFF[ilevel]+LCELLS[ilevel]
                subplot(2,2,1+ilevel)
                plot(SIJ_ARRAY[a:b], 'k.')
                plot(ESC_ARRAY[a:b], 'r.')
                subplot(2,2,3+ilevel)
                plot(NI_ARRAY[a:b,1]/NI_ARRAY[a:b,0], 'k.')            
        #show()
        #sys.exit()
            
            
    else:  # not OCTREE
        
        for ibatch in range(CELLS//BATCH+1):
            a     =  ibatch*BATCH
            b     =  min([a+BATCH, CELLS])
            batch =  b-a
            if (batch<1): break  #   division CELLS//BATCH went even...
            #print(" SolveCL, batch %5d,  [%7d, %7d[,  %4d cells, BATCH %4d" % (ibatch, a, b, batch, BATCH))
            # copy RHO, TKIN, ABU
            cl.enqueue_copy(queue, SOL_RHO_buf,  RHO[a:b].copy())
            cl.enqueue_copy(queue, SOL_TKIN_buf, TKIN[a:b].copy())
            cl.enqueue_copy(queue, SOL_ABU_buf,  ABU[a:b].copy())
            cl.enqueue_copy(queue, SOL_NI_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
            cl.enqueue_copy(queue, SOL_SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
            if (CABFP!=None):  # read abundances of collisional partners, b-a cells
                CABFP.seek(4*4+a*PARTNERS)   # 4 ints + CELLS*PARTNERS
                CABU[0:(b-a),:] = fromfile(CABFP, float32, (b-a)*PARTNERS).reshape(b-a, PARTNERS)
                cl.enqueue_copy(queue, SOL_CABU_buf, CABU)
            if (WITH_ALI>0):
                cl.enqueue_copy(queue, SOL_ESC_buf,  ESC_ARRAY[a:b,:].copy())
            # solve
            kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], batch, 
            MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf,        PARTNERS, NTKIN, NCUL,   
            MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, SOL_CABU_buf,
            SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf,  SOL_NI_buf, SOL_SIJ_buf, SOL_ESC_buf,  RES_buf, SOL_WRK_buf, -1)
            cl.enqueue_copy(queue, res, RES_buf)
            # delta = for each cell, the maximum level populations change amog levels 0:CHECK
            delta             =  np.max((res[0:batch,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / NI_ARRAY[a:b,0:CHECK], axis=1)
            global_max_change =  max([global_max_change, max(delta)])
            ave_max_change   +=  sum(delta)
            NI_ARRAY[a:b,:]   =  res[0:batch]
    ave_max_change /= CELLS
    print("        SolveCL    AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    
    if (CABFP): CABFP.close()
    
    if (1):
        mbad = nonzero(~isfinite(sum(SIJ_ARRAY, axis=1)))
        print('        *** SIJ NOT FINITE: %d' % len(mbad[0]))
        if (WITH_ALI):
            mbad = nonzero(~isfinite(sum(ESC_ARRAY, axis=1)))
            print('        *** ESC NOT FINITE: %d' % len(mbad[0]))
        mbad = nonzero(~isfinite(sum(NI_ARRAY,  axis=1)))
        print('        *** NI  NOT FINITE: %d' % len(mbad[0]))
        for i in mbad[0]:
            NI_ARRAY[i,:] =  LTE_10_pop *RHO[i]*ABU[i]
    return ave_max_change

    



def WriteSpectra(INI, u, l):
    """
    Inputs:
        INI   =   parameter dictionary
        u, l  =   upper and lower level index of the transition
    """
    global MOL, program, queue, WIDTH, LOCAL, NI_ARRAY, WRK, NI_buf, HFS, CHANNELS, HFS, TAUSAVE
    global NTRUE_buf, STAU_buf, NI_buf, CLOUD_buf, GAU_buf, PROFILE_buf
    tmp_1       =  C_LIGHT*C_LIGHT/(8.0*pi)
    tran        =  MOL.L2T(u, l)
    if (tran<0):
        print("*** ERROR:  WriteSpectra  %2d -> %2d not valid transition... skipped" % (u, l))
        return
    if (HFS):
        ncmp    =  BAND[tran].N
        nchn    =  BAND[tran].Channels()
        print("     .... WriteSpectra, tran=%d, %d->%d: %d components, %d channels" % (tran, u, l, ncmp, nchn))
    else:
        nchn    =  CHANNELS     #  it is the original INI['channels']
        ncmp    =  1
    # print("*** WriteSpectra(%d, %d), ncmp=%d, nchn=%d, TAUSAVE %d" % (u, l, ncmp, nchn, TAUSAVE))
    # print("WriteSpectra: %d -> %d, transition %d" % (u, l, tran))
    Aul         =  MOL.A[tran]
    freq        =  MOL.F[tran]
    gg          =  MOL.G[u]/MOL.G[l]
    GNORM       =  (C_LIGHT/(1.0e5*WIDTH*freq))    #  GRID_LENGTH **NOT** multiplied in    
    int2temp    =  C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)
    BG          =  int2temp * Planck(freq, INI['Tbg'])

    # index of the spectrum... to find fwhm value, if given
    ul          = INI['spectra']
    ispectrum   = -1
    for i in range(len(ul)//2):
        if ((ul[2*i]==u)&(ul[2*i+1]==l)): 
            ispectrum = i
            break
    assert(ispectrum>=0)
    # 
    fwhm   = -1.0
    nfwhm  = len(INI['fwhm'])   # could be less than the number of saved spectra...
    if (nfwhm>0):  # yes, fwhm is given
        fwhm = INI['fwhm'][min([nfwhm-1, ispectrum])] / INI['grid']  # beam size in pixels
    
    
    NVIEW       =  len(INI['mapview'])
    for iview in range(NVIEW):
        
        # dimensions, direction, centre -- now all in INI['mapview']
        # print('*** MAPVIEW ', INI['mapview'][iview])
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA, NDE    = int(NRA), int(NDE)                               # 2021-07-14 - map dimensions
        DE          =  0.0    
        GLOBAL      =  IRound(NRA, LOCAL)    
        STEP        =  INI['grid'] / INI['angle']
        emissivity  =  (PLANCK/(4.0*pi))*freq*Aul*int2temp    
        direction   =  cl.cltypes.make_float2(0.0, 0.0)
        direction['x'], direction['y'] = theta, phi                   # 2021-07-14 - map direction
        
        centre       =  cl.cltypes.make_float3(0.0, 0.0, 0.0)
        centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ   
        if (isfinite(xc*yc*zc)):                                      # 2021-07-14 - map centre given by user
            centre['x'], centre['y'], centre['z']  =   xc, yc, zc
        # print("MAP CENTRE:  %8.3f %8.3f %8.3f" % (centre['x'], centre['y'], centre['z']))
        if (ncmp>1): # note -- GAU is for CHANNELS channels
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels (from centre of the spectrum)
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
                # print("       offset  %5.2f channels, weight %5.3f" % (HF[i]['x'], HF[i]['y']))
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
    
        if (WITH_CRT):
            TMP[:] = CRT_EMI[:, tran] * H_K * ((C_LIGHT/MOL.F[tran])**2.0)*C_LIGHT/(1.0e5*WIDTH*8.0*pi)
            cl.enqueue_copy(queue, CRT_EMI_buf, TMP)
            cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))
    
        if (WITH_CRT):
            if (OCTREE>0):
                kernel_spe  = program.Spectra        
                #                                 CLOUD GAU   LIM   GN          D                  NI  
                #                                 0     1     2     3           4                  5   
                kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU CRT_TAU CRT_EMI
                # 6         7         8           9           10          11    12      13      14     
                np.float32, np.int32, np.float32, np.float32, np.float32, None, None,   None,   None,
                #  15      16     17     18  
                #  LCELLS  OFF    PAR    RHO 
                None,      None,  None,  None,
                # 19       20    
                cl.int32,  cl.cltypes.float3])
            else:
                kernel_spe  = program.Spectra        
                #                                 CLOUD GAU   LIM   GN          D                  NI  
                #                                 0     1     2     3           4                  5   
                kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU CRT_TAU CRT_EMI
                # 6         7         8           9           10          11    12      13      14     
                np.float32, np.int32, np.float32, np.float32, np.float32, None, None,   None,   None,
                # 15      16           
                cl.int32, cl.cltypes.float3])
        else:
            if (OCTREE>0):
                kernel_spe  = program.Spectra        
                #                                 CLOUD  GAU   LIM   GN          D                  NI 
                #                                 0      1     2     3           4                  5  
                kernel_spe.set_scalar_arg_dtypes([None,  None, None, np.float32, cl.cltypes.float2, None,
                # DE        NRA       STEP        BG          emit0       NTRUE  SUM_TAU
                # 6         7         8           9           10          11     12    
                np.float32, np.int32, np.float32, np.float32, np.float32, None,  None,
                # LCELLS, OFF,  PAR,  RHO,  FOLLOW    CENTRE            
                # 13      14    15    16    17        18                
                None,     None, None, None, np.int32, cl.cltypes.float3])
            else:
                kernel_spe  = program.Spectra        
                #                                 0     1     2     3           4                  5
                #                                 CLOUD GAU   LIM   GN          D                  NI 
                kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                # 6         7         8           9           10          11    12       14        15
                # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU  FOLLOW    CENTRE
                np.float32, np.int32, np.float32, np.float32, np.float32, None, None,    np.int32, cl.cltypes.float3])
            
        if (ncmp>1):
            # Same kernel used with both OCTREE==0 and OCTREE==4, argument list differs
            if (OCTREE==0):
                kernel_spe_hf  = program.SpectraHF
                #                                    0     1     2     3           4                  5      
                #                                    CLOUD GAU   LIM   GN          D                  NI     
                kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                #  6        7         8           9           10          11      12     
                #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
                np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
                # 13      14        15     16       17                 
                # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE
                np.int32, np.int32, None,  None,    cl.cltypes.float3 ])
            elif (OCTREE==4):
                kernel_spe_hf  = program.SpectraHF
                #                                    0     1     2     3           4                  5      
                #                                    CLOUD GAU   LIM   GN          D                  NI     
                kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                #  6        7         8           9           10          11      12     
                #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
                np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
                # 13      14        15      16        XXX
                # LCELLS  OFF       PAR     RHO       ABU
                None,     None,     None,   None,  #  None, 
                # 17      18        19     20       21                
                # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE        
                np.int32, np.int32, None,  None,    cl.cltypes.float3 ])
            else:
                print("SpectraHF has not been defined for OCTREE=%d" % OCTREE), sys.exit()
            
        wrk         =  (tmp_1 * Aul * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])) / (freq*freq)
        # wrk was clipped to 1e-25 ... and this produced spikes in spectra ??... not the reason....
        if (0):
            wrk     =  np.clip(wrk, -1.0e-12, 1.0e10)    
            wrk[nonzero(abs(wrk)<1.0e-30)] = 1.0e-30
        else:
            wrk     =  clip(wrk, 1.0e-25, 1e10)          #  KILL ALL MASERS  $$$
            
        WRK[:,0]    =  NI_ARRAY[:, u]    # ni
        WRK[:,1]    =  wrk               # nb_nb
        wrk         =  []
        cl.enqueue_copy(queue, NI_buf, WRK)
        
        fptau = None
        if (INI['FITS']==0):
            fp    =  open('%s_%s_%02d-%02d%s.spe' % (INI['prefix'], MOL.NAME, u, l, ['', '.%03d' % iview][NVIEW>1]), 'wb')
            asarray([NRA, NDE, nchn], int32).tofile(fp)
            asarray([-0.5*(nchn-1.0)*WIDTH, WIDTH], float32).tofile(fp)
            if (TAUSAVE>0):            
                fptau = open('%s_%s_%02d-%02d%s.tau' % (INI['prefix'], MOL.NAME, u, l, ['', '%.03d' % iview][NVIEW>1]), 'wb')
        else:
            # print("MakeEmptyFitsDim... nchn = %d" % nchn)
            fp    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE, WIDTH, nchn)
            fp[0].header['BUNIT']    = 'K'
            fp[0].header['RESTFREQ'] = freq
            if (TAUSAVE>0):
                fptau =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE, WIDTH, nchn)
            
        NTRUE       =  zeros((NRA, nchn), float32)
        ANGLE       =  INI['angle']
        ave_tau     =  0.0
        tau         =  zeros(NRA, float32)
        follow      =  -1
        for de in range(NDE):
            DE      =  de-0.5*(NDE-1.0)
            
            if (ncmp>1): # since CHANNELS has been changed, all transitions written using this kernel ???
                if (OCTREE==0):
                    print("HFS:  OCTREE==0   GLOBAL %d  LOCAL %d" % (GLOBAL, LOCAL))
                    kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10         
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                    # 11        12       13    14    15      16           17     
                    NTRUE_buf, STAU_buf, nchn, ncmp, HF_buf, PROFILE_buf, centre)
                elif (OCTREE==4):
                    kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10 
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                    # 11       12        13          14       15       16       17       
                    NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, # ABU_buf, 
                    #  19  19    20      21           22     
                    nchn,  ncmp, HF_buf, PROFILE_buf, centre)
                else: 
                    print("kernel_spe_hfs exists only for OCTREE==0 and OCTREE==4"), sys.exit()
            else:
                # print("---------- kernel_spe ----------")
                if (WITH_CRT):
                    kernel_spe(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10          
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, 
                    # 11         12 13         14           15     
                    NTRUE_buf, STAU_buf, CRT_TAU_buf, CRT_EMI_buf, centre)
                else:
                    if (OCTREE):
                        follow = -1
                        # if (de==90):   follow=170
                        kernel_spe(queue, [GLOBAL,], [LOCAL,],
                        # 0        1        2         3     4          5       6   7    8     9   10         
                        CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, 
                        # 11       12        13          14       15       16       
                        NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, 
                        # 17    18    
                        follow, centre)
                    else:
                        kernel_spe(queue, [GLOBAL,], [LOCAL,],
                        # 0        1        2         3     4          5       6   7    8     9   10          11         12        13  14
                        CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, NTRUE_buf, STAU_buf, -1, centre)
                        
                        
            # save spectrum
            queue.finish()

            # NTRUE[NRA, nchn]
            cl.enqueue_copy(queue, NTRUE, NTRUE_buf)  # NTRUE is only nchn channels, kernel uses nchn channels
            WWW  = sum(NTRUE, axis=1)
            ira  = argmax(WWW)
            # if (WWW[ira]>300): print("de=%3d  max(W)=%7.2f for ra=%d" % (de, WWW[ira], ira))
            
            if (INI['FITS']==0):
                for ra in range(NRA):
                    asarray([(ra-0.5*(NRA-1.0))*ANGLE, (de-0.5*(NDE-1.0))*ANGLE], float32).tofile(fp) # offsets
                    NTRUE[ra,:].tofile(fp)       # spectrum
                    # print("WRITE NTRUE %4d channels, MAX at channel %d" % (NTRUE.shape[1], argmax(NTRUE[ra,:])))
                # save optical depth
                cl.enqueue_copy(queue, NTRUE, STAU_buf)
                for ra in range(NRA):
                    tau[ra]  =  np.max(NTRUE[ra,:])
                ave_tau +=  np.sum(tau)       # sum of the peak tau values of the individual spectra
                if (TAUSAVE):                
                    tau.tofile(fptau)      # file containing peak tau for each spectrum
                
            else:
                # Spectrum
                for ra in range(NRA):
                    fp[0].data[:, de, ra]  =  NTRUE[ra, :]   #  NTRUE[NRA, nchn],  fp[nchn, NDE, NRA]
                # Optical depth
                cl.enqueue_copy(queue, NTRUE, STAU_buf)
                for ra in range(NRA):
                    tau[ra]  =  np.max(NTRUE[ra,:])
                ave_tau +=  np.sum(tau)   # sum of the peak tau values of the individual spectra
                if (TAUSAVE):          # save optical depth
                    for ra in range(NRA):
                        fptau[0].data[:, de, ra]  =  NTRUE[ra,:]
                    
        # --- for de
        if (INI['FITS']==0):
            fp.close()
            if (fptau): fptau.close()
        else:
            # FITS spectra will be convolved if INI containted keyword fwhm            
            if (fwhm>0.0):
                fp[0].data = ConvolveCube(fp[0].data, fwhm, INI['GPU'], INI['platforms'])
                fp[0].header['BEAM'] = fwhm*INI['grid']/3600.0
            fp.writeto('%s_%s_%02d-%02d%s.fits'        % (INI['prefix'], MOL.NAME, u, l, ['', '.%03d' % iview][NVIEW>1]), overwrite=True)
            if (TAUSAVE):
                fptau.writeto('%s_%s_%02d-%02d_tau%s.fits' % (INI['prefix'], MOL.NAME, u, l, ['', '.%03d' % iview][NVIEW>1]), overwrite=True)
                del fptau
            del fp
        print("  SPECTRUM %3d  = %2d -> %2d,  <tau_peak> = %.3e" % (tran, u, l, ave_tau/(NRA*NDE)))


        

        

        
def WriteSpectraOL(INI, OLBAND, iband):
    # Write spectra for set of overlapping lines => 
    global MOL, program, queue, WIDTH, LOCAL, NI_ARRAY, WRK, NI_buf, HFS, CHANNELS
    global HFS, TAUSAVE, LIM_buf
    global CLOUD_buf, GAU_buf
    tmp_1       =  C_LIGHT*C_LIGHT/(8.0*pi)
    NCHN        =  OLBAND.NCHN[iband]
    f0, f1      =  OLBAND.FMIN[iband], OLBAND.FMAX[iband]
    # print("CHANNELS = %d   ==>  NCHN = %d" % (CHANNELS, NCHN))
    freq        =  0.5*(OLBAND.FMIN[iband]+OLBAND.FMAX[iband])
    GNORM       =  (C_LIGHT/(1.0e5*WIDTH*freq))      #  GRID_LENGTH **NOT** multiplied in    
    int2temp    =   C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)    
    BG          =  int2temp * Planck(freq, INI['Tbg'])  # still scalar
    NVIEW       =  len(INI['mapview'])
    TRAN        =  asarray(OLBAND.TRAN[iband], int32)
    NCMP        =  OLBAND.NCMP[iband]
    COFF        =  asarray(OLBAND.COFF[iband], float32)

    # find the spectrum with the closest frequency
    ul          =  INI['spectra']
    ispectrum   =  -1
    dfmin       =  1e10
    for i in range(len(ul)//2):   # loop over spectra selected by user
        itran   =  MOL.L2T(ul[2*i], ul[2*i+1])
        df      =  abs(MOL.F[itran]-freq)
        if (df<dfmin):
            ispectrum = i
            dfmin     = df
    assert(ispectrum>=0)
    # 
    fwhm   = -1.0
    nfwhm  = len(INI['fwhm'])   # could be less than the number of saved spectra...
    if (nfwhm>0):               # yes, fwhm is given
        fwhm = INI['fwhm'][min([nfwhm-1, ispectrum])] / INI['grid']  # beam size in pixels

            
    NRA_MAX     =  0
    for iview in range(len(INI['mapview'])):    
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA_MAX = max([NRA_MAX, int(NRA)])
    
    WRK         =  zeros((2, NCMP, CELLS), float32)  # 2*NCMP*CELLS
    emit0       =  zeros(NCMP, float32)    
    GNORM       =  zeros(NCMP, float32)
    tran0       =  TRAN[0]
    u0, l0      =  MOL.T2L(tran0)
    for icmp in range(NCMP):
        tran    =  TRAN[icmp]
        u, l    =  MOL.T2L(tran)
        Aul     =  MOL.A[tran]
        freq    =  MOL.F[tran]        
        gg      =  MOL.G[u]/MOL.G[l]   
        WRK[0, icmp, :] = NI_ARRAY[:,u]   #  NI[NCMP, CELLS]
        wrk     =  (tmp_1 * Aul * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])) / (freq*freq)
        wrk     =  clip(wrk, 1.0e-25, 1e10)          #  KILL ALL MASERS  $$$
        WRK[1, icmp, :] = wrk             # NBNB[NCMP, CELLS]
        ##
        emit0[icmp]  =  (PLANCK/(4.0*pi))*freq*Aul*int2temp
        ##
        GNORM[icmp]  =  (C_LIGHT/(1.0e5*WIDTH*freq))

        
    OL_NI_buf    =  cl.Buffer(context, mf.READ_ONLY,  4* 2*NCMP*CELLS) # NI[NCMP,CELLS]+NBNB[NCMP,CELLS]

    OL_TRAN_buf  =  cl.Buffer(context, mf.READ_ONLY,  4* NCMP)
    OL_COFF_buf  =  cl.Buffer(context, mf.READ_ONLY,  4* NCMP)
    OL_TAU_buf   =  cl.Buffer(context, mf.READ_WRITE, 4* NRA_MAX*NCHN) 
    OL_EMIT_buf  =  cl.Buffer(context, mf.READ_WRITE, 4* NRA_MAX*NCHN) 

    OL_NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4* NRA_MAX*NCHN) 
    OL_STAU_buf  =  cl.Buffer(context, mf.READ_WRITE, 4* NRA_MAX*NCHN) 
    OL_TT_buf    =  cl.Buffer(context, mf.READ_WRITE, 4* NRA_MAX*NCHN)

    OL_EMIT0_buf =  cl.Buffer(context, mf.READ_ONLY,  4* NCMP)
    GN_buf       =  cl.Buffer(context, mf.READ_ONLY,  4* GNO)    
    cl.enqueue_copy(queue, GN_buf,       GNORM)
    cl.enqueue_copy(queue, OL_NI_buf,    WRK)
    cl.enqueue_copy(queue, OL_EMIT0_buf, emit0)
    cl.enqueue_copy(queue, OL_TRAN_buf,  TRAN)
    cl.enqueue_copy(queue, OL_COFF_buf,  COFF)
    WRK = []
    
    kernel_spe_ol  = program.SpectraOL
    if (OCTREE==0):
        kernel_spe_ol.set_scalar_arg_dtypes(
        # NCMP       NCHN         TRAN   COFF   TAU_ARRAY  EMIT_array
        [ np.int32,  np.int32,    None,  None,  None,      None,
        # CLOUD  GAU    LIM    GN     D                  NI
        None,    None,  None,  None,  cl.cltypes.float2, None,
        #  DE       NRA       STEP        BG          emit0
        np.float32, np.int32, np.float32, np.float32, None,
        # NTRUE STAU   TT    CENTRE
        None,    None, None, cl.cltypes.float3])
    else:
        kernel_spe_ol.set_scalar_arg_dtypes(
        [ np.int32,  np.int32, None, None, None, None,
        None, None, None, None, cl.cltypes.float2, None,
        np.float32, np.int32, np.float32, np.float32, None,
        None, None, None,
        None, None, None, None,
        np.int32, cl.cltypes.float3])

        
        
    for iview in range(NVIEW):
        
        # dimensions, direction, centre -- now all in INI['mapview']
        # print('*** MAPVIEW ', INI['mapview'][iview])
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA, NDE     =  int(NRA), int(NDE)                               # 2021-07-14 - map dimensions
        NTRUE        =  zeros((NRA, NCHN), float32)        
        DE           =  0.0    
        GLOBAL       =  IRound(NRA, LOCAL)    
        STEP         =  INI['grid'] / INI['angle']
        direction    =  cl.cltypes.make_float2(0.0, 0.0)
        direction['x'], direction['y'] = theta, phi                   # 2021-07-14 - map direction
        centre       =  cl.cltypes.make_float3(0.0, 0.0, 0.0)
        centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ   
        if (isfinite(xc*yc*zc)):                                      # 2021-07-14 - map centre given by user
            centre['x'], centre['y'], centre['z']  =   xc, yc, zc
        # print("MAP CENTRE:  %8.3f %8.3f %8.3f" % (centre['x'], centre['y'], centre['z']))
        # print("MakeEmptyFitsDim... nchn = %d" % NCHN)
        fp    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE, WIDTH, NCHN)
        fp[0].header['BUNIT']    = 'K'
        fp[0].header['RESTFREQ'] = freq
        if (TAUSAVE>0):
            fptau =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE, WIDTH, NCHN)
        ###
        ANGLE       =  INI['angle']
        ave_tau     =  0.0
        tau         =  zeros(NRA, float32)
        for de in range(NDE):
            DE      =  de-0.5*(NDE-1.0)
            # print("DE %d/%d" % (de, NDE))
            if (OCTREE==0):
                # print("OL, OCTREE==0   GLOBAL %d  LOCAL %d" % (GLOBAL, LOCAL))
                kernel_spe_ol(queue, [GLOBAL,], [LOCAL,],
                NCMP, NCHN,    OL_TRAN_buf, OL_COFF_buf, OL_TAU_buf, OL_EMIT_buf,
                CLOUD_buf, GAU_buf, LIM_buf, GN_buf, direction, OL_NI_buf, 
                DE, NRA, STEP, BG, OL_EMIT0_buf, 
                OL_NTRUE_buf, OL_STAU_buf, OL_TT_buf, centre)
            elif (OCTREE==4):
                # print("OL, OCTREE==1   GLOBAL %d  LOCAL %d" % (GLOBAL, LOCAL))
                kernel_spe_ol(queue, [GLOBAL,], [LOCAL,],
                NCMP, NCHN,    OL_TRAN_buf, OL_COFF_buf, OL_TAU_buf, OL_EMIT_buf,
                CLOUD_buf, GAU_buf, LIM_buf, GN_buf, direction, OL_NI_buf, 
                DE, NRA, STEP, BG, OL_EMIT0_buf, 
                OL_NTRUE_buf, OL_STAU_buf, OL_TT_buf, centre,
                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)                
            # NTRUE[NRA, NCHN]
            queue.finish()
            # print("*** KERNEL DONE ***")
            cl.enqueue_copy(queue, NTRUE, OL_NTRUE_buf)  # NTRUE is only nchn channels, kernel uses nchn channels
            WWW  = sum(NTRUE, axis=1)
            ira  = argmax(WWW)
            for ra in range(NRA):
                fp[0].data[:, de, ra]  =  NTRUE[ra, :]   #  NTRUE[NRA, nchn],  fp[nchn, NDE, NRA]
            # tau
            cl.enqueue_copy(queue, NTRUE, OL_STAU_buf)
            for ra in range(NRA):
                ave_tau += np.max(NTRUE[ra,:])
                if (TAUSAVE):  # save optical depth
                    fptau[0].data[:, de, ra]  =  NTRUE[ra,:]
        # --- for de
        if (fwhm>0.0):
            fp[0].data = ConvolveCube(fp[0].data, fwhm, INI['GPU'], INI['platforms'])
            fp[0].header['BEAM'] = fwhm*INI['grid']/3600.0            
        fp.writeto('OL_%s_%s_%02d-%02d%s.fits'        % (INI['prefix'], MOL.NAME, u0, l0, ['', '.%03d' % iview][NVIEW>1]), overwrite=True)
        if (TAUSAVE):
            #  TAU not convolved !
            fptau.writeto('%s_%s_%02d-%02d_tau%s.fits' % (INI['prefix'], MOL.NAME, u0, l0, ['', '.%03d' % iview][NVIEW>1]), overwrite=True)
            del fptau
        print("  SPECTRUM %3d  = %2d -> %2d,  <tau_peak> = %.3e" % (tran0, u0, l0, ave_tau/(NRA*NDE)))

        
        
    

        

def WriteLOSSpectrum(INI, u, l):
    """
    Write file for a single line of sight, the contribution of different LOS steps to the final spectrum.
    Input:
        INI['losspectrum'] = 1   =>  escaped (observed radiation)
                           = 2   =>  emitted, without any foreground absorption
    LOS_EMIT[steps, channels] =  contribution [K] of each spectrum to the final spectrum
    LENGTH[steps]             =  length of each step, starting from the observer side
    LOS_RHO[steps], LOS_T[steps], LOS_X[steps] = density, Tkin, and abundance for each step
    """
    global MOL, program, queue, WIDTH, LOCAL, NI_ARRAY, WRK, NI_buf, HFS, CHANNELS, HFS, TAUSAVE
    global NTRUE_buf, STAU_buf, NI_buf, CLOUD_buf, GAU_buf, PROFILE_buf
    TAGABS      =  INI['losspectrum']    #  1 = observed, 2 = no foreground absorptions
    if (not(TAGABS in[1,2])):
        print("********************************************************************************")
        print("WriteLOSSpectrum: check keyword losspectrum!")
        print("Possible values are: 1 (normal, observed) and 2 (no foreground absorption)")
        print("********************************************************************************")
        time.sleep(3)
        return ;
    tmp_1       =  C_LIGHT*C_LIGHT/(8.0*pi)
    tran        =  MOL.L2T(u, l)
    if (tran<0):
        print("*** ERROR:  WriteLOSSpectrum  %2d -> %2d not valid transition" % (u, l))
        return
    if (HFS):
        ncmp    =  BAND[tran].N
        nchn    =  BAND[tran].Channels()
        print("     .... WriteSpectra, tran=%d, %d->%d: %d components, %d channels" % (tran, u, l, ncmp, nchn))
    else:
        nchn    =  CHANNELS     #  it is the original INI['channels']
        ncmp    =  1
    print("*** WriteLOsSpectrum(%d, %d), ncmp=%d, nchn=%d, TAUSAVE %d" % (u, l, ncmp, nchn, TAUSAVE))
    Aul         =  MOL.A[tran]
    freq        =  MOL.F[tran]
    gg          =  MOL.G[u]/MOL.G[l]
    GNORM       =  (C_LIGHT/(1.0e5*WIDTH*freq))    #  GRID_LENGTH **NOT** multiplied in    
    int2temp    =  C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)
    BG          =  int2temp * Planck(freq, INI['Tbg'])

    TKIN_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*CELLS)
    ABU_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*CELLS)
    cl.enqueue_copy(queue, TKIN_buf, TKIN)
    cl.enqueue_copy(queue, ABU_buf,  ABU)

    
    for iview in range(len(INI['mapview'])):    
        
        # dimensions, direction, centre -- now all in INI['mapview']
        print('*** MAPVIEW ', INI['mapview'][iview])
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA, NDE    = int(NRA), int(NDE)                               # 2021-07-14 - map dimensions
        NRA, NDDE   =  1, 1      # a single LOS !!
        DE          =  0.0       # not used
        GLOBAL      =  LOCAL     # single work item used
        STEP        =  INI['grid'] / INI['angle']
        emissivity  =  (PLANCK/(4.0*pi))*freq*Aul*int2temp    
        direction   =  cl.cltypes.make_float2(0.0,0.0)
        direction['x'], direction['y'] = theta, phi                   # 2021-07-14 - map direction
        centre       =  cl.cltypes.make_float3(0.0,0.0,0.0)
        centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ   
        if (isfinite(xc*yc*zc)):                                      # 2021-07-14 - map centre given by user
            centre['x'], centre['y'], centre['z']  =   xc, yc, zc
        print("MAP CENTRE:  %8.3f %8.3f %8.3f" % (centre['x'], centre['y'], centre['z']))

        if (ncmp>1): # note -- GAU is for CHANNELS channels = maximum over all bands!!
            print("*** ncmp>1 not yet possible in WriteLOSSpectrum !!!")
            return
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels (from centre of the spectrum)
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
                print("       offset  %5.2f channels, weight %5.3f" % (HF[i]['x'], HF[i]['y']))
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
    
        if (WITH_CRT):
            TMP[:] = CRT_EMI[:, tran] * H_K * ((C_LIGHT/MOL.F[tran])**2.0)*C_LIGHT/(1.0e5*WIDTH*8.0*pi)
            cl.enqueue_copy(queue, CRT_EMI_buf, TMP)
            cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))
    
        if (WITH_CRT):
            if (OCTREE>0):
                kernel_spe  = program.Spectra_vs_LOS
                #                                 CLOUD GAU   LIM   GN          D                  NI  
                #                                 0     1     2     3           4                  5   
                kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU CRT_TAU CRT_EMI
                # 6         7         8           9           10          11    12      13      14     
                np.float32, np.int32, np.float32, np.float32, np.float32, None, None,   None,   None,
                #  15     16   17      18     
                #  LCELLS OFF  PAR     RHO    
                None,    None,  None,  None,   
                # 19       20                     21     22     23        24          25        26         27        28   
                # FOLLOW   CENTRE                 TKIN   ABU    LOS_EMIT  LOS_LENGTH  LOS_RHO   LOS_TKIN   LOS_ABU   TAGABS 
                cl.int32,  cl.cltypes.float3,     None,  None,  None,     None,       None,     None,      None,     np.int32 ])
            else:
                if (1):
                    print("Spectra_vs_LOS() not implemented for plain Cartesian grid -- consider using octree format")
                    sys.exit()
                else:  # in the case of Cartesian grid, one would need to add density as another parameter 
                    kernel_spe  = program.Spectra_vs_LOS
                    #                                 CLOUD GAU   LIM   GN          D                  NI  
                    #                                 0     1     2     3           4                  5   
                    kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                    # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU CRT_TAU CRT_EMI
                    # 6         7         8           9           10          11    12      13      14     
                    np.float32, np.int32, np.float32, np.float32, np.float32, None, None,   None,   None,
                    # 15      16                     17     18     19        20      21       22        23       24
                    #  FOLLOW   CENTRE               TKIN   ABU    LOS_EMIT  LENGTH  LOS_RHO, LOS_TKIN, LOS_ABU  TAGABS
                    cl.int32, cl.cltypes.float3,     None,  None,  None,     None,   None,    None,     None,    np.int32   ])
        else:
            if (OCTREE>0):
                kernel_spe  = program.Spectra_vs_LOS
                #                                 CLOUD  GAU   LIM   GN          D                  NI 
                #                                 0      1     2     3           4                  5  
                kernel_spe.set_scalar_arg_dtypes([None,  None, None, np.float32, cl.cltypes.float2, None,
                # DE        NRA       STEP        BG          emit0       NTRUE  SUM_TAU
                # 6         7         8           9           10          11     12    
                np.float32, np.int32, np.float32, np.float32, np.float32, None,  None,
                # LCELLS, OFF,  PAR,  RHO,    FOLLOW    CENTRE             
                # 13      14    15    16      17        18                 
                None,     None, None, None,   np.int32, cl.cltypes.float3,  
                #   TKIN    ABU    LOS_EMIT  LENGTH  LOS_RHO  LOS_TKIN    LOS_ABU   TAGABS
                #   19      20     21        22      23       24          15        16
                None,       None,  None,     None,   None,    None,       None,     np.int32   ])
            else:
                kernel_spe  = program.Spectra_vs_LOS
                #                                 0     1     2     3           4                  5
                #                                 CLOUD GAU   LIM   GN          D                  NI 
                kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                # 6         7         8           9           10          11    12       14        15
                # DE        NRA       STEP        BG          emit0       NTRUE SUM_TAU  FOLLOW    CENTRE
                np.float32, np.int32, np.float32, np.float32, np.float32, None, None,    np.int32, cl.cltypes.float3,
                #  16     17     19        20      21       22         13       14
                #  TKIN   ABU    LOS_EMIT  LENGTH  LOS_RHO  LOS_TKIN   LOS_ABU  TAGABS
                None,     None,  None,     None,   None,    None,      None,    np.int32    ])
            
        if (ncmp>1):
            print("WriteLOSSpectrum -- no kernel yet for hfs spectra -- skip calculation!!!")
            return 
            # Same kernel used with both OCTREE==0 and OCTREE==4, argument list differs
            if (OCTREE==0):
                kernel_spe_hf  = program.SpectraHF
                #                                    0     1     2     3           4                  5      
                #                                    CLOUD GAU   LIM   GN          D                  NI     
                kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                #  6        7         8           9           10          11      12     
                #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
                np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
                # 13      14        15     16       17                  18
                # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE          TAGABS
                np.int32, np.int32, None,  None,    cl.cltypes.float3,  np.int32 ])
            elif (OCTREE==4):
                kernel_spe_hf  = program.SpectraHF
                #                                    0     1     2     3           4                  5      
                #                                    CLOUD GAU   LIM   GN          D                  NI     
                kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
                #  6        7         8           9           10          11      12     
                #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
                np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
                # 13      14        15      16   
                # LCELLS  OFF       PAR     RHO  
                None,     None,     None,   None, None, 
                # 17      18        19     20       21                  22
                # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE          TAGABS
                np.int32, np.int32, None,  None,    cl.cltypes.float3,  np.int32 ])
            else:
                print("SpectraHF has not been defined for OCTREE=%d" % OCTREE), sys.exit()
            
        wrk         =  (tmp_1 * Aul * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])) / (freq*freq)
        if (0):
            wrk     =  np.clip(wrk, -1.0e-12, 1.0e10)    
            wrk[nonzero(abs(wrk)<1.0e-30)] = 1.0e-30
        else:
            wrk     =  clip(wrk, 1.0e-25, 1e10)          #  KILL ALL MASERS  $$$
            
        WRK[:,0]    =  NI_ARRAY[:, u]    # ni
        WRK[:,1]    =  wrk               # nb_nb
        wrk         =  []
        cl.enqueue_copy(queue, NI_buf, WRK)

        NTRUE       =  zeros((NRA, nchn), float32)
        ANGLE       =  INI['angle']
        ave_tau     =  0.0
        tau         =  zeros(NRA, float32)
        follow      =  -1

        # we need buffers for  LOS_EMIT[max_steps, nchn]  and LENGTH[max_steps]
        max_steps    =  32768     #  assume no LOS has more than this many cells for any given LOS
        LOS_EMIT     =  zeros((max_steps, nchn), float32)
        tmp          =  zeros(max_steps, float32)
        LOS_EMIT_buf =  cl.Buffer(context, mf.WRITE_ONLY,  4*max_steps*nchn)
        LENGTH_buf   =  cl.Buffer(context, mf.WRITE_ONLY,  4*max_steps)
        LOS_RHO_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*max_steps)
        LOS_TKIN_buf =  cl.Buffer(context, mf.WRITE_ONLY,  4*max_steps)
        LOS_ABU_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*max_steps)        
        cl.enqueue_copy(queue, LOS_EMIT_buf, LOS_EMIT)
        
        for de in range(1):                      # a single LOS !!
            DE      =  de-0.5*(NDE-1.0)
            
            if (ncmp>1): # since CHANNELS has been changed, all transitions written using this kernel ???
                assert(1==0)  # not yet implemented
                if (OCTREE==0):
                    print("HFS:  OCTREE==0   GLOBAL %d  LOCAL %d" % (GLOBAL, LOCAL))
                    kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10         
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                    # 11        12       13    14    15      16           17      18   
                    NTRUE_buf, STAU_buf, nchn, ncmp, HF_buf, PROFILE_buf, centre, TAGABS)
                elif (OCTREE==4):
                    kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10 
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                    # 11       12        13          14       15       16       17       
                    NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, ABU_buf, 
                    #  19  19    20      21           22      23 
                    nchn,  ncmp, HF_buf, PROFILE_buf, centre, TAGABS)
                else:
                    print("kernel_spe_hfs exists only for OCTREE==0 and OCTREE==4"), sys.exit()
            else:
                # print("---------- kernel_spe ----------")
                if (WITH_CRT):
                    kernel_spe(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10          
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, 
                    # 11       12        13           14           15     
                    NTRUE_buf, STAU_buf, CRT_TAU_buf, CRT_EMI_buf, centre,
                    #  16      17       18            19          20           21            22           23
                    TKIN_buf,  ABU_buf, LOS_EMIT_buf, LENGTH_buf, LOS_RHO_buf, LOS_TKIN_buf, LOS_ABU_buf, TAGABS)
                else:
                    if (OCTREE):
                        follow = -1
                        # if (de==90):   follow=170
                        kernel_spe(queue, [GLOBAL,], [LOCAL,],
                        # 0        1        2         3     4          5       6   7    8     9   10         
                        CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, 
                        # 11       12        13          14       15       16       
                        NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, 
                        # 17    18      19        20       21            22          23           24            25           26    
                        follow, centre, TKIN_buf, ABU_buf, LOS_EMIT_buf, LENGTH_buf, LOS_RHO_buf, LOS_TKIN_buf, LOS_ABU_buf, TAGABS)
                    else:
                        kernel_spe(queue, [GLOBAL,], [LOCAL,],
                        # 0        1        2         3     4          5       6   7    8     9   10         
                        CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                        # 11       12        13  14      15        16       17            18          19           20            21           22
                        NTRUE_buf, STAU_buf, -1, centre, TKIN_buf, ABU_buf, LOS_EMIT_buf, LENGTH_buf, LOS_RHO_buf, LOS_TKIN_buf, LOS_ABU_bud, TAGABS)
                        
            # save spectrum
            queue.finish()

            # we pull from device: LOS_EMIT, LENGTH, LOS_RHO, LOS_TKIN, LOS_ABU
            cl.enqueue_copy(queue, LOS_EMIT, LOS_EMIT_buf)
            cl.enqueue_copy(queue, tmp,   LENGTH_buf)
            # file format: steps, step values [steps], emission 
            # last used row in LOS_EMIT corresponds to the background contribution, LENGTH[steps-1]==-1.1e10
            m = nonzero(tmp<-1e10)
            steps = m[0][0]+1  # we have steps entries, plus one row for background contribution to the spectrum
            fp    = open('losspectrum_%s_%02d-%02d_%03d_tag%d.bin' % (MOL.NAME, u, l, iview, TAGABS), 'wb')
            asarray([steps, nchn], int32).tofile(fp)           # actually steps-1 steps + one entry for background
            asarray(tmp[0:steps], float32).tofile(fp)          #  LENGTH
            asarray(LOS_EMIT[0:steps, :], float32).tofile(fp)  #  LOS_EMIT[steps, nchn]
            cl.enqueue_copy(queue, tmp,   LOS_RHO_buf)
            asarray(tmp[0:steps], float32).tofile(fp)          #  LOS_RHO
            cl.enqueue_copy(queue, tmp,   LOS_TKIN_buf)
            asarray(tmp[0:steps], float32).tofile(fp)          #  LOS_TKIN
            cl.enqueue_copy(queue, tmp,   LOS_ABU_buf)
            asarray(tmp[0:steps], float32).tofile(fp)          #  LOS_ABU
            fp.close()
            

    
        
        
def WriteColumndensity(INI):
    """
    Save maps of total column density, column density of the species, and the mass-weighted Tkin 
    (mass-weighted = weighted by rho*abundance) and the mass-weighted LOS velocity
    """
    if (OCTREE<1): 
        print("WriteColumnDensity() only for octree clouds")
        return
    global program, queue, LOCAL, RHO_buf
    ABU_buf   = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
    TKIN_buf  = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
    cl.enqueue_copy(queue, ABU_buf,  ABU)
    cl.enqueue_copy(queue, TKIN_buf, TKIN)
    
    for iview in range(len(INI['mapview'])):            
        print('*** MAPVIEW ', INI['mapview'][iview])
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA, NDE    =  int(NRA), int(NDE)                             # 2021-07-14 - map dimensions
        DE          =  0.0    
        GLOBAL      =  IRound(NRA, LOCAL)    
        STEP        =  INI['grid'] / INI['angle']
        direction   =  cl.cltypes.make_float2(0.0,0.0)
        direction['x'], direction['y'] = theta, phi                   # 2021-07-14 - map direction
        centre       =  cl.cltypes.make_float3(0.0,0.0,0.0)
        centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ   
        if (isfinite(xc*yc*zc)):                                      # 2021-07-14 - map centre given by user
            centre['x'], centre['y'], centre['z']  =   xc, yc, zc
        print("MAP CENTRE:  %8.3f %8.3f %8.3f" % (centre['x'], centre['y'], centre['z']))
        COLDEN_buf  = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # N(H2) for one column of pixels
        MCOLDEN_buf = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # N(mol) -"-
        WTKIN_buf   = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # <Tkin> -"-
        WV_buf      = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # <V_LOS> -"-
        kernel_colden  = program.Columndensity
        if (OCTREE>0):
            kernel_colden.set_scalar_arg_dtypes([
            # 0                 1                   2            3          4         
            # DIRECTION         CENTRE              DE           NRA        STEP      
            cl.cltypes.float2,  cl.cltypes.float3,  np.float32,  np.int32,  np.float32,
            # 5         6      7      8      9      10      11       12        13     14     15
            # LCELLS,   OFF,   PAR,   RHO,   ABU,   TKIN,   COLDEN,  MCOLDEN,  WTKIN  WV     CLOUD
            None,       None,  None,  None,  None,  None,   None,    None,     None,  None,  None
            ])
        else:
            kernel_colden.set_scalar_arg_dtypes([
            # 0                 1                   2            3          4         
            # DIRECTION         CENTRE              DE           NRA        STEP      
            cl.cltypes.float2,  cl.cltypes.float3,  np.float32,  np.int32,  np.float32,
            # 5      6      7      8        9        10     11    12
            # RHO,   ABU,   TKIN,  COLDEN,  MCOLDEN, WTKIN  WV    CLOUD
            None,    None,  None,  None,    None,    None,  None, None   ])
        fp_N    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)  # total N(H2)
        fp_NM   =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)  # N(mol)
        fp_T    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)  # <Tkin>
        fp_V    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)  # <V>
        ANGLE   =  INI['angle']
        tmp     =  zeros(NRA, float32)
        for de in range(NDE):
            DE      =  de-0.5*(NDE-1.0)            
            if (OCTREE>0):
                kernel_colden(queue, [GLOBAL,], [LOCAL,], direction, centre, DE, NRA, STEP, 
                LCELLS_buf, OFF_buf, PAR_buf, 
                RHO_buf, ABU_buf, TKIN_buf, COLDEN_buf, MCOLDEN_buf, WTKIN_buf, WV_buf, CLOUD_buf)
            else:
                kernel_colden(queue, [GLOBAL,], [LOCAL,], direction, centre, DE, NRA, STEP, 
                RHO_buf, ABU_buf, TKIN_buf, COLDEN_buf, MCOLDEN_buf, WTKIN_buf, WV_buf, CLOUD_buf)
            # save spectrum
            queue.finish()
            cl.enqueue_copy(queue, tmp, COLDEN_buf)
            fp_N[0].data[de,:]   =  tmp
            cl.enqueue_copy(queue, tmp, MCOLDEN_buf)
            fp_NM[0].data[de,:]  =  tmp
            cl.enqueue_copy(queue, tmp, WTKIN_buf)
            fp_T[0].data[de,:]  =  tmp
            cl.enqueue_copy(queue, tmp, WV_buf)
            fp_V[0].data[de,:]  =  tmp
        ###
        fp_N.writeto( '%s_NH2.%03d.fits'       % (INI['prefix'], iview), overwrite=True)                
        fp_NM.writeto('%s_N_%s.%03d.fits'      % (INI['prefix'], MOL.NAME, iview), overwrite=True)                
        fp_T.writeto( '%s_Tkin.%03d.fits'      % (INI['prefix'], iview), overwrite=True)
        fp_V.writeto( '%s_VLOS.%03d.fits'      % (INI['prefix'], iview), overwrite=True)
    # for iview

        

    

        
def WriteInfallIndex(INI):
    """
    Save maps of the distance (from the observer) to the LOS density maximum and 
    of the infall index =  sum(max(0, rho-rholim)*(v-v0)) / sum(max(0, rho-rholim)).
    Here rho is the volume density, rholim is some density threshold, v is the radial velocity, and
    v0 is the radial velocity at the location of the LOS density maximum.
    In the summation, the sign is changed behind the density maximum, so that the contributions
    are always positive if the LOS motion is towards the density maximum.
    Therefore, infall indices [km/s] are positive if te density-weighted motion is towards the
    density maximum.
    """
    global program, queue, LOCAL, RHO_buf    
    rholim = INI['infallindex']
    nviews = len(INI['mapview'])
    for iview in range(nviews):   # loop over map views
        print('*** MAPVIEW ', INI['mapview'][iview])
        theta, phi,  NRA, NDE,  xc, yc, zc   =   INI['mapview'][iview] 
        NRA, NDE    =  int(NRA), int(NDE)                             # 2021-07-14 - map dimensions
        DE          =  0.0    
        GLOBAL      =  IRound(NRA, LOCAL)    
        STEP        =  INI['grid'] / INI['angle']
        direction   =  cl.cltypes.make_float2(0.0,0.0)
        direction['x'], direction['y'] = theta, phi                   # 2021-07-14 - map direction
        centre       =  cl.cltypes.make_float3(0.0,0.0,0.0)
        centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ   
        if (isfinite(xc*yc*zc)):                                      # 2021-07-14 - map centre given by user
            centre['x'], centre['y'], centre['z']  =   xc, yc, zc
        print("MAP CENTRE:  %8.3f %8.3f %8.3f" % (centre['x'], centre['y'], centre['z']))
        INFALL_buf  = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # infall index (per pixel) [km/s]
        DIST_buf    = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # LOS distance to maximum [pc]
        RHOMAX_buf  = cl.Buffer(context, mf.WRITE_ONLY, 4*NDE) # LOS peak density  [cm-3]
        kernel_infall = program.LOS_infall
        if (OCTREE>0):
            kernel_infall.set_scalar_arg_dtypes([
            # rholim    D(irection)         CENTRE              DE           NRA        STEP
            np.float32, cl.cltypes.float2,  cl.cltypes.float3,  np.float32,  np.int32,  np.float32,
            # LCELLS   OFF     PAR            RHO     INFALL  DISTANCE  DIST   CLOUD 
            None,      None,   None,          None,   None,   None,     None,  None    ])
        else:
            kernel_infall.set_scalar_arg_dtypes([
            # rholim     DIRECTION           CENTRE              DE           NRA        STEP      
            np.float32,  cl.cltypes.float2,  cl.cltypes.float3,  np.float32,  np.int32,  np.float32,
            # RHO,   INFALL    DISTANCE   DIST    CLOUD
            None,    None,     None,      None,   None     ])
        fp_I    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)
        fp_D    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)
        fp_n    =  MakeEmptyFitsDim(INI['FITS_RA'], INI['FITS_DE'], INI['grid']*ARCSEC_TO_DEGREE, NRA, NDE)
        ANGLE   =  INI['angle']
        tmp     =  zeros(NRA, float32)
        for de in range(NDE):
            DE      =  de-0.5*(NDE-1.0)            
            if (OCTREE>0):
                kernel_infall(queue, [GLOBAL,], [LOCAL,], rholim, direction, centre, DE, NRA, STEP, 
                LCELLS_buf, OFF_buf, PAR_buf,    RHO_buf, INFALL_buf, DIST_buf, RHOMAX_buf, CLOUD_buf)
            else:
                kernel_infall(queue, [GLOBAL,], [LOCAL,], rholim, direction, centre, DE, NRA, STEP, 
                RHO_buf, INFALL_buf, DIST_buf, RHOMAX_buf, CLOUD_buf)
            # save spectrum
            queue.finish()
            cl.enqueue_copy(queue, tmp, INFALL_buf)
            fp_I[0].data[de,:]   =  tmp
            cl.enqueue_copy(queue, tmp, DIST_buf)
            fp_D[0].data[de,:]  =  tmp
            cl.enqueue_copy(queue, tmp, RHOMAX_buf)
            fp_n[0].data[de,:]  =  tmp
        ###
        if (nviews>1): suffix = '%.2e.%03d.fits' % (rholim,  iview)
        else:          suffix = '%.2e.fits'      % (rholim)        
        fp_I.writeto( '%s_LOS_INFALL.%s' % (INI['prefix'], suffix), overwrite=True)   #  [km/s]
        fp_D.writeto( '%s_LOS_DMAX.%s'   % (INI['prefix'], suffix), overwrite=True)   #  [pc]
        fp_n.writeto( '%s_LOS_NMAX.%s'   % (INI['prefix'], suffix), overwrite=True)   #  [cm-3]
    # for iview

        

    
    
    
#================================================================================
#================================================================================
#================================================================================
#================================================================================

    
    
        
# Main loop -- simulation and updates to level populations
max_change, Tsin, Tsol, Tsav = 0.0, 0.0, 0.0, 0.0
print("================================================================================")
for ITER in range(INI['iterations']):
    print('   ITERATION %d/%d' % (1+ITER, INI['iterations']))
    t0    =  time.time()    
    if (INI['multitran']>0):
        SimulateMultitran(INI['multitran'])      # multiple transitions per kernel call
    else:
        if (WITH_OVERLAP):
            SimulateOL()         # general line overlap
        Simulate()               # calculations one transition at a time, all HFS cases, single components
    if (0):
        print("================================================================================")
        for tran in range(TRANSITIONS):
            print("  tran = %02d    SIJ = %10.3e   ESC = %10.3e" % (tran, mean(SIJ_ARRAY[:,tran]), mean(ESC_ARRAY[:,tran])))
        print("  ")
        print("================================================================================")
        ## sys.exit()
    Tsim  =  time.time()-t0
    t0    =  time.time()
    if (INI['clsolve']):
        ave_max_change = SolveCL()
    else:
        ave_max_change = Solve(CELLS, MOL, INI, LEVELS, TKIN, RHO, ABU, ESC_ARRAY)
    Tsol  =  time.time()-t0
    t0    =  time.time()
    if (((ITER%4==3)|(ITER==(INI['iterations']-1))) & (len(INI['save'])>0)): # save level populations
        print("      ... save level populations") 
        fp = open(INI['save'], 'wb')
        asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
    Tsave = time.time()-t0
    print("       SIMULATION %7.2f    SOLVE %7.2f    SAVE %7.2f" % (Tsim, Tsol, Tsav))
    if (ave_max_change<INI['stop']):  break
print("================================================================================")


if ((INI['iterations']>0)&(COOLING)):
    Cooling()        

    
mleaf = nonzero(RHO<=0.0)
NI_ARRAY[mleaf[0], :] = np.nan

if (0):    
    clf()    
    plot(NI_ARRAY[:,0], 'b.')    
    plot(NI_ARRAY[:,1], 'r.')
    show()
    sys.exit()

    
if (1):    
    print("\n********************")
    print("***** save Tex *****")
    print("********************")
    # Save Tex files
    ul = INI['Tex']                # upper and lower level for each transition
    ## print(ul)
    if (0):        
        print("NI[15,15,15]",  NI_ARRAY.reshape(30,30,30,LEVELS)[15,15,15,:])
    for i in range(len(ul)//2):    # loop over transitions
        u, l  =  ul[2*i], ul[2*i+1]
        if (MOL.E[u]<MOL.E[l]):
            print("      warning:  %2d -> %2d  converted to %2d -> %2d for Tex calculation" % (u, l, l, u))
            tmp = u 
            u = l
            l = tmp
        tr    =  MOL.L2T(u,l)
        if (tr<0):
            print("      warning:  Tex %2d -> %2d  (E=%.3e -> %.3e K) not radiative transition" % (u, l, H_K*MOL.E[u], H_K*MOL.E[l]))
            # continue
        gg    =  MOL.G[u]/MOL.G[l]
        fp    =  open('%s_%s_%02d-%02d.tex' % (INI['prefix'], MOL.NAME, u, l), 'wb')
        asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
        tex   =  BOLTZMANN * log(NI_ARRAY[:, l]*gg/NI_ARRAY[:, u])
        if (0):
            m = nonzero((RHO>0.0)&(~isfinite(tex)))
            print("NaNs: %d" % len(m[0]))
            if (len(m[0])>10):
                print("Problem RHO percentiles:", percentile(RHO[m], (0.0, 10.0, 50.0, 90.0, 100.0)))
            elif (len(m[0])>0):
                print("Problem RHO all:", RHO[m])
            nnn = min([len(m[0]), 10])
            for i in range(nnn):
                icell = m[0][i]
                print("--- CELL %8d  RHO %.3e   n[%d] = %11.3e      n[%d] = %11.3e" % (icell, RHO[icell], l, NI_ARRAY[icell, l],   u, NI_ARRAY[icell, u]))
                sys.stdout.write("       SIJ ")
                for j in range(TRANSITIONS):
                    sys.stdout.write(' %11.3e' % (SIJ_ARRAY[icell, j]))
                sys.stdout.write('\n')
                ###
                #icell = m[0][i]+1001
                # print("+++ CELL %8d  RHO %.3e   n[%d] = %11.3e      n[%d] = %11.3e" % (icell, RHO[icell], l, NI_ARRAY[icell, l],   u, NI_ARRAY[icell, u]))
                #sys.stdout.write("       SIJ ")
                #for j in range(TRANSITIONS):
                #    sys.stdout.write(' %11.3e' % (SIJ_ARRAY[icell, j]))
                #sys.stdout.write('\n')
        m     =  nonzero(np.abs(tex)>1.0e-35)
        if (len(m[0])<len(tex)):
            print("      warning: divisions by zero => some Tex ~ 0 K values")
        if (tr>=0):  tex[m]   =  PLANCK * MOL.F[tr] / tex[m]
        else:        tex[m]   =  PLANCK * (MOL.E[u]-MOL.E[l]) / tex[m]
        asarray(tex, float32).tofile(fp)
        fp.close()
        if (OCTREE):
            m = nonzero(RHO>0.0)
            print("  TEX      %3d  = %2d -> %2d,  [%.3f,%.3f]  %.3f K" % (tr, u, l, np.min(tex[m]), np.max(tex[m]), np.mean(tex[m])))
        else:
            print("  TEX      %3d  = %2d -> %2d,  [%.3f,%.3f]  %.3f K" % (tr, u, l, np.min(tex), np.max(tex), np.mean(tex)))
    
        if (0):
            clf()
            plot(PL, tex, 'r.')
            xlabel(r'$\rm Path \/ \/ length$')
            ylabel(r'$T\rm_{ex} \/ \/ (K)$')
            title(r'$%.4f \pm %.4f$' % (mean(tex), std(tex)))
            show()
            sys.exit()
        
        

# Save spectra
print("\n************************")
print("***** save spectra *****")
print("************************")
ul = INI['spectra']
for i in range(len(ul)//2):
    tran   =  MOL.L2T(ul[2*i], ul[2*i+1])
    skip   =  False  # possibly skip transition if that is part of band = overlapping lines
    if (WITH_OVERLAP):
        for iband in range(OLBAND.BANDS):
            if (tran in OLBAND.TRAN[iband]):  skip = True
    ## if (skip): continue
    if (INI['losspectrum']>0): # we write only LOS contributions to a single spectrum
        WriteLOSSpectrum(INI, ul[2*i], ul[2*i+1])
    else:                      # normal spectral maps
        WriteSpectra(INI, ul[2*i], ul[2*i+1])
        
if (WITH_OVERLAP):             # save spectra for bands
    print("\n*************************************")
    print("***** write overlapping spectra *****")
    print("*************************************")
    bands = []
    for i in range(len(ul)//2):
        tran   =  MOL.L2T(ul[2*i], ul[2*i+1])
        for iband in range(OLBAND.BANDS):
            if (tran in OLBAND.TRAN[iband]):  bands.append(iband)
    for iband in bands:
        WriteSpectraOL(INI, OLBAND, iband)
    
        
if (INI['coldensave']>0):      # save  maps of N(H2), N(mol), Tkin, <V_LOS, mass_weighted)
    print("\n*******************************")
    print("***** save column density *****")
    print("*******************************")
    WriteColumndensity(INI)

if (INI['infallindex']>0.0):   # save  maps of N(H2), N(mol), Tkin, <V_LOS, mass_weighted)
    WriteInfallIndex(INI)

    
print("\n================================================================================")

print("LOC_OT.py TOTAL TIME: %.3f SECONDS" % (time.time()-t000))
    
#print(type(NI_ARRAY))    
#print(type(SIJ_ARRAY))    
#print(type(ESC_ARRAY))    
    
