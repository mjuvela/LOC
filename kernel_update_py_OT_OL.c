
#if (WITH_OVERLAP>0)


__kernel void 
SpectraOL(  // both Cartesian and Octree
            const       int      NCMP,         //   number of components
            const       int      NCHN,
            __constant  int     *TRAN,         //  TRAN[NCMP] tran numbers
            __constant  float   *COFF,         //  COFF[NCMP] channel offsets for components
            __global    float   *TAU_ARRAY,    //  wrk  [NRA, NCHN]
            __global    float   *EMIT_ARRAY,   //  wrk  [NRA, NCHN]
            __global    float4  *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
            GAUSTORE    float   *GAU,          //  1 GAU[GNO, CHANNELS]
            __global    int2    *LIM,          //  2 channel limits LIM[GNO]
            __global    float   *GN,           //  3 Gauss normalisation GN[ntran]
            const       float2   D,            //  4 ray direction == theta, phi
            __global    float   *NI,           //  5 NI[NCMP, CELLS] + NBNB[NCMP,CELLS]
            const       float    DE,           //  6 grid units, single offset
            const       int      NRA,          //  7 number of RA points = work items
            const       float    STEP,         //  8 step between spectra (grid units)
            const       float    BG,           //  9 background intensity
            __global    float   *emit0,        // 10 h/(4pi)*freq*Aul*int2temp  emit0[NCMP]
            __global    float   *NTRUE_ARRAY,  // 11 [NRA, NCHN] -- spectrum 
            __global    float   *STAU_ARRAY,   // 12 [NRA, NCHN] -- total optical depths
            __global    float   *TT_ARRAY,     //    [NRA, NCHN] -- wrk (1-exp(-tau))/tau
            const       float3   CENTRE        // 18, 20  map centre in current                          
# if (WITH_OCTREE>0)
            ,
            __global    int     *LCELLS,       // 13, 15
            __constant  int     *OFF,          // 14, 16
            __global    int     *PAR,          // 15, 17
            __global    float   *RHO           // 16, 18
# endif            
         )
{
   // each work item calculates a spectrum for one RA offset => kernel call has fixed DE
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int id = get_global_id(0) ;
   if (id>=NRA) return ; // no more rays
   
# if 0
   if (id==0) {
      for(int icmp=0; icmp<NCMP; icmp++) {
         printf("COFF[%d] = %4.1f\n", icmp, COFF[icmp]) ;
      }
   }
   printf("NCHN = %d\n", NCHN) ;
# endif
   
   __global float *NTRUE   = &(NTRUE_ARRAY[id*NCHN]) ;
   __global float *EMIT    = &(EMIT_ARRAY[ id*NCHN]) ;
   __global float *TAU     = &(TAU_ARRAY[  id*NCHN]) ;
   __global float *STAU    = &(STAU_ARRAY[ id*NCHN]) ;
   __global float *TT      = &(TT_ARRAY[   id*NCHN]) ;
   GAUSTORE float *profile ;
   int   i, itran ;
   float RA, maxi ;       // grid units, offset of current ray
   RA  =   id  ;
   // calculate the initial position of the ray
   REAL3   POS, dr, RPOS ;
   float3  DIR ;
   REAL    dx, dy, dz ;
   DIR.x   =  sin(D.x)*cos(D.y) ;     // D.x = theta,   D.y = phi
   DIR.y   =  sin(D.x)*sin(D.y) ;
   DIR.z   =  cos(D.x)            ;
   REAL3 RV, DV ; 
   // Definition:  DE follows +Z, RA is now right
   if (DIR.z>0.9999f) {
      RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;    // RA = Y
      DV.x=-0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ;    // DE = -X
   } else {
      if (DIR.z<-0.9999f) {                              // view from -Z =>  (Y,X)
         RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;
         DV.x=+0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ; 
      } else {
         // RA orthogonal to DIR and to +Z,   DIR=(1,0,0) => RV=(0,+1,0)
         //                                   DIR=(0,1,0) => RV=(-1,0,0)
         RV.x = -DIR.y ;   RV.y = +DIR.x ;  RV.z = ZERO ;  RV = normalize(RV) ;
         // DV  =   RV x DIR
         DV.x = -RV.y*DIR.z+RV.z*DIR.y ;
         DV.y = -RV.z*DIR.x+RV.x*DIR.z ;
         DV.z = -RV.x*DIR.y+RV.y*DIR.x ;
      }
   }   
   // Offsets in RA and DE directions, (RA, DE) are just indices [0, NRA[, [0,NDE[
   // CENTRE are the indices for the map centre using the current pixel size
   // POS is already at the map centre
   POS.x  =  CENTRE.x + (RA-0.5*(NRA-1.0f))*STEP*RV.x + DE*STEP*DV.x ;
   POS.y  =  CENTRE.y + (RA-0.5*(NRA-1.0f))*STEP*RV.y + DE*STEP*DV.y ;
   POS.z  =  CENTRE.z + (RA-0.5*(NRA-1.0f))*STEP*RV.z + DE*STEP*DV.z ;      
   // int ID = ((fabs(POS.y-1.5)<0.02)&&(fabs(POS.z-0.7)<0.02)) ? id : -1 ;
   int ID = ((fabs(POS.y-2.0)<0.05)&&(fabs(POS.z-1.5)<0.02)) ? id : -1 ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;   
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;   
   // go to front surface, first far enough upstream (towards observer), then step forward to cloud (if ray hits)
   POS.x -= 1000.0*DIR.x ;  POS.y -= 1000.0*DIR.y ;  POS.z -= 1000.0*DIR.z ;
   if (DIR.x>ZERO)  dx = (ZERO-POS.x)/DIR.x ;
   else             dx = (NX  -POS.x)/DIR.x ;
   if (DIR.y>ZERO)  dy = (ZERO-POS.y)/DIR.y ;
   else             dy = (NY  -POS.y)/DIR.y ;
   if (DIR.z>ZERO)  dz = (ZERO-POS.z)/DIR.z ;
   else             dz = (NZ  -POS.z)/DIR.z ;
   dx      =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS.x  +=  dx*DIR.x ;   POS.y  +=  dx*DIR.y ;   POS.z  +=  dx*DIR.z ;   // even for OT, still in root grid units
   
   int level0 ;
   
# if (WITH_OCTREE>0)
   int OTL, OTI, INDEX ;
   // Note: for OCTREE=5,  input is [0,NX], output coordinates are [0,1] for root-grid cells
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
   INDEX =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;
# else
   int INDEX   =  Index(POS) ;
# endif
   
   
   float tmp, emissivity, doppler, nu, sst, gamma ;
   int row, shift, c1, c2  ;
# if (WITH_CRT>0)
   float Ctau, Cemit, pro, distance=0.0f ;
# endif   
   float distance = 0.0f ;
   
   for(int i=0; i<NCHN; i++) {
      NTRUE[i] = 0.0f ;
      STAU[i]  = 0.0f ;
   }
   
   
   while (INDEX>=0) {      
# if (WITH_OCTREE>0)   // INDEX  =  OFF[OTL] + OTI ;  --- update INDEX at the end of the step
      // OCTREE==5 uses level=0 coordinates POS=[0,1], not [0,NX]
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL, -1) ; // updates POS, OTL, OTI
      // RPOS = POS ; RootPos(&RPOS, OTL, OTI, OFF, PAR) ;
# else
      if (DIR.x<0.0f)   dx = -     fmod(POS.x,ONE)  / DIR.x - EPS/DIR.x;
      else              dx =  (ONE-fmod(POS.x,ONE)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -     fmod(POS.y,ONE)  / DIR.y - EPS/DIR.y;
      else              dy =  (ONE-fmod(POS.y,ONE)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -     fmod(POS.z,ONE)  / DIR.z - EPS/DIR.z;
      else              dz =  (ONE-fmod(POS.z,ONE)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;      // actual step
# endif
      
      doppler    =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      profile    =  &GAU[row*CHANNELS] ; // CHANNELS <= NCHN
      
      for(int i=0; i<NCHN; i++) { 
         TAU[i]  = 0.0f ; 
         EMIT[i] = 0.0f ;  
      }
      
      for(int icmp=0; icmp<NCMP; icmp++) {         
         itran   =  TRAN[icmp] ;
         tmp     =  NI[NCMP*CELLS + icmp*CELLS + INDEX] ;    // NBNB[NCMP,CELLS]
         tmp     =  (fabs(tmp)<1.0e-30f) ? (dx*1.0e-30f*GN[icmp]*GL) : clamp((float)(dx*tmp*GN[icmp]*GL), -2.0f, 1.0e10f) ;
         tmp     =  clamp(tmp, 1.0e-30f, 1.0e10f) ;  // $$$  KILL ALL MASERS
         shift   =  round(doppler/WIDTH + COFF[icmp]) ;
         int i1  =  max(-shift,      LIM[row].x) ;
         int i2  =  min(NCHN-shift,  LIM[row].y) ;
         for(int i=i1; i<i2; i++)    TAU[ i+shift]  +=  tmp*profile[i] ;  // TAU 
         nu      =  NI[icmp*CELLS+INDEX] ;     // NI[NCMP, CELLS]
         tmp     =  emit0[icmp] * nu * dx * GL * GN[icmp]  ;
         for(int i=i1; i<i2; i++)    EMIT[i+shift]  +=  tmp*profile[i] ;  // EMIT 
      }
      for(i=0; i<NCHN; i++) {         
         tmp        =  TAU[i] ;
         tmp        =  (fabs(tmp)>1.0e-5f) ? ((1.0f-exp(-tmp))/tmp) : (1.0f-0.5f*tmp) ;
         NTRUE[i]  +=  EMIT[i] * tmp * exp(-STAU[i]) ;
         STAU[i]   +=  TAU[i] ;
      }
      
      distance  += dx ;                  
      
# if (WITH_OCTREE>0)
      INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
# else
      POS.x  += dx*DIR.x ;  POS.y  += dx*DIR.y ;  POS.z  += dx*DIR.z ;
      INDEX = Index(POS) ;         
# endif
   } // while INDEX>=0
   
   
   for (i=0; i<NCHN; i++) {
      tmp       =  STAU[i] ;
      NTRUE[i] -=  BG * ((fabs(tmp)>0.01f ) ?  (1.0f-exp(-tmp))  :  (tmp*(1.0f-tmp*(0.5f-0.166666667f*tmp)))) ;
   } // for i over CHANNELS
   
# if 0   
   if (id==0) {
      for(int icmp=0; icmp<NCMP; icmp++) {
         printf("COFF[%d] = %4.1f\n", icmp, COFF[icmp]) ;
      }
   }
# endif
   
}









# if (WITH_OCTREE==0)


//  @u  Cartesian grid, PL not used, only APL
__kernel void UpdateOL(const int           NCMP,       //   0 - number of spectral components
                       const int           NCHN,       //   1
                       __global   float   *Aul,        //   2 - Aul[NCMP]   Einstein A(upper->lower)
                       __global   float   *Ab,         //   3 - Ab[NCMP]    (g_u/g_l)*B(upper->lower)
                       const      float    GN,         //   4 - GN  Gauss normalisation == C_LIGHT/(1e5*DV*freq) ~ constant for band
                       __global   float   *COFF,       //   5 - COFF[NCMP]
                       __global   float   *NI,         //   6 - NI[2,NCMP,CELLS]  =  { nu, NBNB }
                       __global   float   *NTRUES,     //   7 - NTRUES[GLOBAL, MAXCHN]
                       const      int      id0,        //   8 - index of the first ray
                       __global   float4  *CLOUD,      //   9 - [CELLS]: vx, vy, vz, sigma
                       GAUSTORE   float   *GAU,        //  10 - precalculated gaussian profiles [GNO,CHANNELS]
                       __global   int2    *LIM,        //  11 - limits of ~zero profile function [GNO]
                       __global   float   *PL,         //  12 - just for testing
                       const      float    APL,        //  13 - average path length [GL]
                       const      float    BG,         //  14 - background value (photons)
                       const      float    DIRWEI,     //  15 - <cos(theta)> for rays entering leading edge
                       const      float    EWEI,       //  16 - 1/<1/cosT>/NDIR
                       const      int      LEADING,    //  17 - leading edge
                       const      REAL3    POS0,       //  18 - initial position of id=0 ray
                       const      float3   DIR,        //  19 - ray direction
                       __global   float   *RES,        //  20 - [2,NCMP,CELLS]:  SIJ, ESC
                       __global   float   *TAU_ARRAY,  //  21 - TAU_ARRAY[GLOBAL,MAXCHN]
                       __global   float   *EMIT_ARRAY, //  22 - TAU_ARRAY[GLOBAL,MAXCHN]
                       __global   float   *TT_ARRAY    //  23 - TT_ARRAY[ GLOBAL,MAXCHN]
                      )  {
   float weight ;        
   float dx, doppler, w ;
#  if 0 // DOUBLE
   double  wd, tau, emit, nb_nb, factor, escape, absorbed ;
#  else
   float  tau, emit, nb_nb, factor, escape, absorbed ;
#  endif

   // printf("Aul %.3e Ab %.3e GN %.3e BG %.3e DIRWEI %.3e EWEI %.3e\n", Aul[0], Ab[0], GN, BG, DIRWEI, EWEI) ;
   
   float sij[MAXCMP], esc[MAXCMP] ;
   int  row, shift, INDEX, c1, c2 ;
   int  id  = get_global_id(0) ;
   if  (id>=GLOBAL) return ;
   int steps = 0 ;
   // if (id>0) return ;
   
   
   __global float *NTRUE = &(    NTRUES[id*NCHN]) ;   // NTRUES[GLOBAL, MAXCHN]
   __global float *TAU   = &( TAU_ARRAY[id*NCHN]) ;   //    TAU[GLOBAL, MAXCHN]
   __global float *TT    = &(  TT_ARRAY[id*NCHN]) ;   //     TT[GLOBAL, MAXCHN]
   __global float *EMIT  = &(EMIT_ARRAY[id*NCHN]) ;   //   EMIT[GLOBAL, MAXCHN]
   
   
   id += id0 ;  // from this point on, id is the ray number, not the work item number
   if (id>=NRAY) return ;
   

#  if (ONESHOT<1)
   int nx = (NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ; // dimensions of the current ray grid
#  endif
   GAUSTORE float *profile ;
   
   
   // Initial position of each ray shifted by two grid units == host has a loop over 4 offset positions
   REAL3 POS = POS0 ;
#  if (FIX_PATHS>0)
   REAL3 POS_LE ;
   int count = 0 ;
#  endif
      
   
#  if (ONESHOT<1)
   switch (LEADING) {
    case 0:   POS.x =    EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;      break ;
    case 1:   POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;      break ;
    case 2:   POS.y =    EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;      break ;
    case 3:   POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;      break ;
    case 4:   POS.z =    EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;      break ;
    case 5:   POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;      break ;
   }
#  else
   switch (LEADING) {
    case 0:   POS.x =    EPS ;  POS.y += (id%NY) ;   POS.z += (int)(id/NY) ;      break ;
    case 1:   POS.x = NX-EPS ;  POS.y += (id%NY) ;   POS.z += (int)(id/NY) ;      break ;
    case 2:   POS.y =    EPS ;  POS.x += (id%NX) ;   POS.z += (int)(id/NX) ;      break ;
    case 3:   POS.y = NY-EPS ;  POS.x += (id%NX) ;   POS.z += (int)(id/NX) ;      break ;
    case 4:   POS.z =    EPS ;  POS.x += (id%NX) ;   POS.y += (int)(id/NX) ;      break ;
    case 5:   POS.z = NZ-EPS ;  POS.x += (id%NX) ;   POS.y += (int)(id/NX) ;      break ;
   }
#  endif     

#  if (FIX_PATHS>0)
   POS_LE = POS ;
#  endif   

   INDEX = Index(POS) ;   
   // BG       =  average number of photons per ray
   // DIRWEI   =  cos(theta) / <cos(theta)>,   weight for current direction relative to average
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]  =  BG * DIRWEI ; // DIRWEI ~ cos(theta) / <cos(theta)>
   }


   
   while(INDEX>=0) {      
      
#  if (NX>DIMLIM) // ====================================================================================================
      double dx, dy, dz ;
      dx = (DIR.x>0.0f)  ?  ((1.0+DEPS-fmod(POS.x,ONE))/DIR.x)  :  ((-DEPS-fmod(POS.x,ONE))/DIR.x) ;
      dy = (DIR.y>0.0f)  ?  ((1.0+DEPS-fmod(POS.y,ONE))/DIR.y)  :  ((-DEPS-fmod(POS.y,ONE))/DIR.y) ;
      dz = (DIR.z>0.0f)  ?  ((1.0+DEPS-fmod(POS.z,ONE))/DIR.z)  :  ((-DEPS-fmod(POS.z,ONE))/DIR.z) ;
      dx = min(dx, min(dy, dz)) ;
#  else
      dx=        (DIR.x<0.0f) ? (-fmod(POS.x,ONE)/DIR.x-EPS/DIR.x) : ((ONE-fmod(POS.x,ONE))/DIR.x+EPS/DIR.x) ;
      dx= min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,ONE)/DIR.y-EPS/DIR.y) : ((ONE-fmod(POS.y,ONE))/DIR.y+EPS/DIR.y)) ;
      dx= min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,ONE)/DIR.z-EPS/DIR.z) : ((ONE-fmod(POS.z,ONE))/DIR.z+EPS/DIR.z)) ;
#  endif
      
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
      profile   =  &GAU[row*CHANNELS] ;
      weight    =  (dx/APL)*VOLUME ;
      
      for(int i=0; i<NCHN; i++) {
         TAU[i]    =  1.0e-20f ;    // avoid later division by zero
         EMIT[i]   =  0.0f ;
      }

      for(int icmp=0; icmp<NCMP; icmp++) {
         tau      =  NI[NCMP*CELLS+icmp*CELLS+INDEX] * GN*dx ;
         emit     =  weight *   NI[icmp*CELLS+INDEX] * Aul[icmp] ;
         shift    =  round(doppler/WIDTH + COFF[icmp]) ;
         int i1   =  max(    -shift,  LIM[row].x) ;
         int i2   =  min(NCHN-shift,  LIM[row].y) ;
         for(int i=i1; i<i2; i++) {                    // over profile
            TAU[i+shift]  +=  tau  * profile[i] ;      // sum of all lines
            EMIT[i+shift] +=  emit * profile[i] ;      // photons within one channel
         }         
      }      

      // calculate (1-exp(-TAU))/TAU for all channels
      for(int i=0; i<NCHN; i++) {
         TT[i] =  (fabs(TAU[i])>1.0e-3f)  ?  ((1.0f-exp(-TAU[i]))/TAU[i])  :  (1.0f-0.5f*TAU[i])  ;
      }            

      // sij, esc for current step, all components
      for(int icmp=0; icmp<NCMP; icmp++) {    
         sij[icmp] = 0.0f ;  
         esc[icmp] = 0.0f ;  
      }
      
      // count absorptions of incoming photons =  BLU*ntrue*phi*(1-exp(-tau))/tau
      for(int icmp=0; icmp<NCMP; icmp++) {                // loop over NCMP components
         shift    =  round(doppler/WIDTH + COFF[icmp]) ;  // offset, current component
         int i1   =  max(    -shift,  LIM[row].x) ;
         int i2   =  min(NCHN-shift,  LIM[row].y) ;
         for(int i=i1; i<i2; i++) {              // over the profile of this transition
            sij[icmp]  +=  NTRUE[i+shift] * Ab[icmp]*GN*dx*profile[i]*TT[i+shift] ;
         }         
      }
            
      // emission-absorption
      for(int icmp=0; icmp<NCMP; icmp++) {
         shift     =  round(doppler/WIDTH + COFF[icmp]) ;                  
         int i1    =  max(    -shift,  LIM[row].x) ;
         int i2    =  min(NCHN-shift,  LIM[row].y) ;
         for(int i=i1; i<i2; i++) {
            tau        =  NI[NCMP*CELLS+icmp*CELLS+INDEX] * GN*dx    *profile[i] ;   // NBNB*GN*dx*profile
            emit       =  weight*NI[    icmp*CELLS+INDEX] * Aul[icmp]*profile[i] ;   // weight*nu*Aul*profile
            // this sij==0 for a single component (test case...)
            sij[icmp] +=  (EMIT[i+shift]-emit) * Ab[icmp]*GN*profile[i] * ((1.0f-TT[i+shift]) / TAU[i+shift]) ;
            esc[icmp] +=  emit * ( 1.0f   -   (1.0f-TT[i+shift]) * (tau/TAU[i+shift]) ) ;
         }
      } // for icmp

      // update NTRUE
      for(int i=0; i<NCHN; i++) {
         NTRUE[i]  =  NTRUE[i] * exp(-TAU[i])   +    EMIT[i] * TT[i] ;
      }           
      
#  if 0      
      if (id==0) {
         //   RES[2, NCMP, CELLS]  = { sij, esc }
         //           sij            dsij      tau                                   nt
         printf("@    %12.4e         %12.4e    %12.4e                                %12.4e         %12.4e\n", 
                RES[0*CELLS+INDEX],  sij[0],   1.01*NI[NCMP*CELLS+0*CELLS+INDEX] * GN*dx, NTRUE[NCHN/2],
                EMIT[NCHN/2]*TT[NCHN/2]) ;
         steps += 1 ;
         for(int i=0; i<NCHN; i++) {
            //                                          0      1  2         3       4
            tau = NI[NCMP*CELLS+0*CELLS+INDEX] * GN*dx  * profile[i] ;
            printf("& %2d %3d  %12.5e %12.5e %12.5e\n", steps, i, NTRUE[i], tau, profile[i]) ;
         }
         if (steps>5)  return ;    // NI[2, NCMP, CELLS]
      }
#  endif
      
      // sij => global array   RES[2, NCMP, CELLS]
      for(int icmp=0; icmp<NCMP; icmp++) {
#  if (0)
         RES[             icmp*CELLS+INDEX] += sij[icmp] ;  // RES[2, NCMP, CELLS]
         RES[NCMP*CELLS + icmp*CELLS+INDEX] += esc[icmp] ;
#  else         
         // RES[2, NCMP, CELLS]
         AADD(&(RES[             icmp*CELLS+INDEX]),  sij[icmp]) ;
         AADD(&(RES[NCMP*CELLS + icmp*CELLS+INDEX]),  esc[icmp]) ;       
#  endif
      }
            
      POS.x += dx*DIR.x ;  POS.y += dx*DIR.y ;  POS.z += dx*DIR.z ;
      
#  if 0  // testing, is the path length the same as in Paths
      AADD(&(PL[INDEX]), -dx) ;
#  endif
      
      
#  if (FIX_PATHS>0)
      not used
        // try to precise the position
        count += 1 ;
      if (count%7==2) {
         if (LEADING<2) {
            float s =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y   =  POS_LE.y + s*DIR.y ;
            POS.z   =  POS_LE.z + s*DIR.z ;
            if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else { 
            if (LEADING<4) {
               float s =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.z   =  POS_LE.z + s*DIR.z ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
            } else {
               float s =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.y   =  POS_LE.y + s*DIR.y ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
            }
         }
      }
#  endif
      
      
      INDEX      = Index(POS) ;
      
      if (INDEX<0) {  
         // exits the cloud... but on which side?
         // even when the ray is now entering via non-leading edge (= one of the sides), 
         // DIRWEI is still the same (the rays will hit the side walls at correspondingly larger steps)
         if (POS.x>=NX)   {   if (LEADING!=0)    POS.x =    EPS ;    } ;
         if (POS.x<=ZERO) {   if (LEADING!=1)    POS.x = NX-EPS ;    } ;
         if (POS.y>=NY)   {   if (LEADING!=2)    POS.y =    EPS ;    } ;
         if (POS.y<=ZERO) {   if (LEADING!=3)    POS.y = NY-EPS ;    } ;
         if (POS.z>=NZ)   {   if (LEADING!=4)    POS.z =    EPS ;    } ;
         if (POS.z<=ZERO) {   if (LEADING!=5)    POS.z = NZ-EPS ;    } ;
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<NCHN; ii++)  NTRUE[ii] = BG * DIRWEI ;
         }
      } // if INDEX<0

      
#  if 0
      if (id==1) {
         printf("INDEX %6d   ----- SIJ[0] %12.4e\n", INDEX, RES[0]) ;
      }
#  endif
      
   } // while INDEX>=0
   
}



# endif  // WITH_OCTREE==0


#endif // WITH_OVERLAP
