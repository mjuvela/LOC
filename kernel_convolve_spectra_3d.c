


__kernel void Convolve3D(__global float *SPE, __global float *CON) {
   // SPE[NCHN, NY, NX] convolved with beam with FWHM [pixels].
   const int id = get_global_id(0) ;
   if (id>=(NX*NY)) return ;
   int   i0 = id % NX ;        // one spatial pixel per work item
   int   j0 = id / NX ;
   float K  = -4.0f*log(2.0f)/(FWHM*FWHM) ;
   int   d  = floor(3.9f*FWHM) ;
   float w, W, r2, Z ;
   __global float *S ;
   
   for(int k=0; k<NCHN; k++) {
      S = &(SPE[k*NY*NX]) ;   // image for single velocity channel
      W = 0.0f ;   Z = 0.0f ;      
      for(int i=max(0, i0-d); i<=min(NX-1, i0+d); i++) {
	 for(int j=max(0, j0-d); j<=min(NY-1, j0+d); j++) {	    
	    r2  =  (i-i0)*(i-i0) + (j-j0)*(j-j0) ;
	    w   =  exp(K*r2) ;              // Gaussian weight
	    W  +=  w ;
	    Z  +=  w * S[i+NX*j] ;
	 }
      }
      CON[k*NY*NX + i0 + j0*NX] = Z / W ;
   }
}
	      
