// Natural order, Size 8 Inverse FFT Kernel.
// Written by Jason Groothuis bSc based on work by Vasily Volkov.

#include "codelets.h"

__global__ void oIFFT8_device_ps( float *ps, float2 *worksrc)
{	
    int tid = threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    worksrc +=  bid * 512;
    ps +=  bid * 512;

    float2 a[8]; 
    __shared__ float2 smem[64*9];
//    load<8>( a, worksrc, 64 );    // Replace Original
// ... instead loading to shared mem first, avoiding bank conflicts
   #pragma unroll 
	for (int i=0; i < 8; i++) smem[hi*8*9+lo+i*9] = worksrc[i*64+tid]; // Stride 64 input straight to shared memory, transposing 64x8.

// ...now load the registers from shared mem (faster)
   #pragma unroll 
	for (int i=0; i < 8; i++) a[i] = smem[tid*9+i];

// IFFT8( a );  // Replace Original , Partial de-macroing gains ~1 GFlop
    IFFT2( a[0], a[4] ); 
	IFFT2( a[1], a[5] ); 
    IFFT2( a[2], a[6] ); 
    IFFT2( a[3], a[7] ); 
	float a5x = a[5].x;
	float a6x = a[6].x;
	float a7x = a[7].x;
	a[5].x = (a5x-a[5].y)* M_SQRT1_2f; 
	a[5].y = (a5x+a[5].y)* M_SQRT1_2f;
	a[6].x = -a[6].y;
	a[6].y = a6x;
	a[7].x = (-a7x-a[7].y )* M_SQRT1_2f;
	a[7].y = ( a7x-a[7].y )* M_SQRT1_2f;
    IFFT4( a[0], a[1], a[2], a[3] );
    IFFT4( a[4], a[5], a[6], a[7] );

//store directly from the registers or via shared mem, transposing 8x64, becoming natural order power spectrum
   #pragma unroll 
	for (int i=0; i < 8; i++) smem[tid*9+i] = a[rev<8>(i)];
   #pragma unroll 
	for (int i=0; i < 8; i++) 
	{
		float2 freqData = smem[hi*8*9+lo+i*9];
		// workdst[i*64+tid] = smem[hi*8*9+lo+i*9];  // stride 64, ~72 GFlops
		// PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
        ps[i*64+tid] = freqData.x * freqData.x + freqData.y * freqData.y;
	}

}	

extern "C" void oIFFT8ps( float *ps, float2 *worksrc, int batch )
{	
    oIFFT8_device_ps<<< grid2D(batch/64), 64 >>>( ps, worksrc );
}	
