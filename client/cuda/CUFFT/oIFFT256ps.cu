#include "codelets.h"

__global__ void oIFFT256_device_ps( float *ps, float2 *src )
{	
    int tid = threadIdx.x;
    int hi = tid>>4;
    int lo = tid&15;
    
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * 1024 + lo + hi*256;
    src += index;
    ps += index;
	
    //
    //  no sync in transpose is needed here if warpSize >= 32
    //  since the permutations are within-warp
    //
    
    float2 a[16];
    __shared__ float smem[64*17];
    
    load<16>( a, src, 16 );

    IFFT16( a );
    
    itwiddle<16>( a, lo, 256 );
    transpose<16>( a, &smem[hi*17*16 + 17*lo], 1, &smem[hi*17*16+lo], 17, 0 );
    
    IFFT16( a );

//    store<16>( a, dst, 16 );

//	#pragma unroll
//    for( int i = 0; i < n; i++ )
//        x[i*sx] = a[rev<n>(i)];

	#pragma unroll
    for( int i = 0; i < 16; i++ )
	{
		 float2 freqData = a[rev<16>(i)];
		// PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
        ps[i*16] = freqData.x * freqData.x + freqData.y * freqData.y;
	}
}	
    
extern "C" void oIFFT256ps( float *ps, float2 *worksrc, int batch )
{	
    oIFFT256_device_ps<<< grid2D(batch/4), 64 >>>( ps, worksrc );
}	
