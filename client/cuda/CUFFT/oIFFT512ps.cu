#include "codelets.h"

__global__ void oIFFT512_device_ps( float *ps, float2 *src )
{	
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;
    
    int index = (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
	src += index;
	ps += index;

    float2 a[8];
    __shared__ float smem[8*8*9];
    
    load<8>( a, src, 64 );

    IFFT8( a );
	
    itwiddle<8>( a, tid, 512 );
    transpose<8>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );
	
    IFFT8( a );
	
    itwiddle<8>( a, hi, 64);
    transpose<8>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );
    
    IFFT8( a );

    //store<8>( a, work, 64 );

	#pragma unroll
    for( int i = 0; i < 8; i++ )
	{
		float2 freqData = a[rev<8>(i)];
		// PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
        ps[i*64] = (freqData.x * freqData.x + freqData.y * freqData.y);
	}
}	
    
extern "C" void oIFFT512ps( float *ps, float2 *worksrc, int batch )
{	
    oIFFT512_device_ps<<< grid2D(batch), 64 >>>( ps, worksrc );
}	
