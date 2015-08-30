#include "codelets.h"

__global__ void oIFFT1024_device_ps( float *ps, float2 *src )
{	
    int tid = threadIdx.x;
    
    int iblock = blockIdx.y * gridDim.x + blockIdx.x;
    int index = iblock * 1024 + tid;
    src += index;
    ps += index;
    
    int hi4 = tid>>4;
    int lo4 = tid&15;
    int hi2 = tid>>4;
    int mi2 = (tid>>2)&3;
    int lo2 = tid&3;

    float2 a[16];
    __shared__ float smem[69*16];
    
    load<16>( a, src, 64 );

    IFFT16( a );
    
    itwiddle<16>( a, tid, 1024 );
    int il[] = {0,1,2,3, 16,17,18,19, 32,33,34,35, 48,49,50,51};
    transpose<16>( a, &smem[lo4*65+hi4], 4, &smem[lo4*65+hi4*4], il );
    
    IFFT4x4( a );

    itwiddle4x4( a, lo4 );
    transpose4x4( a, &smem[hi2*17 + mi2*4 + lo2], 69, &smem[mi2*69*4 + hi2*69 + lo2*17 ], 1, 0xE );
    
    IFFT16( a );

    //store<16>( a, dst, 64 );
	#pragma unroll
    for( int i = 0; i < 16; i++ )
	{
		 float2 freqData = a[rev<16>(i)];
		// PowerSpectrum[i] = freqData.x * freqData.x + freqData.y * freqData.y;
        ps[i*64] = freqData.x * freqData.x + freqData.y * freqData.y;
	}
}   
    
extern "C" void oIFFT1024ps( float *ps, float2 *src, int batch )
{	
    oIFFT1024_device_ps<<< grid2D(batch), 64 >>>( ps, src );
}	
