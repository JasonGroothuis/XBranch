#include "cudaAcceleration.h"

#ifdef USE_CUDA
#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

#define B 2

#if 1
__global__ void cudaAcc_GPS_kernel_mod3SM( float2* FreqData, float* PowerSpectrum) {
	int iblock = blockIdx.x + blockIdx.y * gridDim.x;
	int sidx = threadIdx.x + B*iblock*blockDim.x; 

	float2 a[B];

#pragma unroll
	for (int i=0;i<B;i++) {		a[i] = FreqData[sidx+i*blockDim.x]; }

#pragma unroll
	for (int i=0;i<B;i++)PowerSpectrum[sidx+i*blockDim.x] =
			__fadd_rn( __fmul_rn(a[i].x,a[i].x),__fmul_rn(a[i].y,a[i].y)); 
}

#else
__global__ void cudaAcc_GPS_kernel_mod3( int NumDataPoints, float2* FreqData, float* PowerSpectrum) {
	const int sidx = (blockIdx.x * blockDim.x + threadIdx.x); 

	float ax,ay;

	if ( sidx < NumDataPoints )
	{
		ax = FreqData[sidx].x;
		ay = FreqData[sidx].y;
		PowerSpectrum[sidx] =  __fadd_rn( __fmul_rn(ax,ax),__fmul_rn(ay,ay)); 
	}
}
#endif

void cudaAcc_GetPowerSpectrum(int numpoints, int offset, cudaStream_t str) {
	dim3 block(256, 1, 1);
//	dim3 grid((numpoints + (block.x*B) - 1) / (block.x*B), 1, 1);
	dim3 grid = grid2D((numpoints + (block.x*B) - 1) / (block.x*B));

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_GPS_kernel_mod3SM<<<grid, block, 0, str>>>(dev_WorkData+offset, dev_PowerSpectrum+offset)),true);
}

#endif //USE_CUDA