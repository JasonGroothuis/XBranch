#include "cudaAcceleration.h"

#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

#define B 2

float2* dev_AutoCorrIn;
float2* dev_AutoCorrOut;
bool gCudaAutocorrelation = false;

float3 *dev_blockSums;
float3 *blockSums;
float3 *dev_ac_partials;

//float ac_TotalSum;
//float ac_Peak;
//int ac_PeakBin;

__global__ void ac_RepackInputKernel( float* PowerSpectrum, float2* dct_In ) 
{
	int sidx = threadIdx.x + (blockDim.x*blockIdx.x*B);
	int nDestPoints = (gridDim.x*blockDim.x*B)*4;
	int didx1 = sidx*2; 
	int didx2 = (nDestPoints-2)-didx1; 

	float a[B];
#pragma unroll
	for (int i=0;i<B;i++)
	{
		a[i] = PowerSpectrum[sidx+i*blockDim.x];
	}

#pragma unroll
	for (int i=0;i<B;i++)
	{
		dct_In[didx1+2*i*blockDim.x  ] = make_float2(0.0f,0.0f);
		dct_In[didx1+2*i*blockDim.x+1] = make_float2(a[i],0.0f);
		dct_In[didx2-(2*i*blockDim.x)  ] = make_float2(0.0f,0.0f);
		dct_In[(didx2-(2*i*blockDim.x))+1] = make_float2(a[i],0.0f);
	}
}

__global__ void ac_RepackScaleKernel( float2* src, float2* dst) 
{
	int sidx = ((threadIdx.x + blockDim.x*blockIdx.x*B)*2);
	int didx = ((threadIdx.x + blockDim.x*blockIdx.x*B));  //packing into float2s

	float a[B],b[B];

#pragma unroll
	for (int i=0;i<B;i++)
	{
		a[i] = src[sidx+i*blockDim.x*2].x;
		b[i] = src[sidx+i*blockDim.x*2+1].x;
		a[i] = __fmul_rn(a[i],a[i]);
		b[i] = __fmul_rn(b[i],b[i]);
	}
#pragma unroll
	for (int i=0;i<B;i++)
	{
		dst[didx+i*blockDim.x] = make_float2(a[i],b[i]);
	}
}

extern __shared__ float3 acpartial[];

__global__ void ac_reducePartial( float *ac, float3 *devpartials )
{
    const int tid = threadIdx.x;
	const int idx = threadIdx.x + blockIdx.x*blockDim.x;
	const int bid = blockIdx.x;
	int n = blockDim.x>>1;
	
	volatile float3 *acp = acpartial;
	acp[tid].x = ac[idx];
	if ( idx >= 1 )
	{
		acp[tid].y = ac[idx]; 
		acp[tid].z = idx; 
	} else {
		acp[tid].y = 0.0f; 
		acp[tid].z = idx; 
	}
	
	__syncthreads();

	for (;n>32;n>>=1)
	{
		if ( tid < n )
		{
			acp[tid].x = __fadd_rn(acp[tid].x,acp[tid + n].x);
			// peak power & its bin
			float pp = acp[tid+n].y;
			float pb = acp[tid+n].z;
			if ( pp > acp[tid].y )
			{
				acp[tid].y = pp;
				acp[tid].z = pb;
			}
		}
		__syncthreads();
	}
	for (;n>0;n>>=1)
	{
		if ( tid < n )
		{
			acp[tid].x = __fadd_rn(acp[tid].x,acp[tid + n].x);
			// peak power & its bin
			float pp = acp[tid+n].y;
			float pb = acp[tid+n].z;
			if ( pp > acp[tid].y )
			{
				acp[tid].y = pp;
				acp[tid].z = pb;
			}
		}
	}
	if (tid == 0) {        
		devpartials[bid] = make_float3(acp[tid].x,acp[tid].y,acp[tid].z);
    }
}

#if 0
__global__ void ac_reduceMore( float3 *devpartials, int nblks )
{
    const int tid = threadIdx.x;
	const int idx = threadIdx.x + blockIdx.x*blockDim.x;
	const int bid = blockIdx.x;
	int n = blockDim.x>>1;
	
	if ( idx >= nblks ) return;

	volatile float3 *acp = acpartial;
	acp[tid].x = devpartials[idx].x;
	acp[tid].y = devpartials[idx].y;
	acp[tid].z = devpartials[idx].z;
	
	__syncthreads();

	for (;n>32;n>>=1)
	{
		if ( tid < n )
		{
			acp[tid].x = __fadd_rn(acp[tid].x,acp[tid + n].x);
			// peak power & its bin
			float pp = acp[tid+n].y;
			float pb = acp[tid+n].z;
			if ( pp > acp[tid].y )
			{
				acp[tid].y = pp;
				acp[tid].z = pb;
			}
		}
		__syncthreads();
	}
	for (;n>0;n>>=1)
	{
		if ( tid < n )
		{
			acp[tid].x = __fadd_rn(acp[tid].x,acp[tid + n].x);
			// peak power & its bin
			float pp = acp[tid+n].y;
			float pb = acp[tid+n].z;
			if ( pp > acp[tid].y )
			{
				acp[tid].y = pp;
				acp[tid].z = pb;
			}
		}
	}
	if (tid == 0) {        
		devpartials[bid] = make_float3(acp[tid].x,acp[tid].y,acp[tid].z);
    }
}
#endif

int cudaAcc_FindAutoCorrelation(float *AutoCorrelation, int ac_fftlen, int fft_num ) 
{
  //Jason: Use 4N-FFT method for Type 2 Discrete Cosine Tranform for now, to match fftw's REDFT10
  // 1 Autocorrelation from global powerspectrum at fft_num*ac_fft_len  (fft_num*ul_NumDataPoints )
	dim3 block(64, 1, 1);
	dim3 grid((ac_fftlen + (block.x*B) - 1) / (block.x*B), 1, 1);

  //Step 1: Preprocessing - repack relevant powerspectrum into a 4N array with 'real-even symmetry'
	CUDA_ACC_SAFE_LAUNCH( (ac_RepackInputKernel<<<grid, block>>>( &dev_PowerSpectrum[ac_fftlen*fft_num], dev_AutoCorrIn )),true);
  //Step 2: Process the 4N-FFT (Complex to Complex, size is 4 * ac_fft_len)
	CUFFT_SAFE_CALL(cufftExecC2C(  cudaAutoCorr_plan,dev_AutoCorrIn , dev_AutoCorrOut, CUFFT_FORWARD));
  //Step 3: Postprocess the FFT result (Scale, take powers & normalise), discarding unused data packing into AutoCorr_in first half for VRAM reuse
	dim3 grid2( ((ac_fftlen>>1)+block.x*B-1)/(block.x*B), 1, 1);
	CUDA_ACC_SAFE_LAUNCH( (ac_RepackScaleKernel<<<grid2, block>>>( dev_AutoCorrOut, dev_AutoCorrIn )),true);
  //Step 4: Update best autocorrelation if needed, Search for & report any autocorrelations eceeding threshold.
  int len = ac_fftlen/2;
  int blksize = 256; //UNSTDMAX(4, UNSTDMIN(pow2((unsigned int) sqrt((float) (len / 32)) * 32), 512));

  dim3 block3(blksize,1,1);
  dim3 grid3(len/blksize,1,1);
  CUDA_ACC_SAFE_LAUNCH( (ac_reducePartial<<<grid3, block3,blksize*sizeof(float3)>>>( (float *)dev_AutoCorrIn, dev_ac_partials )),true);
//  dim3 block4(64,1,1);
// int nblks = grid3.x;
//  dim3 grid4(nblks,1,1);
//  while ( nblks >= 1 ) {
//	grid4.x = max(nblks/block4.x,1);
//	CUDA_ACC_SAFE_LAUNCH( (ac_reduceMore<<<grid4, block4, block4.x*sizeof(float3)>>>( dev_ac_partials, nblks )),true);	
//	nblks = grid4.x/block4.x;
//  }
//  cudaMemcpy(&blockSums[fft_num],dev_ac_partials,1*sizeof(float3),cudaMemcpyDeviceToHost);
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  cudaMemcpy(&blockSums[0],dev_ac_partials,grid3.x*sizeof(float3),cudaMemcpyDeviceToHost);
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  float rac_TotalSum = 0.0f;
  float ac_Peak = 0.0f;
  float ac_PeakBin = 0.0f;
  for (int b=0;b<len/blksize;b++)
  {
	  rac_TotalSum += blockSums[b].x;
	  if ( blockSums[b].y > ac_Peak )
	  {
		  ac_Peak = blockSums[b].y;
		  ac_PeakBin = blockSums[b].z;
	  }
  }
  blockSums[0].x = rac_TotalSum;
  blockSums[0].y = ac_Peak;
  blockSums[0].z = ac_PeakBin;

  return 0;
}

#endif //USE_CUDA
