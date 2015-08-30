
#include "cudaAcceleration.h"

#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

//#define B 2


#define RPI 64
#define RPIB 8
#define RPBY 4

#define B 8
#define RPS 64
#define RDP 256

float2 *dev_AutoCorrIn[8];
float2 *dev_AutoCorrOut[8];


bool gCudaAutocorrelation = false;

float3 *dev_blockSums[8];
float3 *blockSums[8];
float3 *dev_ac_partials[8];

//float ac_TotalSum;
//float ac_Peak;
//int ac_PeakBin;



__global__ void __launch_bounds__(RPI, 16)
ac_RepackInputKernelR(float *PowerSpectrum, float2 *dct_In) 
{
  int sidx = (threadIdx.x + blockIdx.x*RPI + (blockIdx.y*RPI*RPIB)); 
  int nDestPoints = 524288/4; // (gridDim.y*RPI*RPIB)*4;
  int didx1 = sidx<<2; 
  int didx2 = ((nDestPoints)<<3)-didx1-4; 

  PowerSpectrum += sidx;
  float *dct1 = (float*)((void*)dct_In + didx1);
  float *dct2 = (float*)((void*)dct_In + didx2);
  //  float2 t = make_float2(PowerSpectrum[0], 0);
  float t = PowerSpectrum[0];
      
  *dct1 = t;
  *dct2 = t;
}



__global__ void __launch_bounds__(RPI, 16)
ac_RepackInputKernelP(float *PowerSpectrum, float2 *dct_In) 
{
  int sidx = (threadIdx.x + blockIdx.x*RPI + (blockIdx.y*RPI*RPIB)); 
  int nDestPoints = 524288; // (gridDim.y*RPI*RPIB)*4;
  int didx1 = sidx<<4; 
  int didx2 = ((nDestPoints-2)<<3)-didx1; 

  PowerSpectrum += sidx;
  float4 *dct1 = (float4*)((void*)dct_In + didx1);
  float4 *dct2 = (float4*)((void*)dct_In + didx2);
  float4 t = make_float4(0.0f, 0.0f, PowerSpectrum[0], 0.0f);
      
  *dct1 = t;
  *dct2 = t;
}


__global__ void __launch_bounds__(RPI, 16)
ac_RepackInputKernel(float *PowerSpectrum, float2 *dct_In) 
{
  int sidx = (threadIdx.x + (blockIdx.x*RPI*RPIB)); 
  int nDestPoints = 524288; //(gridDim.x*RPI*RPIB)<<2;
  int didx1 = sidx<<4; 
  int didx2 = ((nDestPoints-2)<<3)-didx1 - 2*(RPIB-1)*RPI<<3; 
  
  float a[RPIB];
  PowerSpectrum += sidx;
  float2 *dct1 = (float2*)((void*)dct_In + didx1);
  float2 *dct2 = (float2*)((void*)dct_In + didx2);

#pragma unroll
  for(int i = 0; i < RPIB; i++)
    {
      a[i] = PowerSpectrum[i*RPI];
    }
  
#pragma unroll
  for(int i = 0; i < RPIB; i++)
    {
      float4 t = make_float4(0.0f, 0.0f, a[i], 0.0f);
      float4 t2 = make_float4(0.0f, 0.0f, a[RPIB-1-i], 0.0f);
      
      *(float4 *)(&dct1[2*i*RPI]) = t;
      *(float4 *)(&dct2[2*i*RPI]) = t2;
    }
}


/*

__global__ void 
ac_RepackInputKernel(float * __restrict__ PowerSpectrum, float2 * __restrict__ dct_In) 
{
  int sidx = (threadIdx.x + (blockIdx.x*RPI*RPIB)); 
  int nDestPoints = (gridDim.x*RPI*RPIB)<<2;
  int didx1 = sidx<<1; 
  int didx2 = (nDestPoints-2)-didx1; 
  
  float a[RPIB];
  PowerSpectrum += sidx;
  float2 * __restrict__ dct1 = (float2*)((void*)dct_In + sizeof(float2)*didx1);
  float2 * __restrict__ dct2 = (float2*)((void*)dct_In + sizeof(float2)*didx2);

#pragma unroll
  for(int i = 0; i < RPIB; i++)
    {
      a[i] = PowerSpectrum[i*RPI];
    }
  
#pragma unroll
  for(int i = 0; i < RPIB; i++)
    {
      float4 t = make_float4(0.0f, 0.0f, a[i], 0.0f);
      
      *(float4 *)(&dct1[2*i*RPI]) = t;
      *(float4 *)(&dct2[-(2*i*RPI)]) = t;
    }
}
*/



__global__ void __launch_bounds__(RPS, 16)
  ac_RepackScaleKernelP(float2 *src, float2 *dst) 
{
  int didx = ((threadIdx.x + blockIdx.x*RPS + blockIdx.y*RPS*B));  //packing into float2s
  int sidx = didx << 1; //((threadIdx.x + blockIdx.x*RPS*B)*2);

//printf("bx=%d, by=%d, tx=%d, sidx=%d\r\n", blockIdx.x, blockIdx.y, threadIdx.x, sidx);
  dst += didx;
  src += sidx;

  float4 t = ((float4 *)src)[0];
  float2 a = make_float2(t.x, t.z);

  a.x *= a.x;
  a.y *= a.y;

  dst[0] = a;
}

__global__ void __launch_bounds__(RPS, 16)
ac_RepackScaleKernel(float2 *src, float2 *dst) 
{
  int didx = ((threadIdx.x + blockIdx.x*RPS*B));  //packing into float2s
  int sidx = didx << 1; //((threadIdx.x + blockIdx.x*RPS*B)*2);
  
  float2 a[B];
  
  dst += didx;
  src += sidx;

#pragma unroll
  for(int i = 0; i < B; i++)
    {
      float4 t = *(float4 *)(&src[i*RPS*2]);
      a[i] = make_float2(t.x, t.z);
    }
  for(int i = 0; i < B; i++)
    {
      a[i].x *= a[i].x;
      a[i].y *= a[i].y;
    }
#pragma unroll
  for(int i = 0; i < B; i++)
    {
      dst[i*RPS] = a[i];
    }
}

extern __shared__ float3 acpartial[];

__global__ void ac_reducePartial(float *ac, float3 *devpartials)
{
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x*RDP;
  const int bid = blockIdx.x;
  int n = RDP>>1;
  
  float3 *acp = acpartial;
  float tmp = ac[idx];
  acp[tid].z = idx;
  acp[tid].y = idx >= 1 ? tmp : 0.0f; 
  acp[tid].x = tmp;
  
  __syncthreads();
  
  volatile float3 *dp = &acp[tid];
  int fadd = n * 12;
#pragma unroll 2
  for(; n > 32; n >>= 1)
    {
      if(tid < n)
	{
	  float a =  __fadd_rn(dp[0].x, (*(float3*)(((void*)dp)+fadd)).x);
	  // peak power & its bin
	  float pp = (*(float3*)(((void*)dp)+fadd)).y;
	  float pb = (*(float3*)(((void*)dp)+fadd)).z;
	  fadd >>= 1;
	  bool b = pp > dp[0].y;
          dp[0].x = a;
	  if(b)
	    {
	      dp[0].y = pp;
	      dp[0].z = pb;
	    }
	}
      __syncthreads();
    }

#pragma unroll 6
  for(; n > 0; n >>= 1)
    {
      if(tid < n)
	{
	  float a =  __fadd_rn(dp[0].x, (*(float3*)(((void*)dp)+fadd)).x);
	  // peak power & its bin
	  float pp = (*(float3*)(((void*)dp)+fadd)).y;
	  float pb = (*(float3*)(((void*)dp)+fadd)).z;
	  fadd >>= 1;
	  bool b = pp > dp[0].y;
          dp[0].x = a;
	  if(b)
	    {
	      dp[0].y = pp;
	      dp[0].z = pb;
	    }
	}
    }

  if(tid == 0) 
    {        
      devpartials[bid] = make_float3(dp[0].x, dp[0].y, dp[0].z);
    }
}



int cudaAcc_FindAutoCorrelations(float *AutoCorrelation, int ac_fftlen) 
{
  for(int fft_num = 0; fft_num < 8; fft_num++)
    {
      //Jason: Use 4N-FFT method for Type 2 Discrete Cosine Tranform for now, to match fftw's REDFT10
      // 1 Autocorrelation from global powerspectrum at fft_num*ac_fft_len  (fft_num*ul_NumDataPoints )
      dim3 block(RPI, 1, 1);
      //  dim3 grid((ac_fftlen + (block.x*RPIB) - 1) / (block.x*RPIB), 1, 1); 
      dim3 grid(RPIB, (ac_fftlen + (block.x*RPIB) - 1) / (block.x*RPIB), 1); 
      
      //Step 1: Preprocessing - repack relevant powerspectrum into a 4N array with 'real-even symmetry'
      //////!!!  CUDA_ACC_SAFE_LAUNCH( (ac_RepackInputKernelR<<<grid, block>>>( &dev_PowerSpectrum[ac_fftlen*fft_num], (float2 *)dev_AutoCorrInR )),true);
      //  CUDA_ACC_SAFE_LAUNCH( (ac_RepackInputKernel<<<grid, block>>>( &dev_PowerSpectrum[ac_fftlen*fft_num], dev_AutoCorrIn )),true);
      CUDA_ACC_SAFE_LAUNCH( (ac_RepackInputKernelP<<<grid, block>>>( &dev_PowerSpectrum[ac_fftlen*fft_num], dev_AutoCorrIn[fft_num] )),true);
      
      //Step 2: Process the 4N-FFT (Complex to Complex, size is 4 * ac_fft_len)
      //////!!!  CUFFT_SAFE_CALL(cufftExecR2C(cudaAutoCorr_planR, (float*)dev_AutoCorrInR , dev_AutoCorrOutR)); //, CUFFT_FORWARD
      cufftExecC2C(cudaAutoCorr_plan, dev_AutoCorrIn[fft_num] , dev_AutoCorrOut[fft_num], CUFFT_FORWARD);
      
      
      //Step 3: Postprocess the FFT result (Scale, take powers & normalise), discarding unused data packing into AutoCorr_in first half for VRAM reuse
      //  block.x = RPS;
      //  block.y = 1;
      //  dim3 grid2( ((ac_fftlen>>1)+block.x*B-1)/(block.x*B), 1, 1);
      //  CUDA_ACC_SAFE_LAUNCH( (ac_RepackScaleKernel<<<grid2, block>>>( dev_AutoCorrOut, dev_AutoCorrIn )),true);
      block.x = RPS;
      block.y = 1;
      dim3 grid2(B, ((ac_fftlen>>1)+block.x*B-1)/(block.x*B), 1);
      CUDA_ACC_SAFE_LAUNCH( (ac_RepackScaleKernelP<<<grid2, block>>>( dev_AutoCorrOut[fft_num], dev_AutoCorrIn[fft_num])),true); //R first

      int len = ac_fftlen/2;
      int blksize = RDP; 
      dim3 block3(blksize,1,1);
      dim3 grid3(len/blksize,1,1);

      cudaFuncSetCacheConfig(ac_reducePartial, cudaFuncCachePreferShared);
      CUDA_ACC_SAFE_LAUNCH( (ac_reducePartial<<<grid3, block3, blksize*sizeof(float3)>>>( (float *)dev_AutoCorrIn[fft_num], dev_ac_partials[fft_num] )),true);
      blockSums[fft_num][0].x = -999;
      cudaMemcpyAsync(&blockSums[fft_num][0], dev_ac_partials[fft_num], len/RDP*sizeof(float3), cudaMemcpyDeviceToHost);

    }
 return 0;
}


// TODO (half done): start all autocorrs. start all datadownloads. do ALL peak finds. cudasync. process all autocorr results.

int cudaAcc_GetAutoCorrelation(float *AutoCorrelation, int ac_fftlen, int fft_num)
{
  int len = ac_fftlen/2;
  int blksize = RDP; 
  float rac_TotalSum = 0, ac_Peak = 0;
  int ac_PeakBin = 0;

  if(blockSums[fft_num][0].x == -999) { CUDASYNC; }
  for(int b = 0; b < len/blksize; b++)
    {
      rac_TotalSum += blockSums[fft_num][b].x;
      if(blockSums[fft_num][b].y > ac_Peak)
	{
	  ac_Peak = blockSums[fft_num][b].y;
	  ac_PeakBin = b;
	}
    }
  blockSums[fft_num][0].x = rac_TotalSum;
  blockSums[fft_num][0].y = ac_Peak;
  blockSums[fft_num][0].z = blockSums[fft_num][ac_PeakBin].z;
  
  return 0;
}

#endif //USE_CUDA
