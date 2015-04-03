/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"

#include "cudaAcc_utilities.h"

float3 *dev_PowerSpectrumSumMax;
float3 *PowerSpectrumSumMax;

extern __shared__ float3 partial[];
// At least 32elems per sum
template <int iterations>
__global__ void cudaAcc_summax32_kernel(float* input, float3* output) 
{    
  const int tid = threadIdx.x;
  const int y = blockIdx.y * blockDim.y;
  const int gridX = /* gridDim.x * */ blockDim.x;
  const int start = y * gridX * iterations;
  const int end = (y + 1) * gridX * iterations;
  
  float sum = 0.0f;
  float maximum = 0.0f;    
  int pos = 0;
  int i = start + tid;
  float *ip = &input[i];
  float val = *ip, val2; 
  sum = sum + val;

  if(tid > 0) 
    { // Original max omits first element of every fft
      if(val > maximum)
	pos = i;
      maximum = max(maximum, val);
    }

  ip = &input[start + tid + gridX];  
  for(i = start + tid + gridX; i < (end-gridX*3); i += gridX) 
    {
      val = *ip;
      ip += gridX;  
      sum = sum + val;
      if (val > maximum)
          {pos = i;
           maximum = val;}
      val2 = *ip;
      i += gridX;
      ip += gridX;  
      sum = sum + val2;
      if (val2 > maximum)
          {pos = i;
           maximum = val2;}
      val = *ip;
      i += gridX;
      ip += gridX;  
      sum = sum + val;
      if (val > maximum)
          {pos = i;
           maximum = val;}
      val2 = *ip;
      i += gridX;
      ip += gridX;  
      sum = sum + val2;
      if (val2 > maximum)
          {pos = i;
           maximum = val2;}
    }
  // tail
  for(; i < end; i += gridX) 
    {
      val = *ip;
      ip += gridX;  
      sum = sum + val;
      if (val > maximum)
          {pos = i;
           maximum = val;}
    }
  
  float3 *pp = &partial[tid];
  *pp = make_float3(sum, maximum, pos - start);
  int padd = (gridX>>1)*12;

  for(i = gridX >> 1; i > 0; i >>= 1) 
    {
      __syncthreads(); 
      if(tid < i) 
	{            
	  float a = pp[0].x, b = (*(float3 *)(((void*)pp)+padd)).x;
	  float s = a + b;
	  float ay = pp[0].y, by = (*(float3 *)(((void*)pp)+padd)).y;
	  bool bb = ay > by;
	  float az = pp[0].z, bz = (*(float3 *)(((void*)pp)+padd)).z;
	  padd >>= 1;
	  pp[0] = make_float3(s, bb ? ay : by, bb ? az : bz); 
	  /*
	    partial[tid] = make_float3(
	    partial[tid].x + partial[tid + i].x, 
	    partial[tid].y > partial[tid + i].y ? partial[tid].y : partial[tid + i].y,
	    partial[tid].y > partial[tid + i].y ? partial[tid].z : partial[tid + i].z            
	    ); 
	  */
	}
    }
  
  if(tid == 0) 
    {        
      output[blockIdx.y] = pp[0];
    }
}


//less than 32 elems
template<int n>
__global__ void cudaAcc_summax_kernel(float* input, float3* output)
{
  const int tid = threadIdx.x;
  const int y = blockIdx.y * blockDim.y;
  const int gridX = gridDim.x * blockDim.x;
  const int start = y * gridX;
  
  const int n1 = n - 1;
  const int width = blockDim.x / n;
  
  float val = input[start + tid];
  if ((tid & n1) == 0)
    partial[tid] = make_float3(val, 0.0f, tid & n1); // Original max omits first element of every fft
  else
    partial[tid] = make_float3(val, val, tid & n1);
  
  for(int i = n >> 1; i > 0; i >>= 1)
    {
      __syncthreads(); 
      if((tid & n1) < i) 
	{
	  float a = partial[tid].x, b = partial[tid+i].x;
	  float s = a + b;
	  float ay = partial[tid].y, by = partial[tid+i].y;
	  bool bb = ay > by;
	  float az = partial[tid].z, bz = partial[tid+i].z;
	  partial[tid] = make_float3(s, bb ? ay : by, bb ? az : bz); 
	}
    }
  __syncthreads();
  
  if (tid < width) {
    output[blockIdx.y * width + tid] = partial[tid * n];
  }
}

//Jason:---------------------------------------------------------------------------
template<class T>
struct SharedMemory
{
  __device__ inline operator       T*()
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
  
  __device__ inline operator const T*() const
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

template <unsigned int fftlen> 
__global__ void cudaAcc_SM( float* PowerSpectrum,float3* devPowerSpectrumSumMax) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx = (iblock*blockDim.x + threadIdx.x); 
  int tid = threadIdx.x;
  
  float3 *sdata = SharedMemory<float3>();
  
  float mySum = sdata[tid].x = PowerSpectrum[sidx];
  sdata[tid].z = tid;
  sdata[tid].y = (tid == 0 ? 0.0f :  mySum);
  __syncthreads();
  
  if (tid < fftlen/2) // last part of reduction is warp synchronous;
    {
      volatile float3* smem = sdata;
      if(fftlen >=  64) 
	{
	  smem[tid].x = mySum = mySum + smem[tid + 32].x;
	  smem[tid].y = max(smem[tid].y, smem[tid + 32].y);
	  smem[tid].z = smem[tid].y > smem[tid + 32].y ? smem[tid].z : smem[tid + 32].z; 
	}
      if(fftlen >=  32) 
	{
	  smem[tid].x = mySum = mySum + smem[tid + 16].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 16].y);
	  smem[tid].z = smem[tid].y > smem[tid + 16].y ? smem[tid].z : smem[tid + 16].z; 
	}
      if(fftlen >=  16) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  8].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 8].y);
	  smem[tid].z = smem[tid].y > smem[tid + 8].y ? smem[tid].z : smem[tid + 8].z; 
	}
      if(fftlen >=   8) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  4].x;
	  smem[tid].y = max(smem[tid].y, smem[tid + 4].y);
	  smem[tid].z = smem[tid].y > smem[tid + 4].y ? smem[tid].z : smem[tid + 4].z; 
	}
      if(fftlen >=   4) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  2].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 2].y);
	  smem[tid].z = smem[tid].y > smem[tid + 2].y ? smem[tid].z : smem[tid + 2].z; 
	}
      if(fftlen >=   2) 
	{
	  smem[tid].x = mySum = mySum + smem[tid +  1].x;  
	  smem[tid].y = max(smem[tid].y, smem[tid + 1].y);
	  smem[tid].z = smem[tid].y > smem[tid + 1].y ? smem[tid].z : smem[tid + 1].z; 
	}
    }
  
  if(tid==0)
    {
      devPowerSpectrumSumMax[sidx/fftlen] = sdata[0];
    }
}

// TODO: optimize Memcpy, download data only when are going to be reported: 
//							if (si.score > best_spike->score || best_spike->s.fft_len == 0)
//							if (si.s.peak_power > (swi.analysis_cfg.spike_thresh))
void cudaAcc_summax(int fftlen) 
{
  //	int smemSize2 = fftlen*sizeof(float3);
  dim3 block2(fftlen, 1, 1);
  //dim3 grid2((cudaAcc_NumDataPoints + block2.x - 1) / block2.x, 1, 1);
  dim3 grid2 = grid2D((cudaAcc_NumDataPoints + block2.x - 1) / block2.x);
  
  //  if ( fftlen == 64 || fftlen == 32)
  //  {
  //switch (fftlen) {
  //case 64:
  //	CUDA_ACC_SAFE_LAUNCH((cudaAcc_SM<64><<<grid2,block2,smemSize2,fftstream0>>>( dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
  //	break;
  //case 32:
  //	CUDA_ACC_SAFE_LAUNCH((cudaAcc_SM<32><<<grid2,block2,smemSize2,fftstream0>>>( dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
  //	break;
  ////case 16:
  ////	CUDA_ACC_SAFE_LAUNCH((cudaAcc_SM<16><<<grid2,block2,smemSize2,fftstream0>>>( dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
  ////	break;
  ////case 8:
  ////	CUDA_ACC_SAFE_LAUNCH((cudaAcc_SM<8><<<grid2,block2,smemSize2,fftstream0>>>( dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
  ////	break;
  //}    
  //  }
  //  else
  if(fftlen >= 32 && cudaAcc_NumDataPoints/fftlen < 65536)
    {
      int optimal_block_x;
      if(gCudaDevProps.major >= 2)
	optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 1024));
      else
	optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 512));
      
      dim3 block(optimal_block_x, 1, 1);;
      dim3 grid(1, cudaAcc_NumDataPoints / fftlen, 1);     
      int iterations = fftlen/block.x;
      if(iterations > 256)
        printf("iterations %d\r\n", iterations);
      switch(iterations)
      {
        case 1:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<1><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 2:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<2><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 4:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<4><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<8><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<16><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<32><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<64><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<128><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 256:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<256><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	default:
	  fprintf(stderr, "cudaAcc_summax32_kernel template error\r\n");
	  fflush(stderr);
	  *(char *)0 = 0;
      }     
    }
  else 
    {
      // Occupancy Calculator: 128 for cc1.x, 256 for cc2.x
      dim3 block(128, 1, 1);
      if(gCudaDevProps.major >= 2) 
	{
#if CUDART_VERSION >= 3000
//	  cudaFuncSetCacheConfig(cudaAcc_summax32_kernel, cudaFuncCachePreferShared); // Set this at init time
#endif
	  block.x = 256; 
	}
      dim3 grid(1, cudaAcc_NumDataPoints / block.x, 1);
      switch (fftlen) 
	{
	case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<128><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<64><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<32><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<16><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<8><<<grid, block, (block.x * sizeof(float3)),fftstream0>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	}
    }
  
  
  cudaMemcpyAsync(PowerSpectrumSumMax, dev_PowerSpectrumSumMax, (cudaAcc_NumDataPoints / fftlen) * sizeof(*dev_PowerSpectrumSumMax), cudaMemcpyDeviceToHost, fftstream0);    
}


void cudaAcc_summax_x(int fftlen) 
{
  dim3 block2(fftlen, 1, 1);
  dim3 grid2 = grid2D((cudaAcc_NumDataPoints + block2.x - 1) / block2.x);
  
  if(fftlen >= 32 && cudaAcc_NumDataPoints/fftlen < 65536)
    {
      int optimal_block_x;
      if(gCudaDevProps.major >= 2)
	optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 1024));
      else
	optimal_block_x = max(32, min(pow2((unsigned int) sqrt((float) (fftlen / 32)) * 32), 512));
      
      dim3 block(optimal_block_x, 1, 1);;
      dim3 grid(1, cudaAcc_NumDataPoints / fftlen, 1);     
      int iterations = fftlen/block.x;
      if(iterations > 256)
        printf("iterations %d\r\n", iterations);
      switch(iterations)
      {
        case 1:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<1><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 2:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<2><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 4:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<4><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<8><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<16><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<32><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<64><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<128><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
        case 256:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax32_kernel<256><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	default:
	  fprintf(stderr, "cudaAcc_summax32_kernel template error\r\n");
	  fflush(stderr);
	  *(char *)0 = 0;
      }     
    }
  else 
    {
      // Occupancy Calculator: 128 for cc1.x, 256 for cc2.x
      dim3 block(128, 1, 1);
      if(gCudaDevProps.major >= 2) 
	{
#if CUDART_VERSION >= 3000
//	  cudaFuncSetCacheConfig(cudaAcc_summax32_kernel, cudaFuncCachePreferShared); // Set this at init time
#endif
	  block.x = 256; 
	}
      dim3 grid(1, cudaAcc_NumDataPoints / block.x, 1);
      switch (fftlen) 
	{
	case 128:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<128><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 64:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<64><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 32:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<32><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 16:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<16><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	case 8:
	  CUDA_ACC_SAFE_LAUNCH((cudaAcc_summax_kernel<8><<<grid, block, (block.x * sizeof(float3))>>>(dev_PowerSpectrum, dev_PowerSpectrumSumMax)),true);
	  break;
	}
    }
  
  
  cudaMemcpyAsync(PowerSpectrumSumMax, dev_PowerSpectrumSumMax, (cudaAcc_NumDataPoints / fftlen) * sizeof(*dev_PowerSpectrumSumMax), cudaMemcpyDeviceToHost);    
}

#endif //USE_CUDA
