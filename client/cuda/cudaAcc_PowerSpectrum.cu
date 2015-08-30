#include "cudaAcceleration.h"

#ifdef USE_CUDA
#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

//2, 8, 64
#define B 2
#define D 16
#define blockDim_x 128

#if 1
__launch_bounds__(blockDim_x, 8)
__global__ void cudaAcc_GPS_kernel_mod3SM(const float2 * const FreqData, float *PowerSpectrum) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx   = threadIdx.x + D * iblock * blockDim_x; 
  
  float4 a[D];
  float4 *ip = ((float4 *)FreqData) + sidx;
  float2 *op = ((float2 *)PowerSpectrum) + sidx;

#pragma unroll
  for(int i = 0; i < D; i++) 
    {
      a[i] = ip[i * blockDim_x]; 
    }

#pragma unroll
  for(int i = 0; i < D; i++) 
    {
      a[i].x = a[i].x * a[i].x;
      a[i].y = a[i].y * a[i].y;
      a[i].z = a[i].z * a[i].z;
      a[i].w = a[i].w * a[i].w;
    }

#pragma unroll
  for(int i = 0; i < D; i++) 
    {
      a[i].x = a[i].x + a[i].y;
      a[i].z = a[i].z + a[i].w;
    }

#pragma unroll
  for(int i = 0; i < D; i++)
    op[i * blockDim_x] = make_float2(a[i].x, a[i].z);
}


#else
__global__ void cudaAcc_GPS_kernel_mod3SM(float2 * __restrict__ FreqData, float * __restrict__ PowerSpectrum) 
{
  int iblock = blockIdx.x + blockIdx.y * gridDim.x;
  int sidx = threadIdx.x + B*iblock*blockDim_x; 
  
  float2 a[B];
  
#pragma unroll
  for(int i = 0; i < B; i++) 
    {
      a[i] = FreqData[sidx+i*blockDim_x]; 
    }

#pragma unroll
  for(int i = 0; i < B; i++)
    PowerSpectrum[sidx+i*blockDim_x] =__fadd_rn( __fmul_rn(a[i].x,a[i].x),__fmul_rn(a[i].y,a[i].y)); 
}

__global__ void cudaAcc_GPS_kernel_mod3( int NumDataPoints, float2* FreqData, float* PowerSpectrum) 
{
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

void cudaAcc_GetPowerSpectrum(int numpoints, int offset, cudaStream_t str) 
{
  dim3 block(blockDim_x, 1, 1);
  //	dim3 grid((numpoints + (block.x*B) - 1) / (block.x*B), 1, 1);
  dim3 grid = grid2D((numpoints + (block.x*B*D) - 1) / (block.x*B*D));
  
  CUDA_ACC_SAFE_LAUNCH( (cudaAcc_GPS_kernel_mod3SM<<<grid, block, 0, str>>>(dev_WorkData+offset, dev_PowerSpectrum+offset)),true);
}

#endif //USE_CUDA
