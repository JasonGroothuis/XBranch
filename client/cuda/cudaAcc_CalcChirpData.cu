#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

const double SPLITTER=(1<<BSPLIT)+1;

#define B 16
#define N_TIMES 8
#define NB 3
//64
#define THREADS 64
#define TB 6


inline float2 splitd(double a) 
{
  double t   = a * SPLITTER; 
  double ahi = t - (t - a);
  double alo = a - ahi;
  return make_float2((float)ahi, (float)alo);
}

#define SB 1

__global__ void __launch_bounds__(THREADS, 4)
#ifdef SB
  cudaAcc_CalcChirpData_kernel_sm13(int NumDataPoints, double ccr, float2 *cx_DataArray, float2 *cx_ChirpDataArray) 
#else
  cudaAcc_CalcChirpData_kernel_sm13(int NumDataPoints, double ccr, float2 * __restrict__ cx_DataArray, float2 * __restrict__ cx_ChirpDataArray) 
#endif
{
  int iblock = blockIdx.x; // + blockIdx.y * 1024*1024/THREADS; //gridDim.x
  int ix  = (iblock * THREADS + threadIdx.x) * B; //blockDim.x
#ifdef SB
  int ix2 = (iblock * THREADS * N_TIMES); // for reading into shared float4 buffer from float2 addresses
#endif

  double time = ix; 
  float time2; 
  float time3; 

//  float4 cx[N_TIMES]; 
//  
//  for(int i = 0; i < N_TIMES; i++) // load 
//    {         
//      cx[i] = *(float4 *)(&cx_DataArray[ix + (i<<1)]); 
//    }

  time = __dmul_rn(time, time); 
  time2 = (((ix + ix) + 1)); 
  float ccf = (float)ccr;
  time3 = ccf + ccf;//__fmul_rn((float)ccr, 2.0f); 

  time = __dmul_rn(ccr, time); 
  time2 = __fmul_rn(ccf, time2); 

  time = __dsub_rn(time, __double2int_rd(time)); 
  int itime = __float2int_rd(time2);
  time2 = __fsub_rn(time2, itime); 
//  time3 = __fsub_rn(time3, __float2int_rd(time3)); 

  float ft1 = time; 
  float ft2 = time2; 
  float ft3 = time3; 
 
  ft2 = __fmul_rn(ft2, M_2PIf); 
  ft3 = __fmul_rn(ft3, M_2PIf); 
  ft1 = __fmul_rn(ft1, M_2PIf); 

  float cf, sf, ca, sa, cb, sb; 
  
  __sincosf(ft1, &sf, &cf); 
  __sincosf(ft2, &sa, &ca); 
  __sincosf(ft3, &sb, &cb); 
  
  float4 *ip = (float4 *)(&cx_DataArray[ix]);
  float4 *op = (float4 *)(&cx_ChirpDataArray[ix]);
  float4 *op2 = (float4 *)(&cx_ChirpDataArray[ix+1179648]);

  float4 tmp = *ip; 
  const float nsb = -sb; 

#ifdef SB
  __shared__ float4 sd[N_TIMES][THREADS+1];

  for(int i = 0; i < N_TIMES; i++)
    {
      int row = threadIdx.x & (N_TIMES-1);
      int col = (i << (TB-NB)) + (threadIdx.x >> NB);
      int addr = ix2 + i*THREADS + threadIdx.x;
      sd[row][col] = ((float4 *)cx_DataArray)[addr];
    }

  if(THREADS > 32)
    __syncthreads();
  tmp = sd[0][threadIdx.x];
#endif

//#pragma unroll 4
  for(int i = 0; i < N_TIMES; i++) // use f and g to rot 
    { 
      float tsca, tcca, sg, cg, sacb, cacb, tsa; 

      float ft1f = __fmul_rn(tmp.y, -sf); 
      float ft2f = __fmul_rn(tmp.y, cf); 

      tsca = __fmul_rn(sf, ca); // rot f by a to make g 
      tcca = __fmul_rn(cf, ca); // 
      sacb = __fmul_rn(sa, cb); // rot a by b 
      cacb = __fmul_rn(ca, cb); // 

      sg = __fmaf_rn(cf, sa, tsca); // 
      cg = __fmaf_rn(sf, -sa, tcca); // rot f to g by a ready 

      tsa = sa; // 
      sa = __fmaf_rn(ca, sb, sacb); // 
      ca = __fmaf_rn(tsa, nsb, cacb); // rot a by b ready 

      float ft3g = __fmul_rn(tmp.w, -sg); 
      float ft4g = __fmul_rn(tmp.w, cg); 
      
      float4 t1;
      t1.y = __fmaf_rn(tmp.x, sf, ft2f); 
      t1.x = __fmaf_rn(tmp.x, cf, ft1f); 
 
      tsca = __fmul_rn(sg, ca); // rot g by a to make f 
      tcca = __fmul_rn(cg, ca); // 

      t1.w = __fmaf_rn(tmp.z, sg, ft4g); 
      t1.z = __fmaf_rn(tmp.z, cg, ft3g); 
      op[i] = t1;
      t1.x = __fmaf_rn(tmp.x, cf, -ft1f); 
      t1.y = __fmaf_rn(tmp.x, -sf, ft2f); 
      t1.z = __fmaf_rn(tmp.z, cg, -ft3g); 
      t1.w = __fmaf_rn(tmp.z, -sg, ft4g); 
      op2[i] = t1;
      
#ifdef SB
      tmp = sd[(i+1)&(N_TIMES-1)][threadIdx.x];
#else
      tmp = ip[(i+1)]; 
#endif

      sacb = __fmul_rn(sa, cb); // rot a by b again 
      cacb = __fmul_rn(ca, cb); // 

      tsa = sa; // 
      sf = __fmaf_rn(cg, sa, tsca); // 
      cf = __fmaf_rn(sg, -sa, tcca); // rot g to f by a ready 
 
      sa = __fmaf_rn(ca, sb, sacb); // 
      ca = __fmaf_rn(tsa, nsb, cacb); // rot a by b ready 
    } 
//
//  for(int i = 0; i < N_TIMES; i++) // store 
//    { 
//      *(float4 *)&(cx_ChirpDataArray[ix + (i<<1)]) = cx[i]; 
//    } 
//
}


__global__ void cudaAcc_CalcChirpData_kernel(int NumDataPoints, float2 chirp_rate, float2 recip_sample_rate, float2* cx_DataArray, float2* cx_ChirpDataArray)
{
   //Dummy function for Fermi+
}


void cudaAcc_CalcChirpData(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray) {
	if (!cudaAcc_initialized()) return;
	dim3 block(64, 1, 1);
	dim3 grid = grid2D((cudaAcc_NumDataPoints + block.x - 1) / block.x);

//	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel<<<grid, block>>>(cudaAcc_NumDataPoints, splitd(0.5*chirp_rate), splitd(recip_sample_rate), dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream)
{
	if (!cudaAcc_initialized()) return;
	dim3 block(64, 1, 1);
	dim3 grid = grid2D((cudaAcc_NumDataPoints + block.x - 1) / block.x);

//	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel<<<grid, block,0,chirpstream>>>(cudaAcc_NumDataPoints, splitd(0.5*chirp_rate), splitd(recip_sample_rate), dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_sm13(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray) {
//	if (!cudaAcc_initialized()) return;

	dim3 block(64, 1, 1);
	// determined from chirp unit tests, cc 2.1 likes 128 threads here due to superscalar warp schedulers..
	// assume the architectural balance for future GPU arch will be similar
	if ( ((gCudaDevProps.major == 2) && (gCudaDevProps.minor >= 1)) || (gCudaDevProps.major > 2) )
	{
		block.x = THREADS;
	}
	dim3 grid = grid2D((cudaAcc_NumDataPoints + (block.x*B - 1)) / (block.x*B));

        double ccr = 0.5*chirp_rate*recip_sample_rate*recip_sample_rate;
	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel_sm13<<<grid, block>>>(cudaAcc_NumDataPoints, ccr, dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_sm13_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream) {
//	if (!cudaAcc_initialized()) return;

	dim3 block(64, 1, 1);
	// determined from chirp unit tests, cc 2.1 likes 128 threads here due to superscalar warp schedulers..
	// assume the architectural balance for future GPU arch will be similar
	if ( ((gCudaDevProps.major == 2) && (gCudaDevProps.minor >= 1)) || (gCudaDevProps.major > 2) )
	{
		block.x = THREADS;
	}

	dim3 grid = grid2D( (cudaAcc_NumDataPoints + (block.x*B - 1)) / (block.x*B));
        double ccr = 0.5*chirp_rate*recip_sample_rate*recip_sample_rate;

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel_sm13<<<grid, block,0,chirpstream>>>(cudaAcc_NumDataPoints, ccr, dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

#endif //USE_CUDA
