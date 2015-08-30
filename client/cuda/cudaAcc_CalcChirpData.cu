#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

const double SPLITTER=(1<<BSPLIT)+1;
#define B 4

inline float2 splitd(double a) {
    double t = a*SPLITTER; 
	double ahi= t-(t-a);
	double alo = a-ahi;
	return make_float2((float)ahi,(float)alo);
}

#if __CUDA_ARCH__ < 130
inline __device__ float2 split(float a)
{
	const float split= 4097;
	float t = __fmul_rn(a,split);
	float ahi = t-(t-a);
	float alo = a - ahi;
	return make_float2(ahi,alo);
}

inline __device__ float2 twoProd(float a, float b)
{
	float p = __fmul_rn(a,b);
	float2 aS = split(a);
	float2 bS = split(b);
	float err = ( ( __fmul_rn(aS.x,bS.x) - p )
				+ __fmul_rn(aS.x,bS.y) + __fmul_rn(aS.y,bS.x) )
				+ __fmul_rn(aS.y,bS.y);
	return make_float2(p,err);
}

inline __device__ float2 quickTwoSum(float a, float b)
{
	float s = a + b;
	float err = b - (s-a);
	return make_float2(s,err);
}

inline __device__ float2 df64_mult(float2 a, float2 b)
{
	float ah=a.x,al=a.y,bh=b.x,bl=b.y;
	float2 p; 
	p = twoProd(ah,bh);
	p.y += ah*bl;
	p.y += al*bh;
	p = quickTwoSum(p.x,p.y);
	return p;
}

__global__ void cudaAcc_CalcChirpData_kernel(int NumDataPoints, float2 chirp_rate, float2 recip_sample_rate, float2* cx_DataArray, float2* cx_ChirpDataArray) {  
	int iblock = blockIdx.x + blockIdx.y * gridDim.x;
	int i = iblock*blockDim.x + threadIdx.x;    
	if (i < NumDataPoints) {        
		float2 cx = cx_DataArray[i];
		float c, d, real, imag;

		//time[i] = recip_sample_rate * i ;
		float2 time = df64_mult(split(__int2float_rn(i)),recip_sample_rate);

		float2 a = chirp_rate;
		//float2 b = time;
		float2 angdf = df64_mult(a,df64_mult(time,time));
		angdf.x = __fadd_rn(angdf.x,-__float2int_rn(angdf.x))*M_2PIf;
		angdf.y = __fadd_rn(angdf.y,-__float2int_rn(angdf.y))*M_2PIf;
		float ang = __fadd_rn(angdf.x,angdf.y); 
		sincosf(ang,&d,&c); 

//		real = cx.x * c - cx.y * d;
//		imag = cx.x * d + cx.y * c;
		real = __fadd_rn( __fmul_rn(cx.x,c), -__fmul_rn(cx.y,d) );
		imag = __fadd_rn( __fmul_rn(cx.x,d), __fmul_rn(cx.y,c) );
		cx_ChirpDataArray[i] = make_float2(real, imag);
	}
}

__global__ void cudaAcc_CalcChirpData_kernel_sm13(int NumDataPoints, double chirp_rate, double recip_sample_rate, float2* cx_DataArray, float2* cx_ChirpDataArray) 
{
	//Dummy function for PreFermis
}
#endif //__CUDA_ARCH__ < 130


#if __CUDA_ARCH__ >= 130
__global__ void 
cudaAcc_CalcChirpData_kernel_sm13(int NumDataPoints, double chirp_rate, double recip_sample_rate, float2* cx_DataArray, float2* cx_ChirpDataArray) 
{  
		int iblock = blockIdx.x + blockIdx.y * gridDim.x;
		int sidx = B*iblock*blockDim.x + threadIdx.x;    //stride on accesses instead

		float2 cx[B];
		float c[B], d[B], real[B], imag[B];
		float angf[B];
		double ang[B];
		double time[B];

		#pragma unroll
		for (int i=0;i<B;i++)
		{
			cx[i] = cx_DataArray[sidx+i*blockDim.x];
			time[i] = __dmul_rn(sidx+i*blockDim.x,recip_sample_rate);
		}
		#pragma unroll
		for (int i=0;i<B;i++)
		{
		    ang[i]  = __dmul_rn(chirp_rate,__dmul_rn(time[i],time[i]));       
			ang[i]  = __dadd_rn( ang[i], -__double2int_rd(ang[i]));
			angf[i] = __double2float_rn(ang[i]);
			angf[i] = __fmul_rn(angf[i],M_2PI);
			__sincosf(angf[i],&d[i],&c[i]);        
		}
		#pragma unroll
		for (int i=0;i<B;i++) {
			// real = cx.x * c - cx.y * d;
			real[i] = __fadd_rn(__fmul_rn(cx[i].x,c[i]), -__fmul_rn(cx[i].y,d[i]) );
			// imag = cx.x * d + cx.y * c;
			imag[i] = __fadd_rn(__fmul_rn(cx[i].x,d[i]), __fmul_rn(cx[i].y,c[i]) );
			cx_ChirpDataArray[sidx+i*blockDim.x] = make_float2(real[i], imag[i]);
		}
}

__global__ void cudaAcc_CalcChirpData_kernel(int NumDataPoints, float2 chirp_rate, float2 recip_sample_rate, float2* cx_DataArray, float2* cx_ChirpDataArray)
{
   //Dummy function for Fermi+
}
#endif


void cudaAcc_CalcChirpData(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray) {
	if (!cudaAcc_initialized()) return;
	dim3 block(64, 1, 1);
	dim3 grid = grid2D((cudaAcc_NumDataPoints + block.x - 1) / block.x);

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel<<<grid, block>>>(cudaAcc_NumDataPoints, splitd(0.5*chirp_rate), splitd(recip_sample_rate), dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream)
{
	if (!cudaAcc_initialized()) return;
	dim3 block(64, 1, 1);
	dim3 grid = grid2D((cudaAcc_NumDataPoints + block.x - 1) / block.x);

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel<<<grid, block,0,chirpstream>>>(cudaAcc_NumDataPoints, splitd(0.5*chirp_rate), splitd(recip_sample_rate), dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_sm13(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray) {
//	if (!cudaAcc_initialized()) return;

	dim3 block(64, 1, 1);
	// determined from chirp unit tests, cc 2.1 likes 128 threads here due to superscalar warp schedulers..
	// assume the architectural balance for future GPU arch will be similar
	if ( ((gCudaDevProps.major == 2) && (gCudaDevProps.minor >= 1)) || (gCudaDevProps.major > 2) )
	{
		block.x = 128;
	}
	dim3 grid = grid2D(  (cudaAcc_NumDataPoints + (block.x*B - 1)) / (block.x*B));

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel_sm13<<<grid, block>>>(cudaAcc_NumDataPoints, 0.5*chirp_rate, recip_sample_rate, dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

void cudaAcc_CalcChirpData_sm13_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream) {
//	if (!cudaAcc_initialized()) return;

	dim3 block(64, 1, 1);
	// determined from chirp unit tests, cc 2.1 likes 128 threads here due to superscalar warp schedulers..
	// assume the architectural balance for future GPU arch will be similar
	if ( ((gCudaDevProps.major == 2) && (gCudaDevProps.minor >= 1)) || (gCudaDevProps.major > 2) )
	{
		block.x = 128;
	}

	dim3 grid = grid2D(  (cudaAcc_NumDataPoints + (block.x*B - 1)) / (block.x*B));

	CUDA_ACC_SAFE_LAUNCH( (cudaAcc_CalcChirpData_kernel_sm13<<<grid, block,0,chirpstream>>>(cudaAcc_NumDataPoints, 0.5*chirp_rate, recip_sample_rate, dev_cx_DataArray, dev_cx_ChirpDataArray)),true);
}

#endif //USE_CUDA
