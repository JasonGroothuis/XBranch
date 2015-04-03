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
//32
#define GFK_BLOCK 32
//128
#define GFP_BLOCK 128


#include "cudaAcceleration.h"
#ifdef USE_CUDA

#ifdef _WIN32
extern volatile bool worker_thread_exit_ack;
#endif

#include "cudaAcc_data.h"
//#include "cudaAcc_scan.h"
#include "cudaAcc_analyzeReport.h"

#include "s_util.h"
#include "cudaAcc_utilities.h"
#include "lcgamm.h"

__constant__ cudaAcc_GaussFit_t cudaAcc_GaussFit_settings;
cudaAcc_GaussFit_t settings;

#define CUDAACC_LCGF_CACHE_SIZE (1024*8) /* 1D texture can have up to 2^13 in width*/
#define CUDAACC_LCGF_MAX_VALUE (11.0f - 1.0f)
cudaArray* dev_gauss_dof_lcgf_cache;
cudaArray* dev_null_dof_lcgf_cache;

texture<float, 1, cudaReadModeElementType> dev_gauss_dof_lcgf_cache_TEX;
texture<float, 1, cudaReadModeElementType> dev_null_dof_lcgf_cache_TEX;


inline float SQUARE(float a) { return a*a;  }

#define NEG_LN_ONE_HALF     0.693f
//#define NEG_LN_ONE_HALF		0.69314718055994530941723212145818f
#define EXP(a,b,c)          exp(-(NEG_LN_ONE_HALF * SQUARE((float)(a))) / (float)(c))

__device__ float cudaAcc_weight(int i) 
{
  //return (EXP(i, 0, cudaAcc_GaussFit_settings.GaussSigmaSq));
  
  float * __restrict__ F_weight = cudaAcc_GaussFit_settings.f_weight;
  return F_weight[i];
}


template <int ul_FftLength>
__device__ float cudaAcc_GetPeak(float * __restrict__ fp_PoT, int ul_TOffset, int ul_HalfSumLength, float f_MeanPower, float f_PeakScaleFactor) 
{
  // Peak power is calculated as the weighted
  // sum of all powers within ul_HalfSumLength
  // of the assumed gaussian peak.
  // The weights are given by the gaussian function itself.
  // BUG WATCH : for the f_PeakScaleFactor to work,
  // ul_HalfSumLength *must* be set to sigma.
  
  float * __restrict__ F_weight = cudaAcc_GaussFit_settings.f_weight;
  
  int i;
  
  float4 f_sum = make_float4(0,0,0,0);
  
  // Find a weighted sum
//printf("HSL %d,", ul_HalfSumLength);
  float * __restrict__ pp = &fp_PoT[(ul_TOffset - ul_HalfSumLength) * ul_FftLength];
  float * __restrict__ pp2 = &fp_PoT[(ul_TOffset + ul_HalfSumLength) * ul_FftLength];

  float f_MeanPower2 = f_MeanPower + f_MeanPower;

#pragma unroll1 
  for(i = ul_TOffset - ul_HalfSumLength; i < (ul_TOffset-2); i += 3) 
    { 
      float fw = F_weight[ul_TOffset - i];
      float fw2 = F_weight[ul_TOffset - (i+1)];
      float fw3 = F_weight[ul_TOffset - (i+2)];
      f_sum.x += (*pp + *pp2 - f_MeanPower2) * fw; pp += ul_FftLength; pp2 -= ul_FftLength;
      f_sum.y += (*pp + *pp2 - f_MeanPower2) * fw2; pp += ul_FftLength; pp2 -= ul_FftLength;
      f_sum.z += (*pp + *pp2 - f_MeanPower2) * fw3; pp += ul_FftLength; pp2 -= ul_FftLength;
    }

#pragma unroll 1
  for(; i < ul_TOffset; i++) 
    {
      float fw =  F_weight[ul_TOffset - i];
      f_sum.w += (*pp + *pp2 - f_MeanPower2) * fw; pp += ul_FftLength; pp2 -= ul_FftLength;
    }

  //last
  float fw =  F_weight[ul_TOffset - i];
  f_sum.x += ((*pp) - f_MeanPower) * fw; pp += ul_FftLength;

  f_sum.x += (f_sum.y + (f_sum.z + f_sum.w));

  return(f_sum.x * f_PeakScaleFactor);
}


__device__ float sqrf(float x) 
{
  return x * x;
}


__device__ float cudaAcc_GetChiSq(float * __restrict__ fp_PoT, const int ul_FftLength, int ul_PowerLen, int ul_TOffset, float f_PeakPower, float f_MeanPower, float& xsq_null) 
{
  // We calculate our assumed gaussian powers
  // on the fly as we try to fit them to the
  // actual powers at each point along the PoT.
  ul_PowerLen = 64;
  float f_ChiSq = 0.0f,f_null_hyp=0.0f;
  float recip_MeanPower = __frcp_rn(f_MeanPower);	
  float rebin =  1024*1024 / (ul_FftLength * ul_PowerLen); //cudaAcc_GaussFit_settings.nsamples
  float recip_powerlen = 1.0f/64.0f; //__frcp_rn(ul_PowerLen);
  f_PeakPower *= recip_MeanPower;

  int iidx = 0;

  // ChiSq in this realm is:
  //  sum[0:i]( (observed power - expected power)^2 / expected variance )
  // The power of a signal is:
  //  power = (noise + signal)^2 = noise^2 + signal^2 + 2*noise*signal
  // With mean power normalization, noise becomes 1, leaving:
  //  power = signal^2 +or- 2*signal + 1

  float * __restrict__ F_weight = cudaAcc_GaussFit_settings.f_weight;
  int i = 0;

//#pragma unroll 8
  for( ;  i < ul_PowerLen; ++i) 
    {	
      int a = ul_TOffset - i, b = i - ul_TOffset;
      int fwi = i < ul_TOffset ? a : b;      

      float f_PredictedPower = f_PeakPower * F_weight[fwi];		
      float recip_noise = 1.0f/(sqrt(f_PredictedPower) + 0.5f);
      float PoTval = (*fp_PoT) * recip_MeanPower - 1.0f; fp_PoT += ul_FftLength;
      f_ChiSq += (recip_noise*sqrf(PoTval - f_PredictedPower));
      f_null_hyp+= (recip_noise*sqrf(PoTval));
    }

    f_null_hyp *= recip_powerlen*rebin*0.5f;   //  /= (float)ul_PowerLen;
    f_ChiSq *= recip_powerlen*rebin*0.5f;      //  /= (float)ul_PowerLen;

    xsq_null = f_null_hyp;
    return f_ChiSq;
}

__device__ float cudaAcc_GetTrueMean(float * __restrict__ fp_PoT, int ul_PowerLen, float f_TotalPower, int ul_TOffset, int ul_ExcludeLen, const int ul_FftLength) 
{
  // TrueMean is the mean power of the data set minus all power
  // out to ExcludeLen from our current TOffset.
  int i, i_start, i_lim;
  float f_ExcludePower = 0;

  // take care that we do not add to exclude power beyond PoT bounds!
  i_start = max(ul_TOffset - ul_ExcludeLen, 0);
  i_lim = min(ul_TOffset + ul_ExcludeLen + 1, ul_PowerLen);
  // TODO: prefix sums
  float * __restrict__ pp = &fp_PoT[i_start * ul_FftLength];
  float a = 0, b = 0;
#pragma unroll 8
  for(i = i_start; i < i_lim-3; i+=4) 
    {
      a += *pp + *(pp+ul_FftLength); 
      b += *(pp+2*ul_FftLength) + *(pp+3*ul_FftLength); 
      pp += 4*ul_FftLength;
    }       
  f_ExcludePower = a + b;
#pragma unroll 1
  for(; i < i_lim; i++) 
    {
      f_ExcludePower += *pp; pp += ul_FftLength;
    }       
  return((f_TotalPower - f_ExcludePower) / (ul_PowerLen - (i_lim - i_start)));
}



__device__ float cudaAcc_GetTrueMean2(float* __restrict__ fp_PoTPrefixSum, int ul_PowerLen, float f_TotalPower, int ul_TOffset, int ul_ExcludeLen, const int ul_FftLength) 
{
  // TrueMean is the mean power of the data set minus all power
  // out to ExcludeLen from our current TOffset.
  ul_PowerLen = 64; f_TotalPower = 64;
  int i_start, i_lim;
  float f_ExcludePower = 0;
  
  // take care that we do not add to exclude power beyond PoT bounds!
  i_start = max(ul_TOffset - ul_ExcludeLen -1, -1);
  i_lim = min(ul_TOffset + ul_ExcludeLen, ul_PowerLen - 1);
  
  f_ExcludePower = fp_PoTPrefixSum[i_lim * ul_FftLength];
  if(i_start >= 0)
    f_ExcludePower -= fp_PoTPrefixSum[i_start * ul_FftLength]; 
  
  return((f_TotalPower - f_ExcludePower) / (ul_PowerLen - (i_lim - i_start)));
}



float cudaAcc_GetPeakScaleFactor(float f_sigma) 
{
  // The PeakScaleFactor is calculated such that when used in f_GetPeak(),
  // the actual peak power can be extracted from a weighted sum.
  // This sum (see f_GetPeak()), is calculated as :
  // sum = SUM[x from -sigma to +sigma] of (gaussian weights * our data)
  // The gaussian weights are e^(-x^2 / sigma^2).
  // Our data is A(e^(-x^2 / sigma^2)), where 'A' is the peak power.
  // Through algebraic manipulation, we have:
  // A = sum * (1 / SUM[x from -sigma to +sigma] of (e^(-x^2 / sigma^2))^2.
  // The factor by which we multiply the sum is the PeakScaleFactor.
  // It is completely determined by sigma.
  
  int i, i_s = static_cast<int>(floor(f_sigma+0.5));
  float f_sigma_sq = f_sigma*f_sigma;
  float f_sum = 0.0;
#pragma unroll 3  
  for(i = 1; i <= i_s; i++) 
    {
      f_sum += static_cast<float>(EXP(i, 0, f_sigma_sq));
    }
  f_sum += f_sum + 1;
  
  return(1 / f_sum);
}



// only if (ul_NumSpectra > ul_PoTLen)
template <int ul_FftLength>
__launch_bounds__(GFP_BLOCK, 8)
__global__ void GetFixedPoT_kernel(void) 
{
  int ul_PoT = (blockIdx.x * GFP_BLOCK + threadIdx.x);
  int ul_PoT_i = blockIdx.y; //(blockIdx.y * blockDim.y);

  if(ul_PoT >= ul_FftLength/4) 
    return;
  
  float4 *  __restrict__ fp_PoT = &((float4 *)cudaAcc_GaussFit_settings.dev_PoT)[ul_PoT];
  float4 *  __restrict__ fp_PowerSpectrum = &((float4 *)cudaAcc_GaussFit_settings.dev_PowerSpectrum)[ul_PoT];
  
  int  ul_PoTChunkSize;
  
  //ul_NumSpectra  = 1048576/(ul_FftLength/4); // cudaAcc_GaussFit_settings.NumDataPoints
  ul_PoTChunkSize  = 1048576/(ul_FftLength/4) / 64;
  
  // If the number of spectra is greater than the number
  // of elements in a PoT array, we add sum adjacent spectra
  // into PoT elements.
  // ul_PoTChunkSize indicates how many time-wise
  // power spectra bins are added together to make one PoT bin.
  
  //	float sum = 0.0f;
  float partials1 = 0;
  float partials2 = 0;
  float partials3 = 0;
  float partials4 = 0;
  float4 *ppp = &((float4 *)fp_PowerSpectrum)[1048576 / 64 * ul_PoT_i];

  int i = 0;

  for(; i < ul_PoTChunkSize-3; i += 4) 
    {
      float4 p1, p2;
      p1 = *ppp; p2 = *(ppp+4*ul_FftLength/4); ppp += ul_FftLength/4; 
      partials1 += p1.x + p2.x; 
      partials2 += p1.y + p2.y; 
      partials3 += p1.z + p2.z; 
      partials4 += p1.w + p2.w; 
      ppp += ul_FftLength/4;
      p1 = *ppp; p2 = *(ppp+4*ul_FftLength/4); ppp += ul_FftLength/4; 
      partials1 += p1.x + p2.x; 
      partials2 += p1.y + p2.y; 
      partials3 += p1.z + p2.z; 
      partials4 += p1.w + p2.w; 
      ppp += ul_FftLength/4;
    }

  // tail if needed

  while(i < ul_PoTChunkSize)
    {
      float4 p1, p2;
      p1 = *ppp; ppp += ul_FftLength/4; 
      partials1 += p1.x; 
      partials2 += p1.y; 
      partials3 += p1.z; 
      partials4 += p1.w; 
    }

  fp_PoT[ul_PoT_i*ul_FftLength/4] = make_float4(partials1, partials2, partials3, partials4);
}




#define D 16

#ifdef USE_FLOAT2

#define NPK_B 2
#define NPK_BLOCK 64

template <int ul_FftLength>
__launch_bounds__(NPK_BLOCK, 4)
__global__ void NormalizePoT_kernel(void) 
{
  int ul_PoT = (blockIdx.x * NPK_BLOCK + threadIdx.x)*NPK_B;
  if (ul_PoT >= (ul_FftLength*NPK_B)) return;
  
  float2 * __restrict__ fp_PoT = (float2 *)&cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
  float2 * __restrict__ pp = fp_PoT;
  float2 * __restrict__ fp_PoTPrefixSum = (float2 *)&cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT];

  float2 f_TotalPower = {0,0}, f_TotalPower2 = {0,0};
  register float2 a[D];
  register float2 b[64];
#pragma unroll
  for(int i = 0; i < 64; i += D) 
    {
#pragma unroll
      for(int j = 0; j < D; j++)
	{
          b[i+j] = a[j] = *pp; *pp += ul_FftLength/NPK_B; //fp_PoT[(i+j) * ul_FftLength/NPK_B];
	}
  
#pragma unroll
      for(int j = 0; j < (D>>1); j++)
        {
	  a[j].x += a[(D>>1) + j].x;
	  a[j].y += a[(D>>1) + j].y;
	}

#pragma unroll
      for(int j = 0; j < (D>>2); j++)
	{
	  a[j].x += a[(D>>2) + j].x;
	  a[j].y += a[(D>>2) + j].y;
	}

#pragma unroll
      for(int j = 0; j < (D>>3); j++)
	{
	  a[j].x += a[(D>>3) + j].x;
	  a[j].y += a[(D>>3) + j].y;
	}

      f_TotalPower.x  += a[0].x;
      f_TotalPower.y  += a[0].y;
      f_TotalPower2.x += a[1].x;
      f_TotalPower2.y += a[1].y;
    }

  f_TotalPower.x += f_TotalPower2.x;
  f_TotalPower.y += f_TotalPower2.y;
  // Normalize power-of-time

  float2 fr_MeanPower;
  fr_MeanPower.x = 64.0f / f_TotalPower.x;
  fr_MeanPower.y = 64.0f / f_TotalPower.y;

  float2 f_NormMaxPower = {0,0}, f_NormMaxPower2 = {0,0};
  float2 sum = {0,0}, sum2 = {0,0};
  for(int i = 0; i < 64; i+=2) 
    {
      float2 PoT  = make_float2(b[i].x * fr_MeanPower.x, b[i].y * fr_MeanPower.y);
      float2 PoT2 = make_float2(b[i+1].x * fr_MeanPower.x, b[i+1].y * fr_MeanPower.y);
      sum.x  += PoT.x;
      sum.y  += PoT.y;
      f_NormMaxPower.x  = max(f_NormMaxPower.x, PoT.x);
      f_NormMaxPower.y  = max(f_NormMaxPower.y, PoT.y);
      fp_PoT[i * ul_FftLength/NPK_B] = PoT;
      sum2.x += (PoT.x + PoT2.x);
      sum2.y += (PoT.y + PoT2.y);
      f_NormMaxPower2.x = max(f_NormMaxPower2.x, PoT2.x);
      f_NormMaxPower2.y = max(f_NormMaxPower2.y, PoT2.y);
      fp_PoT[(i+1) * ul_FftLength/NPK_B] = PoT2;
      fp_PoTPrefixSum[i * ul_FftLength/NPK_B] = sum;
      fp_PoTPrefixSum[(i+1) * ul_FftLength/NPK_B] = sum2;
      sum = sum2;
    }

  f_NormMaxPower.x = max(f_NormMaxPower.x, f_NormMaxPower2.x);
  f_NormMaxPower.y = max(f_NormMaxPower.y, f_NormMaxPower2.y);

  *(float2*)&cudaAcc_GaussFit_settings.dev_NormMaxPower[ul_PoT] = f_NormMaxPower;
}

#else

#define NPK_BLOCK 128

template <int ul_FftLength>
__launch_bounds__(NPK_BLOCK, 4)

__global__ void NormalizePoT_kernel(void) 
{
  int ul_PoT = blockIdx.x * NPK_BLOCK + threadIdx.x;
  if (ul_PoT >= ul_FftLength) return;
  
  float * __restrict__ fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
  float * __restrict__ fp_PoTPrefixSum = &cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT];

  float f_TotalPower = 0.0f, f_TotalPower2 = 0.0f;;
  register float a[D];
  register float b[64];
#pragma unroll
  for(int i = 0; i < 64; i += D) 
    {
#pragma unroll
      for(int j = 0; j < D; j++)
        b[i+j] = a[j] = fp_PoT[(i+j) * ul_FftLength];
  
#pragma unroll
      for(int j = 0; j < (D>>1); j++)
        a[j] += a[(D>>1) + j];

#pragma unroll
      for(int j = 0; j < (D>>2); j++)
        a[j] += a[(D>>2) + j];

#pragma unroll
      for(int j = 0; j < (D>>3); j++)
        a[j] += a[(D>>3) + j];

      f_TotalPower  += a[0];
      f_TotalPower2 += a[1];
    }

  f_TotalPower += f_TotalPower2;
  // Normalize power-of-time

  float fr_MeanPower = 64.0f / f_TotalPower;

  float f_NormMaxPower = 0, f_NormMaxPower2 = 0;
  float sum = 0, sum2 = 0;
  for(int i = 0; i < 64; i+=2) 
    {
      float PoT  = b[i] * fr_MeanPower;
      float PoT2 = b[i+1] * fr_MeanPower;
      sum  += PoT;
      f_NormMaxPower  = max(f_NormMaxPower, PoT);
      fp_PoT[i * ul_FftLength] = PoT;
      sum2 += (PoT + PoT2);
      f_NormMaxPower2 = max(f_NormMaxPower2, PoT2);
      fp_PoT[(i+1) * ul_FftLength] = PoT2;
      fp_PoTPrefixSum[i * ul_FftLength] = sum;
      fp_PoTPrefixSum[(i+1) * ul_FftLength] = sum2;
      sum = sum2;
    }

  f_NormMaxPower = max(f_NormMaxPower, f_NormMaxPower2);

  cudaAcc_GaussFit_settings.dev_NormMaxPower[ul_PoT] = f_NormMaxPower;
}
#endif


#define ITMAX 10000  // Needs to be a few times the sqrt of the max. input to lcgf

__device__ float cudaAcc_gammln(float a) {
  float x,y,tmp,ser;
  float cof[6]={76.18009172947146f,-86.50532032941677f,
		24.01409824083091f,-1.231739572450155f,
		0.1208650973866179e-2f,-0.5395239384953e-5f};
  
  y=x=a;
  tmp=x+5.5f;
  tmp -= (x+0.5f)*log(tmp);
  ser=1.000000000190015f;
  for (int j=0;j<=5;j++) ser += cof[j]/++y;
  return (float)(-tmp+log(2.5066282746310005f*ser/x));
}

__device__ float cudaAcc_lcgf(float a, float x) {
  const float EPS= 1.19209e-006f; //007;//std::numeric_limits<double>::epsilon();
  const float FPMIN= 9.86076e-031f; //032;//std::numeric_limits<double>::min()/EPS;
  float an,b,c,d,del,h,gln=cudaAcc_gammln(a);
  
  // assert(x>=(a+1));
  //BOINCASSERT(x>=(a+1));
  b=x+1.0f-a;
  c=1.0f/FPMIN;
  d=1.0f/b;
  h=d;
  for (int i=1;i<=ITMAX;++i) {
    an = -i*(i-a);
    b += 2.0f;
    d=an*d+b;
    if (fabs(d)<FPMIN) d=FPMIN;
    c=b+an/c;
    if (fabs(c)<FPMIN) c=FPMIN;
    d=1.0f/d;
    del=d*c;
    h*=del;
    if (fabs(del-1.0f)<EPS) break;
  }
  // assert(i<ITMAX);
  //BOINCASSERT(i<ITMAX);
  return (float)(log(h)-x+a*log(x)-gln);
}

__device__ float cudaAcc_calc_GaussFit_score(float chisqr, float null_chisqr, float gauss_pot_length) 
{ // <- gauss_pot_length constant across whole package
  float gauss_bins = gauss_pot_length;
  float gauss_dof = gauss_bins * 0.5f - 1.0f;
  float null_dof = gauss_bins * 0.5f - 0.5f;
  gauss_bins *= 0.5f;
  return  cudaAcc_GaussFit_settings.score_offset +
    cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f))
    //-cudaAcc_lcgf(gauss_dof,cudaAcc_GaussFit_settings.gauss_chi_sq_thresh*gauss_bins) // <- always the same result
    -cudaAcc_lcgf(null_dof,max(null_chisqr*gauss_bins,null_dof+1.0f));
  //+cudaAcc_lcgf(null_dof,cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh*gauss_bins); // <- always the same result	
}


// assuming that chisqr and null_chisqr are less than CUDAACC_LCGF_MAX_VALUE + 1. if not the parameters are clamped to CUDAACC_LCGF_MAX_VALUE
__device__ float cudaAcc_calc_GaussFit_score_cached(float chisqr, float null_chisqr)
 { 	   	 					  // <- gauss_pot_length constant across whole package
  float chisqr_cache = (chisqr - 1.0f) / CUDAACC_LCGF_MAX_VALUE; // texture addresMode is cudaAddressModeClamp so behaviour for x < 0.0 and >= 1.0 are well defined
  float null_chisqr_cache = (null_chisqr - 1.0f) / CUDAACC_LCGF_MAX_VALUE; // texture addresMode is cudaAddressModeClamp so behaviour for x < 0.0 and >= 1.0 are well defined
  return cudaAcc_GaussFit_settings.score_offset 
    // cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f)) +cudaAcc_lcgf(null_dof,cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh*gauss_bins)
    + tex1D(dev_gauss_dof_lcgf_cache_TEX, chisqr_cache)		//cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f))
    + tex1D(dev_null_dof_lcgf_cache_TEX, null_chisqr_cache)	// -cudaAcc_lcgf(null_dof,max(null_chisqr*gauss_bins,null_dof+1.0f))
    ; // using textures linear filtering (interpolation)
}

template <int ul_FftLength>
__global__ void GaussFit_kernel(float best_gauss_score, result_flag* flags, bool noscore) 
{
  int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;    
  int ul_TOffset = blockIdx.y * blockDim.y + threadIdx.y + cudaAcc_GaussFit_settings.GaussTOffsetStart;
  if (ul_PoT >= ul_FftLength) return;
  if (ul_TOffset >= cudaAcc_GaussFit_settings.GaussTOffsetStop) return;
  
  float* fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
  float f_null_hyp;
  float4 * __restrict__ resp = &cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT];
  
  int iSigma = cudaAcc_GaussFit_settings.iSigma;
  
  float f_TotalPower = 64, f_TrueMean, f_ChiSq, f_PeakPower;
  
  // slide dynamic gaussian across the Power Of Time array
  
  // TrueMean is the mean power of the data set minus all power
  // out to 2 sigma from our current TOffset.	
  f_TrueMean = cudaAcc_GetTrueMean2(
				    &cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT],
				    64,
				    f_TotalPower,
				    ul_TOffset,
				    2 * iSigma,
				    ul_FftLength
				    );
  
  float rm = 1.0f/f_TrueMean;
  f_PeakPower = cudaAcc_GetPeak<ul_FftLength>(fp_PoT, ul_TOffset, iSigma, f_TrueMean, cudaAcc_GaussFit_settings.PeakScaleFactor);
  
  // worth looking at ?
  if(f_PeakPower*rm < cudaAcc_GaussFit_settings.GaussPeakPowerThresh3) 
    {
      *resp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      return;
    } 
  
  // look at it - try to fit  
  f_ChiSq = cudaAcc_GetChiSq(
			     fp_PoT,
			     ul_FftLength,
			     64,
			     ul_TOffset,
			     f_PeakPower,
			     f_TrueMean,        
			     f_null_hyp
			     );
  
  if(noscore)
    {
      if(((f_ChiSq <=  cudaAcc_GaussFit_settings.GaussChiSqThresh) && (f_null_hyp >= cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh)) ) 
	{
	  flags->has_results = 1;
	  *resp = make_float4(f_TrueMean, f_PeakPower, f_ChiSq, f_null_hyp);
	} 
      else 
	{
	  *resp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}    
    } 
  else 
    {
      float score = cudaAcc_calc_GaussFit_score_cached(f_ChiSq, f_null_hyp);
      if (((f_ChiSq <=  cudaAcc_GaussFit_settings.gauss_chi_sq_thresh) && (score > best_gauss_score)) 
	  || ((f_ChiSq <=  cudaAcc_GaussFit_settings.GaussChiSqThresh) && (f_null_hyp >= cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh))  ) 
	{
	  flags->has_results = 1;
	  *resp = make_float4(f_TrueMean, f_PeakPower, f_ChiSq, f_null_hyp);
	} 
      else 
	{
	  *resp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}    
    }
} // End of gaussfit()


int cudaAcc_initializeGaussfit(const PoTInfo_t& PoTInfo, int gauss_pot_length, unsigned int nsamples, double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh) {
  
  cudaError_t cu_err;
  
  settings.iSigma = static_cast<int>(floor(PoTInfo.GaussSigma+0.5));
  fprintf(stderr, "Sigma %d\r\n", settings.iSigma);
  settings.GaussSigmaSq = (float) PoTInfo.GaussSigmaSq;
  settings.GaussPowerThresh = (float) PoTInfo.GaussPowerThresh;
  settings.GaussPeakPowerThresh3 = (float) PoTInfo.GaussPeakPowerThresh / 3.0f;
  settings.GaussChiSqThresh = (float) PoTInfo.GaussChiSqThresh;
  settings.gauss_null_chi_sq_thresh = (float) gauss_null_chi_sq_thresh;
  settings.gauss_chi_sq_thresh = (float) gauss_chi_sq_thresh;
  settings.GaussTOffsetStart = PoTInfo.GaussTOffsetStart;
  settings.GaussTOffsetStop = PoTInfo.GaussTOffsetStop;
  settings.gauss_pot_length = 64; //gauss_pot_length;
  settings.dev_PoT = dev_PoT;
  settings.dev_PoTPrefixSum = dev_PoTPrefixSum;
  settings.dev_PowerSpectrum = dev_PowerSpectrum;
  settings.dev_GaussFitResults = dev_GaussFitResults;
  settings.dev_GaussFitResultsReordered = dev_GaussFitResultsReordered;
  settings.dev_GaussFitResultsReordered2 = dev_GaussFitResultsReordered2;
  settings.dev_NormMaxPower = dev_NormMaxPower;    
  settings.dev_outputposition = dev_outputposition;
  settings.PeakScaleFactor = cudaAcc_GetPeakScaleFactor(static_cast<float>(PoTInfo.GaussSigma));
  settings.NumDataPoints = cudaAcc_NumDataPoints;
  settings.nsamples = nsamples;
  if (CUDA_ACC_MAX_GaussTOffsetStop < PoTInfo.GaussTOffsetStop) {
    SETIERROR(UNSUPPORTED_FUNCTION, "cudaAcc_Gaussfit doesn't support (CUDA_ACC_MAX_GaussTOffsetStop < PoTInfo.GaussTOffsetStop) in cudaAcc_initializeGaussfit");
  }

  for(int i = 0; i < PoTInfo.GaussTOffsetStop; i++) 
    {
      settings.f_weight[i] = static_cast<float>(EXP(i, 0, PoTInfo.GaussSigmaSq));
    }

  // creating cache of lcgf for score calculation
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  double gauss_bins=gauss_pot_length;
  double gauss_dof=gauss_bins-2;
  double null_dof=gauss_bins-1;	
  settings.score_offset =	(float)-lcgf(0.5*gauss_dof,gauss_chi_sq_thresh*0.5*gauss_bins) + lcgf(0.5*null_dof,gauss_null_chi_sq_thresh*0.5*gauss_bins);
  
  CUDA_ACC_SAFE_CALL(  (cu_err = cudaMallocArray( &dev_gauss_dof_lcgf_cache, &channelDesc, CUDAACC_LCGF_CACHE_SIZE, 1 )),true);
  if( cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMallocArray( &dev_gauss_dof_lcgf_cache");
      return 1;
    } else { CUDAMEMPRINT(dev_gauss_dof_lcgf_cache,"cudaMallocArray( &dev_gauss_dof_lcgf_cache",1,CUDAACC_LCGF_CACHE_SIZE); };
  
  CUDA_ACC_SAFE_CALL( (cu_err = cudaMallocArray( &dev_null_dof_lcgf_cache, &channelDesc, CUDAACC_LCGF_CACHE_SIZE, 1 )),true); 
  if( cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMallocArray( &dev_null_dof_lcgf_cache");
      return 1;
    } else { CUDAMEMPRINT(dev_null_dof_lcgf_cache,"cudaMallocArray( &dev_null_dof_lcgf_cache",1,CUDAACC_LCGF_CACHE_SIZE); };
  
  float* cache = (float*) malloc(CUDAACC_LCGF_CACHE_SIZE * sizeof(float));
  
  for (int i = 0; i < CUDAACC_LCGF_CACHE_SIZE; ++i) {
    double chisqr = 1.0 + (double) i / CUDAACC_LCGF_CACHE_SIZE * CUDAACC_LCGF_MAX_VALUE;
    cache[i] = (float)lcgf(0.5*gauss_dof,std::max(chisqr*0.5*gauss_bins,0.5*gauss_dof+1));
  }
  
  CUDA_ACC_SAFE_CALL( (cudaMemcpyToArray( dev_gauss_dof_lcgf_cache, 0, 0, cache, CUDAACC_LCGF_CACHE_SIZE * sizeof(float), cudaMemcpyHostToDevice)),true);	
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  for (int i = 0; i < CUDAACC_LCGF_CACHE_SIZE; ++i) {
    double null_chisqr = 1.0 + (double) i / CUDAACC_LCGF_CACHE_SIZE * CUDAACC_LCGF_MAX_VALUE;
    cache[i] = (float)-lcgf(0.5*null_dof,std::max(null_chisqr*0.5*gauss_bins,0.5*null_dof+1));
  }
  
  CUDA_ACC_SAFE_CALL( (cudaMemcpyToArray( dev_null_dof_lcgf_cache, 0, 0, cache, CUDAACC_LCGF_CACHE_SIZE * sizeof(float), cudaMemcpyHostToDevice)),true);	
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  free(cache);

  dev_gauss_dof_lcgf_cache_TEX.normalized = true;
  dev_gauss_dof_lcgf_cache_TEX.filterMode = cudaFilterModeLinear;
  dev_gauss_dof_lcgf_cache_TEX.addressMode[0] = cudaAddressModeClamp;
  dev_null_dof_lcgf_cache_TEX.normalized = true;
  dev_null_dof_lcgf_cache_TEX.filterMode = cudaFilterModeLinear;
  dev_null_dof_lcgf_cache_TEX.addressMode[0] = cudaAddressModeClamp;
  CUDA_ACC_SAFE_CALL( (cudaBindTextureToArray( dev_gauss_dof_lcgf_cache_TEX, dev_gauss_dof_lcgf_cache, channelDesc)),true);	
  CUDA_ACC_SAFE_CALL( (cudaBindTextureToArray( dev_null_dof_lcgf_cache_TEX, dev_null_dof_lcgf_cache, channelDesc)),true);	
  
  CUDA_ACC_SAFE_CALL( (cudaMemcpyToSymbol(cudaAcc_GaussFit_settings, (void*) &settings, sizeof(settings))),true);
  
  return 0;
}



void cudaAcc_free_Gaussfit() 
{
  if (dev_gauss_dof_lcgf_cache) cudaFreeArray(dev_gauss_dof_lcgf_cache);
  if (dev_null_dof_lcgf_cache) cudaFreeArray(dev_null_dof_lcgf_cache);
}



result_flag *Gflags = NULL;	

int cudaAcc_GaussfitStart(int ul_FftLength, double best_gauss_score, bool noscore) 
{
  if (!cudaAcc_initialized()) return -1;
  
  if(Gflags == NULL)
    cudaMallocHost(&Gflags, sizeof(*Gflags));

  dim3 block(NPK_BLOCK, 1, 1);
#ifdef USE_FLOAT2
  dim3 grid((ul_FftLength + (block.x*NPK_B) - 1) / (block.x*NPK_B), 1, 1); 
#else
  dim3 grid((ul_FftLength + (block.x) - 1) / (block.x), 1, 1); 
#endif

  int ul_NumSpectra    = 1024*1024 / ul_FftLength; //cudaAcc_NumDataPoints
  
  if (ul_NumSpectra == settings.gauss_pot_length) 
    {
      CUDA_ACC_SAFE_LAUNCH( (cudaMemcpyAsync(dev_PoT, dev_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_PowerSpectrum), cudaMemcpyDeviceToDevice, fftstream1)),true);        
    } 
  else if (ul_NumSpectra > settings.gauss_pot_length) 
    {
      dim3 blockPOT(GFP_BLOCK, 1, 1);
      dim3 gridPOT((ul_FftLength/4 + blockPOT.x - 1) / (blockPOT.x), settings.gauss_pot_length, 1);         
      switch(ul_FftLength)
	{
	case 8:
	  GetFixedPoT_kernel<8><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 16:
	  GetFixedPoT_kernel<16><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 32:
	  GetFixedPoT_kernel<32><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 64:
	  GetFixedPoT_kernel<64><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 128:
	  GetFixedPoT_kernel<128><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 256:
	  GetFixedPoT_kernel<256><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 512:
	  GetFixedPoT_kernel<512><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 1024:
	  GetFixedPoT_kernel<1024><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 2048:
	  GetFixedPoT_kernel<2048><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 4096:
	  GetFixedPoT_kernel<4096><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 8192:
	  GetFixedPoT_kernel<8192><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 16384:
	  GetFixedPoT_kernel<16384><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 32768:
	  GetFixedPoT_kernel<32768><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 65536:
	  GetFixedPoT_kernel<65536><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 131072:
	  GetFixedPoT_kernel<131072><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	case 262144:
	  GetFixedPoT_kernel<262144><<<gridPOT, blockPOT, 0, fftstream1>>>();
	  break;
	}
     // CUDA_ACC_SAFE_CALL_NO_SYNC("GetFixedPoT_kernel");		
    } 
  else 
    {
      SETIERROR(UNSUPPORTED_FUNCTION, "cudaAcc_Gaussfit doesn't support if (ul_NumSpectra < settings.gauss_pot_length) in GetFixedPoT");
    }

  switch(ul_FftLength)
    {
    case 8:
      NormalizePoT_kernel<8><<<grid, block, 0, fftstream1>>>();
      break;
    case 16:
      NormalizePoT_kernel<16><<<grid, block, 0, fftstream1>>>();
      break;
    case 32:
      NormalizePoT_kernel<32><<<grid, block, 0, fftstream1>>>();
      break;
    case 64:
      NormalizePoT_kernel<64><<<grid, block, 0, fftstream1>>>();
      break;
    case 128:
      NormalizePoT_kernel<128><<<grid, block, 0, fftstream1>>>();
      break;
    case 256:
      NormalizePoT_kernel<256><<<grid, block, 0, fftstream1>>>();
      break;
    case 512:
      NormalizePoT_kernel<512><<<grid, block, 0, fftstream1>>>();
      break;
    case 1024:
      NormalizePoT_kernel<1024><<<grid, block, 0, fftstream1>>>();
      break;
    case 2048:
      NormalizePoT_kernel<2048><<<grid, block, 0, fftstream1>>>();
      break;
    case 4096:
      NormalizePoT_kernel<4096><<<grid, block, 0, fftstream1>>>();
      break;
    case 8192:
      NormalizePoT_kernel<8192><<<grid, block, 0, fftstream1>>>();
      break;
    case 16384:
      NormalizePoT_kernel<16384><<<grid, block, 0, fftstream1>>>();
      break;
    case 32768:
      NormalizePoT_kernel<32768><<<grid, block, 0, fftstream1>>>();
      break;
    case 65536:
      NormalizePoT_kernel<65536><<<grid, block, 0, fftstream1>>>();
      break;
    case 131072:
      NormalizePoT_kernel<131072><<<grid, block, 0, fftstream1>>>();
      break;
    case 262144:
      NormalizePoT_kernel<262144><<<grid, block, 0, fftstream1>>>();
      break;
    }
  //	CUDA_ACC_SAFE_CALL_NO_SYNC("NormalizePoT_kernel");
  
  CUDA_ACC_SAFE_LAUNCH((cudaMemsetAsync(dev_flag, 0, sizeof(*dev_flag), fftstream1)),true);		
//  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  dim3 block2(GFK_BLOCK, 4, 1);
  dim3 grid2((ul_FftLength + block2.x - 1) / block2.x, (settings.GaussTOffsetStop - settings.GaussTOffsetStart + block2.y) / block2.y, 1);
  switch(ul_FftLength)
  {
  case 8:
    GaussFit_kernel<8><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 16:
    GaussFit_kernel<16><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 32:
    GaussFit_kernel<32><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 64:
    GaussFit_kernel<64><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 128:
    GaussFit_kernel<128><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 256:
    GaussFit_kernel<256><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 512:
    GaussFit_kernel<512><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 1024:
    GaussFit_kernel<1024><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 2048:
    GaussFit_kernel<2048><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 4096:
    GaussFit_kernel<4096><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 8192:
    GaussFit_kernel<8192><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 16384:
    GaussFit_kernel<16384><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 32768:
    GaussFit_kernel<32768><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 65536:
    GaussFit_kernel<65536><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 131072:
    GaussFit_kernel<131072><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  case 262144:
    GaussFit_kernel<262144><<<grid2, block2, 0, fftstream1>>>((float) best_gauss_score, dev_flag, noscore);
    break;
  }
  
//  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  Gflags->has_results = -1;
  CUDA_ACC_SAFE_LAUNCH( (cudaMemcpyAsync(Gflags, dev_flag, sizeof(*dev_flag), cudaMemcpyDeviceToHost, fftstream1)),true);

  return 0;
}


int cudaAcc_fetchGaussfitFlags(int ul_FftLength, double best_gauss_score) 
{
  if(Gflags->has_results == -1) // no reults (or not yet ready)
    {
      // CUDA_ACC_SAFE_CALL((CUDASYNC), true); // sync wait for results just in case.
      cudaStreamSynchronize(fftstream1);
    }
  
  if(Gflags->has_results > 0) 
    {
      // Download all the results

      CUDA_ACC_SAFE_LAUNCH( (cudaMemcpyAsync(GaussFitResults,
				      dev_GaussFitResults, 
				      cudaAcc_NumDataPoints * sizeof(*dev_GaussFitResults),
					     cudaMemcpyDeviceToHost, fftstream1)),true);  // TODO: Download a little bit less data

      CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(tmp_PoT, dev_NormMaxPower, ul_FftLength * sizeof(*dev_NormMaxPower), cudaMemcpyDeviceToHost, fftstream1)),true);

      
      // Preparing data for cudaAcc_getPoT
      cudaAcc_transposeGPU(dev_t_PowerSpectrum, dev_PoT, ul_FftLength, settings.gauss_pot_length, fftstream1);

      CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(best_PoT, dev_t_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrum), cudaMemcpyDeviceToHost, fftstream1)),true); 
    }

  // TODO: Download a little bit less data
  return Gflags->has_results;
}


int cudaAcc_processGaussFit(int ul_FftLength, double best_gauss_score)
{
  //CUDA_ACC_SAFE_CALL((CUDASYNC), true); // fetch and wait.
  cudaStreamSynchronize(fftstream1); // fetch and wait
  
  int result_count = 0;
  for(int TOffset = settings.GaussTOffsetStart; TOffset < settings.GaussTOffsetStop; TOffset++) 
    {
      for(int ThisPoT = 1; ThisPoT < ul_FftLength; ThisPoT++) 
	{			
	  int index = (TOffset * ul_FftLength + ThisPoT);
	  float4 res1 = GaussFitResults[index];
	  if(res1.x > 0) 
	    {						
	      float f_TrueMean = res1.x;
	      float f_PeakPower = res1.y;                                
	      float f_ChiSq = res1.z;
	      float f_null_hyp = res1.w;
	      float f_NormMaxPower = tmp_PoT[ThisPoT];
	      float sigma = static_cast<float>(PoTInfo.GaussSigma);    
	      
	      int retval = cudaAcc_ChooseGaussEvent(
						    TOffset,
						    f_PeakPower,
						    f_TrueMean,
						    f_ChiSq,
						    f_null_hyp,
						    ThisPoT,
						    sigma,
						    f_NormMaxPower,
						    &best_PoT[ThisPoT * 64]
						    );						
	      result_count++;
	    }
	}					
    }

  return 0;
}
#endif //USE_CUDA
