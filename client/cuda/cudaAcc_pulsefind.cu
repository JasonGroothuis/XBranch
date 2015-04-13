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
// 192 x1
// 128 x2 fast
// 256 x3 ? before
// pulsefind
#define PFTHR 128

//triplets 128 o
#define FTKTHR 128

#include "cudaAcceleration.h"

#ifdef _WIN32
	extern volatile bool worker_thread_exit_ack;
#endif

#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_analyzeReport.h"
#include "cudaAcc_pulsefind.h"

#include "lcgamm.h"
#include "s_util.h"

#include "cudaAcc_utilities.h"
#include "confsettings.h"

float4* dev_TripletResults;
float4* TripletResults;

double *pangle_range;

#define MAX_TRIPLETS_ABOVE_THRESHOLD 11
#define CUDA_ACC_FOLDS_COUNT 4
#define CUDA_ACC_FOLDS_START 2

#define CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD           0x1
#define CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT          0x2

#define AT_XY(x, y, height) ((y) * (height) + (x))
#define AT(i) AT_XY(ul_PoT, i + TOffset, ul_FftLength)
#define MIN(a,b)	((a) < (b) ? (a) : (b))

#define RECIP_12 0.083333333f

float* dev_t_funct_cache;
float4* dev_PulseResults;
float4* PulseResults;
//float* dev_avg;
float* dev_tmp_potP; // pulse
float* dev_tmp_potT; //triplet
float* dev_best_potP;
float* dev_report_potP;

result_find_pulse_flag* dev_find_pulse_flag;

typedef struct { 
	int NumDataPoints;
	// find_triplets
	float *power_ft;
	result_flag* result_flags_ft; 	
	float* tmp_potT;		// Temporary array
	float* report_potT;		// Copy folded pots for reporting
	float4* resultsT;		// Additional data for reporting

	// find_pulse
	float* PulsePot_fp;		// Input data
//	float* PulsePot8_fp;	// Input data moved 8 bytes forward for coleased reads
	float* tmp_potP;		// Temporary array
	float* best_potP;		// Copy folded pots with best score
	float* report_potP;		// Copy folded pots for reporting
	float4* resultsP;		// Additional data for reporting
//	float* avg;				// averages cache
	result_find_pulse_flag* result_flags_fp;

	float* t_funct_cache_fp; // cached results of cudaAcc_t_funct								   								   
	float rcfg_dis_thresh;
	int PulseMax;								                                      
} cudaAcc_PulseFind_t;

__constant__ cudaAcc_PulseFind_t cudaAcc_PulseFind_settings;
cudaAcc_PulseFind_t PulseFind_settings;

int cudaAcc_PulseMax;
float cudaAcc_rcfg_dis_thresh;

// this needs to be the same as blockDim.x anyway, just use that in the kernel
// Allows us to adjust based on optimal occupancy/bandwidth for different devices
// #define TF_BLOCK_DIM 16

__device__ float mean[262144];



template <int ul_FftLength, int blockx>
__global__ void __launch_bounds__(FTKTHR, 4)
calculate_mean_kernel(int len_power, int AdvanceBy) 
{
  int ul_PoT = blockIdx.x * blockx + threadIdx.x;
  if ((ul_PoT < 1) || (ul_PoT >= ul_FftLength)) return; // Original find_triplets, omits first element

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = 0;
  float * __restrict__ power   = cudaAcc_PulseFind_settings.power_ft;

  int PoTLen = (1024*1024) / ul_FftLength;
  int TOffset = y * AdvanceBy;
  if(TOffset >= PoTLen - len_power) 
    {
      TOffset = PoTLen - len_power;
    }

  float * __restrict__ pp = &power[AT(0)];
  float mean_power = 0, mean_power2 = 0;

#define D 16
  register float a[D];

#pragma unroll
  for(; i < (len_power - (D-1)); i += D) 
    {
#pragma unroll
      for(int j = 0; j < D; j++)
        {
	  a[j] = *pp; pp+=ul_FftLength; //pp[(i+j) * ul_FftLength];
	}
  
#pragma unroll
      for(int j = 0; j < (D>>1); j++)
        a[j] += a[(D>>1) + j];

#pragma unroll
      for(int j = 0; j < (D>>2); j++)
        a[j] += a[(D>>2) + j];

#pragma unroll
      for(int j = 0; j < (D>>3); j++)
        a[j] += a[(D>>3) + j];

      mean_power += a[0];
      mean_power2 += a[1];
    }

  //  pp = pp + i*ul_FftLength;

#pragma unroll 1
  while(i < len_power-3)
    {
      mean_power  += *pp; pp += ul_FftLength; i++;
      mean_power2 += *pp; pp += ul_FftLength; i++;
      mean_power  += *pp; pp += ul_FftLength; i++;
      mean_power2 += *pp; pp += ul_FftLength; i++;
    }

#pragma unroll 1
  while(i < len_power)
    {
      mean_power += *pp; pp += ul_FftLength; i++;
    }

  mean_power += mean_power2;

  mean_power *= (1.0f/(float)len_power);

  mean[ul_PoT+y*ul_FftLength] = mean_power;
}



template <int ul_FftLength, int blockx, int len_power>
__global__ void __launch_bounds__(FTKTHR, 4)
calculate_mean_kernel(int AdvanceBy) 
{
  int ul_PoT = blockIdx.x * blockx + threadIdx.x;
  if ((ul_PoT < 1) || (ul_PoT >= ul_FftLength)) return; // Original find_triplets, omits first element

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = 0;
  float * __restrict__ power   = cudaAcc_PulseFind_settings.power_ft;

  int PoTLen = (1024*1024) / ul_FftLength;
  int TOffset = y * AdvanceBy;
  if(TOffset >= PoTLen - len_power) 
    {
      TOffset = PoTLen - len_power;
    }

  float * __restrict__ pp = &power[AT(0)];
  float mean_power = 0, mean_power2 = 0;

#define D 16
  register float a[D];

#pragma unroll
  for(; i < (len_power - (D-1)); i += D) 
    {
#pragma unroll
      for(int j = 0; j < D; j++)
        {
	  a[j] = *pp; pp+=ul_FftLength; //pp[(i+j) * ul_FftLength];
	}
  
#pragma unroll
      for(int j = 0; j < (D>>1); j++)
        a[j] += a[(D>>1) + j];

#pragma unroll
      for(int j = 0; j < (D>>2); j++)
        a[j] += a[(D>>2) + j];

#pragma unroll
      for(int j = 0; j < (D>>3); j++)
        a[j] += a[(D>>3) + j];

      mean_power += a[0];
      mean_power2 += a[1];
    }

  //  pp = pp + i*ul_FftLength;

#pragma unroll 1
  while(i < len_power-3)
    {
      mean_power  += *pp; pp += ul_FftLength; i++;
      mean_power2 += *pp; pp += ul_FftLength; i++;
      mean_power  += *pp; pp += ul_FftLength; i++;
      mean_power2 += *pp; pp += ul_FftLength; i++;
    }

#pragma unroll 1
  while(i < len_power)
    {
      mean_power += *pp; pp += ul_FftLength; i++;
    }

  mean_power += mean_power2;

  mean_power *= (1.0f/(float)len_power);

  mean[ul_PoT+y*ul_FftLength] = mean_power;
}



int cudaAcc_calculate_mean(int PulsePoTLen, float triplet_thresh, int AdvanceBy, int FftLength) 
{
  int PoTLen = 1024*1024 / FftLength; //cudaAcc_NumDataPoints

  cudaStreamWaitEvent(fftstream0, powerspectrumDoneEvent, 0);	 

  dim3 block(64, 1, 1);
  dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
  if(gCudaDevProps.major >= 2) 
    {
      dim3 block(FTKTHR, 1, 1);
      dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
 
      switch(FftLength)
	{
	case 8:
	  if(PulsePoTLen == 131072)
	    calculate_mean_kernel<8, FTKTHR, 131072><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<8, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 16:
	  if(PulsePoTLen == 65536)
	    calculate_mean_kernel<16, FTKTHR, 65536><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<16, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 32:
	  if(PulsePoTLen == 32768)
	    calculate_mean_kernel<32, FTKTHR, 32768><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<32, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 64:
	  if(PulsePoTLen == 16384)
	    calculate_mean_kernel<64, FTKTHR, 16384><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<64, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 128:
	  if(PulsePoTLen == 8192)
	    calculate_mean_kernel<128, FTKTHR, 8192><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<128, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 256:
	  if(PulsePoTLen == 4096)
	    calculate_mean_kernel<256, FTKTHR, 4096><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<256, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 512:
	  if(PulsePoTLen == 2048)
	    calculate_mean_kernel<512, FTKTHR, 2048><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<512, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 1024:
	  if(PulsePoTLen == 1024)
	    calculate_mean_kernel<1024, FTKTHR, 1024><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<1024, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 2048:
	  if(PulsePoTLen == 512)
	    calculate_mean_kernel<2048, FTKTHR, 512><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<2048, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 4096:
	  if(PulsePoTLen == 256)
	    calculate_mean_kernel<4096, FTKTHR, 256><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<4096, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 8192:
	  if(PulsePoTLen == 128)
	    calculate_mean_kernel<8192, FTKTHR, 128><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<8192, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 16384:
	  if(PulsePoTLen == 64)
	    calculate_mean_kernel<16384, FTKTHR, 64><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<16384, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 32768:
	  if(PulsePoTLen == 32)
	    calculate_mean_kernel<32768, FTKTHR, 32><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<32768, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 65536:
	  if(PulsePoTLen == 16)
	    calculate_mean_kernel<65536, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<65536, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 131072:
	  if(PulsePoTLen == 16)
	    calculate_mean_kernel<131072, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<131072, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	case 262144:
	  if(PulsePoTLen == 16)
	    calculate_mean_kernel<262144, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(AdvanceBy);
	  else
	    calculate_mean_kernel<262144, FTKTHR><<<grid, block, 0, fftstream0>>>(PulsePoTLen, AdvanceBy);
	  break;
	}
    } 
  else 
    {
      // Not Fermi, launch with cc1.x gridsize
      // do nothing (does not work on pre fermi)
    }

  cudaEventRecord(meanDoneEvent, fftstream0);

  return 0;
}




template <int ul_FftLength, int blockx>
__global__ void __launch_bounds__(FTKTHR, 4)
find_triplets_kernel(float triplet_thresh, int len_power, int AdvanceBy) 
{
  result_flag * __restrict__ rflags  = cudaAcc_PulseFind_settings.result_flags_ft;
  if(blockIdx.x == 0 && threadIdx.x == 0)
    {
      rflags->has_results = 0;
      rflags->error = 0;
    }

  int ul_PoT = blockIdx.x * blockx + threadIdx.x;
  if ((ul_PoT < 1) || (ul_PoT >= ul_FftLength)) return; // Original find_triplets, omits first element

  // Clear the result array
  float       * __restrict__ power   = cudaAcc_PulseFind_settings.power_ft;
  float4      * __restrict__ results = cudaAcc_PulseFind_settings.resultsT;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int y2 = y + y;

  int PoTLen = (1024*1024) / ul_FftLength;
  int TOffset = y * AdvanceBy;
  if(TOffset >= PoTLen - len_power) 
    {
      TOffset = PoTLen - len_power;
    }

  results = &results[AT_XY(ul_PoT, y2, ul_FftLength)];
  results[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  results[ul_FftLength] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 

  float thresh = triplet_thresh;
  float mean_power = mean[ul_PoT+y*ul_FftLength];
  thresh *= mean_power;
  float * __restrict__ pp = &power[AT(0)];
  float * __restrict__ ppp = pp;

  // MAXTRIPLETS_ABOVE_THESHOLD must be odd to prevent bank conflicts!

  __shared__ int binsAboveThreshold[blockx][MAX_TRIPLETS_ABOVE_THRESHOLD];
  int *batx = &binsAboveThreshold[threadIdx.x][0];
  int i, n, numBinsAboveThreshold = 0, p, q;
  float midpoint, peak_power, period;

//#pragma unroll 8
  for(i = 0; i < len_power - 7; i += 8) 
    for(int j = 0; j < 8; j++)
    {
      if((*pp) >= thresh) 
	{
	  if(numBinsAboveThreshold == MAX_TRIPLETS_ABOVE_THRESHOLD) 
	    {
	      rflags->error = CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD; // Reporting Error
	      return;
	    }
	  batx[numBinsAboveThreshold] = i+j;
	  numBinsAboveThreshold++;
	}
      pp += ul_FftLength;
    }
#pragma unroll 1
  for( ; i < len_power; i++) 
    {
      if((*pp) >= thresh) 
	{
	  if(numBinsAboveThreshold == MAX_TRIPLETS_ABOVE_THRESHOLD) 
	    {
	      rflags->error = CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD; // Reporting Error
	      return;
	    }
	  batx[numBinsAboveThreshold] = i;
	  numBinsAboveThreshold++;
	}
      pp += ul_FftLength;
    }
  
  // Check each bin combination for a triplet 
  if(numBinsAboveThreshold > 2) 
    { /* THIS CONDITION IS TRUE ONLY once every 300 KERNEL LAUNCH*/
      int already_reported_flag = 0;
      // TODO: Use Texture for reads from power[], Random Reads

      for(i = 0; i < numBinsAboveThreshold - 1; i++) 
	{
	  int tmp = batx[i];
	  float cpeak_power = (ppp[tmp*ul_FftLength]); //(power[AT(tmp)]);
	  for(n = i + 2; n < numBinsAboveThreshold; n++) 
	    {
	      int tmp2 = batx[n];

	      int midfi = (tmp+tmp2)>>1; // new
	      
	      /* Get the peak power of this triplet */
	      peak_power = cpeak_power;
              float ptmp2 = (ppp[tmp2*ul_FftLength]); //power[AT(tmp2)];
	      if(ptmp2 > peak_power)
		peak_power = ptmp2;
	      
	      p = tmp;
	      pp = ppp + tmp*ul_FftLength; //&power[AT(p)];
	      while(((*pp) >= thresh) && (p <= midfi)) 
		{    /* Check if there is a pulse "off" in between init and midpoint */
		  p++; pp += ul_FftLength;
		}
	      
	      q = midfi + 1;
	      pp = ppp + q*ul_FftLength; //&power[AT(q)];
	      while(((*pp) >= thresh) && (q <= tmp2)) 
		{    /* Check if there is a pulse "off" in between midpoint and end */
		  q++; pp += ul_FftLength;
		}
	      
	      if(p >= midfi || q >= tmp2) 
		{
		  /* if this pulse doesn't have an "off" between all the three spikes, it's dropped */
		} 
	      else 
                {
		  float midfip  = (ppp[midfi * ul_FftLength]); //(power[AT(midfi)]);
		  //if((midpoint - midff) > 0.1f) 
		  if((tmp ^ tmp2) & 1) // one of them is odd
		    {    /* if it's spread among two bins */
		      midpoint = (tmp+tmp2)*0.5f;
 		      period = (float)fabs((tmp-tmp2)*0.5f);
		      float midfip2 = (ppp[(midfi+1) * ul_FftLength]); // prefetch second mid bin
		      if(midfip >= thresh ) 
			{
			  if(already_reported_flag >= 2)	
			    {
			      rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
			      return;
			    }

			  if(midfip > peak_power)
			    peak_power = midfip;
			  
			  results[already_reported_flag == 0 ? 0 : ul_FftLength] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
			  rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
			  already_reported_flag++;
			}

		      midfip = midfip2; //ppp[(midfi+1)*ul_FftLength]; //(power[AT(midfi + 1)]);
		      if(midfip >= thresh) 
			{
			  if (already_reported_flag >= 2)	
			    {
			      rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
			      return;
			    }
			  
			  if(midfip > peak_power)
			    peak_power = midfip;
			  
			  results[already_reported_flag == 0 ? 0 : ul_FftLength] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
			  rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
			  already_reported_flag++;
			}
		      
		    } 
		  else 
		    {            /* otherwise just check the single midpoint bin */
		      if(midfip >= thresh) 
			{
			  if(already_reported_flag >= 2)	
			    {
			      rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
			      return;
			    }
			  
			  midpoint = (tmp+tmp2)*0.5f;
	                  period = (float)fabs((tmp-tmp2)*0.5f);
			  if(midfip > peak_power)
			    peak_power = midfip;
			  
			  results[already_reported_flag == 0 ? 0 : ul_FftLength] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
			  rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
			  already_reported_flag++;
			}
		    }
		}
	    }
	}
    }
}


template <int ul_FftLength, int blockx, int len_power>
__global__ void __launch_bounds__(FTKTHR, 4)
find_triplets_kernel(float triplet_thresh, int AdvanceBy) 
{
  result_flag * __restrict__ rflags  = cudaAcc_PulseFind_settings.result_flags_ft;
  if(blockIdx.x == 0 && threadIdx.x == 0)
    {
      rflags->has_results = 0;
      rflags->error = 0;
    }

  int ul_PoT = blockIdx.x * blockx + threadIdx.x;
  if ((ul_PoT < 1) || (ul_PoT >= ul_FftLength)) return; // Original find_triplets, omits first element

  int PoTLen = (1024*1024) / ul_FftLength;
//  int PoTLen = cudaAcc_PulseFind_settings.NumDataPoints / ul_FftLength;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int y2 = y + y;
  int TOffset = y * AdvanceBy;
  
  float thresh = triplet_thresh;
  
  if(TOffset >= PoTLen - len_power) 
    {
      TOffset = PoTLen - len_power;
    }

  float       * __restrict__ power   = cudaAcc_PulseFind_settings.power_ft;
  float4      * __restrict__ results = cudaAcc_PulseFind_settings.resultsT;
  // Clear the result array
  results = &results[AT_XY(ul_PoT, y2, ul_FftLength)];
  results[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  results[ul_FftLength] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 
  
  // MAXTRIPLETS_ABOVE_THESHOLD must be odd to prevent bank conflicts!
  __shared__ int binsAboveThreshold[blockx][MAX_TRIPLETS_ABOVE_THRESHOLD];
  int *batx = &binsAboveThreshold[threadIdx.x][0];
  int i, n, numBinsAboveThreshold = 0, p, q;
  float midpoint, mean_power = 0, peak_power, period;
  
  /* Get all the bins that are above the threshold, and find the power array mean value */
  //float4 partials = {0.0f,0.0f,0.0f,0.0f};
  float * __restrict__ pp = &power[AT(0)];
  float * __restrict__ ppp = pp;

  mean_power = mean[ul_PoT+y*ul_FftLength];

  thresh *= mean_power;

  pp = ppp;

  for(i = 0; i < len_power; i++) 
    {
      if((*pp) >= thresh) 
	{
	  if(numBinsAboveThreshold == MAX_TRIPLETS_ABOVE_THRESHOLD) 
	    {
	      rflags->error = CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD; // Reporting Error
	      return;
	    }
	  batx[numBinsAboveThreshold] = i;
	  numBinsAboveThreshold++;
	}
      pp += ul_FftLength;
    }
  
  /* Check each bin combination for a triplet */
  if(numBinsAboveThreshold > 2) 
    { /* THIS CONDITION IS TRUE ONLY once every 300 KERNEL LAUNCH*/
      int already_reported_flag = 0;
      // TODO: Use Texture for reads from power[], Random Reads

      for(i = 0; i < numBinsAboveThreshold - 1; i++) 
	{
	  int tmp = batx[i];
	  float cpeak_power = (ppp[tmp*ul_FftLength]); //(power[AT(tmp)]);
	  for(n = i + 2; n < numBinsAboveThreshold; n++) 
	    {
	      int tmp2 = batx[n];

	      int midfi = (tmp+tmp2)>>1; // new
	      
	      /* Get the peak power of this triplet */
	      peak_power = cpeak_power;
              float ptmp2 = (ppp[tmp2*ul_FftLength]); //power[AT(tmp2)];
	      if(ptmp2 > peak_power)
		peak_power = ptmp2;
	      
	      p = tmp;
	      pp = ppp + tmp*ul_FftLength; //&power[AT(p)];
	      while(((*pp) >= thresh) && (p <= midfi)) 
		{    /* Check if there is a pulse "off" in between init and midpoint */
		  p++; pp += ul_FftLength;
		}
	      
	      q = midfi + 1;
	      pp = ppp + q*ul_FftLength; //&power[AT(q)];
	      while(((*pp) >= thresh) && (q <= tmp2)) 
		{    /* Check if there is a pulse "off" in between midpoint and end */
		  q++; pp += ul_FftLength;
		}
	      
	      if(p >= midfi || q >= tmp2) 
		{
		  /* if this pulse doesn't have an "off" between all the three spikes, it's dropped */
		} 
	      else 
                {
		  float midfip  = (ppp[midfi * ul_FftLength]); //(power[AT(midfi)]);
		  midpoint = (tmp+tmp2)*0.5f;
		  period = (float)fabs((tmp-tmp2)*0.5f);
		  if(midfip >= thresh) 
		    {
		      if(already_reported_flag >= 2)	
			{
			  rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
			  return;
			}
		      
		      if(midfip > peak_power)
			peak_power = midfip;
		      
		      results[already_reported_flag == 0 ? 0 : ul_FftLength] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
		      rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
		      already_reported_flag++;
		    }
		  
		  if((tmp ^ tmp2) & 1) // one of them is odd
		    {    /* then it's spread among two bins */
		      midfip = (ppp[(midfi+1) * ul_FftLength]); 

		      if(midfip >= thresh) 
			{
			  if(already_reported_flag >= 2)	
			    {
			      rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
			      return;
			    }
			  
			  if(midfip > peak_power)
			    peak_power = midfip;
			  
			  results[already_reported_flag == 0 ? 0 : ul_FftLength] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
			  rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
			  already_reported_flag++;
			}
		      
		    } 
		}
	    }
	}
    }
}


result_flag *Tflags = NULL;	
result_find_pulse_flag *Pflags = NULL;	

extern int find_triplets(const float *power, int len_power, float triplet_thresh, int time_bin, int freq_bin);


int cudaAcc_fetchTripletAndPulseFlags(bool SkipTriplet, bool SkipPulse, int PulsePoTLen, int AdvanceBy, int FftLength)
{
  int retval = 0;
  int PoTLen = 1024*1024 / FftLength; //cudaAcc_NumDataPoints

  if(!SkipTriplet)
    {
//      if(Tflags->has_results == -1)
        cudaEventSynchronize(tripletsDoneEvent);	 

      if(Tflags->error) 
	{
	  fprintf(stderr,"Find triplets Cuda kernel encountered too many triplets, or bins above threshold, reprocessing this PoT on CPU...\n");
	  cudaAcc_transposeGPU(dev_t_PowerSpectrumT, dev_PowerSpectrum, FftLength, cudaAcc_NumDataPoints / FftLength, fftstream0);
	  CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(tmp_PoTT, dev_t_PowerSpectrumT, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrumT), cudaMemcpyDeviceToHost, fftstream0)),true);

	  // loop through frequencies
	  for(int ThisPoT = 1; ThisPoT < FftLength; ThisPoT++) 
	    {
	      // loop through time for each frequency.  PulsePoTNum is
	      // used only for progress calculation.
	      int TOffset = 0;
	      int PulsePoTNum = 1;
	      bool TOffsetOK = true;
	      if(ThisPoT == 1)
		{ // dealyed
		  cudaStreamSynchronize(fftstream0);
		}
	      for(;TOffsetOK; PulsePoTNum++, TOffset += AdvanceBy) 
		{
		  
		  // Create PowerOfTime array for pulse detection.  If there
		  // are not enough points left in this PoT, adjust TOffset
		  // to get the latest possible pulse PoT.
		  if(TOffset + PulsePoTLen >= PoTLen) 
		    {
		      TOffsetOK = false;
		      TOffset = PoTLen - PulsePoTLen;
		    }
		  //memcpy(PulsePoT, &tmp_PoT[ThisPoT * PoTLen + TOffset], PulsePoTLen*sizeof(float));
		  
		  int retval2 = find_triplets(&tmp_PoTT[ThisPoT * PoTLen + TOffset], //PulsePoT,
					     PulsePoTLen,
					     (float)PoTInfo.TripletThresh,
					     TOffset,
					     ThisPoT);
		  if(retval2)	
		    {
		      SETIERROR(retval,"from find_triplets()");// tripletError
		      retval |= 4; // triplet_error
		    }
		  else
		    retval |= 1; // has triplets
		}
	    }
	} 
      else
	if(Tflags->has_results > 0)
	  {
	    dim3 block(64,1,1);
	    dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
	    
	    CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(TripletResults, dev_TripletResults, 2 * grid.x * block.x * grid.y * block.y * sizeof(*dev_TripletResults), cudaMemcpyDeviceToHost, fftstream0)),true);
	    cudaAcc_transposeGPU(dev_t_PowerSpectrumT, dev_PowerSpectrum, FftLength, cudaAcc_NumDataPoints / FftLength, fftstream0);
	    CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(tmp_PoTT, dev_t_PowerSpectrumT, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrumT), cudaMemcpyDeviceToHost, fftstream0)),true);		
            cudaEventRecord(tripletsDoneEvent, fftstream0);

	    retval |= 1; // has triplets
	  }
    }

  if(!SkipPulse)
    {
//      if(Pflags->has_best_pulse == -1)
        cudaEventSynchronize(pulseDoneEvent);

      if(Pflags->has_best_pulse > 0 || Pflags->has_report_pulse > 0) 
	{
	  int PoTLen = cudaAcc_NumDataPoints / FftLength;
	  int PoTStride = (( (PulsePoTLen == PoTLen) ? 1: PoTLen/AdvanceBy) + 1) * AdvanceBy;
	  int max_nb_of_elems = FftLength * PoTStride;
	  
	  CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(PulseResults, dev_PulseResults, 4 * (cudaAcc_NumDataPoints / AdvanceBy + 1) * sizeof(*dev_PulseResults), cudaMemcpyDeviceToHost, fftstream0)),true);

	  if(Pflags->has_best_pulse > 0) 
	    {
	      cudaAcc_transposeGPU(dev_tmp_potP, dev_best_potP, FftLength, PoTStride, fftstream1);
	      CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(best_PoTP, dev_tmp_potP, max_nb_of_elems * sizeof(float), cudaMemcpyDeviceToHost, fftstream1)),true);
	    }
	  
	  if(Pflags->has_report_pulse > 0) 
	    {
	      cudaAcc_transposeGPU(dev_tmp_potP, dev_report_potP, FftLength, PoTStride, fftstream1);
	      CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(tmp_PoTP, dev_tmp_potP, max_nb_of_elems * sizeof(float), cudaMemcpyDeviceToHost, fftstream1)),true);
	    }

          cudaEventRecord(pulseDoneEvent, fftstream1);
	  
	  retval |= 2; // has pulse(s)
	}
    }

  return retval;
}



int cudaAcc_find_triplets(int PulsePoTLen, float triplet_thresh, int AdvanceBy, int FftLength) 
{
  int PoTLen = 1024*1024 / FftLength; //cudaAcc_NumDataPoints
  
  cudaStreamWaitEvent(fftstream0, meanDoneEvent, 0);	 

  if(Tflags == NULL)  
    cudaMallocHost(&Tflags, sizeof(result_flag));
  Tflags->has_results = 0;
  Tflags->has_results = 0;
  
  //CUDA_ACC_SAFE_LAUNCH( (cudaMemsetAsync(dev_flagT, 0, sizeof(*dev_flagT), fftstream0)), true);


  // Occupancy Calculator: cc1.x: 64, cc2.x: 128 & Shared instead of L1
  dim3 block(64,1,1);
  dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
  if(gCudaDevProps.major >= 2) 
    {
      dim3 block(FTKTHR, 1, 1);
      dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);

      switch(FftLength)
	{
	case 8:
	  if(PulsePoTLen == 131072)
	    find_triplets_kernel<8, FTKTHR, 131072><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<8, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 16:
	  if(PulsePoTLen == 65536)
	    find_triplets_kernel<16, FTKTHR, 65536><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<16, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 32:
	  if(PulsePoTLen == 32768)
	    find_triplets_kernel<32, FTKTHR, 32768><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<32, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 64:
	  if(PulsePoTLen == 16384)
	    find_triplets_kernel<64, FTKTHR, 16384><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<64, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 128:
	  if(PulsePoTLen == 8192)
	    find_triplets_kernel<128, FTKTHR, 8192><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<128, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 256:
	  if(PulsePoTLen == 4096)
	    find_triplets_kernel<256, FTKTHR, 4096><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<256, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 512:
	  if(PulsePoTLen == 2048)
	    find_triplets_kernel<512, FTKTHR, 2048><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<512, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 1024:
	  if(PulsePoTLen == 1024)
	    find_triplets_kernel<1024, FTKTHR, 1024><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<1024, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 2048:
	  if(PulsePoTLen == 512)
	    find_triplets_kernel<2048, FTKTHR, 512><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<2048, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 4096:
	  if(PulsePoTLen == 256)
	    find_triplets_kernel<4096, FTKTHR, 256><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<4096, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 8192:
	  if(PulsePoTLen == 128)
	    find_triplets_kernel<8192, FTKTHR, 128><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<8192, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 16384:
	  if(PulsePoTLen == 64)
	    find_triplets_kernel<16384, FTKTHR, 64><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<16384, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 32768:
	  if(PulsePoTLen == 32)
	    find_triplets_kernel<32768, FTKTHR, 32><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<32768, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 65536:
	  if(PulsePoTLen == 16)
	    find_triplets_kernel<65536, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<65536, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 131072:
	  if(PulsePoTLen == 16)
	    find_triplets_kernel<131072, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<131072, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	case 262144:
	  if(PulsePoTLen == 16)
	    find_triplets_kernel<262144, FTKTHR, 16><<<grid, block, 0, fftstream0>>>(triplet_thresh, AdvanceBy);
	  else
	    find_triplets_kernel<262144, FTKTHR><<<grid, block, 0, fftstream0>>>(triplet_thresh, PulsePoTLen, AdvanceBy);
	  break;
	}
    } 
  else 
    {
      // Not Fermi, launch with cc1.x gridsize
      //    find_triplets_kernel<64><<<grid, block>>>(FftLength, PulsePoTLen, triplet_thresh, AdvanceBy);
    }
  
  CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(Tflags, dev_flagT, sizeof(*dev_flagT), cudaMemcpyDeviceToHost, fftstream0)), true); // triplet flags

  cudaEventRecord(tripletsDoneEvent, fftstream0);

  return 0;
}



int cudaAcc_processTripletResults(int PulsePoTLen, int AdvanceBy, int FftLength)
{
  int PoTLen = 1024*1024 / FftLength; //cudaAcc_NumDataPoints

  cudaEventSynchronize(tripletsDoneEvent);

  if(Tflags->has_results)
    {
      //TODO: Check if faster is to scan the results and download only the results, not everything
      // Iterating trough results
      for(int ThisPoT = 1; ThisPoT < FftLength; ThisPoT++) 
	{
	  for(int TOffset = 0, TOffsetOK = true, PulsePoTNum = 0; TOffsetOK; PulsePoTNum++, TOffset += AdvanceBy)
	    {
	      if(TOffset + PulsePoTLen >= PoTLen) 
		{
		  TOffsetOK = false;
		  TOffset = PoTLen - PulsePoTLen;
		}
	      
	      int index = ((PulsePoTNum * 2) * FftLength + ThisPoT);
	      int index2 = ((PulsePoTNum * 2 + 1)* FftLength + ThisPoT);
	      if(ThisPoT == 1 && PulsePoTNum == 0)
		{
		//  cudaStreamSynchronize(fftstream0);
		}		

	      if(TripletResults[index].x > 0) 
		{
		  float4 res = TripletResults[index];
		  cudaAcc_ReportTripletEvent( res.x, res.y, res.z, res.w, TOffset,
					      ThisPoT, PulsePoTLen, &tmp_PoTT[ThisPoT * PoTLen + TOffset], 1 );
		}
	      
	      if(TripletResults[index2].x > 0) 
		{
		  float4 res = TripletResults[index2];
		  cudaAcc_ReportTripletEvent( res.x, res.y, res.z, res.w, TOffset,
					      ThisPoT, PulsePoTLen, &tmp_PoTT[ThisPoT * PoTLen + TOffset], 1 );
		}
	    }
	}
    }

  return 0;
}


template <int num_adds>
__device__ float cudaAcc_t_funct(int di, int j, int PulseMax, float *t_funct_cache) 
{
  PulseMax = 40960; // help compiler
  return __ldg(&t_funct_cache[(j * CUDA_ACC_FOLDS_COUNT* PulseMax) + di]); //
}

template <int num_adds>
__device__ float cudaAcc_t_funct_ldg(int di, int j, int PulseMax, float *t_funct_cache) 
{
  PulseMax = 40960; // help compiler
  	   
  return __ldg(&t_funct_cache[(j * CUDA_ACC_FOLDS_COUNT* PulseMax) +  di]); //
}




template <int fft_len>
__device__ float cudaAcc_sumtop2(float *tab, float* __restrict__ dest, int di, float *tmp0) 
{
  float sum, tmax, tmax2;
  int   i = 0;
  float * __restrict__ one = tab;
  float * __restrict__ two = tmp0; //(float *)((void *)tab + tmp0);// * fft_len * sizeof(float));
  tmax = 0.0f, tmax2 = 0.0f;

  int n = sizeof(float)*fft_len;

#define UNRL 16
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 8)
	{
	  a[j]     = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+1]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+2]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+3]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+4]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+5]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+6]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+7]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  //	}
	  //      for(int j = 0; j < UNRL; j += 4)
	  //	{
	  *dest = a[j];   dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j]);
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);
	  *dest = a[j+4]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+4]);
	  *dest = a[j+5]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+5]);
	  *dest = a[j+6]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+6]);
	  *dest = a[j+7]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+7]);
	}
    }
#define UNRL 4
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 4)
	{
	  a[j] = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+1] = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+2] = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+3] = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	//}
      //for(int j = 0; j < UNRL; j++)
	//{
	  *dest = a[j]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j]);		
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);		
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);		
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);		
	}
    }
#pragma unroll 1
  for( ; i < di; i++ ) 
      {
	sum  = *one + *two;
	one = (float *)((void *)one + n);
	two = (float *)((void *)two + n);
	*dest = sum;
	tmax = max(tmax, sum);		
	dest = (float *)((void *)dest + n);
      }

  tmax = max(tmax, tmax2);
  return tmax;
}


template <int fft_len>
__device__ float cudaAcc_sumtop3(float *tab, float * __restrict__ dest, int di, float *tmp0, float *tmp1) 
{
  float sum, tmax, tmax2;
  int   i = 0;
  float * __restrict__ one = tab;
  float * __restrict__ two = tmp0; //(float *)((void *)tab + tmp0);// * fft_len * sizeof(float));
  float * __restrict__ three = tmp1; //(float *)((void *)tab + tmp1);// * fft_len * sizeof(float));
  tmax = 0.0f, tmax2 = 0.0f;

  int n = sizeof(float)*fft_len;
  
#define UNRL 16
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 8)
	{
	  a[j]     = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+1]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+2]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+3]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+4]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+5]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+6]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+7]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  //	}
	  //      for(int j = 0; j < UNRL; j += 4)
	  //	{
	  *dest = a[j];   dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j]);
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);
	  *dest = a[j+4]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+4]);
	  *dest = a[j+5]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+5]);
	  *dest = a[j+6]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+6]);
	  *dest = a[j+7]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+7]);
	}
    }

#define UNRL 4
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 4)
	{
	  a[j]   = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+1] = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+2] = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	  a[j+3] = *one + *two + *three; one = (float *)((void *)one + n); two = (float *)((void *)two + n); three = (float *)((void *)three + n);
	//}
        //for(int j = 0; j < UNRL; j++)
	//{
	  *dest = a[j];   dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j]);
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);
	}
    }

#pragma unroll 1
  for( ; i < di; i++ ) 
      {
	sum  = *one + *two + *three;
	one = (float *)((void *)one + n);
	two = (float *)((void *)two + n);
	three = (float *)((void *)three + n);
	*dest = sum;
	tmax = max(tmax, sum);		
	dest = (float *)((void *)dest + n);
      }

  tmax = max(tmax, tmax2);
  return tmax;
}


template <int fft_len>
__device__ float cudaAcc_sumtop4(float * tab, float * __restrict__ dest, int di, float *tmp0, float *tmp1, float *tmp2) 
{
  float tmax, tmax2;
  int   i = 0;
  float * __restrict__ one = tab;
  float * __restrict__ two   = tmp0; //(float *)((void *)tab + tmp0);// * fft_len * sizeof(float));
  float * __restrict__ three = tmp1; //(float *)((void *)tab + tmp1);// * fft_len * sizeof(float));
  float * __restrict__ four  = tmp2; //(float *)((void *)tab + tmp2);// * fft_len * sizeof(float));
  tmax = 0.0f, tmax2 = 0.0f;

  int n = sizeof(float) * fft_len;
 
#define UNRL 16
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 8)
	{
	  a[j]     = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+1]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+2]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+3]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+4]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+5]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+6]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+7]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);

	  a[j]     += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+1]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+2]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+3]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+4]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+5]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+6]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+7]   += *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  //	}
	  //      for(int j = 0; j < UNRL; j += 4)
	  //	{
	  *dest = a[j+0]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+0]);
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);
	  *dest = a[j+4]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+4]);
	  *dest = a[j+5]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+5]);
	  *dest = a[j+6]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+6]);
	  *dest = a[j+7]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+7]);
	}
    }

#define UNRL 4
  for( ; i < di - (UNRL-1); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j += 4)
	{
	  a[j]     = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+1]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+2]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j+3]   = *one + *two; one = (float *)((void *)one + n); two = (float *)((void *)two + n);
	  a[j]     = *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+1]   = *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+2]   = *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	  a[j+3]   = *three + *four; three = (float *)((void *)three + n); four = (float *)((void *)four + n);
	//}
        //for(int j = 0; j < UNRL; j++)
	//{
	  *dest = a[j];   dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j]);
	  *dest = a[j+1]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest = (float *)((void *)dest + n);
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest = (float *)((void *)dest + n);
	  tmax2 = max(tmax2, a[j+3]);
	}
    }

#pragma unroll 1
  for( ; i < di; i++ ) 
      {
	float sum  = *one + *two + *three;
	one = (float *)((void *)one + n);
	two = (float *)((void *)two + n);
	three = (float *)((void *)three + n);
	*dest = sum;
	tmax = max(tmax, sum);		
	dest = (float *)((void *)dest + n);
      }

  tmax = max(tmax, tmax2);
  return tmax;

}

/*
#define UNRL 2
  for( ; i < di - (UNRL-1); i += UNRL) 
    for(int j = 0; j < UNRL; j++)
      {
	sum   = (*one + *two);
	sum2  = (*three + *four);
	one   = (float *)((void *)one + n);
	two   = (float *)((void *)two + n);
	sum  += sum2;
	three = (float *)((void *)three + n);
	four  = (float *)((void *)four + n);
	*dest = sum;
	tmax  = max(tmax, sum);		
	dest  = (float *)((void *)dest + n);
      }
#pragma unroll 1
  for( ; i < di; i++) 
      {
	sum   = (*one + *two);
	sum2  = (*three + *four);
	one   = (float *)((void *)one + n);
	two   = (float *)((void *)two + n);
	sum  += sum2;
	three = (float *)((void *)three + n);
	four  = (float *)((void *)four + n);
	*dest = sum;
	tmax  = max(tmax, sum);		
	dest  = (float *)((void *)dest + n);
      }
  return tmax;
*/

template <int fft_len>
__device__ float cudaAcc_sumtop5(float *tab, float * __restrict__ dest, int di, float *tmp0, float *tmp1, float *tmp2, float *tmp3) 
{
  float sum, sum2, tmax;
  int   i = 0;
  float * __restrict__ one = tab;
  float * __restrict__ two = tmp0; //(float *)((void *)tab + tmp0);  // * fft_len * sizeof(float));
  float * __restrict__ three = tmp1; //(float *)((void *)tab + tmp1);// * fft_len * sizeof(float));
  float * __restrict__ four = tmp2; //(float *)((void *)tab + tmp2); // * fft_len * sizeof(float));
  float * __restrict__ five = tmp3; //(float *)((void *)tab + tmp3); // * fft_len * sizeof(float));
  tmax = 0.0f;

  int n = sizeof(float) * fft_len;

#define UNRL 2
  for(; i < di - (UNRL-1); i += UNRL) 
    for(int j = 0; j < UNRL; j++)
      {
	sum    = (*one + *two);
	sum2  = (*three + *four);
	one = (float *)((void *)one + n);
	two = (float *)((void *)two + n);
	sum += *five;
	three = (float *)((void *)three + n);
	four  = (float *)((void *)four + n);
        sum   = (sum + sum2);
	five   = (float *)((void *)five + n);
	*dest  = sum;
	tmax = max(tmax, sum);		
	dest = (float *)((void *)dest + n);
      }
#pragma unroll 1
  for( ; i < di; i++) 
      {
	sum    = (*one + *two);
	sum2  = (*three + *four);
	one = (float *)((void *)one + n);
	two = (float *)((void *)two + n);
	sum += *five;
	three = (float *)((void *)three + n);
	four  = (float *)((void *)four + n);
        sum   = (sum + sum2);
	five   = (float *)((void *)five + n);
	*dest  = sum;
	tmax = max(tmax, sum);		
	dest = (float *)((void *)dest + n);
      }
  return tmax;
}


template <int step>
__device__ void cudaAcc_copy(float * __restrict__ from, float * __restrict__ to, int count) 
{
  int i = 0;
  int n = sizeof(float)*step;

#pragma unroll 1
  for(; i < count-3; i += 4) 
    {
      float a, b, c, d;
      a = *from;
      from = (float *)(((void *)from) + n);
      b = *from;
      from = (float *)(((void *)from) + n);
      c = *from;
      from = (float *)(((void *)from) + n);
      d = *from;
      from = (float *)(((void *)from) + n);
      *to = a; 
      to = (float *)(((void *)to) + n);
      *to = b; 
      to = (float *)(((void *)to) + n);
      *to = c; 
      to = (float *)(((void *)to) + n);
      *to = d; 
      to = (float *)(((void *)to) + n);
    }

#pragma unroll 1
  for(; i < count-1; i += 2) 
    {
      float a, b;
      a = *from;
      from = (float *)(((void *)from) + n);
      b = *from;
      from = (float *)(((void *)from) + n);
      *to = a; 
      to = (float *)(((void *)to) + n);
      *to = b; 
      to = (float *)(((void *)to) + n);
    }

#pragma unroll 1
  for(; i < count; ++i) 
    {
      *to = *from;
      to = (float *)(((void *)to) + n);
      from = (float *)(((void *)from) + n);
    }
}



template <bool load_state, int num_adds, int fft_len>
__launch_bounds__(PFTHR, 4) 
__global__ void find_pulse_kernel(float best_pulse_score, int PulsePotLen, int AdvanceBy, int ndivs) 
{
  if(load_state == false && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
      cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 0;
      cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 0;
    }

  int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;
  //	int tid = threadIdx.x+blockIdx.x*blockDim.x+blockIdx.y*blockDim.x*gridDim.x;
  if ((ul_PoT < 1) || (ul_PoT >= fft_len)) return; // Original find_pulse, omits first element
  int PoTLen = 1024*1024 / fft_len;//cudaAcc_PulseFind_settings.NumDataPoints
  int y = blockIdx.y * blockDim.y;// + threadIdx.y;
  int TOffset1 = y * AdvanceBy;
  int TOffset2 = y * AdvanceBy;
  int y4 = y * 4;
  float *tfc = cudaAcc_PulseFind_settings.t_funct_cache_fp + (num_adds - CUDA_ACC_FOLDS_START) * 40960;
  
  if(TOffset1 > PoTLen - PulsePotLen) 
    {		
      TOffset1 = PoTLen - PulsePotLen;
    }

  int nx = ul_PoT + TOffset1 * fft_len;
  float* __restrict__ fp_PulsePot	= &cudaAcc_PulseFind_settings.PulsePot_fp[nx];
  nx = ul_PoT + TOffset2 * fft_len;
  float* __restrict__ tmp_pot		= &cudaAcc_PulseFind_settings.tmp_potP[nx];
  float* __restrict__ best_pot		= &cudaAcc_PulseFind_settings.best_potP[nx];
  float* __restrict__ report_pot	= &cudaAcc_PulseFind_settings.report_potP[nx];

  int di; //maxs = 0
  float maxd=0,avg=0; //snr=0, maxp=0, max=0, fthresh=0
  float tmp_max, t1;
  
  float4 * __restrict__ resp = &cudaAcc_PulseFind_settings.resultsP[AT_XY(ul_PoT, y4, fft_len)];
  
  if(!load_state)
    {
      //  Calculate (load) average power

      avg = mean[ul_PoT + y*fft_len];
      float4 zero = make_float4(0,0,0,0);
      resp[0]         = zero; 
      resp[fft_len]   = zero; 
      resp[3*fft_len] = zero; 
      zero.y = avg;
      resp[2*fft_len] = zero; 
    }
  else
    {		
      best_pulse_score = fmax(best_pulse_score, resp[0].w);
      float4 *tmp_float4 = &resp[2*fft_len];
      avg = tmp_float4->y;
      //max = tmp_float4.x * avg;
      //maxp = tmp_float4.z;
      maxd = tmp_float4->w;
      
      //tmp_float4 = resp[3*fft_len];
      //snr = tmp_float4.y;
      //fthresh = tmp_float4.z;
      //maxs = tmp_float4.w;
    } 

  
  // save redundant calculations: sqrt is expensive
  float sqrt_num_adds_div_avg = 1.0f/(avg/(float)sqrt((float)num_adds));
  
  //  Periods from PulsePotLen/3 to PulsePotLen/4, and power of 2 fractions of.
  //   then (len/4 to len/5) and finally (len/5 to len/6)
  //	
  
  //for(int num_adds = 3; num_adds <= 5; num_adds++) 
  {
    int firstP, lastP;
    switch(num_adds) 
      {
      case 3: lastP = (PulsePotLen*2)/3;  firstP = (PulsePotLen*1)>>1; break;
      case 4: lastP = (PulsePotLen*3)>>2;  firstP = (PulsePotLen*3)/5; break;
      case 5: lastP = (PulsePotLen*4)/5;  firstP = (PulsePotLen*4)/6; break;
      }

    int olddi = -1;

    for(int p = lastP ; p > firstP ; p--) 
      {
	float cur_thresh, dis_thresh;
	int perdiv;
	int tmp0, tmp1, tmp2, tmp3;
	
	perdiv = num_adds - 1;
        di = (int)p/(int)perdiv;

	if(di != olddi)
	  {
	    dis_thresh = cudaAcc_t_funct_ldg<num_adds>(di, 0, 40960, tfc) * avg; //cudaAcc_PulseFind_settings.PulseMax
	    olddi = di;
	  }

	switch(num_adds) 
	  {
	  case 3:
	    tmp0 = fft_len * ((p + 1) >> 1);
	    tmp1 = fft_len * p;
            tmp_max = cudaAcc_sumtop3<fft_len>(&fp_PulsePot[0], &tmp_pot[0], di, &fp_PulsePot[0] + tmp0, &fp_PulsePot[0] + tmp1);
	    break;

	  case 4:
	    tmp0 = fft_len * (int)((p + p + 1)/6);
	    tmp1 = fft_len * (int)((p + p + 1)/3);
            tmp2 = fft_len * p;
	    tmp_max = cudaAcc_sumtop4<fft_len>(&fp_PulsePot[0], &tmp_pot[0], di, &fp_PulsePot[0] + tmp0, &fp_PulsePot[0] + tmp1, &fp_PulsePot[0] + tmp2);
	    break;

	  case 5:
	    tmp0 = fft_len * ((p + 2) >> 2);
	    tmp1 = fft_len * ((p + 1) >> 1);
	    tmp2 = fft_len * ((p + p + p + 2) >> 2); 
	    tmp3 = fft_len * p;
	    tmp_max = cudaAcc_sumtop5<fft_len>(&fp_PulsePot[0], &tmp_pot[0], di, &fp_PulsePot[0] + tmp0, &fp_PulsePot[0] + tmp1, &fp_PulsePot[0] + tmp2, &fp_PulsePot[0] + tmp3);
	    break;
	  }
	
	if(tmp_max>dis_thresh) 
	  {
	    // unscale for reporting
	    tmp_max /= num_adds;
	    cur_thresh = (dis_thresh / num_adds - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

	    float a, b;	    
	    float _thresh = (b = cur_thresh-avg)*sqrt_num_adds_div_avg;
	    float _snr = (a = tmp_max-avg)*sqrt_num_adds_div_avg;

	    if(a > (best_pulse_score * b)) 
	      {
		best_pulse_score = a / b; //_snr / _thresh;
		cudaAcc_copy<fft_len>(tmp_pot, best_pot, di);
		resp[0] = make_float4(
				      tmp_max/avg,
				      avg,
				      ((float)p)/(float)perdiv,
				      0.0f//TOffset1+PulsePotLen/2
				      );
		resp[fft_len] = make_float4(
					    ul_PoT,
					    _snr,
					    _thresh,
					    num_adds
					    );
		cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;					
	      }

	    if((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) 
	      {
		//maxp  = (float)p/(float)perdiv;
		maxd  = t1;
		//maxs  = num_adds;
		//max = tmp_max;
		//snr = _snr;
		//fthresh= _thresh;
		cudaAcc_copy<fft_len>(tmp_pot, report_pot, di);
		//memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float)); 
		
		// It happens very rarely so it's better to store the results right away instead of
		// storing them in registers or worse local memory						
		resp[2 * fft_len] = make_float4(
						tmp_max/avg,
						avg,
						(float)p/(float)perdiv,
						maxd//TOffset1+PulsePotLen/2
						);
		resp[3 * fft_len] = make_float4(
						ul_PoT,
						_snr,
						_thresh,
						num_adds
						);
		cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
		//ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
		//snr, fthresh, FoldedPOT, maxs, 1);						
	      }
	  }
	
	int num_adds_2 = num_adds << 1;

	for(int j = 1; j < ndivs; j++) 
	  {
	    float sqrt_num_adds_2_div_avg = 1.0f/(avg/(float)sqrt((float)num_adds_2));
	    perdiv = perdiv << 1;
	    tmp0 = di & 1;
	    di = di >> 1;
	    tmp0 = fft_len*(tmp0 + di);

	    float dis_thresh2 = cudaAcc_t_funct<num_adds>(di, j, 40960, tfc) * avg; //cudaAcc_PulseFind_settings.PulseMax = 40960
	    
	    if(di == 1)
	      tmp_max  = tmp_pot[0] + tmp_pot[tmp0];
	    else
	      tmp_max = cudaAcc_sumtop2<fft_len>(&tmp_pot[0], &tmp_pot[0], di, &tmp_pot[0] + tmp0);
	    //tmp_max = w.x;
	    //printf("%f %f\r\n", w.x, w.y);
	    
	    if(tmp_max > dis_thresh2) 
	      {
		// unscale for reporting
		tmp_max /= num_adds_2;
		cur_thresh = (dis_thresh2 / num_adds_2 - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;
		
		float a, b;		
		float _thresh = (b = cur_thresh-avg)*sqrt_num_adds_2_div_avg;
		float _snr = (a = tmp_max-avg)*sqrt_num_adds_2_div_avg;

		if(a > (best_pulse_score * b)) 
		  {
		    best_pulse_score = a/b; //_snr / _thresh;
		    if(di == 1)
		      *best_pot = tmp_max;
		    else
		      cudaAcc_copy<fft_len>(tmp_pot, best_pot, di);
		    resp[0] = make_float4(
					  tmp_max/avg,
					  avg,
					  ((float)p)/(float)perdiv,
					  0.0f//TOffset1+PulsePotLen/2
					  );
		    resp[fft_len] = make_float4(
						ul_PoT,
						_snr,
						_thresh,
						num_adds_2
						);
		    cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;
		  }
		
		if((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) 
		  {
		    //maxp = (float)p/(float)perdiv;
		    maxd = t1;
		    //maxs = num_adds_2;
		    //max  = tmp_max;
		    //snr  = _snr;
		    //fthresh = _thresh;
		    if(di ==1)
		      *report_pot = tmp_max;
		    else
		      cudaAcc_copy<fft_len>(tmp_pot, report_pot, di);
		    //memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float));
		    
		    // It happens very rarely so it's better to store the results right away instead of
		    // storing them in registers or worse local memory						
		    resp[2 * fft_len] = make_float4(
						    tmp_max/avg,
						    avg,
						    (float)p/(float)perdiv,
						    maxd//TOffset1+PulsePotLen/2
						    );
		    resp[3 * fft_len] = make_float4(
						    ul_PoT,
						    _snr,
						    _thresh,
						    num_adds
						    );
		    cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
		    //ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
		    //snr, fthresh, FoldedPOT, maxs, 1);						
		  }
	      }
	    
	    num_adds_2 = num_adds_2 << 1;
	  }  // for (j = 1; j < ndivs
      } // for (p = lastP
  } // for(num_adds =
}


// Assuming AdvanceBy >= PulsePoTLen / 2
int cudaAcc_find_pulse_original(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) 
{	
  int PoTLen = 1024*1024 / FftLength;//cudaAcc_NumDataPoints
	
  dim3 block(64, 1, 1); // Pre-Fermi default
  if (gCudaDevProps.regsPerBlock >= 64 * 1024) block.x = PFTHR; // Kepler GPU tweak;
  else if (gCudaDevProps.regsPerBlock >= 32 * 1024) block.x = 128; // Fermi tweak;
  
  dim3 grid((FftLength + block.x - 1) / block.x,(PulsePoTLen == PoTLen) ? 1 : PoTLen / AdvanceBy,1);
  
  int ndivs;

#if 0 //def _WIN32
  _BitScanReverse((DWORD *)&ndivs,PulsePoTLen);
  ndivs = max(1,ndivs-4);
#else
  int i;
  for (i = 32, ndivs = 1; i <= PulsePoTLen; ndivs++, i *= 2); 
#endif

  switch(FftLength)
    {
      case 8:
	find_pulse_kernel<false, 3, 8><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  4, 8><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  5, 8><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 16:
	find_pulse_kernel<false, 3, 16><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  4, 16><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  5, 16><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 32: 
	find_pulse_kernel<false, 3, 32><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  4, 32><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,  5, 32><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 64:
	find_pulse_kernel<false, 3, 64><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 64><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 5, 64><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 128:
	find_pulse_kernel<false, 3, 128><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 128><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 128><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 256:
	find_pulse_kernel<false, 3, 256><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 256><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 256><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 512:
	find_pulse_kernel<false, 3, 512><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 512><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 512><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 1024:
	find_pulse_kernel<false, 3, 1024><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 1024><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 1024><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 2048:
	find_pulse_kernel<false, 3, 2048><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 2048><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 2048><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 4096:
	find_pulse_kernel<false, 3, 4096><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 4096><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 4096><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 8192:
	find_pulse_kernel<false, 3, 8192><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 8192><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 8192><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 16384:
	find_pulse_kernel<false, 3, 16384><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 16384><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 16384><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 32768:
	find_pulse_kernel<false, 3, 32768><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 32768><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 32768><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 65536:
	find_pulse_kernel<false, 3, 65536><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 65536><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 65536><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 131072:
	find_pulse_kernel<false, 3, 131072><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 131072><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 131072><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
      case 262144:
	find_pulse_kernel<false, 3, 262144><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true, 4, 262144><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	find_pulse_kernel<true,5, 262144><<<grid, block, 0, fftstream1 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, ndivs);	
	break;
    }

  return 0;
}

#if __CUDA_ARCH__ >= 200	// Only use on Fermi or newer: 32 registers is too low for cc13, and does not have __ddiv,__drcp,__dsqrt

#define ITMAX 1000		// Needs to be a few times the sqrt of the max. input to lcgf

__device__ double cudaAcc_pulse_lcgf(int a, double x) {
	double EPS= 1.19209e-006f; //007;//std::numeric_limits<double>::epsilon();
	double FPMIN= 9.86076e-031f; //032;//std::numeric_limits<double>::min()/EPS;
	double an,b,c,d,del,h;
	double gln = lgamma(__int2double_rn(a));
	
	b=__dadd_rn(x,(1.0f - a)); //x+1.0f-a;
	c=__drcp_rn(FPMIN);                   //1.0f/FPMIN;
	d=__drcp_rn(b);                                        //1.0f/b;
	h=d;
	del=0.0f;
	
#pragma unroll
	for (int i=1;i<=ITMAX && fabs(del-1)>=EPS;++i) {
		an = -i*(i-a);
		b  = __dadd_rn(b,2);
		d  = __dadd_rn(__dmul_rn(an,d),b);
		if (fabs(d)<FPMIN) d=FPMIN;
		c  = __dadd_rn(b,__dmul_rn(an,__drcp_rn(c)));
		if (fabs(c)<FPMIN) c=FPMIN;
		d  = __drcp_rn(d);
		del= __dmul_rn(d,c);
		h  = __dmul_rn(h,del);
		if (fabs(__dadd_rn(del,-1.0))<EPS) break;
	}
	return (double)( __dadd_rn( __dadd_rn(log(h),-x) ,  __dadd_rn(__dmul_rn(a,log(x)),-gln)) );
}

__device__ float cudaAcc_pulse_dlcgf(int a, float x) {
	return (float)__dmul_rn((cudaAcc_pulse_lcgf(a,__dadd_rn(x,0.1f)))-cudaAcc_pulse_lcgf(a,__dadd_rn(x,-0.1f)), 5.0f);
}

__device__ float cudaAcc_invert_lcgf(float y, float a) {
	int j;
	float df,dx,dxold,f;
	float temp,xh,xl,rts;
	float frac_err = 1e-6f;
	
	xh= __dadd_rn(a,1.5f);
	xl= __dadd_rn(a, __dmul_rn(__dmul_rn(-2.0f,y),__dsqrt_rn(a)));
	float fl=__fadd_rn(cudaAcc_pulse_lcgf(a,xl),-y);
	float fh=__fadd_rn(cudaAcc_pulse_lcgf(a,xh),-y);
	
	rts=__dmul_rn(0.5f,__dadd_rn(xh,xl));
	dxold=fabs(__dadd_rn(xh,-xl));
	dx=dxold;
	f=__dadd_rn(cudaAcc_pulse_lcgf(a,rts),y);
	df=cudaAcc_pulse_dlcgf(a,rts);
	
	for (j=1;j<=ITMAX;j++) {
		if (__dmul_rn((__dmul_rn(df,rts) - __dmul_rn(df,xh) - f), (__dmul_rn(df,rts) - __dmul_rn(df,xl) -f)) >= 0.0f
				|| (fabs(__dmul_rn(2.0f,f))>fabs(__dmul_rn(dxold,df)))) {
			
			dxold= dx;
			dx   = __dadd_rn(__dmul_rn(0.5f,xh),-__dmul_rn(0.5f,xl));
			rts  = __dadd_rn(xl,dx);
			if ((xl==rts) || (xh==rts))
				return rts;
		} else {
			dxold=dx;
			dx=__dmul_rn(f, __drcp_rn(df));
			temp=rts;
			rts = __dadd_rn(rts,-dx);
			if (temp==rts)
				return rts;
		}
		f=__dadd_rn(cudaAcc_pulse_lcgf(a,rts),-y);
		if (fabs(f)<fabs(frac_err*y)) return rts;
		df=cudaAcc_pulse_dlcgf(a,rts);
		
		(f<0.0f) ? xl=rts : xh=rts;
	}
	return 0;
}

// __launch bounds__(X,Y): we guarantee that we'll never launch more than X threads, and we
// want to fit at least Y blocks. Will override -maxrregcount flag. Can reduce
// register spilling, in exchange for when we know we'll need fewer threads
// Current max threads is 32 * maxdivs. maxdivs is currently at 11, with PulseMax at
// 40960, but it could change in the future. 480 allows PulseMax up to 1048576, and
// eliminates register spilling.
/*
__global__ void __launch_bounds__(480,1) cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, float pulse_display_thresh) 
{
  PulseMax = 40960;
	di = di + (PulseMax * threadIdx.y / 32);
	
	int j = threadIdx.x;
	int num_adds = blockIdx.y;
	
	int l = 1<<j;
	int n = (num_adds+CUDA_ACC_FOLDS_START)*l;
	int idx = PulseMax * (j * CUDA_ACC_FOLDS_COUNT + num_adds) + di;
	float inv_lcgf = cudaAcc_invert_lcgf(__dadd_rn(-PulseThresh,-log(__int2float_rn(di))), __int2float_rn(n));
	dev_t_funct_cache[idx] = (__dadd_rn(__dmul_rn(__dadd_rn(inv_lcgf, -n),pulse_display_thresh), n)); //-0.05f;
}*/

//float version
__global__ void __launch_bounds__(480,1) cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, float pulse_display_thresh) 
{
	PulseMax = 40960;
	di = di + (PulseMax * threadIdx.y / 32);
	int j = threadIdx.x;
	int num_adds = blockIdx.y;
	int l = 1<<j;
	int n = (num_adds+CUDA_ACC_FOLDS_START)*l;
	int idx = PulseMax * (j * CUDA_ACC_FOLDS_COUNT + num_adds) + di;
	float inv_lcgf = cudaAcc_invert_lcgf(__fadd_rn(-PulseThresh,-logf(__int2float_rn(di))), __int2float_rn(n));
	dev_t_funct_cache[idx] = (__fadd_rn(__fmul_rn(__fadd_rn(inv_lcgf, -n),pulse_display_thresh), n)); //-0.05f;
}
#else // __CUDA_ARCH__ >= 200

// dummy version for <200 cards
__global__ void cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, float pulse_display_thresh) {
	// do nothing
}
#endif



float cudaAcc_host_t_funct(double pulse_display_thresh, double PulseThresh, int m, int n)
{	
	return (invert_lcgf((float)(-PulseThresh - log((float)m)),
		(float)n, (float)1e-4) - n) * (float)pulse_display_thresh + n;
}



int cudaAcc_initialize_pulse_find(double pulse_display_thresh, double PulseThresh, int PulseMax) 
{
  PulseMax = 40960; // const for now
  cudaError_t cu_err;
  
  cu_err = cudaMalloc((void**) &dev_find_pulse_flag, sizeof(*dev_find_pulse_flag));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_find_pulse_flag");
      return 1;
    } 
  else 
    { 
      CUDAMEMPRINT(dev_find_pulse_flag,"cudaMalloc((void**) &dev_find_pulse_flag",1,sizeof(*dev_find_pulse_flag)); 
    };
  
  int maxdivs = 1; 
  for (int i = 32; i <= PulseMax; maxdivs++, i *= 2);
  
  cu_err = cudaMalloc((void**) &dev_t_funct_cache, (1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax) * sizeof(float));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_t_funct_cache");
      return 1;
    } 
  else 
    { 
      CUDAMEMPRINT(dev_t_funct_cache,"cudaMalloc((void**) &dev_t_funct_cache",(1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax),sizeof(float)); 
    };
  
  //	cu_err = cudaMalloc((void**) &dev_avg, cudaAcc_NumDataPoints*sizeof(float));
  //    if( cudaSuccess != cu_err) 
  //    {
  //        CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_avg");
  //        return 1;
  //    } else { CUDAMEMPRINT(dev_t_funct_cache,"cudaMalloc((void**) &dev_avg",cudaAcc_NumDataPoints,sizeof(float)); };
  
  cudaMemsetAsync(dev_t_funct_cache, 0, (1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax) * sizeof(float));
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  cu_err = cudaErrorInvalidValue; //default is failure so host will generate it if kernels fail
  if(gCudaDevProps.major >= 2) 
    {	       
      dim3 block(maxdivs,32,1);
      dim3 grid(1,CUDA_ACC_FOLDS_COUNT,1);
      for (int di= 1; di< (PulseMax/32)+1; di++) 
	{
	  cudaAcc_dev_t_funct<<<grid,block>>>((float)PulseThresh, PulseMax, di, dev_t_funct_cache, (float)pulse_display_thresh);
	  cu_err = CUDASYNC;
	  if (cudaSuccess != cu_err) {
	    CUDA_ACC_SAFE_CALL_NO_SYNC("cudaAcc_dev_t_funct");
	    break; // No need to keep going if there's a problem, and we can save the last error status
	  }
	}
    }
  
  if (cudaSuccess != cu_err || gCudaDevProps.major < 2) 
    { // did not process properly on GPU, revert to host code, or we're using a pre-Fermi card
      float* t_funct_cache = (float*) malloc(maxdivs * CUDA_ACC_FOLDS_COUNT  * PulseMax * sizeof(float));
      for(int j = 0, l = 1; j < maxdivs; ++j, l *= 2) 
	{
	  for(int num_adds = 0; num_adds < CUDA_ACC_FOLDS_COUNT; ++num_adds) // cache for 2, 3 ,4, 5 folds
	    for (int di = 1; di < PulseMax; ++di)
	      {
		t_funct_cache[j * PulseMax * CUDA_ACC_FOLDS_COUNT + num_adds * PulseMax + di] =
		  cudaAcc_host_t_funct(pulse_display_thresh, PulseThresh, di, (num_adds+CUDA_ACC_FOLDS_START)*l);
	      }
	}
      
      CUDA_ACC_SAFE_CALL((cudaMemcpyAsync(dev_t_funct_cache, t_funct_cache, maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax * sizeof(float), cudaMemcpyHostToDevice)),true);
      CUDA_ACC_SAFE_CALL((CUDASYNC),true);
      
      free(t_funct_cache);
    }
  
  CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  cudaAcc_PulseMax = PulseMax;
  cudaAcc_rcfg_dis_thresh = (float) (1.0 / pulse_display_thresh);
  
  PulseFind_settings.NumDataPoints = cudaAcc_NumDataPoints;
  // find_triplets
  PulseFind_settings.power_ft = dev_PowerSpectrum;
  PulseFind_settings.resultsT = dev_TripletResults;
  PulseFind_settings.result_flags_ft = dev_flagT;
  PulseFind_settings.tmp_potT = dev_tmp_potT;
  
  // find_pulse
  PulseFind_settings.PulsePot_fp = dev_PowerSpectrum;
  //PulseFind_settings.PulsePot8_fp = dev_t_PowerSpectrum + 8;
  PulseFind_settings.tmp_potP = dev_tmp_potP;
  PulseFind_settings.best_potP = dev_best_potP;
  PulseFind_settings.report_potP = dev_report_potP;
  PulseFind_settings.resultsP = dev_PulseResults;
  //	PulseFind_settings.avg = dev_avg;
  
  PulseFind_settings.result_flags_fp = dev_find_pulse_flag;
  PulseFind_settings.t_funct_cache_fp = dev_t_funct_cache;
  PulseFind_settings.rcfg_dis_thresh =  cudaAcc_rcfg_dis_thresh;
  PulseFind_settings.PulseMax = 40960; //cudaAcc_PulseMax;
  
  CUDA_ACC_SAFE_CALL((cudaMemcpyToSymbol(cudaAcc_PulseFind_settings, (void*) &PulseFind_settings, sizeof(PulseFind_settings))),true);

  return 0;
}



void cudaAcc_free_pulse_find() 
{	
  cudaFree(dev_t_funct_cache);
  dev_t_funct_cache = NULL;
}



template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_reduce_sum(float* sdata) 
{
  int tid = threadIdx.x;
  float *sdatap = sdata + tid;
  if (fft_n * n > 32)
    __syncthreads();
  
  //Jason: Adding to fit fermi hardware
  if(fft_n * n >= 1024 && fft_n < 1024) 
    {
      if(tid < 512) 
	sdatap[0] += sdatap[0 + 512];
      __syncthreads();
    }
  
  if(fft_n * n >= 512 && fft_n < 512) 
    {
      if(tid < 256) 
	sdatap[0] += sdatap[0 + 256];
      __syncthreads();
    }
  
  if(fft_n * n >= 256 && fft_n < 256) 
    {
      if(tid < 128) 
	sdatap[0] += sdatap[0 + 128];
      __syncthreads();
    }
  
  if(fft_n * n >= 128 && fft_n < 128) 
    {
      if (tid < 64) 
	sdatap[0] += sdatap[0 + 64];
      __syncthreads();
    }
  
  if(tid < 32) 
    {
      volatile float *smem = sdatap;
      if(fft_n * n >= 64 && fft_n < 64) 
	smem[0] += smem[0 + 32];
      if(fft_n * n >= 32 && fft_n < 32) 
	smem[0] += smem[0 + 16];
      if(fft_n * n >= 16 && fft_n < 16) 
	smem[0] += smem[0 + 8];
    }
  
  return sdata[tid & (fft_n - 1)];
}


template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_reduce_max(float* sdata) 
{
  int tid = threadIdx.x;		
  float *sdatap = &sdata[tid];
  
  if(fft_n * n > 32)
    __syncthreads();
  
  //Jason: Adding to fit fermi hardware
  if(fft_n * n >= 1024 && fft_n < 1024) 
    {
      if(tid < 512) 
	sdatap[0] = max(sdatap[0], sdatap[512]);
      __syncthreads();
  }
  
  if(fft_n * n >= 512 && fft_n < 512) 
    {
      if(tid < 256) 
	sdatap[0] = max(sdatap[0], sdatap[0 + 256]);
      __syncthreads();
    }
  
  if(fft_n * n >= 256 && fft_n < 256) 
    {
      if(tid < 128) 
	sdatap[0] = max(sdatap[0], sdatap[0 + 128]);
      __syncthreads();
    }
  
  if(fft_n * n >= 128 && fft_n < 128) 
    {
      if(tid < 64) 
	sdatap[0] = max(sdatap[0], sdatap[0 + 64]);
      __syncthreads();
    }
  
  if(tid < 32) 
    {
      volatile float *smem = sdatap;
      
      if(fft_n * n >= 64 && fft_n < 64) *smem = max(*smem, smem[32]);
      if(fft_n * n >= 32 && fft_n < 32) *smem = max(*smem, smem[16]);
      if(fft_n * n >= 16 && fft_n < 16) *smem = max(*smem, smem[8]);
    }

  if(fft_n * n > 32)
    __syncthreads();

  return sdata[tid & (fft_n - 1)];
}



template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop2_2(float *tab, float * __restrict__ dest, int di, float *tmp0) 
{
  int rounds = (n - 1 + di - (int)(threadIdx.x >> (int)log2((float) fft_n))) / n;

  int idx = threadIdx.x & (-fft_n);
  float * __restrict__ one = tab + idx;
  float * __restrict__ two = tmp0 + idx; 
  dest += idx;

  int i = 0;
  float tmax = 0.0f, tmax2 = 0.0f;

#define UNRL 16
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j+=8)
	{
	  a[j+0] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+1] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+2] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+3] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+4] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+5] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+6] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+7] = *one + *two; one += fft_n*n; two += fft_n*n;
//	}
//      for(int j = 0; j < UNRL; j++)
//	{
	  *dest = a[j+0]; dest += fft_n*n;
	  tmax = max(tmax, a[j+0]);
	  *dest = a[j+1]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest += fft_n*n;
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+3]);
	  *dest = a[j+4]; dest += fft_n*n;
	  tmax = max(tmax, a[j+4]);
	  *dest = a[j+5]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+5]);
	  *dest = a[j+6]; dest += fft_n*n;
	  tmax = max(tmax, a[j+6]);
	  *dest = a[j+7]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+7]);
	}
    }

#define UNRL 4
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j+=4)
	{
	  a[j+0] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+1] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+2] = *one + *two; one += fft_n*n; two += fft_n*n;
	  a[j+3] = *one + *two; one += fft_n*n; two += fft_n*n;
//	}
//      for(int j = 0; j < UNRL; j++)
//	{
	  *dest = a[j+0]; dest += fft_n*n;
	  tmax = max(tmax, a[j+0]);
	  *dest = a[j+1]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest += fft_n*n;
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+3]);
	}
    }

#pragma unroll 1
  for(; i < rounds; i += 1) 
    {
      float sum  = *one + *two; one += fft_n*n; two += fft_n*n;
      *dest = sum; dest += fft_n*n;
      tmax = fmaxf(tmax, sum);		
    }

  tmax = max(tmax, tmax2);
  return tmax;
}



template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop3_2(float *tab, float * __restrict__ dest, int ul_PoT, int di, float *tmp0, float *tmp1) 
{
//  int idx = (threadIdx.x / fft_n) * fft_n;
  int idx = threadIdx.x & (-fft_n);
  float * __restrict__ one = tab + idx;
  float * __restrict__ two = tmp0 + idx; 
  float * __restrict__ three = tmp1 + idx; 
  dest += idx;

  int rounds = (n - 1 + di - (int)(threadIdx.x >> (int)log2((float)fft_n))) / n;
  float tmax = 0.0f, tmax2 = 0.0f;
  int i = 0;
  
#define UNRL 8
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[4];
      for(int j = 0; j < UNRL; j += 4)
	{
	  a[0] = *one; one += fft_n*n;
	  a[1] = *one; one += fft_n*n;
	  a[2] = *one; one += fft_n*n;
	  a[3] = *one; one += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  a[0] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[1] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[2] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[3] += *two + *three; two += fft_n*n; three += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  *dest = a[0]; dest += fft_n*n; tmax  = max(tmax,  a[0]);
	  *dest = a[1]; dest += fft_n*n; tmax2 = max(tmax2, a[1]);
	  *dest = a[2]; dest += fft_n*n; tmax  = max(tmax,  a[2]);
	  *dest = a[3]; dest += fft_n*n; tmax2 = max(tmax2, a[3]);
	}
    }

#define UNRL 4
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[4];
      for(int j = 0; j < UNRL; j+=4)
	{
	  a[0] = *one; one += fft_n*n;
	  a[1] = *one; one += fft_n*n;
	  a[2] = *one; one += fft_n*n;
	  a[3] = *one; one += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  a[0] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[1] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[2] += *two + *three; two += fft_n*n; three += fft_n*n;
	  a[3] += *two + *three; two += fft_n*n; three += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  *dest = a[0]; dest += fft_n*n; tmax  = max(tmax,  a[0]);
	  *dest = a[1]; dest += fft_n*n; tmax2 = max(tmax2, a[1]);
	  *dest = a[2]; dest += fft_n*n; tmax  = max(tmax,  a[2]);
	  *dest = a[3]; dest += fft_n*n; tmax2 = max(tmax2, a[3]);
	}
    }
  
#pragma unroll 1
  for(; i < rounds; i += 1) 
    {
      float sum  = *one + *two;  one += fft_n*n; two += fft_n*n; 
      sum += *three; three += fft_n*n;
      *dest = sum; dest += fft_n*n;
      tmax = fmaxf(tmax, sum);		
    }
 
  tmax = max(tmax, tmax2);
  return tmax;
}


template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop4_2(float *tab, float * __restrict__  dest, int ul_PoT, int di, float *tmp0, float *tmp1, float *tmp2) 
{
//  int idx = (threadIdx.x / fft_n) * fft_n;
  int idx = threadIdx.x & (-fft_n);
  float * __restrict__ one = tab + idx;
  float * __restrict__ two = tmp0 + idx;   // (float *)((void *)tab + tmp0);// * fft_n * sizeof(float));
  float * __restrict__ three = tmp1 + idx; //(float *)((void *)tab + tmp1);// * fft_n * sizeof(float));
  float * __restrict__ four = tmp2 + idx;  //(float *)((void *)tab + tmp2);// * fft_n * sizeof(float));
  dest += idx;

  int rounds = (n - 1 + di - (int)(threadIdx.x >> (int)log2((float)fft_n))) / n;
  float tmax = 0.0f, tmax2 = 0.0f;
  int i = 0;

#define UNRL 8
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[4];
      for(int j = 0; j < UNRL; j+=4)
	{
	  a[0] = *one + *two;  one += fft_n*n; two += fft_n*n;
	  a[1] = *one + *two;  one += fft_n*n; two += fft_n*n;
	  a[2] = *one + *two;  one += fft_n*n; two += fft_n*n;
	  a[3] = *one + *two;  one += fft_n*n; two += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  a[0] += *three + *four; three += fft_n*n; four += fft_n*n;
	  a[1] += *three + *four; three += fft_n*n; four += fft_n*n;
	  a[2] += *three + *four; three += fft_n*n; four += fft_n*n;
	  a[3] += *three + *four; three += fft_n*n; four += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  *dest = a[0]; dest += fft_n*n; tmax  = max(tmax,  a[0]);
	  *dest = a[1]; dest += fft_n*n; tmax2 = max(tmax2, a[1]);
	  *dest = a[2]; dest += fft_n*n; tmax  = max(tmax,  a[2]);
	  *dest = a[3]; dest += fft_n*n; tmax2 = max(tmax2, a[3]);
	}
    }

#pragma unroll 1
  for(; i < rounds; i += 1) 
    {
      float sum  = *one + *two; one += fft_n*n; two += fft_n*n;
      sum += *three + *four; three += fft_n*n; four += fft_n*n;
      *dest = sum; dest += fft_n*n;
      tmax = fmaxf(tmax, sum);		
    }

  tmax = max(tmax, tmax2); 
  return tmax;
}


template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop5_2(float *tab, float * __restrict__ dest, int ul_PoT, int di, float *tmp0, float *tmp1, float *tmp2, float *tmp3) 
{
//  int idx = (threadIdx.x / fft_n) * fft_n;
  int idx = threadIdx.x & (-fft_n);
  float * __restrict__ one = tab + idx;
  float * __restrict__ two = tmp0 + idx; //(float *)((void *)tab + tmp0);// * fft_n * sizeof(float));
  float * __restrict__ three = tmp1 + idx; //(float *)((void *)tab + tmp1);// * fft_n * sizeof(float));
  float * __restrict__ four = tmp2 + idx; //(float *)((void *)tab + tmp2);// * fft_n * sizeof(float));
  float * __restrict__ five = tmp3 + idx; //(float *)((void *)tab + tmp3);// * fft_n * sizeof(float));
  dest += idx;

  int rounds = (n - 1 + di - (int)(threadIdx.x >> (int)log2((float)fft_n))) / n;
  float tmax = 0.0f, tmax2 = 0.0f;
  int i = 0;

#define UNRL 8
  for(; i < (rounds-(UNRL-1)); i += UNRL) 
    {
      float a[UNRL];
      for(int j = 0; j < UNRL; j+=4)
	{
	  a[j]  = *one;   one += fft_n*n;
	  a[j+1]  = *one;   one += fft_n*n;
	  a[j+2]  = *one;   one += fft_n*n;
	  a[j+3]  = *one;   one += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  a[j] += *two + *three;   two += fft_n*n; three += fft_n*n; 
	  a[j+1] += *two + *three;   two += fft_n*n; three += fft_n*n; 
	  a[j+2] += *two + *three;   two += fft_n*n; three += fft_n*n; 
	  a[j+3] += *two + *three;   two += fft_n*n; three += fft_n*n; 
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  a[j] += *four + *five;  four += fft_n*n; five += fft_n*n;
	  a[j+1] += *four + *five;  four += fft_n*n; five += fft_n*n;
	  a[j+2] += *four + *five;  four += fft_n*n; five += fft_n*n;
	  a[j+3] += *four + *five;  four += fft_n*n; five += fft_n*n;
	  //	}
	  //      for(int j = 0; j < UNRL; j++)
	  //	{
	  *dest = a[j]; dest += fft_n*n;
	  tmax = max(tmax, a[j+0]);
	  *dest = a[j+1]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+1]);
	  *dest = a[j+2]; dest += fft_n*n;
	  tmax = max(tmax, a[j+2]);
	  *dest = a[j+3]; dest += fft_n*n;
	  tmax2 = max(tmax2, a[j+3]);
	}
    }

#pragma unroll 1
  for(; i < rounds; i += 1) 
    {
      float sum  = *one + *two; one += fft_n*n; two += fft_n*n;
      sum += *three + *four; three += fft_n*n; four += fft_n*n;
      sum += *five; five += fft_n*n;
      *dest = sum; dest += fft_n*n;
      tmax = max(tmax, sum);		
    }

  tmax = max(tmax, tmax2);
  return tmax;
}


template <unsigned int fft_n, unsigned int n>
__device__ void cudaAcc_copy2(float * __restrict__ from, float * __restrict__ to, int count) 
{
//  int j = (threadIdx.x / fft_n) * fft_n;
  int j = threadIdx.x & (-fft_n);
  float * __restrict__ f = &from[j], * __restrict__ t = &to[j];
  int rounds = (n - 1 + count - (int)(threadIdx.x / fft_n)) / n;
  int i = 0;
/*
#define UNRL 8
  for(; i < rounds-(UNRL-1); i += UNRL)
    for(int j = 0; j < UNRL; j++) 
    {
      *t = *f;
      f += (fft_n*n);
      t += (fft_n*n);
    }
*/
#pragma unroll 1
  for(; i < rounds; i += 1) 
    {
      *t = *f;
      f += (fft_n*n);
      t += (fft_n*n);
    }
}




template <int fft_n, int numper, int num_adds, bool load_state>
__global__ void 
find_pulse_kernel2m(
								   float best_pulse_score,
								   int PulsePotLen, 
								   int AdvanceBy,
								   int y_offset,
                                   int ndivs,
                                   int firstP,
                                   int lastP) 
{	
  if(load_state == false && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
      cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 0;
      cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 0;
    }

  int PoTLen = 1024*1024 / fft_n;//cudaAcc_PulseFind_settings.NumDataPoints
  //const int fidx = threadIdx.x/fft_n;
  int ul_PoT = threadIdx.x & (fft_n-1);
  int y = blockIdx.y * blockDim.y + y_offset;
  int TOffset1 = y * AdvanceBy;
  int TOffset2 = y * AdvanceBy;
  int y4 = y << 2;
  float *tfc = cudaAcc_PulseFind_settings.t_funct_cache_fp + (num_adds - CUDA_ACC_FOLDS_START) * 40960;

  if(TOffset1 > PoTLen - PulsePotLen) 
    {		
      TOffset1 = PoTLen - PulsePotLen;
    }
  
  float * __restrict__ fp_PulsePot	= cudaAcc_PulseFind_settings.PulsePot_fp	+ ul_PoT + TOffset1 * fft_n;
  float * __restrict__ tmp_pot		= cudaAcc_PulseFind_settings.tmp_potP		+ ul_PoT + TOffset2 * fft_n;
  float * __restrict__ best_pot		= cudaAcc_PulseFind_settings.best_potP		+ ul_PoT + TOffset2 * fft_n;
  float * __restrict__ report_pot	= cudaAcc_PulseFind_settings.report_potP	+ ul_PoT + TOffset2 * fft_n;
  
  int di; //maxs = 0
  float maxd=0,avg=0; //snr=0, maxp=0, max=0, fthresh=0
  float tmp_max, t1;
  
  __shared__ float sdata[fft_n*numper];
  float *sdatat = &sdata[threadIdx.x];
  float4 * __restrict__ resp = &cudaAcc_PulseFind_settings.resultsP[AT_XY(ul_PoT, y4, fft_n)];

  if(!load_state) 
    {
      //  Calculate average power
      avg = (mean[ul_PoT + y*fft_n]);

      if(threadIdx.x < fft_n) 
        { 
#define zero 0
	  resp[0].x = zero;
	  resp[fft_n].x = zero;
	  resp[2 * fft_n].x = zero;
	  resp[3 * fft_n].x = zero;
	  resp[0].y = zero;
	  resp[fft_n].y = zero;
	  resp[2 * fft_n].y = avg;
	  resp[3 * fft_n].y = zero;
	  resp[0].z = zero;
	  resp[fft_n].z = zero;
	  resp[2 * fft_n].z = zero;
	  resp[3 * fft_n].z = zero;
	  resp[0].w = zero;
	  resp[fft_n].w = zero;
	  resp[3 * fft_n].w = zero;
	  resp[2 * fft_n].w = zero;
	}
    }
  else
    {		
      best_pulse_score = fmax(best_pulse_score, resp[0].w);
      float4 *tmp_float4;
      tmp_float4 = &resp[2 * fft_n];
      avg = tmp_float4->y;
      //max = tmp_float4.x * avg;
      //maxp = tmp_float4.z;
      maxd = tmp_float4->w;
      
      //tmp_float4 = &resp[3 * fft_n];
      //snr = tmp_float4.y;
      //fthresh = tmp_float4->z;
      //maxs = tmp_float4.w;
    } 
  
  
  //  Periods from PulsePotLen/3 to PulsePotLen/4, and power of 2 fractions of.
  //   then (len/4 to len/5) and finally (len/5 to len/6)
  //for(int num_adds = 3; num_adds <= 5; num_adds++)

  float sqrt_num_adds_div_avg = 1.0f/(avg/(float)sqrt((float)num_adds));
  int olddi = -1;
#pragma unroll 1
  for (int p = lastP; p > firstP ; p--)
    //for (int p = lastP-(threadIdx.x/fft_n) ; p > firstP ; p-=numper)
    //for (int p = lastP ; p > firstP ; p-=numper)
    //Jason: we're changing the behaviout to do a different set of periods in each 'subblock'...
    //... The summing reductions will need to adapt to work to reduce the lot ( numper FFTs ), instead of just first FFT...
    //... such that the strongest one is reproted for each ulpot, and best overall.
    {
      float cur_thresh, dis_thresh;
      int perdiv;
      int tmp0, tmp1, tmp2, tmp3;
      
      perdiv = num_adds - 1;

      di = (int)p/(int)perdiv;                      // (int)period			
      if(di != olddi)
        {
          dis_thresh = cudaAcc_t_funct_ldg<num_adds>(di, 0, 40960, tfc) * avg; //cudaAcc_PulseFind_settings.PulseMax=40960
	  olddi = di;
	}
      
      if(num_adds == 3) 
        {
	  tmp0 = fft_n * ((p + 1) >> 1);
	  tmp1 = fft_n * p; 
	  sdatat[0] = cudaAcc_sumtop3_2<fft_n, numper>(&fp_PulsePot[0], &tmp_pot[0], ul_PoT, di, &fp_PulsePot[0] + tmp0, &fp_PulsePot[0] + tmp1);
	}

      if(num_adds == 4)
        {
	  tmp0 = fft_n * (int)((p + p + 1)/6);
	  tmp1 = fft_n * (int)((p + p + 1)/3);
          tmp2 = fft_n * p;
	  sdatat[0] = cudaAcc_sumtop4_2<fft_n, numper>(&fp_PulsePot[0], &tmp_pot[0], ul_PoT, di, &fp_PulsePot[0]+tmp0, &fp_PulsePot[0]+tmp1, &fp_PulsePot[0]+tmp2);
	}

      if(num_adds == 5)
        {
	  tmp0 = fft_n * ((p + 2) >> 2);
	  tmp1 = fft_n * ((p + 1) >> 1);
	  tmp2 = fft_n * ((p * 3 + 2) >> 2); 
	  tmp3 = fft_n * p; 
	  sdatat[0] = cudaAcc_sumtop5_2<fft_n, numper>(&fp_PulsePot[0], &tmp_pot[0], ul_PoT, di, &fp_PulsePot[0]+tmp0, &fp_PulsePot[0]+tmp1, &fp_PulsePot[0]+tmp2, &fp_PulsePot[0]+tmp3);
        }

      tmp_max = cudaAcc_reduce_max<fft_n, numper>(sdata);
      
      if(tmp_max > dis_thresh) 
	{
	  // unscale for reporting
	  tmp_max /= num_adds;
	  cur_thresh = (dis_thresh / num_adds - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

	  float a, b;
	  float _snr = (a = tmp_max-avg) * sqrt_num_adds_div_avg;
	  float _thresh = (b = cur_thresh-avg) * sqrt_num_adds_div_avg;
	  if(a > (best_pulse_score * b)) 
	    {
	      best_pulse_score = a / b;
	      cudaAcc_copy2<fft_n, numper>(tmp_pot, best_pot, di); 				          
	      if (threadIdx.x < fft_n) 
		{
		  resp[0] = make_float4(
												    tmp_max/avg,
												    avg,
												    ((float)p)/(float)perdiv,
												    best_pulse_score//TOffset1+PulsePotLen/2
												    );
		  resp[fft_n] = make_float4(
												    ul_PoT,
												    _snr,
												    _thresh,
												    num_adds
												    );
		  cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;	
                }
            }

	  if ((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) 
	    {
	      //maxp  = (float)p/(float)perdiv;
	      maxd  = t1;
	      //maxs  = num_adds;
	      //max = tmp_max;
	      //snr = _snr;
	      //fthresh= _thresh;
	      cudaAcc_copy2<fft_n, numper>(tmp_pot, report_pot, di);
	      
	      if(threadIdx.x < fft_n) 
		{
		  // It happens very rarely so it's better to store the results right away instead of
		  // storing them in registers or worse local memory
		  resp[2 * fft_n] = make_float4(
												    tmp_max/avg,
												    avg,
												    (float)p/(float)perdiv, //maxp
												    maxd//TOffset1+PulsePotLen/2
												    );
		  resp[3 * fft_n] = make_float4(
												    ul_PoT,
												    _snr,
												    _thresh,
												    num_adds
												    );
		  cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
                }
            }
        }
      
      int num_adds_2 = num_adds << 1;
      //#pragma unroll 1
      for(int j = 1; j < ndivs ; j++) 
        {
          float sqrt_num_adds_2_div_avg = 1.0f/(avg/(float)sqrt((float)num_adds_2));
	  perdiv = perdiv << 1;
	  tmp0 = di & 1;
	  di = di >> 1;
	  tmp0 = (tmp0+di)*fft_n;

	  float dis_thresh2 = cudaAcc_t_funct<num_adds>(di, j, 40960, tfc) * avg; //cudaAcc_PulseFind_settings.PulseMax=40960

	  sdatat[0] = cudaAcc_sumtop2_2<fft_n, numper>(&tmp_pot[0], &tmp_pot[0], di, &tmp_pot[0] + tmp0);

	  tmp_max = cudaAcc_reduce_max<fft_n, numper>(sdata);

	  
	  if(tmp_max>dis_thresh2) 
	    {
	    // unscale for reporting
	      tmp_max /= num_adds_2;
	      cur_thresh = (dis_thresh2 / num_adds_2 - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;
	      float a, b;
	      float _snr = (a = tmp_max-avg) * sqrt_num_adds_2_div_avg;
	      float _thresh = (b = cur_thresh-avg) * sqrt_num_adds_2_div_avg;
	      if(a > (best_pulse_score * b)) 
		{
		  best_pulse_score = a / b;
		  cudaAcc_copy2<fft_n,numper>(tmp_pot, best_pot, di);
		  if(threadIdx.x < fft_n) 
		    {
		      resp[0] = make_float4(
													tmp_max/avg,
													avg,
													((float)p)/(float)perdiv,
													best_pulse_score//TOffset1+PulsePotLen/2
													);
		      resp[fft_n] = make_float4(
													ul_PoT,
													_snr,
													_thresh,
													num_adds_2
													);
		      cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;
                    }
                }
	      
	      if((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) 
		{
		  //maxp = (float)p/(float)perdiv;
		  maxd = t1;
		  //maxs = num_adds_2;
		  //max  = tmp_max;
		  //snr  = _snr;
		  //fthresh = _thresh;
		  cudaAcc_copy2<fft_n, numper>(tmp_pot, report_pot, di);
		  
		  // It happens very rarely so it's better to store the results right away instead of
		  // storing them in registers or worse local memory
		  if (threadIdx.x < fft_n) 
		    {
		      resp[2 * fft_n] = make_float4(
													tmp_max/avg,
													avg,
													(float)p/(float)perdiv, //maxp,
													maxd //TOffset1+PulsePotLen/2
													);
		      resp[3 * fft_n] = make_float4(
													ul_PoT,
													_snr,
													_thresh,
													num_adds_2
													);
		      cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
                    }
                }
            }
	  
	  num_adds_2 = num_adds_2 << 1;
	}  // for (j = 1; j < ndivs
    } // for (p = lastP
}





// Assuming AdvanceBy >= PulsePoTLen / 2
template <unsigned int fft_n, unsigned int numthreads>
int cudaAcc_find_pulse_original2m(float best_pulse_score, int PulsePoTLen, int AdvanceBy) 
{	
  int PoTLen = cudaAcc_NumDataPoints / fft_n;
  int parts = (PulsePoTLen == PoTLen) ? 1 : PoTLen / AdvanceBy;
  int num_iter = pfPeriodsPerLaunch;
  int num_blocks = gCudaDevProps.multiProcessorCount*pfBlocksPerSM;  
  
#if CUDART_VERSION >= 3000
  if(gCudaDevProps.major >= 2)
    {
      //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    }
#endif
  int numdivs, firstP, lastP, itr_sent, nRemaining;
#if 0 //def _WIN32
  _BitScanReverse((DWORD *)&numdivs,PulsePoTLen);
  numdivs = max(1,numdivs-4);
#else
  int i;
  for (i = 32, numdivs = 1; i <= PulsePoTLen; numdivs++, i *= 2); 
#endif
  
  for(int y_offset = 0; y_offset < parts; y_offset += num_blocks ) 
    {
      dim3 block(numthreads, 1, 1);
      dim3 grid(1, min(parts - y_offset, num_blocks), 1);
      lastP = (PulsePoTLen*2)/3;  firstP = (PulsePoTLen*1)>>1;
      nRemaining = lastP - firstP; 
      itr_sent = MIN(nRemaining, num_iter);
      firstP = lastP - itr_sent;
      CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2m<fft_n, numthreads/fft_n, 3, false><<<grid, block, 0, fftstream1>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);

      lastP = firstP;
      nRemaining -= itr_sent;
      while (nRemaining)
        {
	  itr_sent = MIN(nRemaining, num_iter);
	  firstP = lastP - itr_sent;
	  CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2m<fft_n, numthreads/fft_n, 3, true><<<grid, block, 0, fftstream1>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);

	  lastP = firstP;
	  nRemaining -= itr_sent;
        } 
      
      lastP = (PulsePoTLen*3)>>2;  firstP = (PulsePoTLen*3)/5; 
      nRemaining = lastP - firstP; 
      do
        {
	  itr_sent = MIN(nRemaining, num_iter);
	  firstP = lastP - itr_sent;
	  CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2m<fft_n, numthreads/fft_n, 4, true><<<grid, block, 0, fftstream1>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);

	  lastP = firstP;
	  nRemaining -= itr_sent;
        } while (nRemaining);
      
      lastP = (PulsePoTLen*4)/5;  firstP = (PulsePoTLen*4)/6; 
      nRemaining = lastP - firstP; 
      do
        {
	  itr_sent = MIN(nRemaining, num_iter);
	  firstP = lastP - itr_sent;
	  CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2m<fft_n, numthreads/fft_n, 5, true><<<grid, block, 0, fftstream1>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);

	  lastP = firstP;
	  nRemaining -= itr_sent;
        } while (nRemaining);
    }

  return 0;
}


void cudaAcc_choose_best_find_pulse(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) 
{
  if (gCudaDevProps.regsPerBlock >= 32 * 1024) 
    {
      //At least 32 * 1024 registers, so we can launch 1024 threads per block. At least CUDA 2.0 compatibile device
      switch (FftLength) 
	{
	case 8:
	  cudaAcc_find_pulse_original2m<8, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	case 16:
	  cudaAcc_find_pulse_original2m<16, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break; 
	case 32:
	  cudaAcc_find_pulse_original2m<32, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;		
	case 64:
	  cudaAcc_find_pulse_original2m<64, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	case 128:
	  cudaAcc_find_pulse_original2m<128, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	case 256:
	  cudaAcc_find_pulse_original2m<256, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	case 512:
	  cudaAcc_find_pulse_original2m<512, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	case 1024:
	  cudaAcc_find_pulse_original2m<1024, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	  break;
	default:			
	  cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
	}
    }
  else 
    if(gCudaDevProps.regsPerBlock >= 16 * 1024) 
      {
	// more that 16 * 1024 registers so we can luch 512 threads per block. At least CUDA 1.2 compatibile device
	switch(FftLength) 
	  {
	  case 8:
	    cudaAcc_find_pulse_original2m<8, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 16:
	    cudaAcc_find_pulse_original2m<16, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 32:
	    cudaAcc_find_pulse_original2m<32, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;		
	  case 64:
	    cudaAcc_find_pulse_original2m<64, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 128:
	    cudaAcc_find_pulse_original2m<128, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 256:
	    cudaAcc_find_pulse_original2m<256, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 512:
	    cudaAcc_find_pulse_original2m<512, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  default:			
	    cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
	  }
      } 
    else 
      {
	// less that 16 * 1024 registers (assuming 8 * 1024) so we can luch only 2562 threads per block. CUDA 1.0 or 1.1 compatibile device
	switch (FftLength) 
	  {
	  case 8:
	    cudaAcc_find_pulse_original2m<8, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 16:
	    cudaAcc_find_pulse_original2m<16, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 32:
	    cudaAcc_find_pulse_original2m<32, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;		
	  case 64:
	    cudaAcc_find_pulse_original2m<64, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 128:
	    cudaAcc_find_pulse_original2m<128, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  case 256:
	    cudaAcc_find_pulse_original2m<256, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
	    break;
	  default:			
	    cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
	  }
      }
}


int cudaAcc_find_pulses(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) 
{	
  cudaStreamWaitEvent(fftstream1, meanDoneEvent, 0);

  //CUDA_ACC_SAFE_LAUNCH((cudaMemsetAsync(dev_find_pulse_flag, 0, sizeof(*dev_find_pulse_flag), fftstream1)),true);

  cudaAcc_choose_best_find_pulse(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);

  if(Pflags == NULL)
    cudaMallocHost(&Pflags, sizeof(result_find_pulse_flag));

  Pflags->has_best_pulse = -1;
  CUDA_ACC_SAFE_LAUNCH((cudaMemcpyAsync(Pflags, dev_find_pulse_flag, sizeof(*dev_find_pulse_flag), cudaMemcpyDeviceToHost, fftstream1)),true);
  
  cudaEventRecord(pulseDoneEvent, fftstream1);

  return 0;
}


int cudaAcc_processPulseResults(int PulsePoTLen, int AdvanceBy, int FftLength) 
{
  int PoTLen = cudaAcc_NumDataPoints / FftLength;
  int nb_of_results = 0;    
  int PoTStride = (( (PulsePoTLen == PoTLen) ? 1: PoTLen/AdvanceBy) + 1) * AdvanceBy;

  cudaEventSynchronize(pulseDoneEvent); 
 
  // Iterating trough results
  for(int ThisPoT = 1; ThisPoT < FftLength; ThisPoT++) 
    {
      for(int TOffset = 0, TOffsetOK = true, PulsePoTNum = 0; TOffsetOK; PulsePoTNum++, TOffset += AdvanceBy) 
	{
	  int TOffset2 = TOffset;
	  if(TOffset + PulsePoTLen >= PoTLen) 
	    {
	      TOffsetOK = false;
	      TOffset = PoTLen - PulsePoTLen;
	    }
	  
	  int index0 = ((PulsePoTNum * 4 + 0) * FftLength + ThisPoT);
	  int index1 = ((PulsePoTNum * 4 + 1) * FftLength + ThisPoT);
	  int index2 = ((PulsePoTNum * 4 + 2) * FftLength + ThisPoT);
	  int index3 = ((PulsePoTNum * 4 + 3) * FftLength + ThisPoT);
	  
	  if(Pflags->has_best_pulse) 
	    {
	      if(PulseResults[index0].x > 0) 
		{
		  float4 res1 = PulseResults[index0];
		  float4 res2 = PulseResults[index1];
		  
		  nb_of_results++;
		  cudaAcc_ReportPulseEvent(res1.x, res1.y, res1.z, TOffset+PulsePoTLen/2, (int) res2.x, res2.y, res2.z, 
					   &best_PoTP[ThisPoT * PoTStride + TOffset2], (int) res2.w, 0);
		}
	    }
	  
	  if(Pflags->has_report_pulse) 
	    {
	      if(!Pflags->has_best_pulse && ThisPoT == 1 && PulsePoTNum == 0) 
		{
		   //cudaEventSynchronize(pulseDoneEvent); 
		   //cudaStreamSynchronize(fftstream1); 
		} 
	      
	      if(PulseResults[index2].x > 0) 
		{
		  float4 res1 = PulseResults[index2];
		  float4 res2 = PulseResults[index3];
		  
		  nb_of_results++;
		  cudaAcc_ReportPulseEvent(res1.x, res1.y, res1.z, TOffset+PulsePoTLen/2, (int) res2.x, res2.y, res2.z, 
					   &tmp_PoTP[ThisPoT * PoTStride + TOffset2], (int) res2.w, 1);
		}
	    }
	}					
    }

  return 0;
}	


/*
  // Debug Printing of the results	
  CUDA_ACC_SAFE_CALL(cudaMemcpy(PulseResults, dev_PulseResults, 4 * (cudaAcc_NumDataPoints / AdvanceBy + 1) * sizeof(*dev_PulseResults), cudaMemcpyDeviceToHost));
  for (int i = 1; i < FftLength; ++i) {
  for (int j = 0; j < PoTLen / AdvanceBy; ++j) {
  int ThisPoT = i;
  int PulsePoTNum = j;
  int index0 = ((PulsePoTNum * 4 + 0) * FftLength + ThisPoT);
  int index1 = ((PulsePoTNum * 4 + 1) * FftLength + ThisPoT);
  int index2 = ((PulsePoTNum * 4 + 2) * FftLength + ThisPoT);
  int index3 = ((PulsePoTNum * 4 + 3) * FftLength + ThisPoT);
  
  logvalue("ul_PoT", i);
  logvalue("j", j);
  logvalue("TOffset", j * AdvanceBy);
  float4 a;
  a = PulseResults[index0];
  logvalue("tmp_max/avg", a.x);
  logvalue("avg", a.y);
  logvalue("((float)p)/(float)perdiv", a.z);
  logvalue("TOffset+PulsePoTLen/2", a.w);
  a = PulseResults[index1];
  logvalue("ul_PoT", a.x);
  logvalue("_snr", a.y);
  logvalue("_thresh", a.z);
  logvalue("num_adds", a.w);
  logvalue("SCORE: ", a.y / a.z);
  a = PulseResults[index2];
  logvalue("max/avg", a.x);panvalue("avg", a.y);
  logvalue("maxp", a.z);
  logvalue("TOffset+PulsePoTLen/2", a.w);
  a = PulseResults[index3];
  logvalue("ul_PoT", a.x);
  logvalue("snr", a.y);
  logvalue("fthresh", a.z);
  logvalue("maxs", a.w);			
  }		
  }
  */
  
//  return 0;
//}

#endif //USE_CUDA
