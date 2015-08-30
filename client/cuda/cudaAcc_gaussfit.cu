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

#define SQUARE(a) ((a) * (a))
#define NEG_LN_ONE_HALF     0.693f
//#define NEG_LN_ONE_HALF		0.69314718055994530941723212145818f
#define EXP(a,b,c)          exp(-(NEG_LN_ONE_HALF * SQUARE((float)(a) - (float)(b))) / (float)(c))

__device__ float cudaAcc_weight(int i) {
	//return (EXP(i, 0, cudaAcc_GaussFit_settings.GaussSigmaSq));
	return cudaAcc_GaussFit_settings.f_weight[i];
}

__device__ float cudaAcc_GetPeak(float fp_PoT[], int ul_TOffset, int ul_HalfSumLength, float f_MeanPower, float f_PeakScaleFactor, int ul_FftLength) {
	// Peak power is calculated as the weighted
	// sum of all powers within ul_HalfSumLength
	// of the assumed gaussian peak.
	// The weights are given by the gaussian function itself.
	// BUG WATCH : for the f_PeakScaleFactor to work,
	// ul_HalfSumLength *must* be set to sigma.

	int i;
	float f_sum;

	f_sum = 0.0;

	// Find a weighted sum
	for (i = ul_TOffset - ul_HalfSumLength; i <= ul_TOffset + ul_HalfSumLength; i++) {
		f_sum += (fp_PoT[i * ul_FftLength] - f_MeanPower) * cudaAcc_weight(abs(i-ul_TOffset));
	}

	return(f_sum * f_PeakScaleFactor);
}

__device__ float sqrf(float x) {
	return x * x;
}

__device__ float cudaAcc_GetChiSq(float fp_PoT[], int ul_FftLength, int ul_PowerLen, int ul_TOffset, float f_PeakPower, float f_MeanPower, float& xsq_null) {
	// We calculate our assumed gaussian powers
	// on the fly as we try to fit them to the
	// actual powers at each point along the PoT.

	float f_ChiSq = 0.0f,f_null_hyp=0.0f;
	float rebin = cudaAcc_GaussFit_settings.nsamples / ul_FftLength / ul_PowerLen;

	float recip_MeanPower = 1.0f / f_MeanPower;	
	for (int i = 0; i < ul_PowerLen; ++i) {				
		float f_PredictedPower = f_MeanPower + f_PeakPower * cudaAcc_weight(abs(i - ul_TOffset));		
		f_PredictedPower *= recip_MeanPower;

		// ChiSq in this realm is:
		//  sum[0:i]( (observed power - expected power)^2 / expected variance )
		// The power of a signal is:
		//  power = (noise + signal)^2 = noise^2 + signal^2 + 2*noise*signal
		// With mean power normalization, noise becomes 1, leaving:
		//  power = signal^2 +or- 2*signal + 1
		float noise=2.0f*sqrt(max(f_PredictedPower,1.0f)-1.0f)+1.0f;
		float recip_noise = rebin / noise;

		float PoTval = fp_PoT[i * ul_FftLength] * recip_MeanPower;
		f_ChiSq += (recip_noise*sqrf(PoTval - f_PredictedPower));
		f_null_hyp+= (recip_noise*sqrf(PoTval - 1.0f));
	}

	f_ChiSq/=ul_PowerLen;
	f_null_hyp/=ul_PowerLen;

	xsq_null=f_null_hyp;
	return f_ChiSq;
}

__device__ float cudaAcc_GetTrueMean(float fp_PoT[], int ul_PowerLen, float f_TotalPower, int ul_TOffset, int ul_ExcludeLen, int ul_FftLength) {
	// TrueMean is the mean power of the data set minus all power
	// out to ExcludeLen from our current TOffset.
	int i, i_start, i_lim;
	float f_ExcludePower = 0;

	// take care that we do not add to exclude power beyond PoT bounds!
	i_start = max(ul_TOffset - ul_ExcludeLen, 0);
	i_lim = min(ul_TOffset + ul_ExcludeLen + 1, ul_PowerLen);
	// TODO: prefix sums
	for (i = i_start; i < i_lim; i++) {
		f_ExcludePower += fp_PoT[i * ul_FftLength];
	}

	return((f_TotalPower - f_ExcludePower) / (ul_PowerLen - (i_lim - i_start)));
}

__device__ float cudaAcc_GetTrueMean2(float* fp_PoTPrefixSum, int ul_PowerLen, float f_TotalPower, int ul_TOffset, int ul_ExcludeLen, int ul_FftLength) {
	// TrueMean is the mean power of the data set minus all power
	// out to ExcludeLen from our current TOffset.
	int i_start, i_lim;
	float f_ExcludePower = 0;

	// take care that we do not add to exclude power beyond PoT bounds!
	i_start = max(ul_TOffset - ul_ExcludeLen, 0) - 1;
	i_lim = min(ul_TOffset + ul_ExcludeLen + 1, ul_PowerLen) - 1;

	f_ExcludePower = fp_PoTPrefixSum[i_lim * ul_FftLength];
	if (i_start >= 0)
		f_ExcludePower -= fp_PoTPrefixSum[i_start * ul_FftLength]; 

	return((f_TotalPower - f_ExcludePower) / (ul_PowerLen - (i_lim - i_start)));
}

float cudaAcc_GetPeakScaleFactor(float f_sigma) {
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

	for (i = -i_s; i <= i_s; i++) {
		f_sum += static_cast<float>(EXP(i, 0, f_sigma_sq));
	}

	return(1 / f_sum);
}

// only if (ul_NumSpectra > ul_PoTLen)
__global__ void GetFixedPoT_kernel(int ul_FftLength) {
	int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;
	int ul_PoT_i = blockIdx.y * blockDim.y + threadIdx.y;
	if (ul_PoT >= ul_FftLength) return;

	float* fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
	float* fp_PowerSpectrum = &cudaAcc_GaussFit_settings.dev_PowerSpectrum[ul_PoT];

	int   
		ul_PoTChunkSize,
		ul_PoTChunk_i,
		ul_PoTChunkLimit,
		ul_NumSpectra;

	ul_NumSpectra    = cudaAcc_GaussFit_settings.NumDataPoints / ul_FftLength;
	ul_PoTChunkSize  = ul_NumSpectra / cudaAcc_GaussFit_settings.gauss_pot_length;

	// If the number of spectra is greater than the number
	// of elements in a PoT array, we add sum adjacent spectra
	// into PoT elements.
	// ul_PoTChunkSize indicates how many time-wise
	// power spectra bins are added together to make one PoT bin.

	ul_PoTChunk_i = ul_PoTChunkSize * ul_PoT_i;
	ul_PoTChunkLimit = ul_PoTChunk_i + ul_PoTChunkSize;

//	float sum = 0.0f;
	float4 partials = {0.0f,0.0f,0.0f,0.0f};
	for (; ul_PoTChunk_i < ul_PoTChunkLimit-3; ul_PoTChunk_i+=4) {
		partials.x += fp_PowerSpectrum[ul_FftLength * (ul_PoTChunk_i + 0)];
		partials.y += fp_PowerSpectrum[ul_FftLength * (ul_PoTChunk_i + 1)];
		partials.z += fp_PowerSpectrum[ul_FftLength * (ul_PoTChunk_i + 2)];
		partials.w += fp_PowerSpectrum[ul_FftLength * (ul_PoTChunk_i + 3)];
	}
	// tail if needed
	while (ul_PoTChunk_i < ul_PoTChunkLimit)
	{
		partials.w += fp_PowerSpectrum[ul_FftLength * ul_PoTChunk_i];
		ul_PoTChunk_i++;
	}
	fp_PoT[ul_PoT_i*ul_FftLength] = partials.x+partials.y+partials.z+partials.w;
}

__global__ void NormalizePoT_kernel(int ul_FftLength) {
	int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;
	if (ul_PoT >= ul_FftLength) return;

	float* fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
	float* fp_PoTPrefixSum = &cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT];

	float f_TotalPower = 0.0f;
	for (int i = 0; i < cudaAcc_GaussFit_settings.gauss_pot_length; i++) {
		f_TotalPower += fp_PoT[i * ul_FftLength];
	}
	float f_MeanPower = f_TotalPower / cudaAcc_GaussFit_settings.gauss_pot_length;

	// Normalize power-of-time
	float sum = 0.0f;
	float f_NormMaxPower = 0.0f;
	for (int i = 0; i < cudaAcc_GaussFit_settings.gauss_pot_length; i++) {
		float PoT = fp_PoT[i * ul_FftLength] / f_MeanPower;
		f_NormMaxPower = max(f_NormMaxPower, PoT);
		sum += PoT;
		fp_PoT[i * ul_FftLength] = PoT;
		fp_PoTPrefixSum[i * ul_FftLength] = sum;
	}
	cudaAcc_GaussFit_settings.dev_NormMaxPower[ul_PoT] = f_NormMaxPower;
}

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

__device__ float cudaAcc_calc_GaussFit_score(float chisqr, float null_chisqr, float gauss_pot_length) { // <- gauss_pot_length constant across whole package
	float gauss_bins = gauss_pot_length;
	float gauss_dof = (gauss_bins-2.0f) * 0.5f;
	float null_dof = (gauss_bins-1.0f) * 0.5f;
	gauss_bins *= 0.5f;
	return  cudaAcc_GaussFit_settings.score_offset +
		cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f))
		//-cudaAcc_lcgf(gauss_dof,cudaAcc_GaussFit_settings.gauss_chi_sq_thresh*gauss_bins) // <- always the same result
		-cudaAcc_lcgf(null_dof,max(null_chisqr*gauss_bins,null_dof+1.0f));
		//+cudaAcc_lcgf(null_dof,cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh*gauss_bins); // <- always the same result	
}

// assuming that chisqr and null_chisqr are less than CUDAACC_LCGF_MAX_VALUE + 1. if not the parameters are clamped to CUDAACC_LCGF_MAX_VALUE
__device__ float cudaAcc_calc_GaussFit_score_cached(float chisqr, float null_chisqr) { // <- gauss_pot_length constant across whole package
	float chisqr_cache = (chisqr - 1.0f) / CUDAACC_LCGF_MAX_VALUE; // texture addresMode is cudaAddressModeClamp so behaviour for x < 0.0 and >= 1.0 are well defined
	float null_chisqr_cache = (null_chisqr - 1.0f) / CUDAACC_LCGF_MAX_VALUE; // texture addresMode is cudaAddressModeClamp so behaviour for x < 0.0 and >= 1.0 are well defined
	return cudaAcc_GaussFit_settings.score_offset 
		// cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f)) +cudaAcc_lcgf(null_dof,cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh*gauss_bins)
		+ tex1D(dev_gauss_dof_lcgf_cache_TEX, chisqr_cache)		//cudaAcc_lcgf(gauss_dof,max(chisqr*gauss_bins,gauss_dof+1.0f))
		+ tex1D(dev_null_dof_lcgf_cache_TEX, null_chisqr_cache)	// -cudaAcc_lcgf(null_dof,max(null_chisqr*gauss_bins,null_dof+1.0f))
		; // using textures linear filtering (interpolation)
}

__global__ void GaussFit_kernel(int ul_FftLength, float best_gauss_score, result_flag* flags, bool noscore) {
	int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;    
	int ul_TOffset = blockIdx.y * blockDim.y + threadIdx.y + cudaAcc_GaussFit_settings.GaussTOffsetStart;
	if (ul_PoT >= ul_FftLength) return;
	if (ul_TOffset >= cudaAcc_GaussFit_settings.GaussTOffsetStop) return;

	float* fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
	float f_null_hyp;

	int iSigma = cudaAcc_GaussFit_settings.iSigma;

	float f_TotalPower = cudaAcc_GaussFit_settings.gauss_pot_length,
		f_TrueMean,
		f_ChiSq,
		f_PeakPower;

	// slide dynamic gaussian across the Power Of Time array

	// TrueMean is the mean power of the data set minus all power
	// out to 2 sigma from our current TOffset.	
	f_TrueMean = cudaAcc_GetTrueMean2(
		&cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT],
		cudaAcc_GaussFit_settings.gauss_pot_length,
		f_TotalPower,
		ul_TOffset,
		2 * iSigma,
		ul_FftLength
		);

	f_PeakPower = cudaAcc_GetPeak(
		fp_PoT,
		ul_TOffset,
		iSigma,
		f_TrueMean,
		cudaAcc_GaussFit_settings.PeakScaleFactor,
		ul_FftLength
		);

	// worth looking at ?
	if (f_PeakPower / f_TrueMean < cudaAcc_GaussFit_settings.GaussPeakPowerThresh3) {
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		return;
	}

	// look at it - try to fit  
	f_ChiSq = cudaAcc_GetChiSq(
		fp_PoT,
		ul_FftLength,
		cudaAcc_GaussFit_settings.gauss_pot_length,
		ul_TOffset,
		f_PeakPower,
		f_TrueMean,        
		f_null_hyp
		);

  if (noscore)
  {
	if (((f_ChiSq <=  cudaAcc_GaussFit_settings.GaussChiSqThresh) && (f_null_hyp >= cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh))
 		) {
		flags->has_results = 1;
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(f_TrueMean, f_PeakPower, f_ChiSq, f_null_hyp);
	} else {
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}    
  } else {
	float score = cudaAcc_calc_GaussFit_score_cached(f_ChiSq, f_null_hyp);
	if (((f_ChiSq <=  cudaAcc_GaussFit_settings.gauss_chi_sq_thresh) && (score > best_gauss_score)) 
		|| ((f_ChiSq <=  cudaAcc_GaussFit_settings.GaussChiSqThresh) && (f_null_hyp >= cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh))
		) {
		flags->has_results = 1;
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(f_TrueMean, f_PeakPower, f_ChiSq, f_null_hyp);
	} else {
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}    
  }
} // End of gaussfit()

/*__global__ void GaussFit_kernel_noscore(int ul_FftLength, result_flag* flags) {
	int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;    
	int ul_TOffset = blockIdx.y * blockDim.y + threadIdx.y + cudaAcc_GaussFit_settings.GaussTOffsetStart;
	if (ul_PoT >= ul_FftLength) return;
	if (ul_TOffset >= cudaAcc_GaussFit_settings.GaussTOffsetStop) return;

	float* fp_PoT = &cudaAcc_GaussFit_settings.dev_PoT[ul_PoT];
	float f_null_hyp;

	int iSigma = cudaAcc_GaussFit_settings.iSigma;

	float f_TotalPower = cudaAcc_GaussFit_settings.gauss_pot_length,
		f_TrueMean,
		f_ChiSq,
		f_PeakPower;

	// slide dynamic gaussian across the Power Of Time array

	// TrueMean is the mean power of the data set minus all power
	// out to 2 sigma from our current TOffset.	
	f_TrueMean = cudaAcc_GetTrueMean2(
		&cudaAcc_GaussFit_settings.dev_PoTPrefixSum[ul_PoT],
		cudaAcc_GaussFit_settings.gauss_pot_length,
		f_TotalPower,
		ul_TOffset,
		2 * iSigma,
		ul_FftLength
		);

	f_PeakPower = cudaAcc_GetPeak(
		fp_PoT,
		ul_TOffset,
		iSigma,
		f_TrueMean,
		cudaAcc_GaussFit_settings.PeakScaleFactor,
		ul_FftLength
		);

	// worth looking at ?
	if (f_PeakPower / f_TrueMean < cudaAcc_GaussFit_settings.GaussPeakPowerThresh3) {
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		return;
	}

	// look at it - try to fit  
	f_ChiSq = cudaAcc_GetChiSq(
		fp_PoT,
		ul_FftLength,
		cudaAcc_GaussFit_settings.gauss_pot_length,
		ul_TOffset,
		f_PeakPower,
		f_TrueMean,        
		f_null_hyp
		);


	if ((f_ChiSq <=  cudaAcc_GaussFit_settings.gauss_chi_sq_thresh) 
		|| ((f_ChiSq <=  cudaAcc_GaussFit_settings.GaussChiSqThresh) && (f_null_hyp >= cudaAcc_GaussFit_settings.gauss_null_chi_sq_thresh))
		) {
		flags->has_results = 1;
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(f_TrueMean, f_PeakPower, f_ChiSq, f_null_hyp);
	} else {
		cudaAcc_GaussFit_settings.dev_GaussFitResults[ul_TOffset * ul_FftLength + ul_PoT] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}    
} // End of gaussfit_noscore()
*/

int cudaAcc_initializeGaussfit(const PoTInfo_t& PoTInfo, int gauss_pot_length, unsigned int nsamples, double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh) {

    cudaError_t cu_err;

	settings.iSigma = static_cast<int>(floor(PoTInfo.GaussSigma+0.5));
	settings.GaussSigmaSq = (float) PoTInfo.GaussSigmaSq;
	settings.GaussPowerThresh = (float) PoTInfo.GaussPowerThresh;
	settings.GaussPeakPowerThresh3 = (float) PoTInfo.GaussPeakPowerThresh / 3.0f;
	settings.GaussChiSqThresh = (float) PoTInfo.GaussChiSqThresh;
	settings.gauss_null_chi_sq_thresh = (float) gauss_null_chi_sq_thresh;
	settings.gauss_chi_sq_thresh = (float) gauss_chi_sq_thresh;
	settings.GaussTOffsetStart = PoTInfo.GaussTOffsetStart;
	settings.GaussTOffsetStop = PoTInfo.GaussTOffsetStop;
	settings.gauss_pot_length = gauss_pot_length;
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
	for (int i = 0; i < PoTInfo.GaussTOffsetStop; i++) {
		settings.f_weight[i] = static_cast<float>(EXP(i, 0, PoTInfo.GaussSigmaSq));
	}

	// creating cache of lcgf for score calculation
	double gauss_bins=gauss_pot_length;
	double gauss_dof=gauss_bins-2;
	double null_dof=gauss_bins-1;	
	settings.score_offset =	(float)-lcgf(0.5*gauss_dof,gauss_chi_sq_thresh*0.5*gauss_bins) + lcgf(0.5*null_dof,gauss_null_chi_sq_thresh*0.5*gauss_bins);
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
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

void cudaAcc_free_Gaussfit() {
	if (dev_gauss_dof_lcgf_cache) cudaFreeArray(dev_gauss_dof_lcgf_cache);
	if (dev_null_dof_lcgf_cache) cudaFreeArray(dev_null_dof_lcgf_cache);
}

int cudaAcc_Gaussfit(int ul_FftLength, double best_gauss_score, bool noscore) {
	if (!cudaAcc_initialized()) return -1;

	dim3 block(64, 1, 1);
	dim3 grid((ul_FftLength + block.x - 1) / block.x, 1, 1); 
	int ul_NumSpectra    = cudaAcc_NumDataPoints / ul_FftLength;

	if (ul_NumSpectra == settings.gauss_pot_length) {
		CUDA_ACC_SAFE_CALL( (cudaMemcpy(dev_PoT, dev_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_PowerSpectrum), cudaMemcpyDeviceToDevice)),true);        
	} else if (ul_NumSpectra > settings.gauss_pot_length) {
		dim3 blockPOT(64, 1, 1);
		dim3 gridPOT((ul_FftLength + blockPOT.x - 1) / blockPOT.x, settings.gauss_pot_length, 1);         
		GetFixedPoT_kernel<<<gridPOT, blockPOT>>>(ul_FftLength);
		CUDA_ACC_SAFE_CALL_NO_SYNC("GetFixedPoT_kernel");		
	} else {
		SETIERROR(UNSUPPORTED_FUNCTION, "cudaAcc_Gaussfit doesn't support if (ul_NumSpectra < settings.gauss_pot_length) in GetFixedPoT");
	}

	NormalizePoT_kernel<<<grid, block>>>(ul_FftLength);
//	CUDA_ACC_SAFE_CALL_NO_SYNC("NormalizePoT_kernel");

	CUDA_ACC_SAFE_CALL((cudaMemset(dev_flag, 0, sizeof(*dev_flag))),true);		
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	dim3 block2(32, 4, 1);
	dim3 grid2((ul_FftLength + block2.x - 1) / block2.x, (settings.GaussTOffsetStop - settings.GaussTOffsetStart + block2.y) / block2.y, 1);
	//if(noscore) {
	//	GaussFit_kernel_noscore<<<grid2, block2>>>(ul_FftLength, dev_flag);
//		CUDA_ACC_SAFE_CALL_LOW_SYNC("GaussFit_kernel_noscore");	
	//} else {
	GaussFit_kernel<<<grid2, block2>>>(ul_FftLength, (float) best_gauss_score, dev_flag, noscore);
//	CUDA_ACC_SAFE_CALL_LOW_SYNC("GaussFit_kernel");	
	//}

	result_flag flags;	
    CUDA_ACC_SAFE_CALL((CUDASYNC),true);
	CUDA_ACC_SAFE_CALL( (cudaMemcpy(&flags, dev_flag, sizeof(*dev_flag), cudaMemcpyDeviceToHost)),true);
    CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	if (flags.has_results > 0) {
		// Download all the results
		CUDA_ACC_SAFE_CALL( (cudaMemcpy(GaussFitResults,
			dev_GaussFitResults, 
			cudaAcc_NumDataPoints * sizeof(*dev_GaussFitResults),
			cudaMemcpyDeviceToHost)),true);  // TODO: Download a little bit less data
		CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		CUDA_ACC_SAFE_CALL((cudaMemcpy(tmp_PoT, dev_NormMaxPower, ul_FftLength * sizeof(*dev_NormMaxPower), cudaMemcpyDeviceToHost)),true);
        CUDA_ACC_SAFE_CALL((CUDASYNC),true);

		// Preparing data for cudaAcc_getPoT
		cudaAcc_transposeGPU(dev_t_PowerSpectrum, dev_PoT, ul_FftLength, settings.gauss_pot_length);
        CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		CUDA_ACC_SAFE_CALL((cudaMemcpy(best_PoT, dev_t_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrum), cudaMemcpyDeviceToHost)),true); // TODO: Download a little bit less data
        CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		int result_count = 0;
		for(int TOffset = settings.GaussTOffsetStart; TOffset < settings.GaussTOffsetStop; TOffset++) {
			for(int ThisPoT=1; ThisPoT < ul_FftLength; ThisPoT++) {			
				int index = (TOffset * ul_FftLength + ThisPoT);
				float4 res1 = GaussFitResults[index];
				if (res1.x > 0) {						
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
						&best_PoT[ThisPoT * settings.gauss_pot_length]
					);						
					result_count++;
				}
			}					
		}
	}

	return 0;
}
#endif //USE_CUDA
