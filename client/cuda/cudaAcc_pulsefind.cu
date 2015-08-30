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
float* dev_tmp_pot;
float* dev_best_pot;
float* dev_report_pot;

result_find_pulse_flag* dev_find_pulse_flag;

typedef struct { 
	int NumDataPoints;
	// find_triplets
	float *power_ft;
	float4* results_ft;
	result_flag* result_flags_ft; 	

	// find_pulse
	float* PulsePot_fp;		// Input data
	float* PulsePot8_fp;	// Input data moved 8 bytes forward for coleased reads
	float* tmp_pot_fp;		// Temporary array
	float* best_pot_fp;		// Copy folded pots with best score
	float* report_pot_fp;	// Copy folded pots for reporting
	float4* results_fp;		// Additional data for reporting
//	float* avg;				// averages cache
	result_find_pulse_flag* result_flags_fp;

	const float* t_funct_cache_fp; // cached results of cudaAcc_t_funct								   								   
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
template <int blockx>
__global__ void find_triplets_kernel(int ul_FftLength, int len_power, volatile float triplet_thresh, int AdvanceBy) {
	int ul_PoT = blockIdx.x * blockx + threadIdx.x;
	if ((ul_PoT < 1) || (ul_PoT >= ul_FftLength)) return; // Original find_triplets, omits first element

	int PoTLen = cudaAcc_PulseFind_settings.NumDataPoints / ul_FftLength;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int y2 = y * 2;
	int TOffset = y * AdvanceBy;

	float thresh = triplet_thresh;

	if(TOffset >= PoTLen - len_power) {
		TOffset = PoTLen - len_power;
	}
	volatile float* power = cudaAcc_PulseFind_settings.power_ft;
	float4* results = cudaAcc_PulseFind_settings.results_ft;
	volatile result_flag* rflags = cudaAcc_PulseFind_settings.result_flags_ft;
	// Clear the result array
	results[AT_XY(ul_PoT, y2, ul_FftLength)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	results[AT_XY(ul_PoT, y2 + 1, ul_FftLength)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 

	// MAXTRIPLETS_ABOVE_THESHOLD must be odd to prevent bank conflicts!
	__shared__ int binsAboveThreshold[blockx][MAX_TRIPLETS_ABOVE_THRESHOLD];
	int i,n,numBinsAboveThreshold=0,p,q;
	float midpoint,mean_power=0,peak_power,period;

	/* Get all the bins that are above the threshold, and find the power array mean value */
	float4 partials = {0.0f,0.0f,0.0f,0.0f};
	for( i=0;i<len_power-3;i+=4 ) {
		partials.x += power[AT(i+0)];
		partials.y += power[AT(i+1)];
		partials.z += power[AT(i+2)];
		partials.w += power[AT(i+3)];
	}
	while ( i <len_power)
	{
		partials.w += power[AT(i)];
		i++;
	}
	mean_power = partials.x+partials.y+partials.z+partials.w;
	mean_power /= (float)len_power;
	thresh*=mean_power;

	for( i=0;i<len_power;i++ ) {
		if( power[AT(i)] >= thresh ) {
			if (numBinsAboveThreshold == MAX_TRIPLETS_ABOVE_THRESHOLD) {
				rflags->error = CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD; // Reporting Error
				return;
			}
		binsAboveThreshold[threadIdx.x][numBinsAboveThreshold] = i;
		numBinsAboveThreshold++;
		}
	}

	/* Check each bin combination for a triplet */
	if (numBinsAboveThreshold>2) { /* THIS CONDITION IS TRUE ONLY once every 300 KERNEL LUNCH*/
		int already_reported_flag = 0;
		// TODO: Use Texture for reads from power[], Random Reads
		for( i=0;i<numBinsAboveThreshold-1;i++ ) {
			for( n=i+2;n<numBinsAboveThreshold;n++ ) {
				midpoint = (binsAboveThreshold[threadIdx.x][i]+binsAboveThreshold[threadIdx.x][n])/2.0f;
				period = (float)fabs((binsAboveThreshold[threadIdx.x][i]-binsAboveThreshold[threadIdx.x][n])/2.0f);

				int midfi = __float2int_rd(midpoint);
				float midff = __int2float_rn(midfi);

				/* Get the peak power of this triplet */
				peak_power = power[AT(binsAboveThreshold[threadIdx.x][i])];
				if( power[AT(binsAboveThreshold[threadIdx.x][n])] > peak_power )
					peak_power = power[AT(binsAboveThreshold[threadIdx.x][n])];

				p = binsAboveThreshold[threadIdx.x][i];
				while( (power[AT(p)] >= thresh) && (p <= midfi) ) {    /* Check if there is a pulse "off" in between init and midpoint */
					p++;
				}

				q = midfi+1;
				while( (power[AT(q)] >= thresh) && (q <= binsAboveThreshold[threadIdx.x][n])) {    /* Check if there is a pulse "off" in between midpoint and end */
					q++;
				}

				if( p >= midfi || q >= binsAboveThreshold[threadIdx.x][n]) {
					/* if this pulse doesn't have an "off" between all the three spikes, it's dropped */
				} else if( (midpoint - midff) > 0.1f ) {    /* if it's spread among two bins */
					if( power[AT(midfi)] >= thresh ) {
						if (already_reported_flag >= 2)	{
							rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
							return;
						}

						if( power[AT(midfi)] > peak_power )
							peak_power = power[AT(midfi)];

						cudaAcc_PulseFind_settings.results_ft[AT_XY(ul_PoT, y2 + already_reported_flag, ul_FftLength)] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
						rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
						already_reported_flag++;
					}

					if( power[AT(midfi+1)] >= thresh ) {
						if (already_reported_flag >= 2)	{
							rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
							return;
						}

						if( power[AT(midfi+1)] > peak_power )
							peak_power = power[AT(midfi+1)];

						cudaAcc_PulseFind_settings.results_ft[AT_XY(ul_PoT, y2 + already_reported_flag, ul_FftLength)] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
						rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
						already_reported_flag++;
					}

				} else {            /* otherwise just check the single midpoint bin */
					if( power[AT(midfi)] >= thresh ) {
						if (already_reported_flag >= 2)	{
							rflags->error = CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT; // Reporting Error, more than one result per PoT, redo the calculations on CPU
							return;
						}

						if( power[AT(midfi)] > peak_power )
							peak_power = power[AT(midfi)];

						results[AT_XY(ul_PoT, y2 + already_reported_flag, ul_FftLength)] = make_float4(peak_power/mean_power, mean_power, period, midpoint);
						rflags->has_results = 1; // Mark for download <- VERY RARE SITUATION
						already_reported_flag++;
					}
				}
			}
		}
	}
}

extern int find_triplets( const float *power, int len_power, float triplet_thresh, int time_bin, int freq_bin );

int cudaAcc_find_triplets(int PulsePoTLen, float triplet_thresh, int AdvanceBy, int FftLength) {
	int PoTLen = cudaAcc_NumDataPoints / FftLength;

	CUDA_ACC_SAFE_CALL( (cudaMemset(dev_flag, 0, sizeof(*dev_flag))),true);
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	// Occupancy Calculator: cc1.x: 64, cc2.x: 128 & Shared instead of L1
	dim3 block(64,1,1);
	dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
	if (gCudaDevProps.major >= 2) {
		// Fermi, launch with larger gridsize, more shared mem
		//cudaFuncSetCacheConfig(find_triplets_kernel<128>, cudaFuncCachePreferShared);
		dim3 block(128, 1, 1);
		dim3 grid((FftLength + block.x - 1) / block.x, (PoTLen + AdvanceBy - 1) / AdvanceBy, 1);
		find_triplets_kernel<128><<<grid, block>>>(FftLength, PulsePoTLen, triplet_thresh, AdvanceBy);
	} else {
		// Not Fermi, launch with cc1.x gridsize
		find_triplets_kernel<64><<<grid, block>>>(FftLength, PulsePoTLen, triplet_thresh, AdvanceBy);
	}

	CUDA_ACC_SAFE_CALL_LOW_SYNC("find_triplets_kernel");
    cudaAcc_peekLastError_alliscool();

	result_flag flags;	

	flags.error = 0;
	flags.has_results = 0;

	CUDA_ACC_SAFE_CALL((CUDASYNC),true);
	CUDA_ACC_SAFE_CALL( (cudaMemcpy(&flags, dev_flag, sizeof(*dev_flag), cudaMemcpyDeviceToHost)),true);
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);
//	cudaAcc_peekLastError_alliscool();
	if (flags.error) {
		//fprintf(stderr,"Find triplets return flags indicate an error (value: %x)\n",flags.error);
		//fprintf(stderr,"Last Cuda error code indicates: ");
		//cudaAcc_peekLastError();
		//if (flags.error & CUDA_ERROR_TRIPLETS_ABOVE_THRESHOLD) {
			//SETIERROR(UNSUPPORTED_FUNCTION, "cudaAcc_find_triplets doesn't support more than MAX_TRIPLETS_ABOVE_THRESHOLD numBinsAboveThreshold in find_triplets_kernel");		
		//}
		//if (flags.error & CUDA_ERROR_TRIPLETS_DUPLICATE_RESULT) {
			//SETIERROR(UNSUPPORTED_FUNCTION, "cudaAcc_find_triplets erroneously found a triplet thrice in find_triplets_kernel");
		//}
		fprintf(stderr,"Find triplets Cuda kernel encountered too many triplets, or bins above threshold, reprocessing this PoT on CPU...\n");
		cudaAcc_transposeGPU(dev_t_PowerSpectrum, dev_PowerSpectrum, FftLength, cudaAcc_NumDataPoints / FftLength);
        CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		CUDA_ACC_SAFE_CALL((cudaMemcpy(tmp_PoT, dev_t_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrum), cudaMemcpyDeviceToHost)),true);						
        CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		// loop through frequencies
		for(int ThisPoT=1; ThisPoT < FftLength; ThisPoT++) 
		{
			// loop through time for each frequency.  PulsePoTNum is
			// used only for progress calculation.
			int TOffset = 0;
			int PulsePoTNum = 1;
			bool TOffsetOK = true;
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

					int retval = find_triplets(&tmp_PoT[ThisPoT * PoTLen + TOffset], //PulsePoT,
						PulsePoTLen,
						(float)PoTInfo.TripletThresh,
						TOffset,
						ThisPoT);
					if (retval)	SETIERROR(retval,"from find_triplets()");
			}
		}
	} 
	else if (flags.has_results) {
		//TODO: Check if faster is to scan the results and download only the results, not everything
		CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		CUDA_ACC_SAFE_CALL((cudaMemcpy(TripletResults, dev_TripletResults, 2 * grid.x * block.x * grid.y * block.y * sizeof(*dev_TripletResults), cudaMemcpyDeviceToHost)),true);
		cudaAcc_transposeGPU(dev_t_PowerSpectrum, dev_PowerSpectrum, FftLength, cudaAcc_NumDataPoints / FftLength);
		CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		CUDA_ACC_SAFE_CALL((cudaMemcpy(tmp_PoT, dev_t_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_t_PowerSpectrum), cudaMemcpyDeviceToHost)),true);						
		CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		// Iterating trough results
		for(int ThisPoT=1; ThisPoT < FftLength; ThisPoT++) {
			for(int TOffset = 0, TOffsetOK = true, PulsePoTNum = 0; TOffsetOK; PulsePoTNum++, TOffset += AdvanceBy) {
				if(TOffset + PulsePoTLen >= PoTLen) {
					TOffsetOK = false;
					TOffset = PoTLen - PulsePoTLen;
				}
				int index = ((PulsePoTNum * 2) * FftLength + ThisPoT);
				int index2 = ((PulsePoTNum * 2 + 1)* FftLength + ThisPoT);
				if (TripletResults[index].x > 0) {
					float4 res = TripletResults[index];
					cudaAcc_ReportTripletEvent( res.x, res.y, res.z, res.w, TOffset,
						ThisPoT, PulsePoTLen, &tmp_PoT[ThisPoT * PoTLen + TOffset], 1 );
					//ReportTripletEvent( peak_power/mean_power, mean_power, period, midpoint,time_bin, freq_bin, len_power, power, 1 )
				}
				if (TripletResults[index2].x > 0) {
					float4 res = TripletResults[index2];
					cudaAcc_ReportTripletEvent( res.x, res.y, res.z, res.w, TOffset,
						ThisPoT, PulsePoTLen, &tmp_PoT[ThisPoT * PoTLen + TOffset], 1 );
					//ReportTripletEvent( peak_power/mean_power, mean_power, period, midpoint,time_bin, freq_bin, len_power, power, 1 )
				}
			}
		}
	}
	return 0;
}

__device__ float cudaAcc_t_funct(int di, int num_adds, int j, int PulseMax, const float* t_funct_cache) {
	return t_funct_cache[j * PulseMax * CUDA_ACC_FOLDS_COUNT + (num_adds - CUDA_ACC_FOLDS_START) * PulseMax + di];
}

__device__ float cudaAcc_sumtop2(const float *tab, float* dest, int di, int fft_len, int tmp0) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_len;
	tmax = 0.0f;

	for (i = 0; i < di; i++) {
		int idx = i * fft_len;
		sum  = one[idx];
		sum += two[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

__device__ float cudaAcc_sumtop3(const float *tab, float* dest, int di, int fft_len, int tmp0, int tmp1) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_len;
	const float *three = tab + tmp1 * fft_len;
	tmax = 0.0f;

	for (i = 0; i < di; i++) {
		int idx = i * fft_len;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

__device__ float cudaAcc_sumtop4(const float *tab, float* dest, int di, int fft_len, int tmp0, int tmp1, int tmp2) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_len;
	const float *three = tab + tmp1 * fft_len;
	const float *four = tab + tmp2 * fft_len;
	tmax = 0.0f;

	for (i = 0; i < di; i++) {
		int idx = i * fft_len;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];
		sum += four[idx];		
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

__device__ float cudaAcc_sumtop5(const float *tab, float* dest, int di, int fft_len, int tmp0, int tmp1, int tmp2, int tmp3) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_len;
	const float *three = tab + tmp1 * fft_len;
	const float *four = tab + tmp2 * fft_len;
	const float *five = tab + tmp3 * fft_len;
	tmax = 0.0f;

	for (i = 0; i < di; i++) {
		int idx = i * fft_len;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];
		sum += four[idx];
		sum += five[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}	
	return tmax;
}

__device__ void cudaAcc_copy(const float* from, float* to, int step, int count) {
	for (int i = 0; i < count; ++i) {
		to[i * step] = from[i * step];
	}
}

template <bool load_state, int num_adds>
__global__ void find_pulse_kernel(float best_pulse_score, int PulsePotLen, int AdvanceBy, int fft_len, int ndivs) 
{
	int ul_PoT = blockIdx.x * blockDim.x + threadIdx.x;
//	int tid = threadIdx.x+blockIdx.x*blockDim.x+blockIdx.y*blockDim.x*gridDim.x;
	if ((ul_PoT < 1) || (ul_PoT >= fft_len)) return; // Original find_pulse, omits first element
	const int PoTLen = cudaAcc_PulseFind_settings.NumDataPoints / fft_len;
	int y = blockIdx.y * blockDim.y;// + threadIdx.y;
	int TOffset1 = y * AdvanceBy;
	int TOffset2 = y * AdvanceBy;
	int y4 = y * 4;

	if(TOffset1 > PoTLen - PulsePotLen) {		
		TOffset1 = PoTLen - PulsePotLen;
	}

	float* fp_PulsePot	= cudaAcc_PulseFind_settings.PulsePot_fp	+ ul_PoT + TOffset1 * fft_len;
	float* tmp_pot		= cudaAcc_PulseFind_settings.tmp_pot_fp		+ ul_PoT + TOffset2 * fft_len;
	float* best_pot		= cudaAcc_PulseFind_settings.best_pot_fp	+ ul_PoT + TOffset2 * fft_len;
	float* report_pot	= cudaAcc_PulseFind_settings.report_pot_fp	+ ul_PoT + TOffset2 * fft_len;

	int di, maxs = 0;

	float max=0,maxd=0,avg=0,maxp=0, snr=0, fthresh=0;
	float tmp_max, t1;
	int i;

	if (!load_state)
	{
		//  Calculate average power
		float4 partials = {0.0f,0.0f,0.0f,0.0f};
		for (i = 0; i < PulsePotLen-3; i+=4) {
			partials.x += fp_PulsePot[(i+0) * fft_len];
			partials.y += fp_PulsePot[(i+1) * fft_len];
			partials.z += fp_PulsePot[(i+2) * fft_len];
			partials.w += fp_PulsePot[(i+3) * fft_len];
		}
		while (i<PulsePotLen)
		{
			partials.w += fp_PulsePot[i * fft_len];
			i++;
		}
		avg = partials.x+partials.y+partials.z+partials.w;
		avg /= PulsePotLen;
//		cudaAcc_PulseFind_settings.avg[tid] = avg;
	} //else {
	  //  avg = cudaAcc_PulseFind_settings.avg[tid];
//	}

	if (load_state) 
    {		
		best_pulse_score = fmax(best_pulse_score, cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_len)].w);
		float4 tmp_float4;
		tmp_float4 = cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_len)];
		avg = tmp_float4.y;
		max = tmp_float4.x * avg;
		maxp = tmp_float4.z;
		maxd = tmp_float4.w;

		tmp_float4 = cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_len)];
		snr = tmp_float4.y;
		fthresh = tmp_float4.z;
		maxs = tmp_float4.w;
	} 
    else 
    {
        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_len)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_len)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_len)] = make_float4(0.0f, avg, 0.0f, 0.0f);
        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_len)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	
    }

	// save redundant calculations: sqrt is expensive
	float sqrt_num_adds_div_avg = (float)sqrt((float)num_adds) / avg;

	//  Periods from PulsePotLen/3 to PulsePotLen/4, and power of 2 fractions of.
	//   then (len/4 to len/5) and finally (len/5 to len/6)
	//	

	//for(int num_adds = 3; num_adds <= 5; num_adds++) 
    {
		int firstP, lastP;
		switch(num_adds) {
		case 3: lastP = (PulsePotLen*2)/3;  firstP = (PulsePotLen*1)/2; break;
		case 4: lastP = (PulsePotLen*3)/4;  firstP = (PulsePotLen*3)/5; break;
		case 5: lastP = (PulsePotLen*4)/5;  firstP = (PulsePotLen*4)/6; break;
		}

		for (int p = lastP ; p > firstP ; p--) {
			float cur_thresh, dis_thresh;
			int tabofst, mper, perdiv;
			int tmp0, tmp1, tmp2, tmp3;

			tabofst = ndivs*3+2-num_adds;
			mper = p * (12/(num_adds - 1));
			perdiv = num_adds - 1;
			tmp0 = (int)((mper + 6) * RECIP_12);             // round(period)
			tmp1 = (int)((mper * 2 + 6) * RECIP_12);         // round(period*2)
			di = (int)p/perdiv;                      // (int)period			
			//dis_thresh = cudaAcc_t_funct(di, num_adds)*avg;
			dis_thresh = cudaAcc_t_funct(di, num_adds, 0, cudaAcc_PulseFind_settings.PulseMax, cudaAcc_PulseFind_settings.t_funct_cache_fp) * avg;

			switch(num_adds) {
			case 3:
				tmp_max = cudaAcc_sumtop3(fp_PulsePot, tmp_pot, di, fft_len, tmp0, tmp1);
				break;
			case 4:
				tmp2 = (int)((mper * 3 + 6) * RECIP_12);     // round(period*3)
				tmp_max = cudaAcc_sumtop4(fp_PulsePot, tmp_pot, di, fft_len, tmp0, tmp1, tmp2);
				break;
			case 5:
				tmp2 = (int)((mper * 3 + 6) * RECIP_12);     // round(period*3)
				tmp3 = (int)((mper * 4 + 6) * RECIP_12);     // round(period*4)
				tmp_max = cudaAcc_sumtop5(fp_PulsePot, tmp_pot, di, fft_len, tmp0, tmp1, tmp2, tmp3);
				break;
			}

			if (tmp_max>dis_thresh) {
				// unscale for reporting
				tmp_max /= num_adds;
				cur_thresh = (dis_thresh / num_adds - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

				float _snr = (tmp_max-avg)*sqrt_num_adds_div_avg;
				float _thresh = (cur_thresh-avg)*sqrt_num_adds_div_avg;
				if (_snr / _thresh > best_pulse_score) {
					best_pulse_score = _snr / _thresh;
					cudaAcc_copy(tmp_pot, best_pot, fft_len, di);
					cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_len)] = make_float4(
						tmp_max/avg,
						avg,
						((float)p)/(float)perdiv,
						0.0f//TOffset1+PulsePotLen/2
						);
					cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_len)] = make_float4(
						ul_PoT,
						_snr,
						_thresh,
						num_adds
						);
					cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;					
				}
				//          ReportPulseEvent(tmp_max/avg,avg,((float)p)/(float)perdiv*res,
				//TOffset+PulsePotLen/2,FOffset,
				//	(tmp_max-avg)*(float)sqrt((float)num_adds)/avg,
				//	(cur_thresh-avg)*(float)sqrt((float)num_adds)/avg,
				//	PTPln.dest, num_adds, 0);
				if ((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) {
					maxp  = (float)p/(float)perdiv;
					maxd  = t1;
					maxs  = num_adds;
					max = tmp_max;
					snr = _snr;
					fthresh= _thresh;
					cudaAcc_copy(tmp_pot, report_pot, fft_len, di);
					//memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float)); 

					// It happens very rarely so it's better to store the results right away instead of
					// storing them in registers or worse local memory						
					cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_len)] = make_float4(
						max/avg,
						avg,
						maxp,
						maxd//TOffset1+PulsePotLen/2
						);
					cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_len)] = make_float4(
						ul_PoT,
						snr,
						fthresh,
						maxs
						);
					cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
					//ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
					//snr, fthresh, FoldedPOT, maxs, 1);						
				}
			}

			int num_adds_2 = num_adds << 1;

			//	int j = 1;
			for (int j = 1; j < ndivs ; j++) 
			{
				float sqrt_num_adds_2_div_avg = (float)sqrt((float)num_adds_2) / avg;
				perdiv = perdiv << 1;
				tmp0 = di & 1;
				di = di >> 1;
				tmp0 += di;
				tabofst -=3;
				//dis_thresh = cudaAcc_t_funct(di, num_adds_2) * avg;
				dis_thresh = cudaAcc_t_funct(di, num_adds, j, cudaAcc_PulseFind_settings.PulseMax, cudaAcc_PulseFind_settings.t_funct_cache_fp) * avg;

				tmp_max = cudaAcc_sumtop2(tmp_pot, tmp_pot, di, fft_len, tmp0);

				if (tmp_max>dis_thresh) {
					// unscale for reporting
					tmp_max /= num_adds_2;
					cur_thresh = (dis_thresh / num_adds_2 - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

					float _snr = (tmp_max-avg)*sqrt_num_adds_2_div_avg;
					float _thresh = (cur_thresh-avg)*sqrt_num_adds_2_div_avg;
					if (_snr / _thresh > best_pulse_score) {
						best_pulse_score = _snr / _thresh;
						cudaAcc_copy(tmp_pot, best_pot, fft_len, di);
						cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_len)] = make_float4(
							tmp_max/avg,
							avg,
							((float)p)/(float)perdiv,
							0.0f//TOffset1+PulsePotLen/2
							);
						cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_len)] = make_float4(
							ul_PoT,
							_snr,
							_thresh,
							num_adds_2
							);
						cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;
					}
					//ReportPulseEvent(tmp_max/avg,avg,((float)p)/(float)perdiv*res,
					//	TOffset+PulsePotLen/2,FOffset,
					//	(tmp_max-avg)*(float)sqrt((float)num_adds_2)/avg,
					//	(cur_thresh-avg)*(float)sqrt((float)num_adds_2)/avg,
					//	PTPln.dest, num_adds_2, 0);

					if ((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) {
						maxp = (float)p/(float)perdiv;
						maxd = t1;
						maxs = num_adds_2;
						max  = tmp_max;
						snr  = _snr;
						fthresh = _thresh;
						cudaAcc_copy(tmp_pot, report_pot, fft_len, di);
						//memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float));

						// It happens very rarely so it's better to store the results right away instead of
						// storing them in registers or worse local memory						
						cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_len)] = make_float4(
							max/avg,
							avg,
							maxp,
							maxd//TOffset1+PulsePotLen/2
							);
						cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_len)] = make_float4(
							ul_PoT,
							snr,
							fthresh,
							maxs
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
int cudaAcc_find_pulse_original(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) {	
	const int PoTLen = cudaAcc_NumDataPoints / FftLength;
	
	dim3 block(64, 1, 1); // Pre-Fermi default
	if (gCudaDevProps.regsPerBlock >= 64 * 1024) block.x = 256; // Kepler GPU tweak;
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
	find_pulse_kernel<false,3><<<grid, block, 0 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength, ndivs);	
	find_pulse_kernel<true,4><<<grid, block, 0 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength, ndivs);	
	find_pulse_kernel<true,5><<<grid, block, 0 >>>(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength, ndivs);	
	CUDASYNC;  //cudaThreadSynchronize();
	return 0;
}

#if __CUDA_ARCH__ >= 200	// Only use on Fermi or newer: 32 registers is too low for cc13, and does not have __ddiv,__drcp,__dsqrt

#define ITMAX 1000		// Needs to be a few times the sqrt of the max. input to lcgf

__device__ double cudaAcc_pulse_lcgf(int a, double x) {
	const double EPS= 1.19209e-006f; //007;//std::numeric_limits<double>::epsilon();
	const double FPMIN= 9.86076e-031f; //032;//std::numeric_limits<double>::min()/EPS;
	double an,b,c,d,del,h;
	const double gln = lgamma(__int2double_rn(a));
	
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
	const float frac_err = 1e-6f;
	
	xh= __dadd_rn(a,1.5f);
	xl= __dadd_rn(a, __dmul_rn(__dmul_rn(-2.0f,y),__dsqrt_rn(a)));
	const float fl=__fadd_rn(cudaAcc_pulse_lcgf(a,xl),-y);
	const float fh=__fadd_rn(cudaAcc_pulse_lcgf(a,xh),-y);
	
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
__global__ void __launch_bounds__(480,1) cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, const float pulse_display_thresh) {
	di = di + (PulseMax * threadIdx.y / 32);
	
	const int j = threadIdx.x;
	const int num_adds = blockIdx.y;
	
	int l = 1<<j;
	int n = (num_adds+CUDA_ACC_FOLDS_START)*l;
	int idx = PulseMax * (j * CUDA_ACC_FOLDS_COUNT + num_adds) + di;
	float inv_lcgf = cudaAcc_invert_lcgf(__dadd_rn(-PulseThresh,-log(__int2float_rn(di))), __int2float_rn(n));
	dev_t_funct_cache[idx] = (__dadd_rn(__dmul_rn(__dadd_rn(inv_lcgf, -n),pulse_display_thresh), n)); //-0.05f;
}*/
//float version
__global__ void __launch_bounds__(480,1) cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, const float pulse_display_thresh) {
	di = di + (PulseMax * threadIdx.y / 32);
	const int j = threadIdx.x;
	const int num_adds = blockIdx.y;
	int l = 1<<j;
	int n = (num_adds+CUDA_ACC_FOLDS_START)*l;
	int idx = PulseMax * (j * CUDA_ACC_FOLDS_COUNT + num_adds) + di;
	float inv_lcgf = cudaAcc_invert_lcgf(__fadd_rn(-PulseThresh,-logf(__int2float_rn(di))), __int2float_rn(n));
	dev_t_funct_cache[idx] = (__fadd_rn(__fmul_rn(__fadd_rn(inv_lcgf, -n),pulse_display_thresh), n)); //-0.05f;
}
#else // __CUDA_ARCH__ >= 200

// dummy version for <200 cards
__global__ void cudaAcc_dev_t_funct(float PulseThresh, int PulseMax, int di, float *dev_t_funct_cache, const float pulse_display_thresh) {
	// do nothing
}
#endif

float cudaAcc_host_t_funct(double pulse_display_thresh, double PulseThresh, int m, int n) {	
	return (invert_lcgf((float)(-PulseThresh - log((float)m)),
		(float)n, (float)1e-4) - n) * (float)pulse_display_thresh + n;
}

int cudaAcc_initialize_pulse_find(double pulse_display_thresh, double PulseThresh, int PulseMax) {
    cudaError_t cu_err;
	cu_err = cudaMalloc((void**) &dev_find_pulse_flag, sizeof(*dev_find_pulse_flag));
    if( cudaSuccess != cu_err) 
    {
        CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_find_pulse_flag");
        return 1;
    } else { CUDAMEMPRINT(dev_find_pulse_flag,"cudaMalloc((void**) &dev_find_pulse_flag",1,sizeof(*dev_find_pulse_flag)); };

	int maxdivs = 1;
	for (int i = 32; i <= PulseMax; maxdivs++, i *= 2);

	cu_err = cudaMalloc((void**) &dev_t_funct_cache, (1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax) * sizeof(float));
    if( cudaSuccess != cu_err) 
    {
        CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_t_funct_cache");
        return 1;
    } else { CUDAMEMPRINT(dev_t_funct_cache,"cudaMalloc((void**) &dev_t_funct_cache",(1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax),sizeof(float)); };

//	cu_err = cudaMalloc((void**) &dev_avg, cudaAcc_NumDataPoints*sizeof(float));
//    if( cudaSuccess != cu_err) 
//    {
//        CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_avg");
//        return 1;
//    } else { CUDAMEMPRINT(dev_t_funct_cache,"cudaMalloc((void**) &dev_avg",cudaAcc_NumDataPoints,sizeof(float)); };

    cudaMemset(dev_t_funct_cache, 0, (1 + maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax) * sizeof(float));
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	cu_err = cudaErrorInvalidValue; //default is failure so host will generate it if kernels fail
    if ( gCudaDevProps.major >= 2) 
	{	       
	    dim3 block(maxdivs,32,1);
	    dim3 grid(1,CUDA_ACC_FOLDS_COUNT,1);
	    for (int di= 1; di< (PulseMax/32)+1; di++) {
		    cudaAcc_dev_t_funct<<<grid,block>>>((float)PulseThresh, PulseMax, di, dev_t_funct_cache, (float)pulse_display_thresh);
		    cu_err = CUDASYNC;
		    if (cudaSuccess != cu_err) {
			    CUDA_ACC_SAFE_CALL_NO_SYNC("cudaAcc_dev_t_funct");
			    break; // No need to keep going if there's a problem, and we can save the last error status
		    }
	    }
	}

    if (cudaSuccess != cu_err || gCudaDevProps.major < 2) { // did not process properly on GPU, revert to host code, or we're using a pre-Fermi card
	float* t_funct_cache = (float*) malloc(maxdivs * CUDA_ACC_FOLDS_COUNT  * PulseMax * sizeof(float));
	for (int j = 0, l = 1; j < maxdivs; ++j, l *= 2) {
		for (int num_adds = 0; num_adds < CUDA_ACC_FOLDS_COUNT; ++num_adds) // cache for 2, 3 ,4, 5 folds
			for (int di = 1; di < PulseMax; ++di)
			{
				t_funct_cache[j * PulseMax * CUDA_ACC_FOLDS_COUNT + num_adds * PulseMax + di] =
					cudaAcc_host_t_funct(pulse_display_thresh, PulseThresh, di, (num_adds+CUDA_ACC_FOLDS_START)*l);
			}
	}

	CUDA_ACC_SAFE_CALL((cudaMemcpy(dev_t_funct_cache, t_funct_cache, maxdivs * CUDA_ACC_FOLDS_COUNT * PulseMax * sizeof(float), cudaMemcpyHostToDevice)),true);
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	free(t_funct_cache);
    }

	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	cudaAcc_PulseMax = PulseMax;
	cudaAcc_rcfg_dis_thresh = (float) (1.0 / pulse_display_thresh);

	PulseFind_settings.NumDataPoints = cudaAcc_NumDataPoints;
	// find_triplets
	PulseFind_settings.power_ft = dev_PowerSpectrum;
	PulseFind_settings.results_ft = dev_TripletResults;
	PulseFind_settings.result_flags_ft = dev_flag;

	// find_pulse
	PulseFind_settings.PulsePot_fp = dev_PowerSpectrum;
	PulseFind_settings.PulsePot8_fp = dev_t_PowerSpectrum + 8;
	PulseFind_settings.tmp_pot_fp = dev_tmp_pot;
	PulseFind_settings.best_pot_fp = dev_best_pot;
	PulseFind_settings.report_pot_fp = dev_report_pot;
	PulseFind_settings.results_fp = dev_PulseResults;
//	PulseFind_settings.avg = dev_avg;

	PulseFind_settings.result_flags_fp = dev_find_pulse_flag;
	PulseFind_settings.t_funct_cache_fp = dev_t_funct_cache;
	PulseFind_settings.rcfg_dis_thresh =  cudaAcc_rcfg_dis_thresh;
	PulseFind_settings.PulseMax = cudaAcc_PulseMax;

	CUDA_ACC_SAFE_CALL((cudaMemcpyToSymbol(cudaAcc_PulseFind_settings, (void*) &PulseFind_settings, sizeof(PulseFind_settings))),true);

    return 0;
}

void cudaAcc_free_pulse_find() {	
	cudaFree(dev_t_funct_cache);
	dev_t_funct_cache = NULL;
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_reduce_sum(float* sdata) {
	const int tid = threadIdx.x;

	if (fft_n * n > 32)
		__syncthreads();

	//Jason: Adding to fit fermi hardware
	if (fft_n * n >= 1024 && fft_n < 1024) {
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}

	if (fft_n * n >= 512 && fft_n < 512) {
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}

	if (fft_n * n >= 256 && fft_n < 256) {
		if (tid < 128) 
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}

	if (fft_n * n >= 128 && fft_n < 128) {
		if (tid < 64) 
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32) {
		volatile float *smem = sdata;
		if (fft_n * n >= 64 && fft_n < 64) smem[tid] += smem[tid + 32];
		if (fft_n * n >= 32 && fft_n < 32) smem[tid] += smem[tid + 16];
		if (fft_n * n >= 16 && fft_n < 16) smem[tid] += smem[tid + 8];
	}

	return sdata[tid & (fft_n - 1)];
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_reduce_max(float* sdata) {
	const int tid = threadIdx.x;		

	if (fft_n * n > 32)
		__syncthreads();

//Jason: Adding to fit fermi hardware
	if (fft_n * n >= 1024 && fft_n < 1024) {
		if (tid < 512) 
			sdata[tid] = max(sdata[tid], sdata[tid + 512]);
		__syncthreads();
	}

	if (fft_n * n >= 512 && fft_n < 512) {
		if (tid < 256) 
			sdata[tid] = max(sdata[tid], sdata[tid + 256]);
		__syncthreads();
	}

	if (fft_n * n >= 256 && fft_n < 256) {
		if (tid < 128) 
			sdata[tid] = max(sdata[tid], sdata[tid + 128]);
		__syncthreads();
	}

	if (fft_n * n >= 128 && fft_n < 128) {
		if (tid < 64) 
			sdata[tid] = max(sdata[tid], sdata[tid + 64]);
		__syncthreads();
	}

	if (tid < 32) {
		volatile float *smem = sdata;
//		if (fft_n * n >= 64 && fft_n < 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
//		if (fft_n * n >= 32 && fft_n < 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
//		if (fft_n * n >= 16 && fft_n < 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
		if (fft_n * n >= 64 && fft_n < 64) smem[tid] = max(smem[tid], smem[tid + 32]);
		if (fft_n * n >= 32 && fft_n < 32) smem[tid] = max(smem[tid], smem[tid + 16]);
		if (fft_n * n >= 16 && fft_n < 16) smem[tid] = max(smem[tid], smem[tid + 8]);
	}

	if (fft_n * n > 32)
		__syncthreads();
	return sdata[tid & (fft_n - 1)];
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop2_2(const float *tab, float* dest, int di, int tmp0) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_n;
	tmax = 0.0f;

	for (i = threadIdx.x / fft_n; i < di; i += n) {
		int idx = i * fft_n;
		sum  = one[idx];
		sum += two[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop3_2(const float *tab, const float *tab8, float* dest, int ul_PoT, int di, int tmp0, int tmp1) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_n;
	const float *three = tab + tmp1 * fft_n;

	if (fft_n == 8) {
		if (/*fft_n == 8 && */((long int) (one - ul_PoT) & 0x3F)) // address not dividable by 16 * sizeof(float)
			one = tab8;
		if (/*fft_n == 8 && */ ((long int) (two - ul_PoT) & 0x3F)) // address not dividable by 16 * sizeof(float)
			two = tab8 + tmp0 * fft_n;
		if (/*fft_n == 8 && */ ((long int) (three - ul_PoT) & 0x3F)) // address not dividable by 16 * sizeof(float)
			three = tab8 + tmp1 * fft_n;
	}

	tmax = 0.0f;

	for (i = threadIdx.x / fft_n; i < di; i += n) {
		int idx = i * fft_n;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];			
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop4_2(const float *tab, const float *tab8, float* dest, int ul_PoT, int di, int tmp0, int tmp1, int tmp2) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_n;
	const float *three = tab + tmp1 * fft_n;
	const float *four = tab + tmp2 * fft_n;

	if (fft_n == 8) {
		if ((long int) (one - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			one = tab8;
		if ((long int) (two - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			two = tab8 + tmp0 * fft_n;
		if ((long int) (three - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			three = tab8 + tmp1 * fft_n;
		if ((long int) (four - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			four = tab8 + tmp2 * fft_n;
	}

	tmax = 0.0f;

	for (i = threadIdx.x / fft_n; i < di; i += n) {
		int idx = i * fft_n;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];
		sum += four[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}
	return tmax;
}

template <unsigned int fft_n, unsigned int n>
__device__ float cudaAcc_sumtop5_2(const float *tab, const float *tab8, float* dest, int ul_PoT, int di, int tmp0, int tmp1, int tmp2, int tmp3) {
	float sum, tmax;
	int   i;
	const float *one = tab;
	const float *two = tab + tmp0 * fft_n;
	const float *three = tab + tmp1 * fft_n;
	const float *four = tab + tmp2 * fft_n;
	const float *five = tab + tmp3 * fft_n;

	if (fft_n == 8) {
		if ((long int) (one - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			one = tab8;
		if ((long int) (two - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			two = tab8 + tmp0 * fft_n;
		if ((long int) (three - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			three = tab8 + tmp1 * fft_n;
		if ((long int) (four - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			four = tab8 + tmp2 * fft_n;
		if ((long int) (five - ul_PoT) & 0x3F) // address not dividable by 16 * sizeof(float)
			five = tab8 + tmp3 * fft_n;
	}
	tmax = 0.0f;

	for (i = threadIdx.x / fft_n; i < di; i += n) {
		int idx = i * fft_n;
		sum  = one[idx];
		sum += two[idx];
		sum += three[idx];
		sum += four[idx];
		sum += five[idx];
		dest[idx] = sum;
		tmax = max(tmax, sum);		
	}	
	return tmax;
}

template <unsigned int fft_n, unsigned int n>
__device__ void cudaAcc_copy2(const float* from, float* to, int count) {
//	int step = fft_n;
//	for (int i = threadIdx.x / fft_n; i < count; i += n) {
//		to[i * step] = from[i * step];
//	}
	for (int i = threadIdx.x / fft_n; i < count; i += n) {
		to[i * fft_n] = from[i * fft_n];
	}
}

template <unsigned int fft_n, unsigned int numper, int num_adds, bool load_state>
__global__ void find_pulse_kernel2(
								   float best_pulse_score,
								   int PulsePotLen, 
								   int AdvanceBy,
								   int y_offset,
                                   int ndivs,
                                   int firstP,
                                   int lastP) 
{	
	const int PoTLen = cudaAcc_PulseFind_settings.NumDataPoints / fft_n;
	//const int fidx = threadIdx.x/fft_n;
	int ul_PoT = threadIdx.x & (fft_n-1);
	int y = blockIdx.y * blockDim.y + y_offset;
	int TOffset1 = y * AdvanceBy;
	int TOffset2 = y * AdvanceBy;
	int y4 = y * 4;

	if (fft_n == 8) {        
		TOffset2 += TOffset2 & 1; // Make TOffset2 dividable by 2 to enable better coalesced reads/writes
	}

	if(TOffset1 > PoTLen - PulsePotLen) {		
		TOffset1 = PoTLen - PulsePotLen;
	}

	float* fp_PulsePot	= cudaAcc_PulseFind_settings.PulsePot_fp	+ ul_PoT + TOffset1 * fft_n;
	float* fp_PulsePot8 = cudaAcc_PulseFind_settings.PulsePot8_fp	+ ul_PoT + TOffset1 * fft_n;
	float* tmp_pot		= cudaAcc_PulseFind_settings.tmp_pot_fp		+ ul_PoT + TOffset2 * fft_n;
	float* best_pot		= cudaAcc_PulseFind_settings.best_pot_fp	+ ul_PoT + TOffset2 * fft_n;
	float* report_pot	= cudaAcc_PulseFind_settings.report_pot_fp	+ ul_PoT + TOffset2 * fft_n;

	int di, maxs = 0;
	float max=0,maxd=0,avg=0,maxp=0, snr=0, fthresh=0;
	float tmp_max, t1;
	int i;

	__shared__ float sdata[fft_n*numper];

	if (!load_state) 
	{
		//  Calculate average power
		// Jason: these average will be the same for each FFTLength worth of PulsePoTs, so we can use the whole block
		for (i = threadIdx.x/fft_n; i < PulsePotLen; i+=numper ) {
			avg += fp_PulsePot[i * fft_n];
		}
		sdata[threadIdx.x] = avg;
		avg = cudaAcc_reduce_sum<fft_n, numper>(sdata) / PulsePotLen;	
	}

	if (load_state) 
    {		
		best_pulse_score = fmax(best_pulse_score, cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_n)].w);
		float4 tmp_float4;
		tmp_float4 = cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_n)];
		avg = tmp_float4.y;
		max = tmp_float4.x * avg;
		maxp = tmp_float4.z;
		maxd = tmp_float4.w;

		tmp_float4 = cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_n)];
		snr = tmp_float4.y;
		fthresh = tmp_float4.z;
		maxs = tmp_float4.w;
	} 
    else 
    {
		if (threadIdx.x < fft_n) 
        {
			cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_n)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_n)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_n)] = make_float4(0.0f, avg, 0.0f, 0.0f);
			cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_n)] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	
		}
	}

	float sqrt_num_adds_div_avg = (float)sqrt((float)num_adds) / avg;

	//  Periods from PulsePotLen/3 to PulsePotLen/4, and power of 2 fractions of.
	//   then (len/4 to len/5) and finally (len/5 to len/6)
	//for(int num_adds = 3; num_adds <= 5; num_adds++)

    for (int p = lastP; p > firstP ; p--)
    //for (int p = lastP-(threadIdx.x/fft_n) ; p > firstP ; p-=numper)
	//for (int p = lastP ; p > firstP ; p-=numper)
//Jason: we're changing the behaviout to do a different set of periods in each 'subblock'...
//... The summing reductions will need to adapt to work to reduce the lot ( numper FFTs ), instead of just first FFT...
//... such that the strongest one is reproted for each ulpot, and best overall.
    {
        float cur_thresh, dis_thresh;
        int tabofst, mper, perdiv;
        int tmp0, tmp1, tmp2, tmp3;

        tabofst = ndivs*3+2-num_adds;
        mper = p * (12/(num_adds - 1));
        perdiv = num_adds - 1;
        tmp0 = (int)((mper + 6) * RECIP_12);             // round(period)
        tmp1 = (int)((mper * 2 + 6) * RECIP_12);         // round(period*2)
        di = (int)p/perdiv;                      // (int)period			
        //dis_thresh = cudaAcc_t_funct(di, num_adds)*avg;	
        dis_thresh = cudaAcc_t_funct(di, num_adds, 0, cudaAcc_PulseFind_settings.PulseMax, cudaAcc_PulseFind_settings.t_funct_cache_fp) * avg;

        switch(num_adds) 
        {
        case 3:
            sdata[threadIdx.x] = cudaAcc_sumtop3_2<fft_n, numper>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1);				
//            sdata[(fidx*(fft_n+1))+ul_PoT] = cudaAcc_sumtop3_2<fft_n, 1>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1);				
            break;
        case 4:
            tmp2 = (int)((mper * 3 + 6) * RECIP_12);     // round(period*3)
            sdata[threadIdx.x] = cudaAcc_sumtop4_2<fft_n, numper>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1, tmp2);				
//            sdata[(fidx*(fft_n+1))+ul_PoT] = cudaAcc_sumtop4_2<fft_n, 1>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1, tmp2);				
            break;
        case 5:
            tmp2 = (int)((mper * 3 + 6) * RECIP_12);     // round(period*3)
            tmp3 = (int)((mper * 4 + 6) * RECIP_12);     // round(period*4)
            sdata[threadIdx.x] = cudaAcc_sumtop5_2<fft_n, numper>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1, tmp2, tmp3);				
//            sdata[(fidx*(fft_n+1))+ul_PoT] = cudaAcc_sumtop5_2<fft_n, 1>(fp_PulsePot, fp_PulsePot8, tmp_pot, ul_PoT, di, tmp0, tmp1, tmp2, tmp3);				
            break;
        }
        tmp_max = cudaAcc_reduce_max<fft_n, numper>(sdata);
//		tmp_max = cudaAcc_reduce_max<fft_n, 1>(&sdata[fidx*(fft_n+1)]);

        if (tmp_max>dis_thresh) {
            // unscale for reporting
            tmp_max /= num_adds;
            cur_thresh = (dis_thresh / num_adds - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

            float _snr = (tmp_max-avg) * sqrt_num_adds_div_avg;
            float _thresh = (cur_thresh-avg) * sqrt_num_adds_div_avg;
            if (_snr / _thresh > best_pulse_score) {
                best_pulse_score = _snr / _thresh;
				cudaAcc_copy2<fft_n, numper>(tmp_pot, best_pot, di); 				          
                if (threadIdx.x < fft_n) 
				{
                    cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_n)] = make_float4(
                        tmp_max/avg,
                        avg,
                        ((float)p)/(float)perdiv,
                        best_pulse_score//TOffset1+PulsePotLen/2
                        );
                    cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_n)] = make_float4(
                        ul_PoT,
                        _snr,
                        _thresh,
                        num_adds
                        );
                    cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;	
                }
            }
            //          ReportPulseEvent(tmp_max/avg,avg,((float)p)/(float)perdiv*res,
            //TOffset+PulsePotLen/2,FOffset,
            //	(tmp_max-avg)*(float)sqrt((float)num_adds)/avg,
            //	(cur_thresh-avg)*(float)sqrt((float)num_adds)/avg,
            //	PTPln.dest, num_adds, 0);
            if ((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) {
                maxp  = (float)p/(float)perdiv;
                maxd  = t1;
                maxs  = num_adds;
                max = tmp_max;
                snr = _snr;
                fthresh= _thresh;
				cudaAcc_copy2<fft_n, numper>(tmp_pot, report_pot, di);
				//memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float)); 

                if (threadIdx.x < fft_n) 
				{
                    // It happens very rarely so it's better to store the results right away instead of
                    // storing them in registers or worse local memory
                    cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_n)] = make_float4(
                        max/avg,
                        avg,
                        maxp,
                        maxd//TOffset1+PulsePotLen/2
                        );
                    cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_n)] = make_float4(
                        ul_PoT,
                        snr,
                        fthresh,
                        maxs
                        );
                    cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
                    //ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
                    //snr, fthresh, FoldedPOT, maxs, 1);
                }
            }
        }

        int num_adds_2 = num_adds << 1;

        for (int j = 1; j < ndivs ; j++) 
        {
            float sqrt_num_adds_2_div_avg = (float)sqrt((float)num_adds_2) / avg;
			perdiv = perdiv << 1;
            tmp0 = di & 1;
            di = di >> 1;
            tmp0 += di;
            tabofst -=3;
            //dis_thresh = cudaAcc_t_funct(di, num_adds_2) * avg;				
            dis_thresh = cudaAcc_t_funct(di, num_adds, j, cudaAcc_PulseFind_settings.PulseMax, cudaAcc_PulseFind_settings.t_funct_cache_fp) * avg;

            sdata[threadIdx.x] = cudaAcc_sumtop2_2<fft_n, numper>(tmp_pot, tmp_pot, di, tmp0);
			//sdata[(fidx*(fft_n+1))+ul_PoT] = cudaAcc_sumtop2_2<fft_n, 1>(tmp_pot, tmp_pot, di, tmp0);
            tmp_max = cudaAcc_reduce_max<fft_n, numper>(sdata);
			//tmp_max = cudaAcc_reduce_max<fft_n, 1>(&sdata[fidx*(fft_n+1)]);

            if (tmp_max>dis_thresh) {
                // unscale for reporting
                tmp_max /= num_adds_2;
                cur_thresh = (dis_thresh / num_adds_2 - avg) * cudaAcc_PulseFind_settings.rcfg_dis_thresh + avg;

                float _snr = (tmp_max-avg) * sqrt_num_adds_2_div_avg;
                float _thresh = (cur_thresh-avg) * sqrt_num_adds_2_div_avg;
                if (_snr / _thresh > best_pulse_score) {
                    best_pulse_score = _snr / _thresh;
                    cudaAcc_copy2<fft_n,numper>(tmp_pot, best_pot, di);
                    if (threadIdx.x < fft_n) 
					{
                        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4    , fft_n)] = make_float4(
                            tmp_max/avg,
                            avg,
                            ((float)p)/(float)perdiv,
                            best_pulse_score//TOffset1+PulsePotLen/2
                            );
                        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 1, fft_n)] = make_float4(
                            ul_PoT,
                            _snr,
                            _thresh,
                            num_adds_2
                            );
                        cudaAcc_PulseFind_settings.result_flags_fp->has_best_pulse = 1;
                    }
                }
                //ReportPulseEvent(tmp_max/avg,avg,((float)p)/(float)perdiv*res,
                //	TOffset+PulsePotLen/2,FOffset,
                //	(tmp_max-avg)*(float)sqrt((float)num_adds_2)/avg,
                //	(cur_thresh-avg)*(float)sqrt((float)num_adds_2)/avg,
                //	PTPln.dest, num_adds_2, 0);

                if ((tmp_max>cur_thresh) && ((t1=tmp_max-cur_thresh)>maxd)) {
                    maxp = (float)p/(float)perdiv;
                    maxd = t1;
                    maxs = num_adds_2;
                    max  = tmp_max;
                    snr  = _snr;
                    fthresh = _thresh;

                    cudaAcc_copy2<fft_n, numper>(tmp_pot, report_pot, di);
	                //memcpy(best_pot, PTPln.dest, PTPln.di*sizeof(float));

                    // It happens very rarely so it's better to store the results right away instead of
                    // storing them in registers or worse local memory
                    if (threadIdx.x < fft_n) 
					{
                        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 2, fft_n)] = make_float4(
                            max/avg,
                            avg,
                            maxp,
                            maxd//TOffset1+PulsePotLen/2
                            );
                        cudaAcc_PulseFind_settings.results_fp[AT_XY(ul_PoT, y4 + 3, fft_n)] = make_float4(
                            ul_PoT,
                            snr,
                            fthresh,
                            maxs
                            );
                        cudaAcc_PulseFind_settings.result_flags_fp->has_report_pulse = 1;
                        //ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
                        //snr, fthresh, FoldedPOT, maxs, 1);
                    }
                }
            }

            num_adds_2 = num_adds_2 << 1;
			}  // for (j = 1; j < ndivs
		} // for (p = lastP
	
}

// Assuming AdvanceBy >= PulsePoTLen / 2
template <unsigned int fft_n, unsigned int numthreads>
int cudaAcc_find_pulse_original2(float best_pulse_score, int PulsePoTLen, int AdvanceBy) {	
	const int PoTLen = cudaAcc_NumDataPoints / fft_n;
	const int parts = (PulsePoTLen == PoTLen) ? 1 : PoTLen / AdvanceBy;
	int num_iter = pfPeriodsPerLaunch;
	int num_blocks = gCudaDevProps.multiProcessorCount*pfBlocksPerSM;  

	if (fft_n == 8) {
		// moving input data 8 bytes forwar to enable coleased reads.
		// find_pulse_kernel2 checks if it's better to make reads from dev_t_PowersSpectrum + 8 or dev_PowerSpectrum
		// to make reads coalesced
		CUDA_ACC_SAFE_CALL((cudaMemcpy(dev_t_PowerSpectrum + 8, dev_PowerSpectrum, cudaAcc_NumDataPoints * sizeof(*dev_PowerSpectrum), cudaMemcpyDeviceToDevice)),true);
	}

#if CUDART_VERSION >= 3000
	if (gCudaDevProps.major >= 2) {
		cudaFuncSetCacheConfig(find_pulse_kernel2<fft_n, numthreads/fft_n, 3, false>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(find_pulse_kernel2<fft_n, numthreads/fft_n, 3, true>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(find_pulse_kernel2<fft_n, numthreads/fft_n, 4, true>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(find_pulse_kernel2<fft_n, numthreads/fft_n, 5, true>,cudaFuncCachePreferL1);
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

	for (int y_offset = 0; y_offset < parts; y_offset += num_blocks ) {
		dim3 block(numthreads,1, 1);
		dim3 grid(1, min(parts - y_offset, num_blocks), 1);
        lastP = (PulsePoTLen*2)/3;  firstP = (PulsePoTLen*1)/2;
        nRemaining = lastP - firstP; 
        itr_sent = MIN(nRemaining, num_iter);
        firstP = lastP - itr_sent;
        CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2<fft_n, numthreads/fft_n, 3, false><<<grid, block>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);
		CUDASYNC;
		lastP = firstP;
        nRemaining -= itr_sent;
        while (nRemaining)
        {
            itr_sent = MIN(nRemaining, num_iter);
            firstP = lastP - itr_sent;
		    CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2<fft_n, numthreads/fft_n, 3, true><<<grid, block>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);
			CUDASYNC;
            lastP = firstP;
            nRemaining -= itr_sent;
        } 

        lastP = (PulsePoTLen*3)/4;  firstP = (PulsePoTLen*3)/5; 
        nRemaining = lastP - firstP; 
        do
        {
            itr_sent = MIN(nRemaining, num_iter);
            firstP = lastP - itr_sent;
            CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2<fft_n, numthreads/fft_n, 4, true><<<grid, block>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);
			CUDASYNC;
            lastP = firstP;
            nRemaining -= itr_sent;
        } while (nRemaining);

        lastP = (PulsePoTLen*4)/5;  firstP = (PulsePoTLen*4)/6; 
        nRemaining = lastP - firstP; 
        do
        {
            itr_sent = MIN(nRemaining, num_iter);
            firstP = lastP - itr_sent;
            CUDA_ACC_SAFE_LAUNCH( (find_pulse_kernel2<fft_n, numthreads/fft_n, 5, true><<<grid, block>>>(best_pulse_score, PulsePoTLen, AdvanceBy, y_offset, numdivs, firstP, lastP)),true);
			CUDASYNC;
            lastP = firstP;
            nRemaining -= itr_sent;
        } while (nRemaining);
	}

	return 0;
}

void cudaAcc_choose_best_find_pulse(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) {
	if (gCudaDevProps.regsPerBlock >= 32 * 1024) {
		 //At least 32 * 1024 registers, so we can launch 1024 threads per block. At least CUDA 2.0 compatibile device
		switch (FftLength) {
		case 8:
			cudaAcc_find_pulse_original2<8, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 16:
			cudaAcc_find_pulse_original2<16, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 32:
			cudaAcc_find_pulse_original2<32, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;		
		case 64:
			cudaAcc_find_pulse_original2<64, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 128:
			cudaAcc_find_pulse_original2<128, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 256:
			cudaAcc_find_pulse_original2<256, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 512:
			cudaAcc_find_pulse_original2<512, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 1024:
			cudaAcc_find_pulse_original2<1024, 1024>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		default:			
			cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
		}
	}
	else 
		if (gCudaDevProps.regsPerBlock >= 16 * 1024) {
		// more that 16 * 1024 registers so we can luch 512 threads per block. At least CUDA 1.2 compatibile device
		switch (FftLength) {
		case 8:
			cudaAcc_find_pulse_original2<8, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 16:
			cudaAcc_find_pulse_original2<16, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 32:
			cudaAcc_find_pulse_original2<32, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;		
		case 64:
			cudaAcc_find_pulse_original2<64, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 128:
			cudaAcc_find_pulse_original2<128, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 256:
			cudaAcc_find_pulse_original2<256, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 512:
			cudaAcc_find_pulse_original2<512, 512>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		default:			
			cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
		}
	} else {
		// less that 16 * 1024 registers (assuming 8 * 1024) so we can luch only 2562 threads per block. CUDA 1.0 or 1.1 compatibile device
		switch (FftLength) {
		case 8:
			cudaAcc_find_pulse_original2<8, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 16:
			cudaAcc_find_pulse_original2<16, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 32:
			cudaAcc_find_pulse_original2<32, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;		
		case 64:
			cudaAcc_find_pulse_original2<64, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 128:
			cudaAcc_find_pulse_original2<128, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
		case 256:
			cudaAcc_find_pulse_original2<256, 256>(best_pulse_score, PulsePoTLen, AdvanceBy);	
			break;
        default:			
			cudaAcc_find_pulse_original(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);
		}
	}
}

int cudaAcc_find_pulse(float best_pulse_score, int PulsePoTLen, int AdvanceBy, int FftLength) {	
	CUDA_ACC_SAFE_CALL((cudaMemset(dev_find_pulse_flag, 0, sizeof(*dev_find_pulse_flag))),true);
	CUDA_ACC_SAFE_CALL((CUDASYNC),true);

	cudaAcc_choose_best_find_pulse(best_pulse_score, PulsePoTLen, AdvanceBy, FftLength);

	const int PoTLen = cudaAcc_NumDataPoints / FftLength;
	result_find_pulse_flag flags;	
    CUDA_ACC_SAFE_CALL((CUDASYNC),true);
	CUDA_ACC_SAFE_CALL((cudaMemcpy(&flags, dev_find_pulse_flag, sizeof(*dev_find_pulse_flag), cudaMemcpyDeviceToHost)),true);
    CUDA_ACC_SAFE_CALL((CUDASYNC),true);
    if (flags.has_best_pulse || flags.has_report_pulse) {
		int nb_of_results = 0;

		int PoTStride = (( (PulsePoTLen == PoTLen) ? 1: PoTLen/AdvanceBy) + 1) * AdvanceBy;
		int max_nb_of_elems = FftLength * PoTStride;

		CUDA_ACC_SAFE_CALL((cudaMemcpy(PulseResults, dev_PulseResults, 4 * (cudaAcc_NumDataPoints / AdvanceBy + 1) * sizeof(*dev_PulseResults), cudaMemcpyDeviceToHost)),true);
		CUDA_ACC_SAFE_CALL((CUDASYNC),true);
		if (flags.has_best_pulse) {
			cudaAcc_transposeGPU(dev_tmp_pot, dev_best_pot, FftLength, PoTStride);
			CUDASYNC;
			CUDA_ACC_SAFE_CALL((cudaMemcpy(best_PoT, dev_tmp_pot, max_nb_of_elems * sizeof(float), cudaMemcpyDeviceToHost)),true);
			CUDASYNC;
		}
		if (flags.has_report_pulse) {
			cudaAcc_transposeGPU(dev_tmp_pot, dev_report_pot, FftLength, PoTStride);
			CUDASYNC;
			CUDA_ACC_SAFE_CALL((cudaMemcpy(tmp_PoT, dev_tmp_pot, max_nb_of_elems * sizeof(float), cudaMemcpyDeviceToHost)),true);
			CUDASYNC;
		}
		
		// Iterating trough results
		for(int ThisPoT=1; ThisPoT < FftLength; ThisPoT++) {
			for(int TOffset = 0, TOffsetOK = true, PulsePoTNum = 0; TOffsetOK; PulsePoTNum++, TOffset += AdvanceBy) {
				int TOffset2 = TOffset;
				if(TOffset + PulsePoTLen >= PoTLen) {
					TOffsetOK = false;
					TOffset = PoTLen - PulsePoTLen;
				}
				if (FftLength == 8) {
					TOffset2 += TOffset2 & 1; // Corresponding addresss as in cudaAcc_find_pulse_original2 kernel
				}

				int index0 = ((PulsePoTNum * 4 + 0) * FftLength + ThisPoT);
				int index1 = ((PulsePoTNum * 4 + 1) * FftLength + ThisPoT);
				int index2 = ((PulsePoTNum * 4 + 2) * FftLength + ThisPoT);
				int index3 = ((PulsePoTNum * 4 + 3) * FftLength + ThisPoT);

				if (flags.has_best_pulse) {
					if (PulseResults[index0].x > 0) {
						float4 res1 = PulseResults[index0];
						float4 res2 = PulseResults[index1];

						//          ReportPulseEvent(tmp_max/avg,avg,((float)p)/(float)perdiv*res,
						//TOffset+PulsePotLen/2,FOffset,
						//	(tmp_max-avg)*(float)sqrt((float)num_adds)/avg,
						//	(cur_thresh-avg)*(float)sqrt((float)num_adds)/avg,
						//	PTPln.dest, num_adds, 0);						
						nb_of_results++;
						cudaAcc_ReportPulseEvent(res1.x, res1.y, res1.z, TOffset+PulsePoTLen/2, (int) res2.x, res2.y, res2.z, 
							&best_PoT[ThisPoT * PoTStride + TOffset2], (int) res2.w, 0);
					}
				}
				if (flags.has_report_pulse) {
					if (PulseResults[index2].x > 0) {
						float4 res1 = PulseResults[index2];
						float4 res2 = PulseResults[index3];

						//ReportPulseEvent(max/avg,avg,maxp*res,TOffset+PulsePotLen/2,FOffset,
						//snr, fthresh, FoldedPOT, maxs, 1);
						nb_of_results++;
						cudaAcc_ReportPulseEvent(res1.x, res1.y, res1.z, TOffset+PulsePoTLen/2, (int) res2.x, res2.y, res2.z, 
							&tmp_PoT[ThisPoT * PoTStride + TOffset2], (int) res2.w, 1);

					}
				}
			}					
		}
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

	return 0;
}

#endif //USE_CUDA
