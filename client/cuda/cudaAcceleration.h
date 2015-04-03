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

#ifndef _CUDA_ACCELERATION_H
#define _CUDA_ACCELERATION_H

#include <cufft.h>

#define BSPLIT 29

#if defined(_MSC_VER) || defined(__linux__) || defined(__APPLE__)
#include <builtin_types.h>
#include "boinc_api.h"
#include "s_util.h"
#include "analyzePoT.h"

extern double *angle_range;
extern unsigned cmem_rtotal;

extern cudaStream_t fftstream0; // fft, autocorr, triplet
extern cudaStream_t fftstream1; // gauss

//Public function prototypes
int  cudaAcc_initializeDevice(int devPref, int usePolling);
int  cudaAcc_initialize(sah_complex* cx_DataArray, int NumDataPoints, int gauss_pot_length, unsigned long nsamples, 
			double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh,
			double pulse_display_thresh, double PulseThresh, int PulseMax,
			double sample_rate, long acfftlen );
int  cudaAcc_initialize_pulse_find(double pulse_display_thresh, double PulseThresh, int PulseMax);
int cudaAcc_InitializeAutocorrelation(int ac_fftlen);
void cudaAcc_free();
void cudaAcc_free_Gaussfit();
void cudaAcc_free_pulse_find();
void cudaAcc_free_AutoCorrelation();
void cudaAcc_CalcChirpData(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray);
void cudaAcc_CalcChirpData_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream);
void cudaAcc_CalcChirpData_sm13(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray);
void cudaAcc_CalcChirpData_sm13_async(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, cudaStream_t chirpstream);
int cudaAcc_fftwf_plan_dft_1d(int FftNum, int FftLen, int NumDataPoints);
void cudaAcc_execute_dfts(int FftNum, int offset);
void cudaAcc_GetPowerSpectrum(int numpoints, int offset, cudaStream_t str);
void cudaAcc_SetPowerSpectrum(float* PowerSpectrum);
int  cudaAcc_initializeGaussfit(const PoTInfo_t& PoTInfo, int gauss_pot_length, unsigned int nsamples, double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh);
int cudaAcc_GaussfitStart(int gauss_pot_length, double best_gauss_score, bool noscore);
int cudaAcc_fetchGaussfitFlags(int gauss_pot_length, double best_gauss_score);
int cudaAcc_processGaussFit(int gauss_pot_length, double best_gauss_score);
void cudaAcc_summax(int fftlen);
void cudaAcc_summax_x(int fftlen);
int cudaAcc_calculate_mean(int PulsePoTLen, float triplet_thresh, int AdvanceBy, int FftLength);
int cudaAcc_find_triplets(int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength);
int cudaAcc_fetchTripletAndPulseFlags(bool SkipTriplet, bool SkipPulse, int PulsePoTLen, int AdvanceBy, int FftLength);
int cudaAcc_processTripletResults(int PulsePoTLen, int AdvanceBy, int FftLength);//int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength);
int cudaAcc_find_pulses(float best_pulse_score, int PulsePotLen, int AdvanceBy, int fft_length);
int cudaAcc_processPulseResults(int PulsePoTLen, int AdvanceBy, int FftLength);//int PulsePotLen, float triplet_thresh, int AdvanceBy, int ul_FftLength);

//V7 Autocorrelation 
int cudaAcc_FindAutoCorrelations(float *AutoCorrelation, int ac_fftlen);
int cudaAcc_GetAutoCorrelation(float *AutoCorrelation, int ac_fftlen, int fft_num);

//Referenced globals
extern float3* PowerSpectrumSumMax;
extern int gSetiUseCudaDevice;
extern cudaDeviceProp gCudaDevProps;
//extern float ac_TotalSum;
//extern float ac_Peak;
//extern int ac_PeakBin;
extern float3 *dev_ac_partials[8];
extern float3 *blockSums[8];
// DEFINES
#define USE_CUDA 1   // Allows the CUDA code path to be compiled in and used if defined

//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 ) 
        slices *= 2;
    return dim3( nblocks/slices, slices );
}

#if (CUDART_VERSION < 4000)
	#define CUDASYNC cudaThreadSynchronize()
#else
	#define CUDASYNC cudaDeviceSynchronize()
#endif

#endif //_MSC_VER

#endif //_CUDA_ACCELERATION_H
