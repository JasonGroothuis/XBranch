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

#ifndef _CUDA_ACC_DATA_H
#define _CUDA_ACC_DATA_H

//#include <cufft.h>

#define MAX_NUM_FFTS       64
#ifdef M_PI // already defined on linux
#undef M_PI
#endif
#define M_PI               3.14159265358979323846
#define M_2PI              6.28318530717958647692
#define M_2PIf              6.28318530717958647692f

#define CUDA_ACC_MAX_GaussTOffsetStop 128

typedef struct {double Sin, Cos; } double_SinCos;
typedef struct {  
    int iSigma;
    int gauss_pot_length;
    float GaussPowerThresh;
    float GaussPeakPowerThresh3;
    float GaussSigmaSq;
    int GaussTOffsetStart;
    int GaussTOffsetStop;
    float PeakScaleFactor;
    float GaussChiSqThresh;
    float gauss_null_chi_sq_thresh;
    float gauss_chi_sq_thresh;
    float* dev_PoT;
    float* dev_PoTPrefixSum;
    float* dev_PowerSpectrum;
    float4* dev_GaussFitResults;
    float4* dev_GaussFitResultsReordered;
    float4* dev_GaussFitResultsReordered2;
    float* dev_NormMaxPower;    
    float* dev_outputposition;
    float *dev_tmp_pot2; //2xxxx
    float score_offset;
    int NumDataPoints;
    float f_weight[CUDA_ACC_MAX_GaussTOffsetStop]; // cached  static_cast<float>(EXP(i, 0, PoTInfo.GaussSigmaSq));
    unsigned int nsamples;
} cudaAcc_GaussFit_t;

typedef struct {
	int has_results;
	int error;
} result_flag;

typedef struct {
	int has_best_pulse;	
	int has_report_pulse;
} result_find_pulse_flag;

extern float2* dev_cx_DataArray;
extern float2* dev_cx_ChirpDataArray;
extern double2* dev_CurrentTrig;

extern float2* dev_WorkData;

extern float* dev_PowerSpectrum;
extern float* dev_t_PowerSpectrum;
extern float* dev_PoT;
extern float* dev_PoTPrefixSum;
extern float4* dev_GaussFitResults;
extern float4* dev_GaussFitResultsReordered;
extern float4* dev_GaussFitResultsReordered2;
extern float4* dev_TripletResults; // In the same place as dev_GaussFitResults
extern float4* dev_PulseResults; // In the same place as dev_GaussFitResults
extern float4* TripletResults; // In the same place as PulseResults
extern float4* PulseResults; // In the same place as PulseResults
extern float* dev_tmp_pot;
extern float* dev_tmp_pot2; // 2xxx
extern float* dev_best_pot;
extern float* dev_report_pot;
//extern float2* dev_sample_rate;

extern float* dev_NormMaxPower;
extern float3* dev_PowerSpectrumSumMax;

extern bool gCudaAutocorrelation;
extern float2* dev_AutoCorrIn[8];
extern float2* dev_AutoCorrOut[8];
//extern cufftHandle cudaAutoCorr_plan[8];
extern cufftHandle cudaAutoCorr_plan;
//extern cudaStream_t cudaAutocorrStream[8];

extern float* dev_flagged;
extern float* dev_outputposition;
extern result_flag* dev_flag;
extern float* tmp_small_PoT;
extern float* tmp_PoT;
extern float* best_PoT;
extern float* tmp_PoT2; // triplets
extern float* best_PoT2; // triplets

extern float4* GaussFitResults;

extern int cudaAcc_NumDataPoints;
//extern int cudaps_blksize;
//extern int cuda_tmax;

extern int cudaAcc_initialized();
extern cudaAcc_GaussFit_t settings;

extern void cudaAcc_transposeGPU(float *odata, float *idata, int width, int height);
extern void cudaAcc_transposeGPU(float *odata, float *idata, int width, int height, cudaStream_t stream);

extern void cudaAcc_CalcChirpData_sm13(double chirp_rate, double recip_sample_rate, sah_complex* cx_ChirpDataArray, int NumDataPoints);


extern void cudaAcc_fft_free();

extern __device__ __host__ double fraction(double x);



#endif //_CUDA_ACC_DATA_H
