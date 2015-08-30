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


#include "cudaAcc_analyzeReport.h"
#include "analyzeReport.h"
#include "gaussfit.h"

int cudaAcc_ReportPulseEvent(float PulsePower, float MeanPower, float PulsePeriod, int time_bin, int freq_bin, float snr, float thresh, float *foldedPOT, int scale, int write) {
	return ReportPulseEvent(
		PulsePower,
		MeanPower,
		PulsePeriod,
		time_bin,
		freq_bin,
		snr,
		thresh,
		foldedPOT,
		scale,
		write
		);
}

int cudaAcc_ReportTripletEvent(float PulsePower, float MeanPower, float PulsePeriod, float mid_time_bin, int start_time_bin, int freq_bin, int pot_len, const float *PoT, int write) {
	return ReportTripletEvent(
		PulsePower,
		MeanPower,
		PulsePeriod,
		mid_time_bin,
		start_time_bin,
		freq_bin,
		pot_len,
		PoT,
		write
		);
}

int cudaAcc_ChooseGaussEvent(int ifft, float PeakPower, float TrueMean, float ChiSq, float null_ChiSq, int bin, float sigma, float PoTMaxPower, float fp_PoT[]) {
	//double old_score = best_gauss->score;
	int retval = ChooseGaussEvent(
		ifft,
		PeakPower,
		TrueMean,
		ChiSq,
		null_ChiSq,
		bin,
		sigma,
		PoTMaxPower,
		fp_PoT
	);
	return retval;
}
