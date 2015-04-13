#include "cudaAcceleration.h"
#ifdef USE_CUDA

#include "cudaAcc_data.h"
#include "cudaAcc_utilities.h"

cufftHandle fft_analysis_plans[MAX_NUM_FFTS][2];
int cufft_init[MAX_NUM_FFTS];
int cufft_len[MAX_NUM_FFTS];

int nbofffts = 0;
int cufftplans_done = 0;

int cudaAcc_fftwf_plan_dft_1d(int FftNum, int FftLen, int NumDataPoints) 
{    
  cufftResult cu_err;
  
  cu_err = cufftPlan1d(&fft_analysis_plans[FftNum][0], FftLen, CUFFT_C2C, NumDataPoints / FftLen);
  if( CUFFT_SUCCESS != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cufftPlan1d(&fft_analysis_plans[FftNum][0], FftLen, CUFFT_C2C, NumDataPoints / FftLen)");
      return 1;
    }
#if CUDART_VERSION >= 3000
  cufftSetStream(fft_analysis_plans[FftNum][0], fftstream0);
  cufftSetCompatibilityMode(fft_analysis_plans[FftNum][0],CUFFT_COMPATIBILITY_NATIVE);
#endif
  
  cufft_init[FftNum] = 1; 
  cufft_len[FftNum] = FftLen;
  
  nbofffts = max(nbofffts, FftNum);
  return 0;
}



void cudaAcc_execute_dfts(int FftNum, int offset) 
{
  CUFFT_SAFE_CALL((cufftExecC2C(fft_analysis_plans[FftNum][0], dev_cx_ChirpDataArray + offset, dev_WorkData, CUFFT_INVERSE)));
  cudaEventRecord(fftDoneEvent, fftstream0);
}


void cudaAcc_fft_free() 
{
  for(int i = 0; i < nbofffts; ++i)
    {
      if(fft_analysis_plans[i][0]) 
	{
	  CUFFT_SAFE_CALL(cufftDestroy(fft_analysis_plans[i][0]));
	  fft_analysis_plans[i][0] = 0; //Make sure it's gone
	}
      cufft_init[i] = 0;
    }
  nbofffts = 0;
  cufftplans_done = 0; //Early paranoid plans (no longer used) can be clleard by a free+initialisation retry, makes sure they're remade with first chirp
}

#endif //USE_CUDA
