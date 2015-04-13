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

#define PINNED
// includes, project
#include <cufft.h>
#include <cuda.h>

#include "malloc_a.h"
#include "cudaAcceleration.h"
#include "cudaAcc_data.h"
//#ifndef __linux__ //Pretty sure this is unused on windows as well... Can we delete?
//#include "cudaAcc_CleanExit.h" // TODO: DELETE THAT, only for testing
//#endif // not __linux__

#include "cudaAcc_utilities.h"
#include "nvapi_device.h"

unsigned cmem_rtotal = 0;

float2* dev_cx_DataArray;
float2* dev_cx_ChirpDataArray;
//float2* float_CurrentTrig;
double2* dev_CurrentTrig;

float2* dev_WorkData;
float *dev_PowerSpectrum;
float *dev_t_PowerSpectrumP; // pulse
float *dev_t_PowerSpectrumT; // triplet
float *dev_t_PowerSpectrumG; // gauss
float *dev_PoTP; // for pulsefind
float *dev_PoTG;
float *dev_PoTPrefixSumP;
float *dev_PoTPrefixSumT;
float *dev_PoTPrefixSumG;

cudaEvent_t fftDoneEvent;
cudaEvent_t summaxDoneEvent;
cudaEvent_t powerspectrumDoneEvent;
cudaEvent_t autocorrelationDoneEvent[8];
cudaEvent_t ac_reduce_partialEvent[8];

cudaEvent_t meanDoneEvent;
cudaEvent_t tripletsDoneEvent;
cudaEvent_t pulseDoneEvent;
cudaEvent_t gaussDoneEvent;

cudaStream_t fftstream0 = 0;
cudaStream_t fftstream1 = 0;

#define CUDA_MAXNUMSTREAMS 2
int cudaAcc_NumDataPoints;
//int cudaps_blksize=64;
//int cuda_tmax=256;
bool cuda_pinned = false;
cudaStream_t cudapsStream[CUDA_MAXNUMSTREAMS];
cudaStream_t cudaAutocorrStream[8];


extern __global__ void cudaAcc_summax32_kernel(float *input, float3* output, int iterations);
template <int blockx> __global__ void find_triplets_kernel(int ul_FftLength, int len_power, volatile float triplet_thresh, int AdvanceBy);
template <bool load_state, int num_adds> __global__ void find_pulse_kernel(float best_pulse_score, int PulsePotLen, int AdvanceBy, int fft_len, int ndivs);

float4* dev_GaussFitResults;
float4* dev_GaussFitResultsReordered;
float4* dev_GaussFitResultsReordered2;
float *dev_NormMaxPower;
result_flag* dev_flagT; // triplet
result_flag* dev_flagG; // gauss
float4* GaussFitResults;
float4* GaussFitResults2;

float *dev_flagged;
float *dev_outputposition;
float *tmp_small_PoT; // Space for PoTs for reporting
float *tmp_PoTP; // pulse
float *tmp_PoTT; // triplet
float *tmp_PoTG; // gauss
float *best_PoTP;
float *best_PoTG;
cudaDeviceProp gCudaDevProps;

int cudaAcc_init = 0;  // global count variable for CUDA mem allocations.


int cudaAcc_initialized() 
{
  return cudaAcc_init;
}


bool cudaAcc_setBlockingSync(int device) 
{
  //CUDA_ACC_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), false);
  CUDA_ACC_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleYield), false);
  
  return true;
}


int cudaAcc_initializeDevice(int devPref, int usePolling) 
{
  int numCudaDevices = 0;
  int i = 0, bestDevFound =0;
  cudaDeviceProp cDevProp[8];
  bool bCapableGPUFound = false;
  cudaError_t cerr;
    
  // init our global DevProp var and query how many CUDA devices
  // are present.
  memset(&gCudaDevProps, 0, sizeof(cudaDeviceProp));
  memset(cDevProp, 0, sizeof(cudaDeviceProp)*8);
  
  //Jason: Don't use safecall with exit here, return 1 if an error finding devices etc.
  cerr = cudaGetDeviceCount(&numCudaDevices);
  CUDA_ACC_SAFE_CALL_NO_SYNC("Couldn't get cuda device count\n");
  if(cerr != cudaSuccess)  //Jason; Extra paranoia
    {
      fprintf(stderr, "setiathome_CUDA: cudaGetDeviceCount() call failed.\n");
    }
  
  if(!numCudaDevices)
    {
      fprintf(stderr, "setiathome_CUDA: No CUDA devices found\n");
    }
  
  //limit to 16 GPU's for now
  if(numCudaDevices > 16) numCudaDevices = 16;
  
  fprintf(stderr, "setiathome_CUDA: Found %d CUDA device(s):\n", numCudaDevices);
  
  // Let's enumerate the CUDA devices avail and 
  // pick the best one.
  for(i = 0; i < numCudaDevices; i++)
    {
      CUDA_ACC_SAFE_CALL(cudaGetDeviceProperties(&cDevProp[i], i), true);
#ifdef _WIN32
      fprintf(stderr, "  Device %d: %s, %u MiB, ",
	      i+1,
	      cDevProp[i].name,
	      (ULONGLONG)(cDevProp[i].totalGlobalMem>>20));
#else
      fprintf(stderr, "  Device %d: %s, %zu MiB, ",
	      i+1,
	      cDevProp[i].name,
	      (size_t)(cDevProp[i].totalGlobalMem>>20));
#endif	
      fprintf(stderr, "regsPerBlock %u\n",cDevProp[i].regsPerBlock);
	fprintf(stderr, "     computeCap %d.%d, multiProcs %d \n", 	
		cDevProp[i].major, cDevProp[i].minor,
		cDevProp[i].multiProcessorCount);
	//fprintf(stderr, "           totalGlobalMem = %d \n", cDevProp[i].totalGlobalMem);
	//fprintf(stderr, "           sharedMemPerBlock = %d \n",cDevProp[i].sharedMemPerBlock);
	//fprintf(stderr, "           regsPerBlock = %d \n", cDevProp[i].regsPerBlock);
        //fprintf(stderr, "           warpSize = %d \n",cDevProp[i].warpSize);
        //fprintf(stderr, "           memPitch = %d \n",cDevProp[i].memPitch);
        //fprintf(stderr, "           maxThreadsPerBlock = %d \n",cDevProp[i].maxThreadsPerBlock);
#if CUDART_VERSION >= 3000
	fprintf(stderr, "     pciBusID = %d, pciSlotID = %d\n", cDevProp[i].pciBusID, cDevProp[i].pciDeviceID);
#endif
	if(cDevProp[i].major < 3)
	  {
	    //Pre Kepler GPU, Cuda Runtime should report the clock rate correctly.
	    fprintf(stderr, "     clockRate = %d MHz\n", cDevProp[i].clockRate/1000);
	  } 
	//fprintf(stderr, "           totalConstMem = %d \n",cDevProp[i].totalConstMem);
	//fprintf(stderr, "           major = %d \n",cDevProp[i].major);
	//fprintf(stderr, "           minor = %d \n",cDevProp[i].minor);
	//fprintf(stderr, "           textureAlignment = %d \n",cDevProp[i].textureAlignment);
        //fprintf(stderr, "           deviceOverlap = %d \n",cDevProp[i].deviceOverlap);
        //fprintf(stderr, "           multiProcessorCount = %d \n",cDevProp[i].multiProcessorCount);
    }
  //nvFreeAPI();
  
  for(i = 0; i < numCudaDevices; i++)
    {
#if CUDART_VERSION >= 6050
      // Check the supported major revision to ensure it's valid and not some pre-Fermi
      if((cDevProp[i].major < 2))
	{
	  fprintf(stderr, "setiathome_CUDA: device %d is Pre-Fermi CUDA 2.x compute compatibility, only has %d.%d\n", 
		  i+1, cDevProp[i].major, cDevProp[i].minor);
	  continue;
	}
#else
      // Check the supported major revision to ensure it's valid and not emulation mode
      if((cDevProp[i].major < 1))
	{
	  fprintf(stderr, "setiathome_CUDA: device %d does not support CUDA 1.x compute compatibility, supports %d.%d\n", 
		  i+1, cDevProp[i].major, cDevProp[i].minor);
	  continue;
	  }
#endif
      
      // Check the supported major revision to ensure it's valid and not emulation mode
      if((cDevProp[i].major >= 9999))
	{
	  fprintf(stderr, "setiathome_CUDA: device %d is emulation device and should not be used, supports %d.%d\n", 
		  i+1, cDevProp[i].major, cDevProp[i].minor);
          continue;
	}
      
#if CUDART_VERSION < 3000
#pragma message (">>>PRE_FERMI_ONLY<<< Build\n")
      // Check the supported major revision for Pre-Fermi only Cuda 2.2 & 2.3 builds
      if((cDevProp[i].major > 1))
	{
	  fprintf(stderr, "setiathome_CUDA: device %d, compute capability %d.%d is not supported by this application\n", 
		  i+1, cDevProp[i].major, cDevProp[i].minor);
	  continue;
	}
#endif
      //Check if there is enough memory resources to handle our CUDA version of SETI
      if(cDevProp[i].totalGlobalMem < 128*1024*1024)
	  {
            fprintf(stderr, "setiathome_CUDA: device %d not have enough available global memory. Only found %d\n",
                    i+1, (int)cDevProp[i].totalGlobalMem);
            continue;
	  }
      
      //Check if this is a more powerful GPU than any others we found
      if(cDevProp[i].multiProcessorCount > gCudaDevProps.multiProcessorCount)
	{
	  memcpy(&gCudaDevProps, &cDevProp[i], sizeof(cudaDeviceProp));
	  bestDevFound = i;
	  bCapableGPUFound = true;
	}
    }
  
  
  if(!devPref)
    {
      //fprintf(stderr,"In cudaAcc_initializeDevice(): Boinc passed DevPref %d, Which is choose best device %d\n",devPref,bestDevFound);
      
      if(bCapableGPUFound)
	{
	  fprintf(stderr, "setiathome_CUDA: No device specified, determined to use CUDA device %d: %s\n", bestDevFound+1, (char *)&cDevProp[bestDevFound].name);
	  CUDA_ACC_SAFE_CALL(cudaSetDevice(bestDevFound),true);
	  if(!usePolling)
	    cudaAcc_setBlockingSync(bestDevFound);
	}
      else
	{
	  fprintf(stderr, "setiathome_CUDA: No SETI@home capabale CUDA GPU found...\n");
	  return 0;
	}
    }
  else 
    {
      fprintf(stderr,"In cudaAcc_initializeDevice(): Boinc passed DevPref %d\n",devPref);
      
      fprintf(stderr, "setiathome_CUDA: CUDA Device %d specified, checking...\n", devPref);
      // user must want a specific device, check it's qualifications
      if((devPref <= numCudaDevices)                             // Make sure it's a valid device
	 && (cDevProp[devPref-1].major >= 1)                     // that has at least 1.x compute
#if CUDART_VERSION < 3000
#pragma message (">>>PRE_FERMI_ONLY<<< Build\n")
	 // Check the supported major revision for Pre-Fermi only Cuda 2.2 & 2.3 builds
	 && (cDevProp[devPref-1].major < 2)
#endif
	 && (cDevProp[devPref-1].totalGlobalMem > 128*1024*1024) // and more than 128MB of memeory
	 && (cDevProp[devPref-1].major != 9999))                         // and is not an emulation device
	{
	  fprintf(stderr, "   Device %d: %s is okay\n", devPref, (char *)&cDevProp[devPref-1].name);
	  memcpy(&gCudaDevProps, &cDevProp[devPref - 1], sizeof(cudaDeviceProp));
	  CUDA_ACC_SAFE_CALL(cudaSetDevice(devPref - 1), true);
	  if(!usePolling)
	    cudaAcc_setBlockingSync(devPref - 1);                
	}
      else
	{
	  fprintf(stderr, "   Device cannot be used\n");
	  return 0;
	}
    }
  
#if(CUDART_VERSION >= 4000)
  // find_pulse_kernels are limited by memory bandwidth and suboptimal fetches.
  // Prefer a larger L1 cache, as we don't need all the shared memory offered (48k)
  // Override for specific kernels where we need the shared memory instead
  // (e.g. find_triplets_kernel)
  
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#endif
  
  //  gpu device heuristics
  //if(gCudaDevProps.major >= 2) { cuda_tmax = 1024; cudaps_blksize = 256; }
  //else if(gCudaDevProps.minor == 3) { cuda_tmax = 512; cudaps_blksize = 128; }
  //else { cuda_tmax = 256; cudaps_blksize = 64; }
  
  //fprintf(stderr, "   Guru says: Max threads is %d/blk, best for Powerspectrum is %d/blk\n", cuda_tmax, cudaps_blksize);
  
  //fprintf(stderr,"-->In cudaAcc_initializeDevice(): 'Supposed' active Cuda device has %d multiProcessors.\n",gCudaDevProps.multiProcessorCount);
  //fprintf(stderr,"-->In cudaAcc_initializeDevice(): 'Supposed' active Cuda device has %d regsPerBlock.\n",gCudaDevProps.regsPerBlock);
  
  //cudaAcc_init_exit_proc();
  
  return 1;
}

//const double SPLITTER=(1<<BSPLIT)+1;
//__host__ float2 splitdd(double a) {
//    double t = a*SPLITTER; 
//	double ahi= t-(t-a);
//	double alo = a-ahi;
//
//	return make_float2((float)ahi,(float)alo);
//}

int cudaAcc_initialize(sah_complex* cx_DataArray, int NumDataPoints, int gauss_pot_length, unsigned long nsamples,
		       double gauss_null_chi_sq_thresh, double gauss_chi_sq_thresh,
		       double pulse_display_thresh, double PulseThresh, int PulseMax,					   
		       double sample_rate, long acfftlen)
{	
  cudaError_t cu_err;

  //Prevent cudaAcc_initialize to be re-entrant
  if(cudaAcc_init)
    return 0;

  //
  // create events, 
  // disable timig so that they do not introduce any global syncing
  //

  cudaEventCreateWithFlags(&fftDoneEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&summaxDoneEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&powerspectrumDoneEvent, cudaEventDisableTiming);

  for(int i = 0; i < 8; i++)
    {
      cudaEventCreateWithFlags(&(autocorrelationDoneEvent[i]), cudaEventDisableTiming);
      cudaEventCreateWithFlags(&(ac_reduce_partialEvent[i]), cudaEventDisableTiming);
    }

  cudaEventCreateWithFlags(&meanDoneEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&tripletsDoneEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&pulseDoneEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&gaussDoneEvent, cudaEventDisableTiming);

  // events created

  
  cu_err = cudaMalloc((void**) &dev_cx_DataArray, sizeof(*dev_cx_DataArray) * (NumDataPoints*PADVAL));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_cx_DataArray");
      return 1;
    } else { CUDAMEMPRINT(dev_cx_DataArray,"cudaMalloc((void**) &dev_cx_DataArray",NumDataPoints,sizeof(*dev_cx_DataArray)); };
  cudaAcc_init++;

  CUDA_ACC_SAFE_CALL(cudaMemcpyAsync(dev_cx_DataArray, cx_DataArray, NumDataPoints * sizeof(*cx_DataArray), cudaMemcpyHostToDevice),true);
  //CUDA_ACC_SAFE_CALL((CUDASYNC),true);
  
  cu_err = cudaMalloc((void**) &dev_cx_ChirpDataArray, sizeof(*dev_cx_ChirpDataArray) * (NumDataPoints*2*PADVAL));    
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_cx_ChirpDataArray");
      return 1;
    }  else { CUDAMEMPRINT(dev_cx_ChirpDataArray,"cudaMalloc((void**) &dev_cx_ChirpDataArray",NumDataPoints*PADVAL,sizeof(*dev_cx_ChirpDataArray)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_flagT, sizeof(*dev_flagT));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_flagT");
      return 1;
    }  else { CUDAMEMPRINT(dev_flagT, "cudaMalloc((void**) &dev_flagT",1,sizeof(*dev_flagT)); };
  cu_err = cudaMalloc((void**) &dev_flagG, sizeof(*dev_flagG));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_flagG");
      return 1;
    }  else { CUDAMEMPRINT(dev_flagG,"cudaMalloc((void**) &dev_flagG",1,sizeof(*dev_flagG)); };
  cudaAcc_init++;
  
  
  cu_err = cudaMalloc((void**) &dev_WorkData, sizeof(*dev_WorkData) * NumDataPoints * PADVAL); // + 1/8 for find_pulse));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_WorkData");
      return 1;
    } else { CUDAMEMPRINT(dev_WorkData,"cudaMalloc((void**) &dev_WorkData",NumDataPoints * PADVAL,sizeof(*dev_WorkData)); };

  cu_err = cudaMalloc((void**) &dev_best_potP, sizeof(*dev_best_potP) * NumDataPoints * PADVAL); // + 1/8 for find_pulse));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_best_potP");
      return 1;
    } else { CUDAMEMPRINT(dev_best_potP,"cudaMalloc((void**) &dev_best_potP",NumDataPoints * PADVAL,sizeof(*dev_best_potP)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_PowerSpectrum, sizeof(*dev_PowerSpectrum) * NumDataPoints * PADVAL);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PowerSpectrum");
      return 1;
    } else { CUDAMEMPRINT(dev_PowerSpectrum,"cudaMalloc((void**) &dev_PowerSpectrum",NumDataPoints,sizeof(*dev_PowerSpectrum)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_t_PowerSpectrumP, sizeof(*dev_t_PowerSpectrumP) * (NumDataPoints+8));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_t_PowerSpectrumP");
      return 1;
    } else { CUDAMEMPRINT(dev_t_PowerSpectrumP,"cudaMalloc((void**) &dev_t_PowerSpectrumP",NumDataPoints+8,sizeof(*dev_t_PowerSpectrumP)); };

  cu_err = cudaMalloc((void**) &dev_t_PowerSpectrumT, sizeof(*dev_t_PowerSpectrumT) * (NumDataPoints+8));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_t_PowerSpectrumT");
      return 1;
    } else { CUDAMEMPRINT(dev_t_PowerSpectrumT,"cudaMalloc((void**) &dev_t_PowerSpectrumT",NumDataPoints+8,sizeof(*dev_t_PowerSpectrumT)); };

  cu_err = cudaMalloc((void**) &dev_t_PowerSpectrumG, sizeof(*dev_t_PowerSpectrumG) * (NumDataPoints+8));
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_t_PowerSpectrumG");
      return 1;
    } else { CUDAMEMPRINT(dev_t_PowerSpectrumG,"cudaMalloc((void**) &dev_t_PowerSpectrumG",NumDataPoints+8,sizeof(*dev_t_PowerSpectrumG)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_GaussFitResults, sizeof(*dev_GaussFitResults) * NumDataPoints);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_GaussFitResults");
      return 1;
    } else { CUDAMEMPRINT(dev_GaussFitResults,"cudaMalloc((void**) &dev_GaussFitResults",NumDataPoints,sizeof(*dev_GaussFitResults)); };
  cudaAcc_init++;

  cu_err = cudaMalloc((void**) &dev_TripletResults, sizeof(*dev_TripletResults) * NumDataPoints);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_TripletResults");
      return 1;
    } else { CUDAMEMPRINT(dev_TripletResults,"cudaMalloc((void**) &dev_TripletResults",NumDataPoints,sizeof(*dev_TripletResults)); };
  cudaAcc_init++;

  cu_err = cudaMalloc((void**) &dev_PulseResults, sizeof(*dev_PulseResults) * NumDataPoints);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PulseResults");
      return 1;
    } else { CUDAMEMPRINT(dev_PulseResults,"cudaMalloc((void**) &dev_PulseResults",NumDataPoints,sizeof(*dev_PulseResults)); };
  cudaAcc_init++;
  
  dev_GaussFitResultsReordered = dev_GaussFitResults + NumDataPoints;
  dev_GaussFitResultsReordered2 = dev_GaussFitResultsReordered + NumDataPoints;
  //CUDA_ACC_SAFE_CALL(cudaMalloc((void**) &dev_GaussFitResultsReordered, sizeof(*dev_GaussFitResultsReordered) * NumDataPoints)); // TODO: it can be smaller
  //CUDA_ACC_SAFE_CALL(cudaMalloc((void**) &dev_GaussFitResultsReordered2, sizeof(*dev_GaussFitResultsReordered2) * NumDataPoints)); // TODO: it can be smaller
  
#ifdef PINNED
  cudaMallocHost((void **)&GaussFitResults, sizeof(*GaussFitResults) * NumDataPoints);
  cudaMallocHost((void **)&TripletResults, sizeof(*TripletResults) * NumDataPoints);
  cudaMallocHost((void **)&PulseResults, sizeof(*PulseResults) * NumDataPoints);
#else
  GaussFitResults = (float4*) malloc(sizeof(*GaussFitResults) * NumDataPoints);
#endif
  GaussFitResults2 = GaussFitResults + NumDataPoints;
  
  cu_err = cudaMalloc((void**) &dev_PoTP, sizeof(*dev_PoTP) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PoTP");
      return 1;
    } else { CUDAMEMPRINT(dev_PoTP,"cudaMalloc((void**) &dev_PoTP", NumDataPoints * PADVAL_PULSE, sizeof(*dev_PoTP)); };

  cu_err = cudaMalloc((void**) &dev_PoTG, sizeof(*dev_PoTG) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PoTG");
      return 1;
    } else { CUDAMEMPRINT(dev_PoT,"cudaMalloc((void**) &dev_PoTG", NumDataPoints * PADVAL_PULSE, sizeof(*dev_PoTG)); };

  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_PoTPrefixSumP, sizeof(*dev_PoTPrefixSumP) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PoTPrefixSumP");
      return 1;
    } else { CUDAMEMPRINT(dev_PoTPrefixSumP,"cudaMalloc((void**) &dev_PoTPrefixSumP", NumDataPoints * PADVAL_PULSE,sizeof(*dev_PoTPrefixSumP)); };

  cu_err = cudaMalloc((void**) &dev_PoTPrefixSumT, sizeof(*dev_PoTPrefixSumT) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PoTPrefixSumT");
      return 1;
    } else { CUDAMEMPRINT(dev_PoTPrefixSumT,"cudaMalloc((void**) &dev_PoTPrefixSumT", NumDataPoints * PADVAL_PULSE,sizeof(*dev_PoTPrefixSumT)); };

  cu_err = cudaMalloc((void**) &dev_PoTPrefixSumG, sizeof(*dev_PoTPrefixSumG) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PoTPrefixSumG");
      return 1;
    } else { CUDAMEMPRINT(dev_PoTPrefixSumG,"cudaMalloc((void**) &dev_PoTPrefixSumG", NumDataPoints * PADVAL_PULSE,sizeof(*dev_PoTPrefixSumG)); };

  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_NormMaxPower, sizeof(*dev_NormMaxPower) * NumDataPoints / gauss_pot_length);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_NormMaxPower");
      return 1;
    } else { CUDAMEMPRINT(dev_NormMaxPower,"cudaMalloc((void**) &dev_NormMaxPower", NumDataPoints / gauss_pot_length,sizeof(*dev_NormMaxPower)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_flagged, sizeof(*dev_flagged) * NumDataPoints);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_flagged");
      return 1;
    } else { CUDAMEMPRINT(dev_flagged,"cudaMalloc((void**) &dev_flagged",NumDataPoints,sizeof(*dev_flagged)); };
  cudaAcc_init++;
  
  cu_err = cudaMalloc((void**) &dev_outputposition, sizeof(*dev_outputposition) * NumDataPoints);
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_outputposition");
      return 1;
    } else { CUDAMEMPRINT(dev_outputposition,"cudaMalloc((void**) &dev_outputposition",NumDataPoints,sizeof(*dev_outputposition)); };
  cudaAcc_init++;
  
//  dev_best_pot = (float*) dev_WorkData;
  dev_report_potP = dev_PoTP;

#ifdef PINNED
  cudaMallocHost((void **)&tmp_PoTP, NumDataPoints * sizeof(*tmp_PoTP) * 3 / 2);
  cudaMallocHost((void **)&tmp_PoTT, NumDataPoints * sizeof(*tmp_PoTT) * 3 / 2);
  cudaMallocHost((void **)&tmp_PoTG, NumDataPoints * sizeof(*tmp_PoTG) * 3 / 2);
  cudaMallocHost((void **)&best_PoTP, NumDataPoints * sizeof(*best_PoTP) * 3 / 2);
  cudaMallocHost((void **)&best_PoTG, NumDataPoints * sizeof(*best_PoTG) * 3 / 2);
#else
  tmp_PoT = (float*) malloc(NumDataPoints * sizeof(*tmp_PoT) * 3 / 2);
  best_PoT = (float*) malloc(NumDataPoints * sizeof(*best_PoT) * 3 / 2);
#endif
  
  CUDA_ACC_SAFE_CALL( (cu_err = cudaMalloc((void**) &dev_PowerSpectrumSumMax, sizeof(*dev_PowerSpectrumSumMax) * NumDataPoints*2 / 8)),true); // The ffts are at least 8 elems long
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_PowerSpectrumSumMax");
      return 1;
    } else { CUDAMEMPRINT(dev_PowerSpectrumSumMax,"cudaMalloc((void**) &dev_PowerSpectrumSumMax",NumDataPoints*2 / 8,sizeof(*dev_PowerSpectrumSumMax)); };
  cudaAcc_init++;
  

  cu_err = cudaMalloc((void**) &dev_tmp_potP, sizeof(*dev_PoTPrefixSumP) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_tmp_potP");
      return 1;
    } else { CUDAMEMPRINT(dev_tmp_potP,"cudaMalloc((void**) &dev_tmp_potP", NumDataPoints * PADVAL_PULSE,sizeof(*dev_PoTPrefixSumP)); };

  cu_err = cudaMalloc((void**) &dev_tmp_potT, sizeof(*dev_PoTPrefixSumT) * NumDataPoints * PADVAL_PULSE); // + 1/2 for find_pulse
  if(cudaSuccess != cu_err) 
    {
      CUDA_ACC_SAFE_CALL_NO_SYNC("cudaMalloc((void**) &dev_tmp_potT");
      return 1;
    } else { CUDAMEMPRINT(dev_tmp_potT,"cudaMalloc((void**) &dev_tmp_potT", NumDataPoints * PADVAL_PULSE,sizeof(*dev_PoTPrefixSumT)); };
  cudaAcc_init++;
  
  cudaAcc_NumDataPoints = NumDataPoints;
#ifdef PINNED
  cudaMallocHost((void **)&tmp_small_PoT, NumDataPoints / 8 * sizeof(*tmp_small_PoT));
  cudaMallocHost((void **)&PowerSpectrumSumMax, sizeof(*dev_PowerSpectrumSumMax) * NumDataPoints / 8);		
#else
  tmp_small_PoT = (float*) malloc(NumDataPoints / 8 * sizeof(*tmp_small_PoT));
  PowerSpectrumSumMax = (float3*) malloc(sizeof(*dev_PowerSpectrumSumMax) * NumDataPoints / 8);		
#endif
  
//  cu_err = cudaStreamCreate(&cudaAutocorrStream);
//  if(cudaSuccess != cu_err) 
//    {fprintf(stderr, "Autocorr stream create 0 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[0]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 0 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[1]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 1 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[2]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 2 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[3]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 3 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[4]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 4 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[5]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 5 failed\r\n"); return 1;} 

  cu_err = cudaStreamCreate(&cudaAutocorrStream[6]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 6 failed\r\n"); return 1;}

  cu_err = cudaStreamCreate(&cudaAutocorrStream[7]);
  if(cudaSuccess != cu_err) 
    {fprintf(stderr, "Autocorr stream create 7 failed\r\n"); return 1;}

  
  cudaStreamCreate(&fftstream0);
  cudaStreamCreate(&fftstream1);

  if(cudaAcc_initializeGaussfit(PoTInfo, gauss_pot_length, nsamples, gauss_null_chi_sq_thresh, gauss_chi_sq_thresh))
    {
      fprintf(stderr, "GaussFit Init failed...\n");
      return 1;
    }    
  
  if(cudaAcc_initialize_pulse_find(pulse_display_thresh, PulseThresh, PulseMax))
    {
      fprintf(stderr, "PulseFind Init failed...\n");
      return 1;		
    }
  
#ifdef _WIN32
#if CUDART_VERSION >= 3000
  if(gCudaDevProps.major >= 3)
    {
      //Kepler GPU, has complex clock setup, dig into nvapi on Windows.
      int crate = nvGetCurrentClock( gCudaDevProps.pciBusID, gCudaDevProps.pciDeviceID);		
      if(crate)
	fprintf(stderr, "\nGPU current clockRate = %d MHz\n\n",crate/1000);
      nvFreeAPI();
    }
#endif
#endif //_WIN32
  
  if(acfftlen && cudaAcc_InitializeAutocorrelation(acfftlen))
    {
      fprintf(stderr, "Not enough VRAM for Autocorrelations...\n");
      return 1;
    }
  
  //...All good
#if CUDART_VERSION >= 3000
  if(gCudaDevProps.major >= 2)
    {
      size_t threadlimit = 0;
      cudaThreadGetLimit(&threadlimit,cudaLimitStackSize);
      fprintf(stderr,"Thread call stack limit is: %dk\n", (int)threadlimit/1024);
      //if(threadlimit < 10240)
      //{
      //	cudaError_t cerr =  cudaThreadSetLimit(cudaLimitStackSize, 10240);
      //	if(cerr != cudaSuccess)
      //	{
      //		fprintf(stderr,"CudaThreadSetLimit() returned code %s\n", cerr);
      //	}
      //	else
      //	{
      //		threadlimit = 0;
      //		cudaThreadGetLimit(&threadlimit,cudaLimitStackSize);
      //		fprintf(stderr,"Cuda Thread Limit was adjusted to %dk\n", threadlimit/1024);
      //	}
      //}
    }
#endif
  
  CUDASYNC;  //clear any error codes.
  
  return 0;
}


int cudaAcc_InitializeAutocorrelation(int ac_fftlen)
{
  //    cudaError_t cu_err;
  cufftResult cu_errf;
  // Failure to initialise Cuda device memory for Autocorrelation isn't fatal, but we need to keep track of things...
  gCudaAutocorrelation = (ac_fftlen > 0); // initially assume we're going to do Autocorrelations on GPU if needed

//  cudaAutoCorr_plan = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[0] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[1] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[2] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[3] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[4] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[5] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[6] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
  cudaAutoCorr_plan[7] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"

  int ac_size = sizeof(float2)*ac_fftlen*4;
  
  if(gCudaAutocorrelation)
    {
      for(int i = 0; i < 8; i++)
	{   
	  cudaError_t err;
	  err = cudaMalloc((void **)&dev_AutoCorrIn[i], ac_size);
	  if(cudaSuccess != err)
	    {
	      fprintf(stderr, "cudaMalloc error %d", i);
	      return 1;
	    }
	  err = cudaMalloc((void **)&dev_AutoCorrOut[i], ac_size);
	  if(cudaSuccess != err)
	    {
	      fprintf(stderr, "cudaMalloc error %d", i);
	      return 1;
	    }
	  err = cudaMallocHost((void **)&blockSums[i], 131072/256*sizeof(float3));
	  if(cudaSuccess != err) 
	    {
	      fprintf(stderr, "cudaMallocHost blockSums %d failed", i);
	      return 1;
	    }
	}
    }
  
  if(gCudaAutocorrelation)
    {
      //      cu_errf = cufftPlan1d(&cudaAutoCorr_plan, ac_fftlen*4, CUFFT_C2C, 1); //4N FFT method
      //      if(CUFFT_SUCCESS != cu_errf) 
      //	fprintf(stderr,"Not enough room for autocorrelation CuFFT plan 0(4NFFT method)\n");
      //      cu_errf = cufftSetStream(cudaAutoCorr_plan, cudaAutocorrStream);
      //      if(CUFFT_SUCCESS != cu_errf) 
      //        fprintf(stderr, "cufftSetStream autocorr AutocorrStream failed");
      for(int i = 0; i < 8; i++)
	{
	  cu_errf = cufftPlan1d(&cudaAutoCorr_plan[i], ac_fftlen*4, CUFFT_C2C, 1); //4N FFT method
	  if(CUFFT_SUCCESS != cu_errf) 
	    fprintf(stderr,"Not enough room for autocorrelation CuFFT plan %d(4NFFT method)\n", i);


	  cu_errf = cufftSetStream(cudaAutoCorr_plan[i], cudaAutocorrStream[i]);
	  if(CUFFT_SUCCESS != cu_errf) 
	    fprintf(stderr, "cufftSetStream %d failed", i);
	  
	  cufftSetCompatibilityMode(cudaAutoCorr_plan[i], CUFFT_COMPATIBILITY_NATIVE);
	  dev_ac_partials[i] = (float3 *) dev_AutoCorrOut[i];
	}

      if(CUFFT_SUCCESS != cu_errf) 
	{
	  fprintf(stderr,"Not enough room for autocorrelation\n");
	  gCudaAutocorrelation = false;

	  for(int i = 0; i < 8; i++)
	    {
	      if(dev_AutoCorrOut[i]) cudaFree(dev_AutoCorrOut[i]);  // If we can't do the fft, won't be needing the output either.
	      if(dev_AutoCorrIn[i]) cudaFree(dev_AutoCorrIn[i]);  // If we can't do the output, won;t be needing the input either.
	      if(blockSums[i]) cudaFreeHost(blockSums[i]);  // If we can't do the output, won;t be needing the input either.
	      dev_AutoCorrIn[i]  = NULL;
	      dev_AutoCorrOut[i] = NULL;
	      blockSums[i] = NULL;
	      cudaAutoCorr_plan[i] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
	    }
	  return 1;
	}
#if CUDART_VERSION >= 3000
      
#endif
    }

  return 0;
}


void cudaAcc_free_AutoCorrelation()
{
  //  if(cudaAutoCorr_plan) cufftDestroy(cudaAutoCorr_plan);

  gCudaAutocorrelation = false;

  for(int i = 0; i < 8; i++)
    {
      if(cudaAutoCorr_plan[i]) cufftDestroy(cudaAutoCorr_plan[i]);

      if(dev_AutoCorrOut[i]) cudaFree(dev_AutoCorrOut[i]);
      if(dev_AutoCorrIn[i])  cudaFree(dev_AutoCorrIn[i]);  
      if(blockSums[i])       cudaFreeHost(blockSums[i]);
      
      dev_AutoCorrIn[i]  = NULL;
      dev_AutoCorrOut[i] = NULL;
      blockSums[i]       = NULL;

      cudaAutoCorr_plan[i] = 0; // cufftHandle is not a pointer. Cannot be set to "NULL"
    }

  return;
}

#define CF(_ptr) do { cudaFree(_ptr);_ptr = NULL; } while (0);


void cudaAcc_free() 
{
  fprintf(stderr,"cudaAcc_free() called...\n");
  if(!cudaAcc_init) return;
  fprintf(stderr,"cudaAcc_free() running...\n");
  cudaAcc_free_pulse_find();
  fprintf(stderr,"cudaAcc_free() PulseFind freed...\n");
  cudaAcc_free_Gaussfit();
  fprintf(stderr,"cudaAcc_free() Gaussfit freed...\n");
  cudaAcc_free_AutoCorrelation();
  fprintf(stderr,"cudaAcc_free() AutoCorrelation freed...\n");
  
  //cudaStreamDestroy(fftstream1);
  //cudaStreamDestroy(fftstream0);
  
  switch(cudaAcc_init) 
    {
    case 16:
      CF(dev_PulseResults);
    case 15:
      CF(dev_TripletResults);
    case 14:
      CF(dev_tmp_potT);
      CF(dev_tmp_potP);
    case 13:
      CF(dev_PowerSpectrumSumMax);
    case 12:
      CF(dev_outputposition);
    case 11:
      CF(dev_flagged);
    case 10:
      CF(dev_NormMaxPower);
    case 9:
      CF(dev_PoTPrefixSumG);
      CF(dev_PoTPrefixSumP);
    case 8:
      CF(dev_PoTP);
      CF(dev_PoTG);
    case 7:
      CF(dev_GaussFitResults);
    case 6:
      CF(dev_t_PowerSpectrumP); // pulse
      CF(dev_t_PowerSpectrumT); // triplet
      CF(dev_t_PowerSpectrumG); // gauss
    case 5:
      CF(dev_PowerSpectrum);
    case 4:
      CF(dev_WorkData);
      CF(dev_best_potP);
    case 3:
      CF(dev_flagT);
      CF(dev_flagG);
    case 2:
      CF(dev_cx_ChirpDataArray);
    case 1:
      CF(dev_cx_DataArray);
    case 0:
    default:
      //CUDA_ACC_SAFE_CALL(CF(dev_GaussFitResultsReordered));
      //CUDA_ACC_SAFE_CALL(CF(dev_GaussFitResultsReordered2));
      //cudaAcc_deallocBlockSums(); // scans are not used at the moment
#ifdef PINNED
      fprintf(stderr, "1");
      cudaFreeHost(GaussFitResults);
      fprintf(stderr, "2");
      cudaFreeHost(TripletResults);
      fprintf(stderr, "3");
      cudaFreeHost(PulseResults);
      fprintf(stderr, "4");
      cudaFreeHost(tmp_small_PoT);
      fprintf(stderr, "5");
      cudaFreeHost(tmp_PoTP);
      fprintf(stderr, "6");
      cudaFreeHost(tmp_PoTT);
      fprintf(stderr, "7");
      cudaFreeHost(tmp_PoTG);
      fprintf(stderr, "8");
      cudaFreeHost(best_PoTP);
      fprintf(stderr, "9");
      cudaFreeHost(best_PoTG);
      fprintf(stderr, "10");
#else
      free(GaussFitResults);	
      free(tmp_small_PoT);
#endif
    }
  
  cudaAcc_fft_free();
  fprintf(stderr, "10½");
  cudaAcc_init = 0;
  cmem_rtotal = 0;
#if(CUDART_VERSION >= 4000)
  fprintf(stderr, "11");
  cudaDeviceReset();
  fprintf(stderr, "12");
#else
  cudaThreadExit();
#endif
  fprintf(stderr,"cudaAcc_free() DONE.\n");
  fprintf(stderr, "13");

}
#endif
