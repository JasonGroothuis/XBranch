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

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "cudaAcc_utilities.h"

const char* cufft_codestr[] = {
  "CUFFT_SUCCESS        = 0x0",
  "CUFFT_INVALID_PLAN   = 0x1",
  "CUFFT_ALLOC_FAILED   = 0x2",
  "CUFFT_INVALID_TYPE   = 0x3",
  "CUFFT_INVALID_VALUE  = 0x4",
  "CUFFT_INTERNAL_ERROR = 0x5",
  "CUFFT_EXEC_FAILED    = 0x6",
  "CUFFT_SETUP_FAILED   = 0x7",
  "CUFFT_INVALID_SIZE   = 0x8",
  "CUFFT_UNALIGNED_DATA = 0x9"
};

int cudaAcc_SafeCall_No_Sync(cudaError_t err, char* errMessage, char* file, int line) {
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error '%s' in file '%s' in line %i : %s.\n", errMessage, file, line, cudaGetErrorString( err));
		//exit(EXIT_FAILURE);
	}
	return 0;
}

int cudaAcc_SafeCall_No_Sync_wExit(cudaError_t err, char* errMessage, char* file, int line) {
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error '%s' in file '%s' in line %i : %s.\n", errMessage, file, line, cudaGetErrorString( err));
		exit(EXIT_FAILURE);
	}
	return 0;
}

int cudaAcc_SafeCall_Sync(cudaError_t err, char* errMessage, char* file, int line) {
	cudaAcc_SafeCall_No_Sync(err, errMessage, file, line);
	err = CUDASYNC;
	return cudaAcc_SafeCall_No_Sync(err, errMessage, file, line);
}

int cudaAcc_SafeCall_No_Sync(char* errMessage, char* file, int line) {
	return cudaAcc_SafeCall_No_Sync_wExit(cudaGetLastError(), errMessage, file, line);
}

int cudaAcc_SafeCall_Sync(char* errMessage, char* file, int line) {
	return cudaAcc_SafeCall_Sync(cudaGetLastError(),errMessage, file, line);
}

int cudaAcc_peekLastError()
{
	cudaError_t  lasterr = cudaSuccess;

#if CUDART_VERSION >= 3000
	lasterr = cudaPeekAtLastError();

	switch ( lasterr )
	{
		case cudaSuccess: fprintf(stderr,"Success - No errors.\n"); break;
		case cudaErrorMissingConfiguration: fprintf(stderr,"Missing configuration error.\n"); break;
		case cudaErrorMemoryAllocation: fprintf(stderr,"Memory allocation error.\n"); break;
		case cudaErrorInitializationError: fprintf(stderr,"Initialization error.\n"); break;
		case cudaErrorLaunchFailure: fprintf(stderr,"Launch failure.\n"); break;
		case cudaErrorPriorLaunchFailure: fprintf(stderr,"Prior launch failure.\n"); break;
		case cudaErrorLaunchTimeout: fprintf(stderr,"Launch timeout error.\n"); break;
		case cudaErrorLaunchOutOfResources: fprintf(stderr,"Launch out of resources error.\n"); break;
		case cudaErrorInvalidDeviceFunction: fprintf(stderr,"Invalid device function.\n"); break;
		case cudaErrorInvalidConfiguration: fprintf(stderr,"Invalid configuration.\n"); break;
		case cudaErrorInvalidDevice: fprintf(stderr,"Invalid device.\n"); break;
		case cudaErrorInvalidValue: fprintf(stderr,"Invalid value.\n"); break;
		case cudaErrorInvalidPitchValue: fprintf(stderr,"Invalid pitch value.\n"); break;
		case cudaErrorInvalidSymbol: fprintf(stderr,"Invalid symbol.\n"); break;
		case cudaErrorMapBufferObjectFailed: fprintf(stderr,"Map buffer object failed.\n"); break;
		case cudaErrorUnmapBufferObjectFailed: fprintf(stderr,"Unmap buffer object failed.\n"); break;
		case cudaErrorInvalidHostPointer: fprintf(stderr,"Invalid host pointer.\n"); break;
		case cudaErrorInvalidDevicePointer: fprintf(stderr,"Invalid device pointer.\n"); break;
		case cudaErrorInvalidTexture: fprintf(stderr,"Invalid texture.\n"); break;
		case cudaErrorInvalidTextureBinding: fprintf(stderr,"Invalid texture binding.\n"); break;
		case cudaErrorInvalidChannelDescriptor: fprintf(stderr,"Invalid channel descriptor.\n"); break;
		case cudaErrorInvalidMemcpyDirection: fprintf(stderr,"Invalid memcpy direction.\n"); break;
		default: fprintf(stderr,"Unknown Error.\n"); break;
	}
#endif
	return lasterr;
}

int cudaAcc_peekLastError_alliscool()
{
	cudaError_t  lasterr = cudaSuccess;

#if CUDART_VERSION >= 3000
	lasterr = cudaPeekAtLastError();

	switch ( lasterr )
	{
		case cudaSuccess: break;
		case cudaErrorMissingConfiguration: fprintf(stderr,"Missing configuration error.\n"); break;
		case cudaErrorMemoryAllocation: fprintf(stderr,"Memory allocation error.\n"); break;
		case cudaErrorInitializationError: fprintf(stderr,"Initialization error.\n"); break;
		case cudaErrorLaunchFailure: fprintf(stderr,"Launch failure.\n"); break;
		case cudaErrorPriorLaunchFailure: fprintf(stderr,"Prior launch failure.\n"); break;
		case cudaErrorLaunchTimeout: fprintf(stderr,"Launch timeout error.\n"); break;
		case cudaErrorLaunchOutOfResources: fprintf(stderr,"Launch out of resources error.\n"); break;
		case cudaErrorInvalidDeviceFunction: fprintf(stderr,"Invalid device function.\n"); break;
		case cudaErrorInvalidConfiguration: fprintf(stderr,"Invalid configuration.\n"); break;
		case cudaErrorInvalidDevice: fprintf(stderr,"Invalid device.\n"); break;
		case cudaErrorInvalidValue: fprintf(stderr,"Invalid value.\n"); break;
		case cudaErrorInvalidPitchValue: fprintf(stderr,"Invalid pitch value.\n"); break;
		case cudaErrorInvalidSymbol: fprintf(stderr,"Invalid symbol.\n"); break;
		case cudaErrorMapBufferObjectFailed: fprintf(stderr,"Map buffer object failed.\n"); break;
		case cudaErrorUnmapBufferObjectFailed: fprintf(stderr,"Unmap buffer object failed.\n"); break;
		case cudaErrorInvalidHostPointer: fprintf(stderr,"Invalid host pointer.\n"); break;
		case cudaErrorInvalidDevicePointer: fprintf(stderr,"Invalid device pointer.\n"); break;
		case cudaErrorInvalidTexture: fprintf(stderr,"Invalid texture.\n"); break;
		case cudaErrorInvalidTextureBinding: fprintf(stderr,"Invalid texture binding.\n"); break;
		case cudaErrorInvalidChannelDescriptor: fprintf(stderr,"Invalid channel descriptor.\n"); break;
		case cudaErrorInvalidMemcpyDirection: fprintf(stderr,"Invalid memcpy direction.\n"); break;
		default: fprintf(stderr,"Unknown Error.\n"); break;
	}
#endif

	return lasterr;
}

#endif //USE_CUDA
