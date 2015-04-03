#ifndef _CUDA_ACC_UTILITIES_H
#define _CUDA_ACC_UTILITIES_H

extern int cudaAcc_SafeCall_No_Sync(cudaError_t err, char* errMessage, char* file, int line);
extern int cudaAcc_SafeCall_Sync(cudaError_t err, char* errMessage, char* file, int line);
extern int cudaAcc_SafeCall_No_Sync_wExit(cudaError_t err, char* errMessage, char* file, int line);

extern int cudaAcc_SafeCall_No_Sync(char* errMessage, char* file, int line);
extern int cudaAcc_SafeCall_Sync(char* errMessage, char* file, int line);

extern int cudaAcc_peekLastError();
extern int cudaAcc_peekLastError_alliscool();

//#define CUDAACC_SYNC_ALL             /* For logging the performance of every kernel */

#ifdef _WIN32

extern const char* cufft_codestr[];
extern volatile bool worker_thread_exit_ack;

#define CUFFT_ERRORSTR(_code) (_code <= 9) ? cufft_codestr[_code]:"UNKNOWN"

    #define CUFFT_SAFE_CALL(_call) {                                             \
	    cufftResult err = _call;						 \
		if( CUFFT_SUCCESS != err) {                                              \
			fprintf(stderr, "CUFFT error in file '%s' in line %i. code %s\n",    \
					__FILE__, __LINE__,CUFFT_ERRORSTR(err));                     \
			CUDASYNC; cudaAcc_free(); fprintf(stderr,"Cuda sync'd & freed.\n");  \
			fflush(NULL);														 \
			fprintf(stderr,"  Preemptively Acknowledging temporary exit -> ");	\
				worker_thread_exit_ack = true;									\
				boinc_temporary_exit(180,"CUFFT Error");										\
		}																		\
	}
#else
	#define CUFFT_SAFE_CALL( call) {                                           \
		 call;                                                  \
                cudaError_t err = cudaGetLastError();	\
		if( CUFFT_SUCCESS != err) {                                              \
		fprintf(stderr, "CUFFT error in file '%s' in line %i.\n",            \
					__FILE__, __LINE__);                                         \
			exit(EXIT_FAILURE);                                                  \
	} }
#endif

// Use for cuda runtime functions like cudaMalloc, cudaMemcpy etc.
#ifdef _WIN32
	#define TEMP_EXIT(_delay) do {																		\
		if (gSetiUseCudaDevice) { CUDASYNC; cudaAcc_free(); fprintf(stderr,"Cuda sync'd & freed.\n"); } \
			fprintf(stderr,"Preemptively acknowledging a safe temporary exit->\n");						\
			fflush(NULL);																				\
			worker_thread_exit_ack = true;																\
			boinc_temporary_exit(180,"Cuda runtime, memory related failure, threadsafe temporary Exit");																	\
	} while (0);
	//Jason: kernel launch error checking
	#define CUDA_ACC_SAFE_LAUNCH(_launch,_exitonfail) do {									\
				cudaError_t err = cudaGetLastError();										\
				if ( err != cudaSuccess ) { fprintf(stderr,									\
					"uncaptured error before launch %s, file %s, line %d: %s\n", 			\
					#_launch,__FILE__,__LINE__,cudaGetErrorString(err));					\
					if (_exitonfail){ fprintf(stderr,"Exiting\n"); TEMP_EXIT(180); }	\
				}																			\
				_launch;																	\
				err = cudaGetLastError();													\
				if ( err != cudaSuccess ) { fprintf(stderr,									\
					"Error on launch %s, file %s, line %d: %s\n",							\
					#_launch,__FILE__,__LINE__,cudaGetErrorString(err));					\
					if (_exitonfail){ fprintf(stderr,"Exiting\n"); TEMP_EXIT(180); }	\
				}																			\
			} while (0);
	//Jason: On Windows, using cudaGetLastError() so that error get cleared, before & after a call
	#define CUDA_ACC_SAFE_CALL(_call,_exitonfail)   do {									\
				cudaError_t err = cudaGetLastError();										\
				if ( err != cudaSuccess ) { fprintf(stderr,									\
					"uncaptured error before call %s, file %s, line %d: %s\n", 				\
					#_call,__FILE__,__LINE__,cudaGetErrorString(err));						\
					if (_exitonfail){ fprintf(stderr,"Exiting\n"); TEMP_EXIT(180); }		\
				}																			\
				err = _call;																\
				if ( err != cudaSuccess ) { fprintf(stderr,									\
					"Error on call %s, file %s, line %d: %s\n",								\
						#_call,__FILE__,__LINE__,cudaGetErrorString(err));					\
					if (_exitonfail){ fprintf(stderr,"Exiting\n"); TEMP_EXIT(180); }		\
				}																			\
			} while (0);
#else
	#define CUDA_ACC_SAFE_LAUNCH(_launch,_exitonfail) _launch;
	#define CUDA_ACC_SAFE_CALL(_call,_exitonfail)  do {										\
		cudaError_t err = _call;															\
		cudaAcc_SafeCall_No_Sync_wExit( err, #_call, __FILE__, __LINE__);					\
		} while (0)
#endif

#define PADVAL		 9/8
#define PADVAL_PULSE 3/2

#ifdef _WIN32
#define CUDAMEMPRINT(_ptr,_name,_elemcount,_elemsize)  do { \
	cmem_rtotal += _elemcount*_elemsize; \
/*	fprintf(stderr,"VRAM: %50s,%8dx%8dbytes = %10dbytes, offs256=%d, rtotal=%10dbytes\n",*/	\
/*		_name,_elemcount,_elemsize,_elemcount*_elemsize,(unsigned)_ptr & 255,cmem_rtotal);*/ \
	} while (0)
#else
#define CUDAMEMPRINT(_ptr,_name,_elemcount,_elemsize)  do { \
	cmem_rtotal += _elemcount*_elemsize; \
/*	fprintf(stderr,"VRAM: %50s,%8dx%8dbytes = %10dbytes, rtotal=%10dbytes\n",  */ \
/*		_name,(int)_elemcount,(int)_elemsize,(int)(_elemcount*_elemsize),(int)cmem_rtotal); */ \
	} while (0)
#endif

// Use this version to gein peak performance but waste one CPU Core.
#  define CUDA_ACC_SAFE_CALL_NO_SYNC(errMessage) do {				\
    cudaAcc_SafeCall_No_Sync(errMessage, __FILE__, __LINE__);		\
    } while (0)

#  define CUDA_ACC_SAFE_CALL_LOW_SYNC(errMessage) do {				\
    cudaAcc_SafeCall_Sync(errMessage, __FILE__, __LINE__);		\
    } while (0)

#  define CUDA_ACC_SAFE_CALL_SYNC(errMessage) do {					\
    cudaAcc_SafeCall_Sync(errMessage, __FILE__, __LINE__);			\
    } while (0)

#ifdef CUDAACC_SYNC_ALL
#undef CUDA_ACC_SAFE_CALL_NO_SYNC
#undef CUDA_ACC_SAFE_CALL_LOW_SYNC
#undef CUDA_ACC_SAFE_CALL_SYNC

// For logging the performance of every kernel
#  define CUDA_ACC_SAFE_CALL_NO_SYNC(errMessage) do {				\
    cudaAcc_SafeCall_Sync(errMessage, __FILE__, __LINE__);		\
    } while (0)

#  define CUDA_ACC_SAFE_CALL_LOW_SYNC(errMessage) do {				\
    cudaAcc_SafeCall_Sync(errMessage, __FILE__, __LINE__);		\
    } while (0)

#  define CUDA_ACC_SAFE_CALL_SYNC(errMessage) do {					\
    cudaAcc_SafeCall_Sync(errMessage, __FILE__, __LINE__);			\
    } while (0)

#endif

#endif //_CUDA_ACC_UTILITIES_H
