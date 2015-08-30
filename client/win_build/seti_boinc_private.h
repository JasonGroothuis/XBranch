
#ifndef SETI_BOINC_PRIVATE_H
#define SETI_BOINC_PRIVATE_H
/* VERSION DEFINITIONS */
#define VER_STRING	"6.98.0.0"
#define VER_MAJOR	6
#define VER_MINOR	98
#define VER_RELEASE	0
#define VER_BUILD	0
#define COMPANY_NAME	"Jason Groothuis bSc"
#define FILE_VERSION	"x41zc"
#define FILE_DESCRIPTION	"setiathome_enhanced"
#define INTERNAL_NAME	"setiathome_enhanced"
#define LEGAL_COPYRIGHT	"Copyright (c) 2012, Jason Richard Groothuis bSc, All Rights Reserved"
#define LEGAL_TRADEMARKS	""
#ifdef WIN64
	#if defined(CUDA65)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_winx64_cuda65.exe"
	#else
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_winx64_cudaxx.exe"
	#endif
#else
	#if defined(CUDA65)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda65.exe"
	#elif defined(CUDA60)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda60.exe"
	#elif defined(CUDA55)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda55.exe"
	#elif defined(CUDA50)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda50.exe"
	#elif defined(CUDA42)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda42.exe"
	#elif defined(CUDA41)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda41.exe"
	#elif defined(CUDA40)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda40.exe"
	#elif defined(CUDA32)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda32.exe"
	#elif defined(CUDA23)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda23.exe"
	#elif defined(CUDA22)
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cuda22.exe"
	#else
		#define ORIGINAL_FILENAME	"Lunatics_x41zc_win32_cudaxx.exe"
	#endif
#endif
#define PRODUCT_NAME	"setiathome_enhanced"
#define PRODUCT_VERSION	"x41zc"

#endif /*SETI_BOINC_PRIVATE_H*/
