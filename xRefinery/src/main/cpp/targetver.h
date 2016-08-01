#if !defined(__TARGETVER_H)
	#define __TARGETVER_H 1

	#define WIN32_LEAN_AND_MEAN
	#include <WinSDKVer.h>
	#define  NTDDI_VERSION  NTDDI_WINXP 
	#define WINVER _WIN32_WINNT_WINXP 
	#define _WIN32_WINNT _WIN32_WINNT_WINXP 	
	
	#include <SDKDDKVer.h>
#endif