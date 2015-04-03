#ifdef _WIN32
	#include <windows.h>
#endif
	#include "app_ipc.h"
	#include "cuda_runtime_api.h"
	#include "cuda/cudaAcceleration.h"
	#include "confsettings.h"

#ifndef max
	#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
	#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

//Conservative Boinc default for Cuda application process priority
priority_t confSetPriority = pt_BELOWNORMAL;

//defaults for long pulsefinds, need to be set at runtime in InitConfig()
int pfBlocksPerSM = 0;
int pfPeriodsPerLaunch = 0;

extern APP_INIT_DATA app_init_data;

void initConfig(int pcibusid,int pcislotid)
{
// mbcuda.cfg only used for Windows at the moment, curently uses Windows API
// should switch to standard C functions, if Linux/MAC needs similar functrionality
// Current non-Windows path just sets sensible defaults.

#ifdef _WIN32
	_TCHAR setstr[64];
	int setint;
	#if CUDART_VERSION >= 3000
		_TCHAR pcisection[64];
	#endif

	char cfgname[266];
    boinc_resolve_filename(".//mbcuda.cfg",cfgname,266);

	#if CUDART_VERSION >= 3000
		sprintf((char*)pcisection,"bus%1dslot%1d",pcibusid,pcislotid);
	#endif

  #if CUDART_VERSION >= 3000
	if ( GetPrivateProfileString(pcisection,TEXT("processpriority"),TEXT(""),setstr,64,cfgname) )
	{
		fprintf(stderr,"mbcuda.cfg, matching pci device processpriority key detected\n");
		if (!strncmp(setstr,"bel",3)) confSetPriority = pt_BELOWNORMAL;
		else if (!strncmp(setstr,"nor",3)) confSetPriority = pt_NORMAL;
		else if (!strncmp(setstr,"abo",3)) confSetPriority = pt_ABOVENORMAL;
		else if (!strncmp(setstr,"hig",3)) confSetPriority = pt_HIGH;
	} else
  #endif
	if ( GetPrivateProfileString(TEXT("mbcuda"),TEXT("processpriority"),TEXT(""),setstr,64,cfgname) )
	{
		fprintf(stderr,"mbcuda.cfg, processpriority key detected\n");
		if (!strncmp(setstr,"bel",3)) confSetPriority = pt_BELOWNORMAL;
		else if (!strncmp(setstr,"nor",3)) confSetPriority = pt_NORMAL;
		else if (!strncmp(setstr,"abo",3)) confSetPriority = pt_ABOVENORMAL;
		else if (!strncmp(setstr,"hig",3)) confSetPriority = pt_HIGH;
	} 

	int def;
	def = (gCudaDevProps.major < 2) ? DEFAULT_PFBLOCKSPERSM:DEFAULT_PFBLOCKSPERSM_FERMI;
	setint = 0;
  #if CUDART_VERSION >= 3000
	if ( (setint = GetPrivateProfileInt(pcisection,TEXT("pfblockspersm"),0,cfgname)) > 0)
	{
		fprintf(stderr,"mbcuda.cfg, matching pci device pfblockspersm key detected\n");
		pfBlocksPerSM = max(1,min(16,setint)); //Sane Limits
	} else 
  #endif
	if ( (setint = GetPrivateProfileInt(TEXT("mbcuda"),TEXT("pfblockspersm"),0,cfgname)) > 0)
	{
		fprintf(stderr,"mbcuda.cfg, Global pfblockspersm key being used for this device\n");
		pfBlocksPerSM = max(1,min(16,setint)); //Sane Limits
	} else if ( pfBlocksPerSM == 0 )
	{
		pfBlocksPerSM = def;
	}
	if (gCudaDevProps.major < 2) fprintf(stderr,"pulsefind: blocks per SM %d %s\n", pfBlocksPerSM, (pfBlocksPerSM == def) ? "(Pre-Fermi default)":"" );
	else fprintf(stderr,"pulsefind: blocks per SM %d %s\n", pfBlocksPerSM, (pfBlocksPerSM == def) ? "(Fermi or newer default)":"" );

	def = DEFAULT_PFPERIODSPERLAUNCH;
	setint = 0;
  #if CUDART_VERSION >= 3000
	if ( (setint = GetPrivateProfileInt(pcisection,TEXT("pfperiodsperlaunch"),0,cfgname)) > 0 )
	{
		fprintf(stderr,"mbcuda.cfg, matching pci device pfperiodsperlaunch key detected\n");
		pfPeriodsPerLaunch = max(1,min(1000,setint)); // Sane Limits
	} else
  #endif
	if ( (setint = GetPrivateProfileInt(TEXT("mbcuda"),TEXT("pfperiodsperlaunch"),0,cfgname)) > 0 )
	{
		fprintf(stderr,"mbcuda.cfg, Global pfperiodsperlaunch key being used for this device\n");
		pfPeriodsPerLaunch = max(1,min(1000,setint)); // Sane Limits
	} 
	else if ( pfPeriodsPerLaunch == 0 )
	{
		pfPeriodsPerLaunch = def;
	}
	fprintf(stderr,"pulsefind: periods per launch %d %s\n", pfPeriodsPerLaunch, (pfPeriodsPerLaunch == def) ? "(default)":"" );
#else // not win, is other
	confSetPriority = pt_BELOWNORMAL;
	if(pfBlocksPerSM == 0)
	  pfBlocksPerSM = (gCudaDevProps.major < 2) ? DEFAULT_PFBLOCKSPERSM:DEFAULT_PFBLOCKSPERSM_FERMI;
	else
	  fprintf(stderr, "Using pfb = %d from command line args\n", pfBlocksPerSM);
	if(pfPeriodsPerLaunch == 0)
	  pfPeriodsPerLaunch = DEFAULT_PFPERIODSPERLAUNCH;
	else
	  fprintf(stderr, "Using pfp = %d from command line args\n", pfPeriodsPerLaunch);
#endif //_WIN32

	return;
}
