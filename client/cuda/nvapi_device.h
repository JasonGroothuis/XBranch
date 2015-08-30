#ifndef __NVAPI_DEVICE_H
	#define __NVAPI_DEVICE_H
#ifdef _WIN32
	extern void nvInitAPI();
	extern void nvFreeAPI();
	extern int nvGetCurrentClock( int pcibusid, int pcislotid );
#endif _WIN32
#endif // !__NVAPI_DEVICE_H