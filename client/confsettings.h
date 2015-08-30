#ifndef __CONFSETTINGS_H
  #define __CONFSETTINGS_H

    enum priority_t { pt_BELOWNORMAL=0,pt_NORMAL,pt_ABOVENORMAL,pt_HIGH };

//  For non-Windows, initConfig() will just setup defaults.  
//  initConfig should switch to use stadard C fucntions, so we can gave some configurability on Linux/Mac as well
	extern void initConfig(int pcibusid,int pcislotid);
	
	extern priority_t confSetPriority;
	extern int pfBlocksPerSM;
	extern int pfPeriodsPerLaunch;

	#define DEFAULT_PFBLOCKSPERSM		1
	#define DEFAULT_PFBLOCKSPERSM_FERMI 4
	#define DEFAULT_PFPERIODSPERLAUNCH	100
#endif // !CONFSETTINGS_H