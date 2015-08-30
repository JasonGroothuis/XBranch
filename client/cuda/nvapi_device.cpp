#include "nvapi_device.h"
#include "nvapi_types.h"

static bool nvapiInitialised = false;
static HMODULE hNvApiDll = NULL;

static nvapi_QueryInterface             pnvapi_QueryInterface             = NULL;
static NvAPI_Initialize                 pNvAPI_Initialize                 = NULL;
static NvAPI_Unload						pNvAPI_Unload	                  = NULL;
static NvAPI_EnumPhysicalGPUs           pNvAPI_EnumPhysicalGPUs           = NULL;
static NvAPI_GPU_GetAllClockFrequencies pNvAPI_GPU_GetAllClockFrequencies = NULL;
static NvAPI_GetDisplayDriverVersion    pNvAPI_GetDisplayDriverVersion    = NULL;
static NvAPI_GPU_GetBusId				pNvAPI_GPU_GetBusId				  = NULL;
static NvAPI_GPU_GetBusSlotId			pNvAPI_GPU_GetBusSlotId			  = NULL;

NvPhysicalGpuHandle NvHandles[NVAPI_MAX_PHYSICAL_GPUS];
ULONG GpuCount;
NV_GPU_CLOCK_FREQUENCIES gclocksall[NVAPI_MAX_PHYSICAL_GPUS];
NV_PCILOC gpciloc[NVAPI_MAX_PHYSICAL_GPUS];

void nvInitAPI()
{
	if (nvapiInitialised) return;

	if (!hNvApiDll)
	{
		hNvApiDll = LoadLibrary(_TEXT("nvapi.dll"));
		if (!hNvApiDll)	goto Failed;
	}

    pnvapi_QueryInterface = (nvapi_QueryInterface)GetProcAddress(hNvApiDll, "nvapi_QueryInterface");
    if (!pnvapi_QueryInterface) goto Failed;

    pNvAPI_Initialize = (NvAPI_Initialize)pnvapi_QueryInterface(0x0150E828);
    if (!pNvAPI_Initialize) goto Failed;

    pNvAPI_Unload = (NvAPI_Unload)pnvapi_QueryInterface(0xD22BDD7E);
    if (!pNvAPI_Unload) goto Failed;

    pNvAPI_EnumPhysicalGPUs = (NvAPI_EnumPhysicalGPUs)pnvapi_QueryInterface(0xE5AC921F);
    if (!pNvAPI_EnumPhysicalGPUs) goto Failed;

    pNvAPI_GetDisplayDriverVersion = (NvAPI_GetDisplayDriverVersion)pnvapi_QueryInterface(0xF951A4D1);
    if (!pNvAPI_GetDisplayDriverVersion) goto Failed;

	pNvAPI_GPU_GetAllClockFrequencies = (NvAPI_GPU_GetAllClockFrequencies)pnvapi_QueryInterface(0xDCB616C3);
    if (!pNvAPI_GPU_GetAllClockFrequencies) goto Failed;

    pNvAPI_GPU_GetBusId = (NvAPI_GPU_GetBusId)pnvapi_QueryInterface(0x1BE0B8E5);
    if (!pNvAPI_GPU_GetBusId) goto Failed;

    pNvAPI_GPU_GetBusSlotId = (NvAPI_GPU_GetBusSlotId)pnvapi_QueryInterface(0x2A0A350F);
    if (!pNvAPI_GPU_GetBusSlotId) goto Failed;

    if (pNvAPI_Initialize() != NVAPI_OK) goto Failed;

    if (pNvAPI_EnumPhysicalGPUs(NvHandles, &GpuCount) != NVAPI_OK) goto Failed;

	nvapiInitialised = true;
	return;
Failed:
    nvFreeAPI();
    return;
}

void nvFreeAPI()
{
	if (!nvapiInitialised) return;

	if (pNvAPI_Unload) pNvAPI_Unload();

    if (hNvApiDll) FreeLibrary(hNvApiDll);
    hNvApiDll = NULL;

    pNvAPI_Initialize                 = NULL;
	pNvAPI_Unload					  = NULL;
    pNvAPI_EnumPhysicalGPUs           = NULL;
	pNvAPI_GPU_GetAllClockFrequencies = NULL;
    pNvAPI_GetDisplayDriverVersion    = NULL;
	pNvAPI_GPU_GetBusId				  = NULL;
	pNvAPI_GPU_GetBusSlotId			  = NULL;

	nvapiInitialised = false;
}

int nvGetCurrentClock( int pcibusid, int pcislotid )
{
	if (!nvapiInitialised) nvInitAPI();
	if (!nvapiInitialised) return 0;

	for ( int i = 0; i< (int)GpuCount; i++ )
	{
		NvU32 bid,did;
		/* Get GPU clock strings (GetAllClockFrequencies Method)*/
		gclocksall[i].version = MAKE_NVAPI_VERSION(NV_GPU_CLOCK_FREQUENCIES,2);
		gclocksall[i].ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;

		if ( (pNvAPI_GPU_GetBusId)
			&& (pNvAPI_GPU_GetBusId(NvHandles[i],&bid) == NVAPI_OK) && (bid == pcibusid)
			&& (pNvAPI_GPU_GetBusSlotId)
			&& (pNvAPI_GPU_GetBusSlotId(NvHandles[i],&did) == NVAPI_OK) && (did == pcislotid) 
			&& (pNvAPI_GPU_GetAllClockFrequencies)
			&& (pNvAPI_GPU_GetAllClockFrequencies(NvHandles[i], &gclocksall[i]) == NVAPI_OK) )
		{
			if ( gclocksall[i].domain[0].bIsPresent )
				return gclocksall[i].domain[0].frequency;
			else
				return 0;
		}
	}
	return 0;
}
