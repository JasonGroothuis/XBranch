#ifndef __NVAPI_TYPES_H
	#define __NVAPI_TYPES_H
#ifdef _WIN32
	//#define NVAPI_INTERFACE extern NvAPI_Status __cdecl
	typedef enum _NvAPI_Status
	{
		NVAPI_OK                                    =  0,
		NVAPI_ERROR                                 = -1,
		NVAPI_LIBRARY_NOT_FOUND                     = -2,
		NVAPI_NO_IMPLEMENTATION                     = -3,
		NVAPI_API_NOT_INITIALIZED                   = -4,
		NVAPI_INVALID_ARGUMENT                      = -5,
		NVAPI_NVIDIA_DEVICE_NOT_FOUND               = -6,
		NVAPI_END_ENUMERATION                       = -7,
		NVAPI_INVALID_HANDLE                        = -8,
		NVAPI_INCOMPATIBLE_STRUCT_VERSION           = -9,
		NVAPI_HANDLE_INVALIDATED                    = -10,
		NVAPI_OPENGL_CONTEXT_NOT_CURRENT            = -11,
		NVAPI_INVALID_POINTER                       = -14,
		NVAPI_NO_GL_EXPERT                          = -12,
		NVAPI_INSTRUMENTATION_DISABLED              = -13,
		NVAPI_NO_GL_NSIGHT                          = -15,
		NVAPI_EXPECTED_LOGICAL_GPU_HANDLE           = -100,
		NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE          = -101,
		NVAPI_EXPECTED_DISPLAY_HANDLE               = -102,
		NVAPI_INVALID_COMBINATION                   = -103,
		NVAPI_NOT_SUPPORTED                         = -104,
		NVAPI_PORTID_NOT_FOUND                      = -105,
		NVAPI_EXPECTED_UNATTACHED_DISPLAY_HANDLE    = -106,
		NVAPI_INVALID_PERF_LEVEL                    = -107,
		NVAPI_DEVICE_BUSY                           = -108,
		NVAPI_NV_PERSIST_FILE_NOT_FOUND             = -109,
		NVAPI_PERSIST_DATA_NOT_FOUND                = -110,
		NVAPI_EXPECTED_TV_DISPLAY                   = -111,
		NVAPI_EXPECTED_TV_DISPLAY_ON_DCONNECTOR     = -112,
		NVAPI_NO_ACTIVE_SLI_TOPOLOGY                = -113,
		NVAPI_SLI_RENDERING_MODE_NOTALLOWED         = -114,
		NVAPI_EXPECTED_DIGITAL_FLAT_PANEL           = -115,
		NVAPI_ARGUMENT_EXCEED_MAX_SIZE              = -116,
		NVAPI_DEVICE_SWITCHING_NOT_ALLOWED          = -117,
		NVAPI_TESTING_CLOCKS_NOT_SUPPORTED          = -118,
		NVAPI_UNKNOWN_UNDERSCAN_CONFIG              = -119,
		NVAPI_TIMEOUT_RECONFIGURING_GPU_TOPO        = -120,
		NVAPI_DATA_NOT_FOUND                        = -121,
		NVAPI_EXPECTED_ANALOG_DISPLAY               = -122,
		NVAPI_NO_VIDLINK                            = -123,
		NVAPI_REQUIRES_REBOOT                       = -124,
		NVAPI_INVALID_HYBRID_MODE                   = -125,
		NVAPI_MIXED_TARGET_TYPES                    = -126,
		NVAPI_SYSWOW64_NOT_SUPPORTED                = -127,
		NVAPI_IMPLICIT_SET_GPU_TOPOLOGY_CHANGE_NOT_ALLOWED = -128,
		NVAPI_REQUEST_USER_TO_CLOSE_NON_MIGRATABLE_APPS = -129,
		NVAPI_OUT_OF_MEMORY                         = -130,
		NVAPI_WAS_STILL_DRAWING                     = -131,
		NVAPI_FILE_NOT_FOUND                        = -132,
		NVAPI_TOO_MANY_UNIQUE_STATE_OBJECTS         = -133,
		NVAPI_INVALID_CALL                          = -134,
		NVAPI_D3D10_1_LIBRARY_NOT_FOUND             = -135,
		NVAPI_FUNCTION_NOT_FOUND                    = -136,
		NVAPI_INVALID_USER_PRIVILEGE                = -137,
		NVAPI_EXPECTED_NON_PRIMARY_DISPLAY_HANDLE   = -138,
		NVAPI_EXPECTED_COMPUTE_GPU_HANDLE           = -139,
		NVAPI_STEREO_NOT_INITIALIZED                = -140,
		NVAPI_STEREO_REGISTRY_ACCESS_FAILED         = -141,
		NVAPI_STEREO_REGISTRY_PROFILE_TYPE_NOT_SUPPORTED = -142,
		NVAPI_STEREO_REGISTRY_VALUE_NOT_SUPPORTED   = -143,
		NVAPI_STEREO_NOT_ENABLED                    = -144,
		NVAPI_STEREO_NOT_TURNED_ON                  = -145,
		NVAPI_STEREO_INVALID_DEVICE_INTERFACE       = -146,
		NVAPI_STEREO_PARAMETER_OUT_OF_RANGE         = -147,
		NVAPI_STEREO_FRUSTUM_ADJUST_MODE_NOT_SUPPORTED = -148,
		NVAPI_TOPO_NOT_POSSIBLE                     = -149,
		NVAPI_MODE_CHANGE_FAILED                    = -150,
		NVAPI_D3D11_LIBRARY_NOT_FOUND               = -151,
		NVAPI_INVALID_ADDRESS                       = -152,
		NVAPI_STRING_TOO_SMALL                      = -153,
		NVAPI_MATCHING_DEVICE_NOT_FOUND             = -154,
		NVAPI_DRIVER_RUNNING                        = -155,
		NVAPI_DRIVER_NOTRUNNING                     = -156,
		NVAPI_ERROR_DRIVER_RELOAD_REQUIRED          = -157,
		NVAPI_SET_NOT_ALLOWED                       = -158,
		NVAPI_ADVANCED_DISPLAY_TOPOLOGY_REQUIRED    = -159,
		NVAPI_SETTING_NOT_FOUND                     = -160,
		NVAPI_SETTING_SIZE_TOO_LARGE                = -161,
		NVAPI_TOO_MANY_SETTINGS_IN_PROFILE          = -162,
		NVAPI_PROFILE_NOT_FOUND                     = -163,
		NVAPI_PROFILE_NAME_IN_USE                   = -164,
		NVAPI_PROFILE_NAME_EMPTY                    = -165,
		NVAPI_EXECUTABLE_NOT_FOUND                  = -166,
		NVAPI_EXECUTABLE_ALREADY_IN_USE             = -167,
		NVAPI_DATATYPE_MISMATCH                     = -168,
		NVAPI_PROFILE_REMOVED                       = -169,
		NVAPI_UNREGISTERED_RESOURCE                 = -170,
		NVAPI_ID_OUT_OF_RANGE                       = -171,
		NVAPI_DISPLAYCONFIG_VALIDATION_FAILED       = -172,
		NVAPI_DPMST_CHANGED                         = -173,
		NVAPI_INSUFFICIENT_BUFFER                   = -174,
		NVAPI_ACCESS_DENIED                         = -175,
		NVAPI_MOSAIC_NOT_ACTIVE                     = -176,
		NVAPI_SHARE_RESOURCE_RELOCATED              = -177,
		NVAPI_REQUEST_USER_TO_DISABLE_DWM           = -178,
		NVAPI_D3D_DEVICE_LOST                       = -179,
		NVAPI_INVALID_CONFIGURATION                 = -180,
		NVAPI_STEREO_HANDSHAKE_NOT_DONE             = -181,
		NVAPI_EXECUTABLE_PATH_IS_AMBIGUOUS          = -182,
		NVAPI_DEFAULT_STEREO_PROFILE_IS_NOT_DEFINED = -183,
		NVAPI_DEFAULT_STEREO_PROFILE_DOES_NOT_EXIST = -184,
		NVAPI_CLUSTER_ALREADY_EXISTS                = -185,
		NVAPI_DPMST_DISPLAY_ID_EXPECTED             = -186,
		NVAPI_INVALID_DISPLAY_ID                    = -187,
		NVAPI_STREAM_IS_OUT_OF_SYNC                 = -188,
	} NvAPI_Status;

	#define NV_DECLARE_HANDLE(name) struct name##__ { int unused; }; typedef struct name##__ *name
	#define MAKE_NVAPI_VERSION(typeName,ver) (ULONG)(sizeof(typeName) | ((ver)<<16))

	NV_DECLARE_HANDLE(NvDisplayHandle);
	NV_DECLARE_HANDLE(NvPhysicalGpuHandle);

	#define NVAPI_SHORT_STRING_MAX      64

	#define NVAPI_MAX_THERMAL_SENSORS_PER_GPU 3
	#define NVAPI_MAX_PHYSICAL_GPUS           64

	typedef char NvAPI_ShortString[NVAPI_SHORT_STRING_MAX];
	typedef UINT32 NvU32;

	typedef struct
	{
		NvU32 busId;
		NvU32 busSlotId;
	} NV_PCILOC;

	typedef struct 
	{
		ULONG              version;
		ULONG              drvVersion;
		ULONG              bldChangeListNum;
		NvAPI_ShortString  szBuildBranchString;
		NvAPI_ShortString  szAdapterString;
	} NV_DISPLAY_DRIVER_VERSION;

	//! Used in NvAPI_GPU_GetAllClockFrequencies()
	typedef enum
	{
		NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ =   0,
		NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK   =   1,
		NV_GPU_CLOCK_FREQUENCIES_BOOST_CLOCK  =   2,
		NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE_NUM = 3
	}   NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE;	 
	#define NVAPI_MAX_GPU_PUBLIC_CLOCKS     32
	//! Used in NvAPI_GPU_GetAllClockFrequencies()
	typedef struct
	{
		NvU32   version;        //!< Structure version
		NvU32   ClockType:2;    //!< One of NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE. Used to specify the type of clock to be returned.
		NvU32   reserved:22;    //!< These bits are reserved for future use. Must be set to 0.
		NvU32   reserved1:8;    //!< These bits are reserved.
		struct
		{
			NvU32 bIsPresent:1;         //!< Set if this domain is present on this GPU
			NvU32 reserved:31;          //!< These bits are reserved for future use.
			NvU32 frequency;            //!< Clock frequency (kHz)
		}domain[NVAPI_MAX_GPU_PUBLIC_CLOCKS];
	} NV_GPU_CLOCK_FREQUENCIES;

	typedef INT *(*nvapi_QueryInterface)(UINT Offset);

	typedef NvAPI_Status (__cdecl *NvAPI_Initialize)(void);
	typedef NvAPI_Status (__cdecl *NvAPI_Unload)(void);
	typedef NvAPI_Status (__cdecl *NvAPI_EnumPhysicalGPUs)(NvPhysicalGpuHandle nvGPUHandle[NVAPI_MAX_PHYSICAL_GPUS], ULONG *pGpuCount);
	typedef NvAPI_Status (__cdecl *NvAPI_GPU_GetAllClockFrequencies)(NvPhysicalGpuHandle hPhysicalGpu, NV_GPU_CLOCK_FREQUENCIES *pClkFreqs );
	typedef NvAPI_Status (__cdecl *NvAPI_GetDisplayDriverVersion)(NvDisplayHandle hNvDisplay, NV_DISPLAY_DRIVER_VERSION *pVersion);
	typedef NvAPI_Status (__cdecl *NvAPI_GPU_GetBusId)(NvPhysicalGpuHandle hPhysicalGpu,NvU32 *pBusId);
	typedef NvAPI_Status (__cdecl *NvAPI_GPU_GetBusSlotId)(NvPhysicalGpuHandle hPhysicalGpu, NvU32 *pBusSlotId);

#endif //_WIN32
#endif // !__NVAPI_TYPES_H