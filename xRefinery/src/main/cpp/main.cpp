#include "stdafx.h"
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	cout << "xRefinery Test Harness" << endl;
	
	#if defined(_WIN32)
		HINSTANCE hGetProcIDDLL = LoadLibrary("nvcuda.dll");
		if (!hGetProcIDDLL) {
			cout << "could not load the nvcuda dynamic library" << endl;
			return 1;
		} else cout << "nvcuda.dll loaded OK!" << endl;    
		FreeLibrary(hGetProcIDDLL);
	#endif
	
	return 0;
}
