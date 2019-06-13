#include "sdtbs.h"

extern "C" BOOL
select_gpu_device(unsigned devno)
{
	if (cudaSetDevice(devno) != 0)
		return FALSE;
	return TRUE;
}

