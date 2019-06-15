#include "sdtbs.h"

unsigned	n_sm_count;
unsigned	n_threads_per_MTB;	/* per macro TB */

extern "C" BOOL
select_gpu_device(unsigned devno)
{
	if (cudaSetDevice(devno) != 0)
		return FALSE;
	return TRUE;
}

extern "C" void
error(const char *fmt, ...)
{
	char	*msg;
	va_list	ap;
	int	n;

	va_start(ap, fmt);
	n = vasprintf(&msg, fmt, ap);
	va_end(ap);
	if (n >= 0) {
		fprintf(stderr, "error: %s\n", msg);
		free(msg);
	}
}

BOOL
setup_gpu_devinfo(void)
{
	struct cudaDeviceProp	prop;
	cudaError_t	err;

	err = cudaGetDeviceProperties(&prop, devno);
	if (err != cudaSuccess) {
		error("device prop error: %s\n", cudaGetErrorString(err));
		return FALSE;
	}

	n_sm_count = prop.multiProcessorCount;
	n_threads_per_MTB = prop.maxThreadsPerBlock;

	return TRUE;
}
