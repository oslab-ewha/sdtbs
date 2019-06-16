#include "sdtbs.h"

#include <sys/times.h>

unsigned	n_sm_count;
unsigned	n_threads_per_MTB;	/* per macro TB */

static struct timespec  started_ts;

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
		error("failed to get gpu device properties: %s", cudaGetErrorString(err));
		return FALSE;
	}

	n_sm_count = prop.multiProcessorCount;
	n_threads_per_MTB = prop.maxThreadsPerBlock;

	return TRUE;
}

void
init_tickcount(void)
{
        clock_gettime(CLOCK_MONOTONIC, &started_ts);
}

unsigned
get_tickcount(void)
{
	struct timespec	ts;
	unsigned	ticks;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	if (ts.tv_nsec < started_ts.tv_nsec) {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec - 1)) * 1000;
		ticks += (1000000000 + ts.tv_nsec - started_ts.tv_nsec) / 1000000;
	}
	else {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec)) * 1000;
		ticks += (ts.tv_nsec - started_ts.tv_nsec) / 1000000;
        }

	return ticks;
}
