#include "../../config.h"

#include "sdtbs_cu.h"

#include <sys/times.h>

unsigned	n_sm_count;
unsigned	n_threads_per_MTB;	/* per macro TB */
unsigned	n_MTBs_per_sm;

extern "C" unsigned	arg_n_MTBs_per_sm;
extern "C" unsigned	arg_n_threads_per_MTB;

static struct timespec  started_ts;

__device__ uint
get_smid(void)
{
	uint	ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret));
	return ret;
}

__device__ void
sleep_in_kernel(void)
{
#if CUDA_COMPUTE >= 70
	asm("nanosleep.u32 1;");
#endif
}

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
	if (sched->type == TBS_TYPE_SOLO)
		n_sm_count--;

	if (arg_n_MTBs_per_sm == 0 && arg_n_threads_per_MTB == 0) {
		n_threads_per_MTB = prop.maxThreadsPerBlock;
		n_MTBs_per_sm = prop.maxThreadsPerMultiProcessor / n_threads_per_MTB;
	}
	else if (arg_n_MTBs_per_sm > 0) {
		n_MTBs_per_sm = arg_n_MTBs_per_sm;
		if (arg_n_threads_per_MTB > 0)
			n_threads_per_MTB = arg_n_threads_per_MTB;
		else
			n_threads_per_MTB = prop.maxThreadsPerMultiProcessor / n_MTBs_per_sm;
	}
	else {
		n_threads_per_MTB = arg_n_threads_per_MTB;
		n_MTBs_per_sm = prop.maxThreadsPerMultiProcessor / n_threads_per_MTB;
	}

	if (n_threads_per_MTB > prop.maxThreadsPerBlock)
		n_threads_per_MTB = prop.maxThreadsPerBlock;
	if (n_threads_per_MTB < 32) {
		error("Too small threads per MTB: %d", n_threads_per_MTB);
		return FALSE;
	}
	if (n_threads_per_MTB % 32) {
		error("Invalid thread count per MTB: %d", n_threads_per_MTB);
		return FALSE;
	}

	return TRUE;
}

void
init_tickcount(void)
{
        clock_gettime(CLOCK_MONOTONIC, &started_ts);
}

/* microsecs */
unsigned
get_tickcount(void)
{
	struct timespec	ts;
	unsigned	ticks;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	if (ts.tv_nsec < started_ts.tv_nsec) {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec - 1)) * 1000000;
		ticks += (1000000000 + ts.tv_nsec - started_ts.tv_nsec) / 1000;
	}
	else {
		ticks = ((unsigned)(ts.tv_sec - started_ts.tv_sec)) * 1000000;
		ticks += (ts.tv_nsec - started_ts.tv_nsec) / 1000;
        }

	return ticks;
}
