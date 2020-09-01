#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <math_constants.h>

#include "../../benchapi.h"

/* NOTE: nvcc over 9.0 has a problem of compilation freeze.
 * This seems to relate to ptx optimization.
 * nvcc with -Xptxas -O0 disables all optimizations. loopcalc Makefile.am has this option.
 */

__device__ int
mklc(void *args[])
{
	int	n_iters = (int)(long long)args[0];
	double	value = 32.192123123213;
	int	i;

	for (i = 0; i < n_iters; i++) {
		if (value == CUDART_INF_F)
			value = 329.99128493;
		else
			value = value * 2911.2134324 + 1.992812932;
	}
	return (int)value;
}

int
bench_mklc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrid_t	*skrids;
	int	count;
	int	res;
	int	i;

	count = (int)(long long)args[0];
	skrids = (skrid_t *)malloc(sizeof(skrid_t) * count);
	for (i = 0; i < count; i++)
		skrids[i] = launch_kernel(MKLC, strm, dimGrid, dimBlock, args + 1);
	
	for (i = 0; i < count; i++)
		wait_kernel(skrids[i], strm, &res);

	return res;
}
