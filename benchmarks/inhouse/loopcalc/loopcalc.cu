#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <math_constants.h>

#include "../../benchapi.h"

/* NOTE: nvcc over 9.0 has a problem of compilation freeze.
 * This seems to relate to ptx optimization.
 * nvcc with -Xptxas -O0 disables all optimizations. loopcalc Makefile.am has this option.
 */

__device__ static int
calc_empty(int n_iters)
{
        int     value = 0;
	int     i;

        for (i = 0; i < n_iters; i++) {
		value++;
        }
        return value;
}

__device__ static int
calc_int(int n_iters)
{
	int	value = 9;
	int     i;

        for (i = 0; i < n_iters; i++) {
		value = value * 1923192273 + 3921192123;
        }
        return value;
}

__device__ static float
calc_float(int n_iters)
{
	float	value = 9.99f;
	int     i;

        for (i = 0; i < n_iters; i++) {
		if (value == CUDART_INF_F)
			value = 9.99f;
		else
			value = value * 2911.2123f + 1.992813f;
        }
        return value;
}

__device__ static double
calc_double(int n_iters)
{
	double	value = 32.192123123213;
	int	i;

	for (i = 0; i < n_iters; i++) {
		if (value == CUDART_INF_F)
			value = 329.99128493;
		else
			value = value * 2911.2134324 + 1.992812932;
	}
	return value;
}

__device__ int
loopcalc(void *args[])
{
	int	calctype = (int)(long long)args[0];
	int	n_iters1 = (int)(long long)args[1];
	int	ret = 0;

	switch (calctype) {
	case 1:
		ret = (int)calc_int(n_iters1);
		break;
	case 2:
		ret = (int)fmodf(calc_float(n_iters1), 10000000.0f);
		break;
	case 3:
		ret = (int)fmod(calc_double(n_iters1), 100000000.0);
		break;
	default:
		ret = (int)calc_empty(n_iters1);
		break;
	}
	return ret;
}

int
bench_loopcalc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrid_t	skrid;
	int	res;

	skrid = launch_kernel(LOOPCALC, strm, dimGrid, dimBlock, args);
	wait_kernel(skrid, strm, &res);

	return res;
}
