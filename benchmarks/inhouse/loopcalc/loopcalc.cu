#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <math_constants.h>

/* NOTE: nvcc over 9.0 has a problem of compilation freeze.
 * This seems to relate to ptx optimization.
 * nvcc with -G disables all optimizations. loopcalc Makefile.am has this option.
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
		int	j;

		for (j = 0; j < 100000; j++) {
			value = value * 1923192273 + 3921192123;
		}
        }
        return value;
}

__device__ static float
calc_float(int n_iters)
{
	float	value = 9.99f;
	int     i;

        for (i = 0; i < n_iters; i++) {
		int	j;

		for (j = 0; j < 10000; j++) {
			if (value == CUDART_INF_F)
				value = 9.99f;
			else
				value = value * 2911.2123f + 1.992813f;
		}
        }
        return value;
}

__device__ static double
calc_double(int n_iters)
{
	double	value = 32.192123123213;
	int	i;

	for (i = 0; i < n_iters; i++) {
		int	j;

		for (j = 0; j < 10000; j++) {
			if (value == CUDART_INF_F)
				value = 329.99128493;
			else
				value = value * 2911.2134324 + 1.992812932;
		}
	}
	return value;
}

__device__ static double
calc_float_double(int n_iters1, int n_iters2)
{
	if (!(threadIdx.x / 32) % 2)
		return calc_float(n_iters1);
	else
		return calc_double(n_iters2);
}

__device__ static double
calc_int_float(int n_iters1, int n_iters2)
{
	if (!(threadIdx.x / 32) % 2)
		return calc_int(n_iters1);
	else
		return calc_float(n_iters2);
}

__device__ static double
calc_int_double(int n_iters1, int n_iters2)
{
	if (!(threadIdx.x / 32) % 2)
		return calc_int(n_iters1);
	else
		return calc_double(n_iters2);
}

__device__ static double
calc_float_double_tb(int n_iters1, int n_iters2)
{
	if (!(blockIdx.x % 2))
		return calc_float(n_iters1);
	else
		return calc_double(n_iters2);
}

__device__ static double
calc_int_float_tb(int n_iters1, int n_iters2)
{
	if (!(blockIdx.x % 2))
		return calc_int(n_iters1);
	else
		return calc_float(n_iters2);
}

__device__ static double
calc_int_double_tb(int n_iters1, int n_iters2)
{
	if (!(blockIdx.x % 2))
		return calc_int(n_iters1);
	else
		return calc_double(n_iters2);
}

__device__ int
loopcalc(void *args[])
{
	int	calctype = (int)(long long)args[0];
	int	n_iters1 = (int)(long long)args[1];
	int	n_iters2 = (int)(long long)args[2];
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
	case 4:
		ret = (int)calc_float_double(n_iters1, n_iters2);
		break;
	case 5:
		ret = (int)calc_int_float(n_iters1, n_iters2);
		break;
	case 6:
		ret = (int)calc_int_double(n_iters1, n_iters2);
		break;
	case 7:
		ret = (int)calc_float_double_tb(n_iters1, n_iters2);
		break;
	case 8:
		ret = (int)calc_int_float_tb(n_iters1, n_iters2);
		break;
	case 9:
		ret = (int)calc_int_double_tb(n_iters1, n_iters2);
		break;
	default:
		ret = (int)calc_empty(n_iters1);
		break;
	}
	return ret;
}

__global__ static void
kernel_loopcalc(void *args[])
{
	int	res;

	res = loopcalc(args);
	if (threadIdx.x == 0 && threadIdx.y == 0)
		args[0] = (void *)(long long)res;
}

int
bench_loopcalc(cudaStream_t strm, int n_grid_width, int n_grid_height, int n_tb_width, int n_tb_height, void *args[])
{
	cudaError_t	err;

	dim3 dimGrid(n_grid_width, n_grid_height);
	dim3 dimBlock(n_tb_width, n_tb_height);

	kernel_loopcalc<<<dimGrid, dimBlock, 0, strm>>>(args);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return -1;
	}

	return 0;
}
