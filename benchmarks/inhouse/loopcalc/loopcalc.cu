#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

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
	int	value1 = 9, value2 = 32, value3 = 2911, value4 = 1992;
	int     i;

        for (i = 0; i < n_iters; i++) {
		value1 *= value2;
		value2 *= value3;
		value4 *= (value2 * value3 * value1);
        }
        return value1 * value2 * value3 * value4;
}

__device__ static float
calc_float(int n_iters)
{
	float	value1 = 9.99f, value2 = 32.19f, value3 = 2911.2f, value4 = 1.9921f;
	int     i;

        for (i = 0; i < n_iters; i++) {
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
        }
        return value1 * value2 * value3 * value4;
}

__device__ static double
calc_double(int n_iters)
{
	double	value1 = 9.9923123132, value2 = 32.192123123213, value3 = 2911.2123123123123, value4 = 1.9921123123123;
	int	i;

	for (i = 0; i < n_iters; i++) {
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= value2; value2 *= value3; value3 *= value4; value4 *= value1;
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
		value1 *= (value2 * value3 * value4);
		value2 *= (value1 * value3 * value4);
		value3 *= (value2 * value1 * value4);
		value4 *= (value2 * value3 * value1);
	}
	return value1 * value2 * value3 * value4;
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
loopcalc(int args[])
{
	int	calctype = args[0];
	int	n_iters1 = args[1];
	int	n_iters2 = args[2];
	int	ret = 0;

	switch (calctype) {
	case 1:
		ret = (int)calc_int(n_iters1);
		break;
	case 2:
		ret = (int)calc_float(n_iters1);
		break;
	case 3:
		ret = (int)calc_double(n_iters1);
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
loopcalc_kernel(int args[])
{
	args[0] = loopcalc(args);
}

int
bench_native_loopcalc(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, int args[])
{
	cudaError_t	err;

	dim3 dimGrid(n_tbs_x, 1);
	dim3 dimBlock(n_threads_x, 1);

	loopcalc_kernel<<<dimGrid, dimBlock, 0, strm>>>(args);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	return 0;
}
