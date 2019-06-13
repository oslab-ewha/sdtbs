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

__global__ static void
loopcalc(double *result, unsigned calctype, int n_iters1, int n_iters2)
{
	double	*my;

	my = result + threadIdx.x + (blockDim.x * threadIdx.y);
	switch (calctype) {
	case 1:
		*(int *)my = calc_int(n_iters1);
		break;
	case 2:
		*(float *)my = calc_float(n_iters1);
		break;
	case 3:
		*my = calc_double(n_iters1);
		break;
	case 4:
		*my = calc_float_double(n_iters1, n_iters2);
		break;
	case 5:
		*my = calc_int_float(n_iters1, n_iters2);
		break;
	case 6:
		*my = calc_int_double(n_iters1, n_iters2);
		break;
	case 7:
		*my = calc_float_double_tb(n_iters1, n_iters2);
		break;
	case 8:
		*my = calc_int_float_tb(n_iters1, n_iters2);
		break;
	case 9:
		*my = calc_int_double_tb(n_iters1, n_iters2);
		break;
	default:
		*(int *)my = calc_empty(n_iters1);
		break;
	}
}

static int
parse_args(const char *c_args, unsigned *pcalctype, int *pgridsize, int *pblksize, int *pn_iters1, int *pn_iters2)
{
	char	*args, *comma;

	if (c_args == NULL)
		return 0;
	args = strdup(c_args);
	comma = strchr(args, ',');
	if (comma != NULL) {
		char	*str_gridsize = comma + 1;

		*comma = '\0';
		comma = strchr(str_gridsize, ',');
		if (comma != NULL) {
			char	*str_blksize = comma + 1;

			*comma = '\0';
			comma = strchr(str_blksize, ',');
			if (comma != NULL) {
				char	*str_iters1 = comma + 1;

				*comma = '\0';
				comma = strchr(str_iters1, ',');
				if (comma != NULL) {
					if (sscanf(comma + 1, "%u", pn_iters2) != 1) {
						printf("invalid argument: %s\n", c_args);
						return -1;
					}
				}
				if (sscanf(str_iters1, "%u", pn_iters1) != 1) {
					printf("invalid argument: %s\n", c_args);
					return -1;
				}
			}
			if (sscanf(str_blksize, "%u", pblksize) != 1) {
				printf("invalid argument: %s\n", c_args);
				return -1;
			}
		}
		if (sscanf(str_gridsize, "%u", pgridsize) != 1) {
			printf("invalid argument: %s\n", c_args);
			return -1;
		}
	}

	if (sscanf(args, "%u", pcalctype) != 1) {
		printf("invalid argument: %s\n", c_args);
		return -1;
	}
	free(args);
	return 0;
}

extern "C" int
bench_loopcalc(cudaStream_t strm, const char *args)
{
	double		*result;
	cudaError_t	err;
	unsigned	calctype;
	int	gridsize = 4;
	int	blksize = 16;
	int	n_iters1 = 10;
	int	n_iters2 = 10;

	if (parse_args(args, &calctype, &blksize, &gridsize, &n_iters1, &n_iters2) < 0)
		return 1;

	err = cudaMalloc((void **)&result, blksize * sizeof(double));
	if (err != cudaSuccess) {
		printf("cudaMalloc failed: err: %s\n", cudaGetErrorString(err));
		return 1;
	}

	dim3 dimBlock(blksize, 1);
	dim3 dimGrid(gridsize, 1);

	loopcalc<<<dimGrid, dimBlock, 0, strm>>>(result, calctype, n_iters1, n_iters2);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaStreamSynchronize(strm);

	cudaFree(result);

	return 0;
}
