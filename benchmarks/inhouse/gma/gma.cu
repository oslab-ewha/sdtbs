#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../benchapi.h"

__device__ int
gma(void *args[])
{
	int	gmemsize = (int)(long long)args[0];
	int	n_iters = (int)(long long)args[1];
	unsigned char	*gmem = (unsigned char *)args[2];
	unsigned	memidx_max = gmemsize * 1024;
	unsigned	randx;
	int	value = 0;
	int	i;

	randx = 0x12345678 + clock() * 19239913 * get_threadIdxX();
	for (i = 0; i < n_iters; i++) {
		unsigned	memidx = randx % memidx_max;
		value += gmem[memidx];
		randx = get_random(randx);
	}
	return value;
}

int
cookarg_gma(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	unsigned char	*gmem;
	int	gmemsize = (int)(long long)args[0];
	char	buf[1024];
	int	i;
	cudaError_t	err;

	err = cudaMalloc((void **)&gmem, gmemsize * 1024);
	if (err != cudaSuccess) {
		printf("cudaMalloc failed: err: %s\n", cudaGetErrorString(err));
		return -1;
	}
	for (i = 0; i < 1024; i++) {
		buf[i] = i;
	}
	for (i = 0; i < gmemsize; i++) {
		cudaMemcpy(gmem + i * 1024, buf, 1024, cudaMemcpyHostToDevice);
	}
	args[2] = gmem;
	return 0;
}

int
bench_gma(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	skrid_t	skrid;
	int	res;

	skrid = launch_kernel(GMA, strm, dimGrid, dimBlock, args);
	wait_kernel(skrid, strm, &res);

	return res;
}
