#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../benchapi.h"

__device__ int
gma(void *args[])
{
	int	gmemsize = (int)(long long)args[0];
	int	stride = (int)(long long)args[1];
	int	n_iters = (int)(long long)args[2];
	unsigned char	*gmem = (unsigned char *)args[3];
	unsigned	memidx_max = gmemsize * 1024;
	unsigned	memidx;
	int	value = 0;
	int	i;

	memidx = (unsigned)(clock() * 19239913 * get_threadIdxX()) % memidx_max;
	for (i = 0; i < n_iters; i++, memidx += stride) {
		if (memidx >= memidx_max)
			memidx -= memidx_max;
		value += (gmem[memidx] + gmem[memidx + stride / 2] + gmem[memidx + stride / 4]);
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
	args[3] = gmem;
	return 0;
}

__global__ static void
kernel_gma(void *args[], int *pres)
{
	native_mode = 1;
	*pres = gma(args);
}

void
bench_gma(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_gma<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
