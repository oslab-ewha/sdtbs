#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

__device__ int
gma(void *args[])
{
	int	gmemsize = (int)(long long)args[0];
	int	stride = (int)(long long)args[1];
	int	n_iters = (int)(long long)args[2];
	unsigned char	*gmem = (unsigned char *)args[3];
	unsigned	memidx;
	int	value = 0;
	int	i, j;

	memidx = clock() * 19239913 * threadIdx.x;
	for (i = 0; i < n_iters; i++) {
		for (j = 0; j < 10000; j ++, memidx += stride) {
			value += gmem[memidx % (gmemsize * 1024)];
		}
	}
	return value;
}

__global__ static void
kernel_gma(void *args[])
{
	gma(args);
}

int
cookarg_gma(void *args[])
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

int
bench_gma(cudaStream_t strm, int n_grid_width, int n_grid_height, int n_tb_width, int n_tb_height, void *args[])
{
	cudaError_t	err;

	dim3 dimGrid(n_grid_width, n_grid_height);
	dim3 dimBlock(n_tb_width, n_tb_height);

	kernel_gma<<<dimGrid, dimBlock, 0, strm>>>(args);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return -1;
	}

	return 0;
}
