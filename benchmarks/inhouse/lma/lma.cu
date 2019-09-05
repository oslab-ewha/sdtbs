#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "../../benchapi.h"

__device__ int
lma(void *args[])
{
	int	chunksize = (int)(long long)args[0];
	int	refspan = (int)(long long)args[1];
	int	n_iters = (int)(long long)args[2];
	int	chunk_idx = get_blockIdxX() + get_gridDimX() * get_blockIdxY();
	int	chunk_start;
	unsigned char	**chunks = (unsigned char **)args[3];
	unsigned	n_chunks = get_gridDimX() * get_gridDimY();
	unsigned	randx;
	int	value = 0;
	int	i;

	chunk_start = chunk_idx - refspan;
	if (chunk_start < 0)
		chunk_start += n_chunks;

	randx = 0x12345678 + clock() * 19239913 * get_threadIdxX();
	for (i = 0; i < n_iters; i++) {
		unsigned	rand_chunk_idx = (chunk_start + randx % (1 + 2 * refspan)) % n_chunks;

		randx = get_random(randx);
		value += chunks[rand_chunk_idx][randx % chunksize];
		randx = get_random(randx);
	}
	return value;
}

int
cookarg_lma(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	unsigned char	**chunks, **d_chunks;
	int	chunksize = (int)(long long)args[0];
	int	refspan = (int)(long long)args[1];
	char	*buf;
	int	i;
	cudaError_t	err;

	if (dimGrid.x * dimGrid.y < refspan * 2) {
		printf("too small TB's\n");
		return -1;
	}
	chunks = (unsigned char **)malloc(dimGrid.x * dimGrid.y * sizeof(unsigned char *));
	buf = (char *)malloc(chunksize);
	for (i = 0; i < chunksize; i++) {
		buf[i] = (char)i;
	}
	for (i = 0; i < dimGrid.x * dimGrid.y; i++) {
		err = cudaMalloc(&chunks[i], chunksize);
		if (err != cudaSuccess) {
			printf("cudaMalloc failed: err: %s\n", cudaGetErrorString(err));
		}
		cudaMemcpy(chunks[i], buf, chunksize, cudaMemcpyHostToDevice);
	}
	free(buf);

	cudaMalloc(&d_chunks, dimGrid.x * dimGrid.y * sizeof(unsigned char *));
	cudaMemcpy(d_chunks, chunks, dimGrid.x * dimGrid.y * sizeof(unsigned char *), cudaMemcpyHostToDevice);

	args[3] = d_chunks;
	return 0;
}

__global__ static void
kernel_lma(void *args[], int *pres)
{
	*pres = lma(args);
}

void
bench_lma(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_lma<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
