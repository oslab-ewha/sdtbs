#include "../benchapi.h"

extern __device__ int kmeans(void *args[]);

__device__ int
kmeans_reloc(void *args[])
{
	return kmeans(args);
}

int
bench_kmeans_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	void	**d_args;
	int	res, *d_pres;

	cudaMalloc(&d_args, sizeof(void *) * 5);
	cudaMalloc(&d_pres, sizeof(int));
	cudaMemcpyAsync(d_args, args, sizeof(void *) * 5, cudaMemcpyHostToDevice, strm);

	launch_kernel(KMEANS_RELOC, strm, dimGrid, dimBlock, d_args, d_pres);

	cudaMemcpyAsync(&res, d_pres, sizeof(int), cudaMemcpyDeviceToHost, strm);
	cudaStreamSynchronize(strm);
	cudaFree(d_args);
	cudaFree(d_pres);

	return res;
}
