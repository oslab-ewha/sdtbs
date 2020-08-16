#include "../../benchapi.h"

extern __device__ int lma(void *args[]);

__device__ int
lma_reloc(void *args[])
{
	return lma(args);
}

int
bench_lma_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	void    **d_args;
	int	res, *d_pres;

	cudaMalloc(&d_args, sizeof(void *) * 4);
	cudaMalloc(&d_pres, sizeof(int));
	cudaMemcpyAsync(d_args, args, sizeof(void *) * 4, cudaMemcpyHostToDevice, strm);

	launch_kernel(LMA_RELOC, strm, dimGrid, dimBlock, d_args, d_pres);

	cudaMemcpyAsync(&res, d_pres, sizeof(int), cudaMemcpyDeviceToHost, strm);
	cudaStreamSynchronize(strm);
	cudaFree(d_args);
	cudaFree(d_pres);

	return res;
}
