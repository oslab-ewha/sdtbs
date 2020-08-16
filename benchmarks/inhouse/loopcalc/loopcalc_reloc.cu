#include "../../benchapi.h"

extern __device__ int loopcalc(void *args[]);

__device__ int
loopcalc_reloc(void *args[])
{
	return loopcalc(args);
}

int
bench_loopcalc_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[])
{
	void    **d_args;
	int	res, *d_pres;

	cudaMalloc(&d_args, sizeof(void *) * 2);
	cudaMalloc(&d_pres, sizeof(int));
	cudaMemcpyAsync(d_args, args, sizeof(void *) * 2, cudaMemcpyHostToDevice, strm);

	launch_kernel(LOOPCALC_RELOC, strm, dimGrid, dimBlock, d_args, d_pres);

	cudaMemcpyAsync(&res, d_pres, sizeof(int), cudaMemcpyDeviceToHost, strm);
	cudaStreamSynchronize(strm);
	cudaFree(d_args);
	cudaFree(d_pres);

	return res;
}
