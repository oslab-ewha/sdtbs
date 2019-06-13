#include <stdio.h>

__global__ static void
run_macro_TB(void)
{
	for (int i = 0;; i++) {
	}
}

extern "C" void
run_macrotb(void)
{
	cudaError_t	err;

	dim3 dimBlock(1, 1);
	dim3 dimGrid(1, 1);

	run_macro_TB<<<dimGrid, dimBlock, 0>>>();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return;
	}

	cudaDeviceSynchronize();
}
