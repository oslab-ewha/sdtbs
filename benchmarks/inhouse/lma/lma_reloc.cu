#include "../../benchapi.h"

extern __device__ int lma(void *args[]);

__global__ static void
kernel_lma_reloc(void *args[], int *pres)
{
	*pres = lma(args);
}

void
bench_lma_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_lma_reloc<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
