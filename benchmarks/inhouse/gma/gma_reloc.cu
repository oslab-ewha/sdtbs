#include "../../benchapi.h"

extern __device__ int gma(void *args[]);

__global__ static void
kernel_gma_reloc(void *args[], int *pres)
{
	*pres = gma(args);
}

void
bench_gma_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_gma_reloc<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
