#include "../../benchapi.h"

extern __device__ int loopcalc(void *args[]);

__global__ static void
kernel_loopcalc_reloc(void *args[], int *pres)
{
	native_mode = 1;
	*pres = loopcalc(args);
}

void
bench_loopcalc_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_loopcalc_reloc<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
