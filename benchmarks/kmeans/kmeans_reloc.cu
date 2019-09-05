#include "../benchapi.h"

extern __device__ int kmeans(void *args[]);

__global__ static void
kernel_kmeans_reloc(void *args[], int *pres)
{
	*pres = kmeans(args);
}

void
bench_kmeans_reloc(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_kmeans_reloc<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}
