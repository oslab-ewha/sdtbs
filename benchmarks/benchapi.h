#ifndef _BENCHAPI_H_
#define _BENCHAPI_H_

#define LOOPCALC	((skid_t)1)
#define MKLC		((skid_t)2)
#define GMA		((skid_t)3)
#define LMA		((skid_t)4)
#define KMEANS		((skid_t)5)

typedef unsigned char	skid_t;
typedef unsigned short	skrid_t;

__device__ unsigned get_random(unsigned randx);

__device__ int get_gridDimX(void);
__device__ int get_gridDimY(void);
__device__ int get_blockIdxX(void);
__device__ int get_blockIdxY(void);
__device__ int get_blockDimX(void);
__device__ int get_blockDimY(void);
__device__ int get_threadIdxX(void);

__device__ void sync_threads(void);

skrid_t launch_kernel(skid_t skid, cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[]);
void wait_kernel(skrid_t skrid, cudaStream_t strm, int *res);

#endif
