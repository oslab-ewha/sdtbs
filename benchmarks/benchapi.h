#ifndef _BENCHAPI_H_
#define _BENCHAPI_H_

extern __device__ int	native_mode;

__device__ unsigned get_random(unsigned randx);

__device__ int get_gridDimX(void);
__device__ int get_gridDimY(void);
__device__ int get_blockIdxX(void);
__device__ int get_blockIdxY(void);
__device__ int get_blockDimX(void);
__device__ int get_blockDimY(void);
__device__ int get_threadIdxX(void);

__device__ void sync_threads(void);

#endif
