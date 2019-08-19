#include "sdtbs_cu.h"

extern __device__ benchrun_k_t *get_brk(void);
extern __device__ unsigned get_offset_TB(void);
extern __device__ void sync_TB_threads(void);

__device__ int	native_mode;
/* TODO: benchmark API is not ready for a static scheduler */
__device__ BOOL	static_sched;

__device__ int
get_gridDimX(void)
{
        benchrun_k_t    *brk;

        if (native_mode || static_sched)
		return gridDim.x;
	brk = get_brk();
	return brk->dimGrid.x;
}

__device__ int
get_gridDimY(void)
{
        benchrun_k_t    *brk;

        if (native_mode || static_sched)
		return gridDim.y;
	brk = get_brk();
	return brk->dimGrid.y;
}

__device__ int
get_blockIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode || static_sched)
		return blockIdx.x;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) / brk->dimBlock.x) % brk->dimGrid.x;
}

__device__ int
get_blockIdxY(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode || static_sched)
		return blockIdx.y;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) / brk->dimBlock.x) / brk->dimGrid.x;
}

__device__ int
get_blockDimX(void)
{
	benchrun_k_t	*brk;

	if (native_mode || static_sched)
		return blockDim.x;
	brk = get_brk();
	return brk->dimBlock.x;
}

__device__ int
get_blockDimY(void)
{
	benchrun_k_t	*brk;

	if (native_mode || static_sched)
		return blockDim.y;
	brk = get_brk();
	return brk->dimBlock.y;
}

__device__ int
get_threadIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode || static_sched)
		return threadIdx.x;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) % brk->dimBlock.x) + threadIdx.x % N_THREADS_PER_mTB;
}

__device__ void
sync_threads(void)
{
	if (native_mode) {
		__syncthreads();
		return;
	}
	sync_TB_threads();
}
