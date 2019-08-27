#include "sdtbs_cu.h"

extern __device__ benchrun_k_t *get_brk_static(void);
extern __device__ benchrun_k_t *get_brk_dyn(void);
extern __device__ unsigned short get_offset_TB_static(void);
extern __device__ unsigned short get_offset_TB_dyn(void);
extern __device__ void sync_TB_threads(void);

__device__ int	native_mode;
/* TODO: benchmark API is not ready for a static scheduler */

extern __device__ BOOL	static_sched;

__device__ int
get_gridDimX(void)
{
        benchrun_k_t    *brk;

        if (native_mode)
		return gridDim.x;

	if (static_sched)
		brk = get_brk_static();
	else
		brk = get_brk_dyn();

	return brk->dimGrid.x;
}

__device__ int
get_gridDimY(void)
{
        benchrun_k_t    *brk;

        if (native_mode)
		return gridDim.y;

	if (static_sched)
		brk = get_brk_static();
	else
		brk = get_brk_dyn();

	return brk->dimGrid.y;
}

__device__ int
get_blockIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return blockIdx.x;
	if (static_sched) {
		brk = get_brk_static();
		offset = get_offset_TB_static();
	}
	else {
		brk = get_brk_dyn();
		offset = get_offset_TB_dyn();
	}
	return ((offset * N_THREADS_PER_mTB) / brk->dimBlock.x) % brk->dimGrid.x;
}

__device__ int
get_blockIdxY(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return blockIdx.y;
	if (static_sched) {
		brk = get_brk_static();
		offset = get_offset_TB_static();
	}
	else {
		brk = get_brk_dyn();
		offset = get_offset_TB_dyn();
	}
	return ((offset * N_THREADS_PER_mTB) / brk->dimBlock.x) / brk->dimGrid.x;
}

__device__ int
get_blockDimX(void)
{
	benchrun_k_t	*brk;

	if (native_mode)
		return blockDim.x;

	if (static_sched)
		brk = get_brk_static();
	else
		brk = get_brk_dyn();

	return brk->dimBlock.x;
}

__device__ int
get_blockDimY(void)
{
	benchrun_k_t	*brk;

	if (native_mode)
		return blockDim.y;

	if (static_sched)
		brk = get_brk_static();
	else
		brk = get_brk_dyn();

	return brk->dimBlock.y;
}

__device__ int
get_threadIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return threadIdx.x;
	if (static_sched) {
		brk = get_brk_static();
		offset = get_offset_TB_static();
	}
	else {
		brk = get_brk_dyn();
		offset = get_offset_TB_dyn();
	}
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
