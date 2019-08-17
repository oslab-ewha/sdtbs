#include "sdtbs_cu.h"

extern __device__ benchrun_k_t *get_brk(void);
extern __device__ unsigned get_offset_TB(void);
extern __device__ void sync_TB_threads(void);

__device__ int	native_mode;

__device__ int
get_gridDimX(void)
{
        benchrun_k_t    *brk;

        if (native_mode)
		return gridDim.x;
	brk = get_brk();
	return brk->n_grid_width;
}

__device__ int
get_gridDimY(void)
{
        benchrun_k_t    *brk;

        if (native_mode)
		return gridDim.y;
	brk = get_brk();
	return brk->n_grid_height;
}

__device__ int
get_blockIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return blockIdx.x;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) / brk->n_tb_width) % brk->n_grid_width;
}

__device__ int
get_blockIdxY(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return blockIdx.y;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) / brk->n_tb_width) / brk->n_grid_width;
}

__device__ int
get_blockDimX(void)
{
	benchrun_k_t	*brk;

	if (native_mode)
		return blockDim.x;
	brk = get_brk();
	return brk->n_tb_width;
}

__device__ int
get_blockDimY(void)
{
	benchrun_k_t	*brk;

	if (native_mode)
		return blockDim.y;
	brk = get_brk();
	return brk->n_tb_height;
}

__device__ int
get_threadIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

	if (native_mode)
		return threadIdx.x;
	brk = get_brk();
	offset = get_offset_TB();
	return ((offset * N_THREADS_PER_mTB) % brk->n_tb_width) + threadIdx.x % N_THREADS_PER_mTB;
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
