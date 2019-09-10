#include "sdtbs_cu.h"

extern __device__ benchrun_k_t *get_brk_static(void);
extern __device__ benchrun_k_t *get_brk_dyn(void);
extern __device__ benchrun_k_t *get_brk_prl(void);
extern __device__ unsigned short get_offset_TB_static(void);
extern __device__ unsigned short get_offset_TB_dyn(void);
extern __device__ unsigned short get_offset_TB_prl(void);
extern __device__ void sync_TB_threads_dyn(void);

static __device__ tbs_type_t	tbs_type;

/* TODO: benchmark API is not ready for a static scheduler */

__device__ unsigned
get_random(unsigned randx)
{
	randx ^= (randx << 13);
	randx ^= (randx >> 17);
	randx ^= (randx << 5);
	return randx;
}

__device__ int
get_gridDimX(void)
{
        benchrun_k_t    *brk;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return gridDim.x;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		break;
	default:
		brk = get_brk_dyn();
		break;
	}

	return brk->dimGrid.x;
}

__device__ int
get_gridDimY(void)
{
        benchrun_k_t    *brk;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return gridDim.y;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		break;
	default:
		brk = get_brk_dyn();
		break;
	}

	return brk->dimGrid.y;
}

__device__ int
get_blockIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return blockIdx.x;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		offset = get_offset_TB_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		offset = get_offset_TB_prl();
		break;
	default:
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

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return blockIdx.y;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		offset = get_offset_TB_static();
		break;
	default:
		brk = get_brk_dyn();
		offset = get_offset_TB_dyn();
		break;
	}
	return ((offset * N_THREADS_PER_mTB) / brk->dimBlock.x) / brk->dimGrid.x;
}

__device__ int
get_blockDimX(void)
{
	benchrun_k_t	*brk;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return blockDim.x;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		break;
	default:
		brk = get_brk_dyn();
		break;
	}

	return brk->dimBlock.x;
}

__device__ int
get_blockDimY(void)
{
	benchrun_k_t	*brk;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return blockDim.y;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		break;
	default:
		brk = get_brk_dyn();
		break;
	}

	return brk->dimBlock.y;
}

__device__ int
get_threadIdxX(void)
{
	benchrun_k_t	*brk;
	unsigned	offset;

        switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		return threadIdx.x;
	case TBS_TYPE_STATIC:
		brk = get_brk_static();
		offset = get_offset_TB_static();
		break;
	case TBS_TYPE_PARALLEL:
		brk = get_brk_prl();
		offset = get_offset_TB_prl();
		break;
	default:
		brk = get_brk_dyn();
		offset = get_offset_TB_dyn();
		break;
	}
	return ((offset * N_THREADS_PER_mTB) % brk->dimBlock.x) + threadIdx.x % N_THREADS_PER_mTB;
}

__device__ void
sync_threads(void)
{
	switch (tbs_type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		__syncthreads();
		return;
	case TBS_TYPE_STATIC:
		break;
	default:
		sync_TB_threads_dyn();
		break;
	}
}

static __global__ void
kernel_init_benchapi(tbs_type_t type)
{
	tbs_type = type;
}

void
init_benchapi(tbs_type_t type)
{
	dim3	dimGrid(1, 1);
	dim3	dimBlock(1, 1);

	kernel_init_benchapi<<<dimGrid, dimBlock>>>(type);
}
