#include "sdtbs_cu.h"

extern __device__ skrun_t *get_skr_dyn(void);
extern __device__ unsigned short get_offset_TB_dyn(void);
extern __device__ unsigned short get_offset_TB_prl(void);
extern __device__ void sync_TB_threads_dyn(void);

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
        skrun_t    *skr;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return gridDim.x;
	default:
		skr = get_skr_dyn();
		break;
	}

	return skr->dimGrid.x;
}

__device__ int
get_gridDimY(void)
{
        skrun_t	*skr;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return gridDim.y;
	default:
		skr = get_skr_dyn();
		break;
	}

	return skr->dimGrid.y;
}

__device__ int
get_blockIdxX(void)
{
	skrun_t	*skr;
	unsigned	offset;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return blockIdx.x;
	default:
		skr = get_skr_dyn();
		offset = get_offset_TB_dyn();
	}
	return ((offset * N_THREADS_PER_mTB) / skr->dimBlock.x) % skr->dimGrid.x;
}

__device__ int
get_blockIdxY(void)
{
	skrun_t	*skr;
	unsigned	offset;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return blockIdx.y;
	default:
		skr = get_skr_dyn();
		offset = get_offset_TB_dyn();
		break;
	}
	return ((offset * N_THREADS_PER_mTB) / skr->dimBlock.x) / skr->dimGrid.x;
}

__device__ int
get_blockDimX(void)
{
	skrun_t	*skr;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return blockDim.x;
	default:
		skr = get_skr_dyn();
		break;
	}

	return skr->dimBlock.x;
}

__device__ int
get_blockDimY(void)
{
	skrun_t	*skr;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return blockDim.y;
	default:
		skr = get_skr_dyn();
		break;
	}

	return skr->dimBlock.y;
}

__device__ int
get_threadIdxX(void)
{
	skrun_t	*skr;
	unsigned	offset;

        switch (d_tbs_type) {
	case TBS_TYPE_HW:
		return threadIdx.x;
	default:
		skr = get_skr_dyn();
		offset = get_offset_TB_dyn();
		break;
	}
	return ((offset * N_THREADS_PER_mTB) % skr->dimBlock.x) + threadIdx.x % N_THREADS_PER_mTB;
}

__device__ void
sync_threads(void)
{
	switch (d_tbs_type) {
	case TBS_TYPE_HW:
		__syncthreads();
		return;
	default:
		sync_TB_threads_dyn();
		break;
	}
}
