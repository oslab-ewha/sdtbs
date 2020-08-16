#include "sdtbs_cu.h"

extern __device__ skrun_t *get_skr_dyn(void);
extern __device__ skrun_t *get_skr_prl(void);
extern __device__ unsigned short get_offset_TB_dyn(void);
extern __device__ unsigned short get_offset_TB_prl(void);
extern __device__ void sync_TB_threads_dyn(void);

extern skrid_t submit_skrun(skid_t skid, dim3 dimGrid, dim3 dimBlock, void *d_args[], int *d_pres);
extern void wait_skrun(skrid_t skrid);

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
	case TBS_TYPE_HW_RELOC:
		return gridDim.x;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
		return gridDim.y;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
		return blockIdx.x;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		offset = get_offset_TB_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
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
	case TBS_TYPE_HW_RELOC:
		return blockDim.x;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
		return blockDim.y;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
		return threadIdx.x;
	case TBS_TYPE_PARALLEL:
		skr = get_skr_prl();
		offset = get_offset_TB_prl();
		break;
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
	case TBS_TYPE_HW_RELOC:
		__syncthreads();
		return;
	default:
		sync_TB_threads_dyn();
		break;
	}
}

__global__ void
sub_kernel_func(skid_t skid, void *args[], int *pres)
{
	int	res;

	res = run_sub_kernel(skid, args);
	if (get_threadIdxX() == 0)
		*pres = res;
}

skrid_t
launch_kernel(skid_t skid, cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *d_args[], int *d_pres)
{
	switch (sched->type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		sub_kernel_func<<<dimGrid, dimBlock, 0, strm>>>(skid, d_args, d_pres);
		return 0;
	case TBS_TYPE_DYNAMIC:
		return submit_skrun(skid, dimGrid, dimBlock, d_args, d_pres);
	default:
		return 0;
	}
}

void
wait_kernel(skrid_t skrid, cudaStream_t strm)
{
	switch (sched->type) {
	case TBS_TYPE_HW:
	case TBS_TYPE_HW_RELOC:
		cudaStreamSynchronize(strm);
		return;
	default:
		wait_skrun(skrid);
		break;
	}
}
