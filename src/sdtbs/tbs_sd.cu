#include "sdtbs_cu.h"

extern void init_sched(void);

extern void wait_fedkern_initialized(fedkern_info_t *d_fkinfo);

extern __device__ void setup_static_sched(fedkern_info_t *fkinfo);
extern __device__ void setup_dyn_sched(fedkern_info_t *fkinfo);
extern __device__ unsigned char get_brid_static(BOOL *pis_primary_mtb);
extern __device__ unsigned char get_brid_dyn(BOOL *pis_primary_mtb);
extern __device__ void advance_epoch_static(void);
extern __device__ void advance_epoch_dyn(void);

__device__ BOOL	going_to_shutdown;

__device__ int loopcalc(void *args[]);
__device__ int gma(void *args[]);
__device__ int lma(void *args[]);
__device__ int kmeans(void *args[]);

__device__ static int
run_bench(int skid, void *args[])
{
	switch (skid) {
	case 1:
		return loopcalc(args);
	case 2:
		return gma(args);
	case 3:
		return lma(args);
	case 4:
		return kmeans(args);
	default:
		return 0;
	}
}

static __device__ void
kernel_macro_TB_static_sched(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		setup_static_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		unsigned char	brid;
		benchrun_k_t	*brk;
		BOOL	is_primary_mtb;
		int	res;

		brid = get_brid_static(&is_primary_mtb);
		if (brid == 0)
			return;

		brk = &fkinfo->bruns[brid - 1];
		res = run_bench(brk->skid, brk->args);

		advance_epoch_static();

		if (is_primary_mtb && threadIdx.x % N_THREADS_PER_mTB == 0) {
			brk->res = res;
		}
	}
}

static __device__ void
kernel_macro_TB_dynamic_sched(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		setup_dyn_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		benchrun_k_t	*brk;
		unsigned char	brid;
		int	res;
		BOOL	is_primary_mtb;

		brid = get_brid_dyn(&is_primary_mtb);
		if (brid == 0)
			return;

		brk = &fkinfo->bruns[brid - 1];
		res = run_bench(brk->skid, brk->args);

		advance_epoch_dyn();

		if (is_primary_mtb)
			brk->res = res;
	}
}

__global__ static void
kernel_macro_TB(fedkern_info_t *fkinfo)
{
	if (fkinfo->sched_id == 0)
		kernel_macro_TB_static_sched(fkinfo);
	else
		kernel_macro_TB_dynamic_sched(fkinfo);
}

static BOOL
launch_macro_TB(fedkern_info_t *d_fkinfo, fedkern_info_t *fkinfo)
{
	cudaStream_t	strm;
	cudaError_t	err;

	dim3 dimGrid(n_sm_count, n_MTBs_per_sm);
	dim3 dimBlock(n_threads_per_MTB, 1);

	cudaStreamCreate(&strm);
	kernel_macro_TB<<<dimGrid, dimBlock, 0, strm>>>(d_fkinfo);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		error("kernel launch error: %s\n", cudaGetErrorString(err));
		return FALSE;
	}

	wait_fedkern_initialized(d_fkinfo);

	init_tickcount();

	if (!sched->use_semi_dynamic_sched && !sched->use_static_sched)
		run_schedule_dyn(fkinfo);

	cudaDeviceSynchronize();
	return TRUE;
}

static void
collect_results(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i;

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		brun->res = fkinfo->bruns[i].res;
	}
}

BOOL
run_sd_tbs(unsigned *pticks)
{
	fedkern_info_t	*fkinfo;
	fedkern_info_t	*d_fkinfo;

	if (!setup_gpu_devinfo()) {
		error("no gpu found");
		return FALSE;
	}

	init_sched();

	fkinfo = create_fedkern_info();

	if (!run_schedule(fkinfo))
		return FALSE;

	d_fkinfo = create_fedkern_info_kernel(fkinfo);

	if (!launch_macro_TB(d_fkinfo, fkinfo))
		return FALSE;

	*pticks = get_tickcount();

	cudaMemcpy(fkinfo, d_fkinfo, sizeof(fedkern_info_t), cudaMemcpyDeviceToHost);
	collect_results(fkinfo);

	free_fedkern_info(fkinfo);
	cudaFree(d_fkinfo);

	return TRUE;
}
