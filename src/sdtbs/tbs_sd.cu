#include "sdtbs_cu.h"

extern void init_sched(void);

extern void wait_fedkern_initialized(fedkern_info_t *d_fkinfo);

extern __device__ void try_setup_dyn_sched(fedkern_info_t *fkinfo);
extern __device__ void run_schedule_as_solo(fedkern_info_t *fkinfo);
extern __device__ void setup_host_sched(fedkern_info_t *fkinfo);
extern __device__ void setup_prl_sched(fedkern_info_t *fkinfo);

extern __device__ skrid_t get_skrid_dyn(BOOL *pis_primary_mtb);
extern __device__ unsigned char get_skrid_host(BOOL *pis_primary_mtb);
extern __device__ unsigned char get_skrid_prl(BOOL *pis_primary_mtb);
extern __device__ void advance_epoch_dyn(skrid_t skrid);
extern __device__ void advance_epoch_host(void);
extern __device__ void advance_epoch_prl(void);

__device__ BOOL	going_to_shutdown;

static __device__ void
kernel_macro_TB_dynamic_sched(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		try_setup_dyn_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		skrun_t	*skr;
		unsigned char	skrid;
		int	res;
		BOOL	is_primary_mtb;

		skrid = get_skrid_dyn(&is_primary_mtb);
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		res = run_sub_kernel((skid_t)skr->skid, skr->d_args);

		advance_epoch_dyn(skrid);

		if (is_primary_mtb)
			*skr->d_pres = res;
	}
}

static __device__ void
kernel_macro_TB_host_sched(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		setup_host_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		unsigned char	skrid;
		skrun_t	*skr;
		BOOL	is_primary_mtb;
		int	res;

		skrid = get_skrid_host(&is_primary_mtb);
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		res = run_sub_kernel((skid_t)skr->skid, skr->d_args);

		advance_epoch_host();

		if (is_primary_mtb && threadIdx.x % N_THREADS_PER_mTB == 0) {
			*skr->d_pres = res;
		}
	}
}

static __device__ void
kernel_macro_TB_parallel_sched(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		setup_prl_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		unsigned char   skrid;
		skrun_t    *skr;
		BOOL    is_primary_mtb;
		int     res;

		skrid = get_skrid_prl(&is_primary_mtb);
		if (skrid == 0)
			return;

		skr = &d_skruns[skrid - 1];
		res = run_sub_kernel((skid_t)skr->skid, skr->d_args);

		advance_epoch_prl();

		if (is_primary_mtb && threadIdx.x % N_THREADS_PER_mTB == 0) {
			*skr->d_pres = res;
		}
	}
}

__global__ static void
kernel_macro_TB(fedkern_info_t *fkinfo)
{
	switch (d_tbs_type) {
	case TBS_TYPE_DYNAMIC:
		kernel_macro_TB_dynamic_sched(fkinfo);
		break;
	case TBS_TYPE_HOST:
		kernel_macro_TB_host_sched(fkinfo);
		break;
	case TBS_TYPE_PARALLEL:
		kernel_macro_TB_parallel_sched(fkinfo);
		break;
	default:
		break;
	}
}

__global__ static void
kernel_solo_sched(fedkern_info_t *fkinfo)
{
	run_schedule_as_solo(fkinfo);
}

static BOOL
launch_macro_TB(fedkern_info_t *d_fkinfo, fedkern_info_t *fkinfo)
{
	cudaStream_t	strm, strm_solo;
	cudaError_t	err;
	dim3	dimGrid(n_sm_count, n_MTBs_per_sm);
	dim3	dimBlock(n_threads_per_MTB, 1);

	cudaStreamCreate(&strm);
	kernel_macro_TB<<<dimGrid, dimBlock, 0, strm>>>(d_fkinfo);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		error("kernel launch error: %s\n", cudaGetErrorString(err));
		return FALSE;
	}

	if (sched->type == TBS_TYPE_SOLO) {
		cudaStreamCreate(&strm_solo);
		dim3	dimGrid(1, 1), dimBlock(1, 1);
		kernel_solo_sched<<<dimGrid, dimBlock, 0, strm_solo>>>(d_fkinfo);
	}

	wait_fedkern_initialized(d_fkinfo);

	if (sched->type == TBS_TYPE_DYNAMIC) {
		///run_schedule_dyn(fkinfo);
	}
	else if (sched->type == TBS_TYPE_HOST)
		run_schedule_host(fkinfo);

	return TRUE;
}

static void
collect_results(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i;

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		///TODO brun->res = fkinfo->skruns[i].res;
	}
}

////TODO
extern void start_benchruns(void);
extern void wait_benchruns(void);

BOOL
run_sd_tbs(unsigned *pticks)
{
	fedkern_info_t	*fkinfo;
	fedkern_info_t	*d_fkinfo;
	extern fedkern_info_t	*fkinfo_dyn;

	if (!setup_gpu_devinfo()) {
		error("no gpu found");
		return FALSE;
	}

	init_sched();

	fkinfo_dyn = fkinfo = create_fedkern_info();

	d_fkinfo = create_fedkern_info_kernel(fkinfo);

	if (!launch_macro_TB(d_fkinfo, fkinfo))
		return FALSE;

	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	cudaMemcpy(fkinfo, d_fkinfo, sizeof(fedkern_info_t), cudaMemcpyDeviceToHost);
	collect_results(fkinfo);

	free_fedkern_info(fkinfo);
	fkinfo_dyn = NULL;
	cudaFree(d_fkinfo);

	return TRUE;
}
