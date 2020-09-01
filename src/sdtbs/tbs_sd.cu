#include "sdtbs_cu.h"

extern void init_sched(void);

extern void wait_fedkern_initialized(fedkern_info_t *d_fkinfo);

extern __device__ void try_setup_dyn_sched(fedkern_info_t *fkinfo);

extern __device__ skrid_t get_skrid_dyn(void);
extern __device__ void advance_epoch_dyn(skrid_t skrid);

__device__ BOOL	going_to_shutdown;

static __global__ void
kernel_macro_TB(fedkern_info_t *fkinfo)
{
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		try_setup_dyn_sched(fkinfo);
	}
	__syncthreads();

	while (!going_to_shutdown) {
		skrid_t	skrid;

		skrid = get_skrid_dyn();
		if (skrid == 0)
			return;

		run_sub_kernel(skrid);

		advance_epoch_dyn(skrid);
	}
}

static BOOL
launch_macro_TB(fedkern_info_t *fkinfo)
{
	cudaStream_t	strm;
	cudaError_t	err;
	dim3	dimGrid(n_sm_count, n_MTBs_per_sm);
	dim3	dimBlock(n_threads_per_MTB, 1);

	cudaStreamCreate(&strm);
	kernel_macro_TB<<<dimGrid, dimBlock, 0, strm>>>(fkinfo);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		error("kernel launch error: %s\n", cudaGetErrorString(err));
		return FALSE;
	}

	wait_fedkern_initialized(fkinfo);

	return TRUE;
}

static void
stop_macro_TB(fedkern_info_t *fkinfo)
{
	cudaStream_t	strm;
	BOOL	done = TRUE;

	cudaStreamCreate(&strm);
	cudaMemcpyAsync(&fkinfo->sched_done, &done, sizeof(BOOL), cudaMemcpyHostToDevice, strm);
	cudaStreamSynchronize(strm);
}

////TODO
extern void start_benchruns(void);
extern void wait_benchruns(void);

BOOL
run_sd_tbs(unsigned *pticks)
{
	fedkern_info_t	*fkinfo;

	if (!setup_gpu_devinfo()) {
		error("no gpu found");
		return FALSE;
	}

	init_sched();

	fkinfo = create_fedkern_info();

	if (!launch_macro_TB(fkinfo))
		return FALSE;

	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	stop_macro_TB(fkinfo);

	free_fedkern_info(fkinfo);

	return TRUE;
}
