#include "sdtbs_cu.h"

__device__ int loopcalc(void *args[]);
__device__ int gma(void *args[]);

__device__ static uint
get_smid(void)
{
	uint	ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret));
	return ret;
}

__global__ static void
kernel_macro_TB(int n_mtbs_per_sm, micro_tb_t *mtbs)
{
	micro_tb_t	*mtb;
	int	n_mtbs_per_width;
	int	res;

	n_mtbs_per_width = blockDim.x / N_THREADS_PER_mTB;
	mtb = mtbs + get_smid() * n_mtbs_per_sm + n_mtbs_per_width * threadIdx.y + threadIdx.x / N_THREADS_PER_mTB;

	switch (mtb->skid) {
	case 1:
		res = loopcalc(mtb->args);
		break;
	case 2:
		res = gma(mtb->args);
		break;
	default:
		goto out;
	}
	if (threadIdx.x % 32 == 0)
		mtb->args[0] = (void *)(long long)res;
out:
	__syncthreads();
}

static void
launch_macro_TB(int n_mtbs_per_sm, micro_tb_t *mtbs)
{
	cudaError_t	err;

	dim3 dimGrid(n_sm_count, 1);
	dim3 dimBlock(n_threads_per_tb, 1);

	kernel_macro_TB<<<dimGrid, dimBlock, 0>>>(n_mtbs_per_sm, mtbs);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		error("kernel launch error: %s\n", cudaGetErrorString(err));
		return;
	}

	cudaDeviceSynchronize();
}

extern "C" void
run_sd_tbs(void)
{
	micro_tb_t	*d_mtbs;

	setup_gpu_devinfo();
	setup_micro_tbs();

	cudaMalloc(&d_mtbs, n_mtbs * sizeof(micro_tb_t));

	run_schedule();

	cudaMemcpy(d_mtbs, mtbs, n_mtbs * sizeof(micro_tb_t), cudaMemcpyHostToDevice);

	launch_macro_TB(n_mtbs_per_sm, d_mtbs);

	cudaFree(d_mtbs);
}
