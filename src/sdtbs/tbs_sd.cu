#include "sdtbs.h"

#include "benchmgr.h"

typedef struct {
	int	skid;
	int	n_tbs_x, n_tbs_y;
	int	n_threads_x, n_threads_y;
	int	args[MAX_ARGS];
} subkernel_t;

__device__ int loopcalc(int args[]);

__global__ static void
kernel_macro_TB(int n_benches, subkernel_t *sks)
{
	int	i;

	for (i = 0; i < n_benches; i++, sks++) {
		switch (sks->skid) {
		case 1:
			sks->args[0] = loopcalc(sks->args);
			break;
		default:
			break;
		}
	}
}

static void
launch_macro_TB(subkernel_t *d_subkernels)
{
	cudaError_t	err;

	dim3 dimGrid(9, 1);
	dim3 dimBlock(32, 1);

	kernel_macro_TB<<<dimGrid, dimBlock, 0>>>(n_benches, d_subkernels);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(err));
		return;
	}

	cudaDeviceSynchronize();
}

extern "C" void
run_sd_tbs(void)
{
	subkernel_t	subkernels[MAX_BENCHES];
	subkernel_t	*d_subkernels;
	benchrun_t	*brun;
	int	i;

	cudaMalloc(&d_subkernels, sizeof(subkernel_t) * n_benches);
	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		subkernels[i].skid = brun->info->skid;
		subkernels[i].n_tbs_x = brun->n_tbs_x;
		subkernels[i].n_tbs_y = brun->n_tbs_y;
		subkernels[i].n_threads_x = brun->n_threads_x;
		subkernels[i].n_threads_y = brun->n_threads_y;
		memcpy(subkernels[i].args, brun->args, sizeof(int) * MAX_ARGS);
	}

	cudaMemcpy(d_subkernels, subkernels, n_benches * sizeof(subkernel_t), cudaMemcpyHostToDevice);

	launch_macro_TB(d_subkernels);
}
