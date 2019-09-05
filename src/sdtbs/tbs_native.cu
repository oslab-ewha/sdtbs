#include "sdtbs_cu.h"

/* use MAX_ARGS + 1 for result value */
#define SIZE_ARGS	(sizeof(void *) * MAX_ARGS)

static cudaStream_t	strms[MAX_BENCHES];

BOOL
run_native_tbs(unsigned *pticks)
{
	benchrun_t	*brun;
	void 	**d_args_brun;
	int	*d_benches_res;
	int	i;

	cudaMalloc(&d_benches_res, sizeof(int) * n_benches);
	cudaMalloc(&d_args_brun, SIZE_ARGS * n_benches);
	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		cudaStreamCreate(strms + i);
		cudaMemcpy((char *)d_args_brun + SIZE_ARGS * i, brun->args, SIZE_ARGS, cudaMemcpyHostToDevice);
	}

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		bench_func_t	bench;
		cudaError_t	err;

		bench = sched->type == TBS_TYPE_HW_RELOC ? brun->info->bench_func: brun->info->bench_func_noreloc;
		bench(strms[i], brun->dimGrid, brun->dimBlock, (void **)((char *)d_args_brun + SIZE_ARGS * i), d_benches_res + i);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("error: %s\n", cudaGetErrorString(err));
			return FALSE;
		}
	}

	init_tickcount();

	for (i = 0; i < n_benches; i++)
		cudaStreamSynchronize(strms[i]);

	*pticks = get_tickcount();

	cudaFree(d_args_brun);

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		cudaMemcpy(&brun->res, d_benches_res + i, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_benches_res);

	return TRUE;
}
