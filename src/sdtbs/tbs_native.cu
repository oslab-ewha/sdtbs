#include "sdtbs_cu.h"

#define SIZE_ARGS	(sizeof(void *) * MAX_ARGS)

static cudaStream_t	strms[MAX_BENCHES];

extern "C" BOOL
run_native_tbs(unsigned *pticks)
{
	benchrun_t	*brun;
	void 	**d_args_brun;
	int	i;

	cudaMalloc(&d_args_brun, SIZE_ARGS * n_benches);
	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		cudaStreamCreate(strms + i);
		cudaMemcpy((char *)d_args_brun + SIZE_ARGS * i, brun->args, SIZE_ARGS, cudaMemcpyHostToDevice);
	}

	init_tickcount();

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		int	ret;

		ret = brun->info->bench_func(strms[i], brun->n_grid_width, brun->n_grid_height, brun->n_tb_width,
					     brun->n_tb_height, (void **)((char *)d_args_brun + SIZE_ARGS * i));
		if (ret < 0)
			return FALSE;
	}

	for (i = 0; i < n_benches; i++)
		cudaStreamSynchronize(strms[i]);

	*pticks = get_tickcount();

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		cudaMemcpy(brun->args, d_args_brun + SIZE_ARGS * i, SIZE_ARGS, cudaMemcpyDeviceToHost);
		brun->res = (int)(long long)brun->args[0];
	}
	cudaFree(d_args_brun);

	return TRUE;
}
