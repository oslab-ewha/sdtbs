#include "sdtbs_cu.h"

extern "C" void
run_native_tbs(void)
{
	benchrun_t	*brun;
	cudaStream_t	strm;
	int	i;

	brun = benchruns;
	cudaStreamCreate(&strm);
	for (i = 0; i < n_benches; i++, brun++) {
		void 	**d_args;

		cudaMalloc(&d_args, sizeof(void *) * MAX_ARGS);
		cudaMemcpy(d_args, brun->args, sizeof(void *) * MAX_ARGS, cudaMemcpyHostToDevice);
		brun->info->bench_func(strm, brun->n_tbs_x, brun->n_tbs_y, brun->n_threads_x, brun->n_threads_y, d_args);
		cudaStreamSynchronize(strm);
		cudaFree(d_args);
	}
}
