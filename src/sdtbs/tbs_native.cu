#include "sdtbs.h"

#include "benchmgr.h"

extern "C" void
run_native_tbs(void)
{
	benchrun_t	*brun;
	cudaStream_t	strm;
	int	i;

	brun = benchruns;
	cudaStreamCreate(&strm);
	for (i = 0; i < n_benches; i++, brun++) {
		int	*d_args;

		cudaMalloc(&d_args, sizeof(int) * MAX_ARGS);
		cudaMemcpy(d_args, brun->args, sizeof(int) * MAX_ARGS, cudaMemcpyHostToDevice);
		brun->info->bench_native(strm, brun->n_tbs_x, brun->n_tbs_y, brun->n_threads_x, brun->n_threads_y, d_args);
		cudaStreamSynchronize(strm);
		cudaFree(d_args);
	}
}
