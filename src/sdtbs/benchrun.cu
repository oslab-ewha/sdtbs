#include "sdtbs_cu.h"

#include <cuda.h>

#include <pthread.h>

static CUcontext	context;

pthread_t	threads[MAX_BENCHES];

static void *
bench_func(void *ctx)
{
	benchrun_t	*brun = (benchrun_t *)ctx;
	bench_func_t	bench;
	cudaStream_t	strm;
	int		res;

	cuCtxSetCurrent(context);
	cudaStreamCreate(&strm);
	bench = brun->info->bench_func;
	res = bench(strm, brun->dimGrid, brun->dimBlock, brun->args);
	cudaStreamSynchronize(strm);
	cudaStreamDestroy(strm);

	brun->res = res;
	return NULL;
}

void
start_benchruns(void)
{
	benchrun_t	*brun;
	int	i;

	for (i = 0, brun = benchruns; i < n_benches; i++, brun++) {
		pthread_create(&threads[i], NULL, bench_func, brun);
	}
}

void
wait_benchruns(void)
{
	int	i;

	cuCtxGetCurrent(&context);
	for (i = 0; i < n_benches; i++) {
		void	*ret;
		pthread_join(threads[i], &ret);
	}
}
