#include "sdtbs_cu.h"

#include <pthread.h>

pthread_t	threads[MAX_BENCHES];

static void *
bench_func(void *ctx)
{
	benchrun_t	*brun = (benchrun_t *)ctx;
	bench_func_t	bench;
	cudaStream_t	strm;
	int		res;

	cudaStreamCreate(&strm);
	bench = sched->type == TBS_TYPE_HW_RELOC ? brun->info->bench_func: brun->info->bench_func_noreloc;
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

	for (i = 0; i < n_benches; i++) {
		void	*ret;
		pthread_join(threads[i], &ret);
	}
}
