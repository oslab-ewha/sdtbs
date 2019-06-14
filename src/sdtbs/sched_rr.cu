#include "sdtbs_cu.h"

static int	id_sm;

extern micro_tb_t *get_mtb(int id_sm);

static void
sched_rr_tb(benchrun_t *brun)
{
	int	i;

	ASSERT(brun->n_threads_x % N_THREADS_PER_mTB == 0 &&
	       brun->n_threads_x * brun->n_threads_y % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->n_threads_x * brun->n_threads_y; i += N_THREADS_PER_mTB) {
		micro_tb_t	*mtb;

		mtb = get_mtb(id_sm);
		mtb->skid = brun->info->skid;
		memcpy(mtb->args, brun->args, sizeof(int) * MAX_ARGS);
	}
}

static void
sched_rr_brun(benchrun_t *brun)
{
	int	i, j;

	for (i = 0; i < brun->n_tbs_y; i++) {
		for (j = 0; j < brun->n_tbs_x; j++) {
			sched_rr_tb(brun);
			id_sm = (id_sm + 1) % n_sm_count;
		}
	}
}

void
sched_rr(void)
{
	benchrun_t	*brun;
	int	i;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		sched_rr_brun(brun);
	}
}
