#include "sdtbs_cu.h"

extern sched_t	sched_rr;

static sched_t	*all_sched[] = {
	&sched_rr, NULL
};

sched_t	*sched = &sched_rr;

extern "C" void
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		if (strcmp(strpol, all_sched[i]->name) == 0) {
			sched = all_sched[i];
			return;
		}
	}

	FATAL(1, "unknown scheduling policy: %s", strpol);
}

static void
sched_micro_tb(benchrun_t *brun, int id_sm)
{
	int	i;

	ASSERT(brun->n_threads_x % N_THREADS_PER_mTB == 0 &&
	       brun->n_threads_x * brun->n_threads_y % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->n_threads_x * brun->n_threads_y; i += N_THREADS_PER_mTB) {
		micro_tb_t	*mtb;

		mtb = get_mtb(id_sm);
		mtb->skid = brun->info->skid;
		memcpy(mtb->args, brun->args, sizeof(void *) * MAX_ARGS);
	}
}

static void
sched_brun(benchrun_t *brun)
{
	int	i, j;

	for (i = 0; i < brun->n_tbs_y; i++) {
		for (j = 0; j < brun->n_tbs_x; j++) {
			int	id_sm = sched->get_tb_sm(j, i);
			sched_micro_tb(brun, id_sm);
		}
	}
}

void
run_schedule(void)
{
	benchrun_t	*brun;
	int	i;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		sched_brun(brun);
	}
}
