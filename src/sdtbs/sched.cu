#include "sdtbs_cu.h"

extern sched_t	sched_rr;
extern sched_t	sched_rrf;

static sched_t	*all_sched[] = {
	&sched_rr, &sched_rrf, NULL
};

sched_t	*sched = &sched_rr;

unsigned n_grid_width, n_grid_height;
unsigned n_tb_width, n_tb_height;

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
sched_micro_tb(benchrun_t *brun, unsigned id_sm)
{
	int	i;

	ASSERT(brun->n_tb_width % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->n_tb_width * brun->n_tb_height; i += N_THREADS_PER_mTB) {
		micro_tb_t	*mtb;

		mtb = get_mtb(id_sm);
		if (mtb == NULL) {
			FATAL(3, "no micro tb avaiable in SM[%u]", id_sm);
		}
		mtb->skid = brun->info->skid;
		memcpy(mtb->args, brun->args, sizeof(void *) * MAX_ARGS);
	}
}

static void
sched_brun(benchrun_t *brun)
{
	int	i, j;

	n_grid_width = brun->n_grid_width;
	n_grid_height = brun->n_grid_height;
	n_tb_width = brun->n_tb_width;
	n_tb_height = brun->n_tb_height;

	for (i = 0; i < brun->n_grid_height; i++) {
		for (j = 0; j < brun->n_grid_width; j++) {
			unsigned	id_sm = sched->get_tb_sm(j, i);
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
