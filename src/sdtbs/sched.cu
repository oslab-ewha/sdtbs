#include "sdtbs_cu.h"

extern sched_t	sched_rr;
extern sched_t	sched_rrf;

static sched_t	*all_sched[] = {
	&sched_rr, &sched_rrf, NULL
};

sched_t	*sched = &sched_rr;
unsigned	sched_id = 1;

unsigned n_grid_width, n_grid_height;
unsigned n_tb_width, n_tb_height;

extern "C" void
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		if (strcmp(strpol, all_sched[i]->name) == 0) {
			sched = all_sched[i];
			sched_id = i + 1;
			return;
		}
	}

	FATAL(1, "unknown scheduling policy: %s", strpol);
}

static void
sched_micro_tb(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid, unsigned id_sm)
{
	int	i;

	ASSERT(brun->n_tb_width % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->n_tb_width * brun->n_tb_height; i += N_THREADS_PER_mTB) {
		if (!assign_brid(fkinfo, id_sm, brid)) {
			FATAL(3, "no micro tb avaiable in SM[%u]", id_sm);
		}
	}
}

static void
sched_brun(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid)
{
	benchrun_k_t	*brk;
	int	i, j;

	brk = &fkinfo->bruns[brid - 1];
	brk->skid = brun->info->skid;
	memcpy(brk->args, brun->args, sizeof(void *) * MAX_ARGS);
	brk->n_mtbs_per_tb = brun->n_tb_width * brun->n_tb_height / N_THREADS_PER_mTB;

	n_grid_width = brun->n_grid_width;
	n_grid_height = brun->n_grid_height;
	n_tb_width = brun->n_tb_width;
	n_tb_height = brun->n_tb_height;

	for (i = 0; i < brun->n_grid_height; i++) {
		for (j = 0; j < brun->n_grid_width; j++) {
			if (use_static_sched) {
				unsigned	id_sm = sched->get_tb_sm(j, i);
				sched_micro_tb(fkinfo, brun, brid, id_sm);
			}
			else {
				assign_brid(fkinfo, 0, brid);
			}
		}
	}
}

void
run_schedule(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		sched_brun(fkinfo, brun, i + 1);
	}
}
