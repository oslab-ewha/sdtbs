#include "sdtbs_cu.h"

extern sched_t	sched_rr;
extern sched_t	sched_rrf;
extern sched_t	sched_fca;

static sched_t	*all_sched[] = {
	&sched_rr, &sched_rrf, &sched_fca, NULL
};

sched_t	*sched = &sched_rr;
unsigned	sched_id = 1;

unsigned n_grid_width, n_grid_height;
unsigned n_tb_width, n_tb_height;

unsigned	n_max_mtbs_per_sm;

static unsigned	*n_cur_mtbs_per_sm;
static unsigned n_cur_mtbs;

extern void assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid);

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

void
init_sched(void)
{
	n_max_mtbs_per_sm = n_threads_per_MTB / N_THREADS_PER_mTB * n_MTBs_per_sm;
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned));
}

static BOOL
assign_brid(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid)
{
	if (use_static_sched) {
		unsigned	idx;

		if (n_cur_mtbs_per_sm[id_sm] == n_max_mtbs_per_sm)
			return FALSE;
		idx = id_sm * n_max_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm];
		fkinfo->brids[idx] = brid;
		n_cur_mtbs_per_sm[id_sm]++;
		if (fkinfo->bruns[brid - 1].primary_mtb_idx == 0)
			fkinfo->bruns[brid - 1].primary_mtb_idx = n_cur_mtbs + 1;
	}
	else {
		fkinfo->brids[n_cur_mtbs] = brid;
	}
	n_cur_mtbs++;

	return TRUE;
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
				assign_fedkern_brid(fkinfo, brid);
			}
		}
	}
}

BOOL
run_schedule(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i;

	if (sched->get_tb_sm == NULL && use_static_sched) {
		error("static scheduling not supported");
		return FALSE;
	}

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		sched_brun(fkinfo, brun, i + 1);
	}

	return TRUE;
}

BOOL
is_sm_avail(int id_sm, unsigned n_threads)
{
	unsigned	n_mtbs_new = (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm] + n_mtbs_new <= n_max_mtbs_per_sm)
		return TRUE;
	return FALSE;
}
