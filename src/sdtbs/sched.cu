#include "sdtbs_cu.h"

extern sched_t	sched_hw;
extern sched_t	sched_rr;
extern sched_t	sched_rrf;
extern sched_t	sched_fca;
extern sched_t	sched_rrm;

static sched_t	*all_sched[] = {
	&sched_hw,
	&sched_rr, &sched_rrf, &sched_fca, &sched_rrm, NULL
};

sched_t	*sched = &sched_hw;
unsigned	sched_id = 1;
char		*sched_argstr;

unsigned	n_max_mtbs_per_sm;

static unsigned	*n_cur_mtbs_per_sm;

static unsigned n_cur_mtbs;

extern BOOL run_native_tbs(unsigned *pticks);
extern BOOL run_sd_tbs(unsigned *pticks);

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char brid);
extern void assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid);
extern void assign_fedkern_brid_dyn(fedkern_info_t *d_fkinfo, unsigned char brid);

extern "C" void
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		int	len = strlen(all_sched[i]->name);

		if (strncmp(strpol, all_sched[i]->name, len) == 0) {
			sched = all_sched[i];
			sched_id = i + 1;
			if (strpol[len] ==':')
				sched_argstr = strdup(strpol + len + 1);
			else if (strpol[len] == 'D' && strpol[len + 1] == '\0') {
				sched->use_semi_dynamic_sched = TRUE;
			}
			else if (strpol[len] == 'R' && strpol[len + 1] == '\0') {
				sched->use_relocatable = TRUE;
			}
			else if (strpol[len] == 'S' && strpol[len + 1] == '\0') {
				sched->use_static_sched = TRUE;
			}
			else if (strpol[len] != '\0') {
				continue;
			}
			sched->name = strdup(strpol);
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
	if (sched->use_static_sched) {
		unsigned	idx;

		if (n_cur_mtbs_per_sm[id_sm - 1] == n_max_mtbs_per_sm)
			return FALSE;
		idx = (id_sm - 1) * n_max_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm - 1];
		fkinfo->brids[idx] = brid;
		n_cur_mtbs_per_sm[id_sm - 1]++;
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

	ASSERT(brun->dimBlock.x % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->dimBlock.x * brun->dimBlock.y; i += N_THREADS_PER_mTB) {
		if (!assign_brid(fkinfo, id_sm, brid)) {
			FATAL(3, "no micro tb avaiable in SM[%u]", id_sm);
		}
	}
}

static void
sched_brun(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid)
{
	int	i, j;

	for (i = 0; i < brun->dimGrid.y; i++) {
		for (j = 0; j < brun->dimGrid.x; j++) {
			if (sched->use_static_sched) {
				unsigned	id_sm = sched->get_tb_sm(brun->dimBlock, j, i);
				if (id_sm == 0) {
					FATAL(3, "schedule failed");
				}
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

	if (sched->get_tb_sm == NULL && sched->use_static_sched) {
		error("static scheduling not supported");
		return FALSE;
	}
	if (sched->parse_arg != NULL) {
		if (sched_argstr == NULL) {
			error("empty policy argument");
			return FALSE;
		}
		fkinfo->sched_arg = sched->parse_arg(sched_argstr);
	}

	if (!sched->use_semi_dynamic_sched && !sched->use_static_sched)
		fkinfo->fully_dynamic = TRUE;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		assign_fedkern_brun(fkinfo, brun, i + 1);
		if (sched->use_semi_dynamic_sched || sched->use_static_sched)
			sched_brun(fkinfo, brun, i + 1);
	}

	return TRUE;
}

void
run_schedule_dyn(fedkern_info_t *d_fkinfo)
{
	benchrun_t	*brun;
	int	i, j;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		for (j = 0; j < brun->dimGrid.y * brun->dimGrid.x; j++) {
			assign_fedkern_brid_dyn(d_fkinfo, i + 1);
		}
	}
}

BOOL
is_sm_avail(int id_sm, unsigned n_threads)
{
	unsigned	n_mtbs_new = (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm - 1] + n_mtbs_new <= n_max_mtbs_per_sm)
		return TRUE;
	return FALSE;
}

unsigned
get_sm_n_sched_mtbs(int id_sm)
{
	return n_cur_mtbs_per_sm[id_sm - 1];
}

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	if (sched->direct_mode)
		return run_native_tbs(pticks);
	else
		return run_sd_tbs(pticks);
}
