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

unsigned	n_max_mtbs;
unsigned	n_max_mtbs_per_sm;

extern BOOL run_native_tbs(unsigned *pticks);
extern BOOL run_sd_tbs(unsigned *pticks);

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char brid);
extern void assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid);
extern void assign_fedkern_brid_dyn(fedkern_info_t *d_fkinfo, unsigned char brid);

extern void init_sched_static(void);
extern void init_sched_dyn(void);
extern void assign_brid_static(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid, unsigned short offset);
extern void assign_brid_dyn(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid);

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
	n_max_mtbs = n_sm_count * n_max_mtbs_per_sm;

	if (sched->use_static_sched)
		init_sched_static();
	else
		init_sched_dyn();
}

static void
sched_micro_tb(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid, unsigned id_sm, unsigned short *poffset)
{
	int	i;

	ASSERT(brun->dimBlock.x % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->dimBlock.x * brun->dimBlock.y; i += N_THREADS_PER_mTB, (*poffset)++) {
		if (sched->use_static_sched)
			assign_brid_static(fkinfo, id_sm, brid, *poffset);
		else
			assign_brid_dyn(fkinfo, id_sm, brid);
	}
}

static void
sched_brun(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid)
{
	unsigned short	offset = 0;
	int	i, j;

	for (i = 0; i < brun->dimGrid.y; i++) {
		for (j = 0; j < brun->dimGrid.x; j++) {
			if (sched->use_static_sched) {
				unsigned	id_sm = sched->get_tb_sm(brun->dimBlock, j, i);
				if (id_sm == 0) {
					FATAL(3, "schedule failed");
				}
				sched_micro_tb(fkinfo, brun, brid, id_sm, &offset);
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

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	if (sched->direct_mode)
		return run_native_tbs(pticks);
	else
		return run_sd_tbs(pticks);
}
