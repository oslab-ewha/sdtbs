#include "sdtbs_cu.h"

extern sched_t	sched_hw;
extern sched_t	sched_rr;
extern sched_t	sched_rrf;
extern sched_t	sched_fca;
extern sched_t	sched_rrm;
extern sched_t	sched_rrp;

static sched_t	*all_sched[] = {
	&sched_hw,
	&sched_rr, &sched_rrf, &sched_fca, &sched_rrm, &sched_rrp, NULL
};

sched_t	*sched = &sched_hw;
unsigned	sched_id = 1;
char		*sched_argstr;

unsigned	n_max_mtbs;
unsigned	n_max_mtbs_per_sm;

extern BOOL run_native_tbs(unsigned *pticks);
extern BOOL run_sd_tbs(unsigned *pticks);

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char brid);
extern void assign_fedkern_brid_static(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid, unsigned id_sm, unsigned short *poffset);
extern void assign_fedkern_brid_dyn(fedkern_info_t *fkinfo, unsigned char brid);
extern void assign_fedkern_brid_kernel(fedkern_info_t *d_fkinfo, unsigned char brid);

extern void init_benchapi(tbs_type_t type);

extern "C" void
setup_sched(const char *strpol)
{
	unsigned	i;

	for (i = 0; all_sched[i]; i++) {
		int	len = strlen(all_sched[i]->name);

		if (strncmp(strpol, all_sched[i]->name, len) == 0) {
			tbs_type_t	type;

			sched = all_sched[i];
			type = sched->type;
			sched_id = i + 1;

			if (strpol[len] == 'D') {
				type = TBS_TYPE_SEMI_DYNAMIC;
				len++;
			}
			else if (strpol[len] == 'R') {
				if (sched->type != TBS_TYPE_HW)
					FATAL(1, "invalid policy: %s", strpol);
				type = TBS_TYPE_HW_RELOC;
				len++;
			}
			else if (strpol[len] == 'S') {
				type = TBS_TYPE_STATIC;
				len++;
			}
			else if (strpol[len] == 'O') {
				type = TBS_TYPE_SOLO;
				len++;
			}
			else if (strpol[len] == 'H') {
				type = TBS_TYPE_HOST;
				len++;
			}
                        else if (strpol[len] == 'P') {
				type = TBS_TYPE_PARALLEL;
				len++;
			}

			if (strpol[len] ==':')
				sched_argstr = strdup(strpol + len + 1);
			else if (strpol[len] != '\0')
				continue;

			sched->name = strdup(strpol);
			sched->type = type;
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
}

static void
sched_brun(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid)
{
	unsigned short	offset = 0;
	int	i, j;

	ASSERT(brun->dimBlock.x % N_THREADS_PER_mTB == 0);

	for (i = 0; i < brun->dimGrid.y; i++) {
		for (j = 0; j < brun->dimGrid.x; j++) {
			if (sched->type == TBS_TYPE_STATIC) {
				unsigned	id_sm = sched->get_tb_sm(brun->dimBlock, j, i);
				if (id_sm == 0) {
					FATAL(3, "schedule failed");
				}
				assign_fedkern_brid_static(fkinfo, brun, brid, id_sm, &offset);
			}
			else {
				assign_fedkern_brid_dyn(fkinfo, brid);
			}
		}
	}
}

BOOL
run_schedule(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i;

	if (sched->get_tb_sm == NULL && sched->type == TBS_TYPE_STATIC) {
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

	fkinfo->n_bruns = n_benches;
	fkinfo->tbs_type = sched->type;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		assign_fedkern_brun(fkinfo, brun, i + 1);
		if (sched->type == TBS_TYPE_SEMI_DYNAMIC || sched->type == TBS_TYPE_SOLO || sched->type == TBS_TYPE_STATIC)
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
			assign_fedkern_brid_kernel(d_fkinfo, i + 1);
		}
	}
}

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	init_benchapi(sched->type);

	if (sched->type == TBS_TYPE_HW || sched->type == TBS_TYPE_HW_RELOC)
		return run_native_tbs(pticks);
	else
		return run_sd_tbs(pticks);
}
