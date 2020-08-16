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

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char skrid);
extern void assign_fedkern_brid_dyn(fedkern_info_t *fkinfo, unsigned char brid);
extern void assign_fedkern_brid_kernel(fedkern_info_t *d_fkinfo, unsigned char brid);

extern void init_skrun(void);

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

			if (strpol[len] == 'R') {
				if (sched->type != TBS_TYPE_HW)
					FATAL(1, "invalid policy: %s", strpol);
				type = TBS_TYPE_HW_RELOC;
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

void
run_schedule_dyn(fedkern_info_t *fkinfo)
{
	benchrun_t	*brun;
	int	i, j;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		for (j = 0; j < brun->dimGrid.y * brun->dimGrid.x; j++) {
			assign_fedkern_brid_kernel(fkinfo, i + 1);
		}
	}
}

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	init_skrun();

	if (sched->type == TBS_TYPE_HW || sched->type == TBS_TYPE_HW_RELOC)
		return run_native_tbs(pticks);
	else
		return run_sd_tbs(pticks);
}
