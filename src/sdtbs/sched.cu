#include "sdtbs_cu.h"

extern sched_t	sched_hw;
extern sched_t	sched_rr;
extern sched_t	sched_rrf;
extern sched_t	sched_fca;
extern sched_t	sched_rrm;
extern sched_t	sched_rrp;

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

extern void assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char skrid);

extern void init_skrun(void);
extern void fini_skrun(void);

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

extern "C" BOOL
run_tbs(unsigned *pticks)
{
	BOOL	res;

	init_skrun();

	if (sched->type == TBS_TYPE_HW)
		res = run_native_tbs(pticks);
	else
		res = run_sd_tbs(pticks);

	fini_skrun();

	return res;
}
