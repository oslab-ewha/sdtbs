#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

static unsigned	n_cur_mtbs_fedkern;

fedkern_info_t *
setup_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	unsigned	size_fkinfo;
	unsigned	n_max_mtbs;

	n_max_mtbs = n_sm_count * n_max_mtbs_per_sm;

	if (use_static_sched)
		size_fkinfo = sizeof(fedkern_info_t) + n_max_mtbs * sizeof(unsigned);
	else
		size_fkinfo = sizeof(fedkern_info_t) + n_tbs_submitted * sizeof(unsigned);
	fkinfo = (fedkern_info_t *)calloc(1, size_fkinfo);

	fkinfo->size = size_fkinfo;
	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = use_static_sched ? 0: sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->n_tbs = n_tbs_submitted;

	return fkinfo;
}

void
assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid)
{
	fkinfo->brids[n_cur_mtbs_fedkern] = brid;
	n_cur_mtbs_fedkern++;
}
