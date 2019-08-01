#include "sdtbs_cu.h"

static unsigned	n_cur_mtbs;
static unsigned	*n_cur_mtbs_per_sm;

static unsigned	n_max_mtbs_per_sm;

fedkern_info_t *
setup_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	unsigned	size_fkinfo;
	unsigned	n_max_mtbs;

	n_max_mtbs_per_sm = n_threads_per_MTB / N_THREADS_PER_mTB * n_MTBs_per_sm;
	n_max_mtbs = n_sm_count * n_max_mtbs_per_sm;

	if (use_static_sched)
		size_fkinfo = sizeof(fedkern_info_t) + n_max_mtbs * sizeof(unsigned);
	else
		size_fkinfo = sizeof(fedkern_info_t) + n_tbs_submitted * sizeof(unsigned);
	fkinfo = (fedkern_info_t *)calloc(1, size_fkinfo);
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned));

	fkinfo->size = size_fkinfo;
	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = use_static_sched ? 0: sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->n_tbs = n_tbs_submitted;

	return fkinfo;
}

BOOL
assign_brid(fedkern_info_t *fkinfo, unsigned id_sm, unsigned brid)
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

BOOL
is_sm_avail(int id_sm, unsigned n_threads)
{
	unsigned	n_mtbs_new = (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm] + n_mtbs_new <= n_max_mtbs_per_sm)
		return TRUE;
	return FALSE;
}
