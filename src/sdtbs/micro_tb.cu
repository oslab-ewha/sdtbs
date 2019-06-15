#include "sdtbs_cu.h"

unsigned	n_mtbs;
unsigned	n_mtbs_per_sm;

micro_tb_t	*mtbs;

static unsigned	*n_cur_mtbs_per_sm;

void
setup_micro_tbs(void)
{
	n_mtbs_per_sm = n_threads_per_MTB / N_THREADS_PER_mTB;
	n_mtbs = n_sm_count * n_mtbs_per_sm;

	mtbs = (micro_tb_t *)calloc(n_mtbs, sizeof(micro_tb_t));
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned));
}

micro_tb_t *
get_mtb(unsigned id_sm)
{
	micro_tb_t	*mtb;

	if (n_cur_mtbs_per_sm[id_sm] == n_mtbs_per_sm)
		return NULL;
	mtb = mtbs + id_sm * n_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm];
	n_cur_mtbs_per_sm[id_sm]++;
	return mtb;
}

BOOL
is_sm_avail(int id_sm, unsigned n_threads)
{
	unsigned	n_mtbs_new = (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm] + n_mtbs_new <= n_mtbs_per_sm)
		return TRUE;
	return FALSE;
}
