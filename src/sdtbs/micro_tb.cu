#include "sdtbs_cu.h"

unsigned	n_mtbs;
unsigned	n_mtbs_per_sm;

micro_tb_t	*mtbs;

static unsigned	*n_cur_mtbs_per_sm;

void
setup_micro_tbs(void)
{
	n_mtbs_per_sm = n_threads_per_tb / N_THREADS_PER_mTB;
	n_mtbs = n_sm_count * n_mtbs_per_sm;

	mtbs = (micro_tb_t *)calloc(n_mtbs, sizeof(micro_tb_t));
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned));
}

micro_tb_t *
get_mtb(int id_sm)
{
	micro_tb_t	*mtb;

	mtb = mtbs + id_sm * n_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm];
	n_cur_mtbs_per_sm[id_sm]++;
	return mtb;
}
