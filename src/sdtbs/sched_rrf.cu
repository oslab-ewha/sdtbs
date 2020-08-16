#include "sdtbs_cu.h"

#if 0 ///DEL
static unsigned	id_sm = 1;
#endif

static unsigned
get_tb_sm_rrf(dim3 dimBlock, unsigned n_tb_x, unsigned n_tb_y)
{
#if 0 ////DEL
	unsigned	id_sm_old = id_sm;
	unsigned	n_mTBs = get_n_mTBs_for_threads(dimBlock.x * dimBlock.y);

	while (!is_sm_avail(id_sm, n_mTBs)) {
		if (id_sm == n_sm_count)
			id_sm = 1;
		else
			id_sm++;
		if (id_sm == id_sm_old)
			break;
	}
	return id_sm;
#endif
	return 0;
}

sched_t	sched_rrf = {
	"rrf",
	TBS_TYPE_DYNAMIC,
	NULL,
	get_tb_sm_rrf
};
