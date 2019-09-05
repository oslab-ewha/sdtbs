#include "sdtbs_cu.h"

static unsigned	id_sm = 1;

static unsigned
get_tb_sm_rrf(dim3 dimBlock, unsigned n_tb_x, unsigned n_tb_y)
{
	unsigned	id_sm_old = id_sm;

	while (!is_sm_avail(id_sm, dimBlock.x * dimBlock.y)) {
		if (id_sm == n_sm_count)
			id_sm = 1;
		else
			id_sm++;
		if (id_sm == id_sm_old)
			break;
	}
	return id_sm;
}

sched_t	sched_rrf = {
	"rrf",
	TBS_TYPE_DYNAMIC,
	NULL,
	get_tb_sm_rrf
};
