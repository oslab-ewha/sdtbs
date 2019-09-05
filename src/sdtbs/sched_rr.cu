#include "sdtbs_cu.h"

static unsigned	id_sm = 1;

static unsigned
get_tb_sm_rr(dim3 dimBlock, unsigned n_tb_x, unsigned n_tb_y)
{
	int	id_sm_cur = id_sm;

	if (id_sm == n_sm_count)
		id_sm = 1;
	else
		id_sm++;
	return id_sm_cur;
}

sched_t	sched_rr = {
	"rr",
	TBS_TYPE_DYNAMIC,
	NULL,
	get_tb_sm_rr
};

sched_t	sched_hw = {
	"hw",
	TBS_TYPE_HW,
	NULL,
	NULL,
};
