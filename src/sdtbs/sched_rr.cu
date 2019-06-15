#include "sdtbs_cu.h"

static unsigned	id_sm;

static unsigned
get_tb_sm_rr(unsigned n_tb_x, unsigned n_tb_y)
{
	int	id_sm_cur = id_sm;

	id_sm = (id_sm + 1) % n_sm_count;
	return id_sm_cur;
}

sched_t	sched_rr = {
	"rr",
	get_tb_sm_rr
};
