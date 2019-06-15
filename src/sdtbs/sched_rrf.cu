#include "sdtbs_cu.h"

static unsigned	id_sm;

static unsigned
get_tb_sm_rrf(unsigned n_tb_x, unsigned n_tb_y)
{
	unsigned	id_sm_old = id_sm;

	while (!is_sm_avail(id_sm, n_tb_width * n_tb_height)) {
		id_sm = (id_sm + 1) % n_sm_count;
		if (id_sm == id_sm_old)
			break;
	}
	return id_sm;
}

sched_t	sched_rrf = {
	"rrf",
	get_tb_sm_rrf
};
