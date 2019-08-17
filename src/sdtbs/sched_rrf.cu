#include "sdtbs_cu.h"

static unsigned	id_sm = 1;

static unsigned
get_tb_sm_rrf(unsigned n_tb_width, unsigned n_tb_height, unsigned n_tb_x, unsigned n_tb_y)
{
	unsigned	id_sm_old = id_sm;

	while (!is_sm_avail(id_sm, n_tb_width * n_tb_height)) {
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
	FALSE, FALSE, FALSE,
	NULL,
	get_tb_sm_rrf
};

sched_t	sched_rrfS = {
	"rrfS",
	FALSE, TRUE, FALSE,
	NULL,
	get_tb_sm_rrf
};
