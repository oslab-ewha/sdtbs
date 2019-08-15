#include "sdtbs_cu.h"

static unsigned	id_sm = 1;
static int	max_mtbs_per_sm;

static void *
parse_arg_rrm(const char *argstr)
{
	if (sscanf(argstr, "%u", &max_mtbs_per_sm) != 1) {
		error("invalid max mTB's per sm: %s", argstr);
		return NULL;
	}
	if (max_mtbs_per_sm == 0) {
		error("zero max mTB's per sm is not allowed");
		return NULL;
	}
	return (void *)(long long)max_mtbs_per_sm;
}

static unsigned
get_tb_sm_rrm(unsigned n_tb_x, unsigned n_tb_y)
{
	unsigned	id_sm_start = id_sm;

	while (!is_sm_avail(id_sm, n_tb_width * n_tb_height) || get_sm_n_sched_mtbs(id_sm) == max_mtbs_per_sm) {
		if (id_sm == n_sm_count)
			id_sm = 1;
		else
			id_sm++;
		if (id_sm == id_sm_start)
			return 0;
	}
	return id_sm;
}

sched_t	sched_rrm = {
	"rrm",
	parse_arg_rrm,
	get_tb_sm_rrm
};
