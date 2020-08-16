#include "sdtbs_cu.h"

#if 0 ///DEL
static unsigned	id_sm = 1;
#endif
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
get_tb_sm_rrm(dim3 dimBlock, unsigned n_tb_x, unsigned n_tb_y)
{
#if 0 ////DEL
	unsigned	id_sm_start = id_sm;
	unsigned	id_sm_res;
	unsigned	n_mTBs = get_n_mTBs_for_threads(dimBlock.x * dimBlock.y);

	while (!is_sm_avail(id_sm, n_mTBs) || get_sm_n_sched_mTBs(id_sm) + n_mTBs > max_mtbs_per_sm) {
		if (id_sm == n_sm_count)
			id_sm = 1;
		else
			id_sm++;
		if (id_sm == id_sm_start)
			use_next_mAT(id_sm);
	}
	id_sm_res = id_sm;
	if (id_sm == n_sm_count)
		id_sm = 1;
	else
		id_sm++;
	return id_sm_res;
#endif
	return 0;
}

sched_t	sched_rrm = {
	"rrm",
	TBS_TYPE_DYNAMIC,
	parse_arg_rrm,
	get_tb_sm_rrm
};
