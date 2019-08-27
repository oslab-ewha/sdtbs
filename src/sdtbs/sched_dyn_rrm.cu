#include "sdtbs_cu.h"

static __device__ volatile unsigned	id_sm = 1;

static __device__ unsigned
next_id_sm(fedkern_info_t *fkinfo, unsigned id_sm)
{
	if (id_sm == fkinfo->n_sm_count)
		return 1;
	return (id_sm + 1);
}

__device__ unsigned
cu_get_tb_sm_rrm(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start)
{
	unsigned	id_sm_start = id_sm;
	int	id_sm_cur = id_sm;
	unsigned	n_max_mtbs_per_sm = (unsigned)(long long)fkinfo->sched_arg;

	do {
		if (get_n_active_mtbs(id_sm_cur) + n_mtbs <= n_max_mtbs_per_sm) {
			int	mtb_start;

			mtb_start = find_mtb_start(id_sm_cur, 1, n_mtbs);
			if (mtb_start > 0) {
				id_sm = next_id_sm(fkinfo, id_sm_cur);
				*pidx_mtb_start = mtb_start;
				return id_sm_cur;
			}
		}
		id_sm_cur = next_id_sm(fkinfo, id_sm_cur);
	}
	while (id_sm_start != id_sm_cur);

	return 0;
}
