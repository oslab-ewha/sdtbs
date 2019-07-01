#include "sdtbs_cu.h"

__device__ int find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs);

static __device__ unsigned	id_sm = 1;

__device__ unsigned
cu_get_tb_sm_rr(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start)
{
	unsigned	id_sm_start = id_sm;
	int	id_sm_cur = id_sm;

	do {
		int	mtb_start;

		mtb_start = find_mtb_start(id_sm_cur, 1, n_mtbs);

		if (mtb_start > 0) {
			id_sm = (id_sm_cur + 1) % fkinfo->n_sm_count + 1;
			*pidx_mtb_start = mtb_start;
			return id_sm_cur;
		}
		id_sm_cur = (id_sm_cur + 1) % fkinfo->n_sm_count + 1;
	}
	while (id_sm_start != id_sm_cur);

	return 0;
}
