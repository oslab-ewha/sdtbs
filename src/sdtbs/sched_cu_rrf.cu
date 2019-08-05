#include "sdtbs_cu.h"

static __device__ volatile unsigned	id_sm = 1;

__device__ unsigned
cu_get_tb_sm_rrf(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start)
{
	unsigned	id_sm_start = id_sm;
	int	id_sm_cur = id_sm;

	do {
		int	mtb_start;

		mtb_start = find_mtb_start(id_sm_cur, 1, n_mtbs);

		if (mtb_start > 0) {
			*pidx_mtb_start = mtb_start;
			return id_sm_cur;
		}
		id_sm_cur = (id_sm_cur + 1) % fkinfo->n_sm_count;
		if (id_sm_cur == 0)
			id_sm_cur = fkinfo->n_sm_count;
	}
	while (id_sm_start != id_sm_cur);

	return 0;
}
