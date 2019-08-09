#include "sdtbs_cu.h"

__device__ unsigned
cu_get_tb_sm_fca(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start)
{
	unsigned	id_sm_my = get_smid() + 1;
	unsigned	mtb_start;

	mtb_start = find_mtb_start(id_sm_my, 1, n_mtbs);
	if (mtb_start > 0) {
		*pidx_mtb_start = mtb_start;
		return id_sm_my;
	}

	return 0;
}
