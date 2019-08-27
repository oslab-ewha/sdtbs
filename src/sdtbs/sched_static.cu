#include "sdtbs_cu.h"

#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)
#define BRK_INDEX_MY(id_sm)	d_fkinfo->brids[mTB_INDEX_MY(id_sm) - 1]
#define mTB_OFFSET_TB_MY(id_sm)	d_fkinfo->offsets[mTB_INDEX_MY(id_sm) - 1]

__device__ static fedkern_info_t	*d_fkinfo;
__device__ BOOL	static_sched;

__device__ unsigned
get_offset_TB_static(void)
{
	unsigned	id_sm = get_smid() + 1;

	return mTB_OFFSET_TB_MY(id_sm);
}

__device__ unsigned char
get_brid_static(BOOL *pis_primary_mtb)
{
	unsigned	id_sm;
	unsigned char	brid;

	id_sm = get_smid() + 1;

	brid = BRK_INDEX_MY(id_sm);
	if (brid != 0 && pis_primary_mtb != NULL) {
		if (get_offset_TB_static() == 0)
			*pis_primary_mtb = TRUE;
		else
			*pis_primary_mtb = FALSE;
	}
	return brid;
}

__device__ benchrun_k_t *
get_brk_static(void)
{
	unsigned char	brid;

	brid = get_brid_static(NULL);
	if (brid == 0)
		return NULL;
	return &d_fkinfo->bruns[brid - 1];
}

__device__ void
setup_static_sched(fedkern_info_t *_fkinfo)
{
	if (blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}

	d_fkinfo = _fkinfo;
	static_sched = TRUE;
	d_fkinfo->initialized = TRUE;
}
