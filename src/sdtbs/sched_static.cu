#include "sdtbs_cu.h"
#include "sched_cu.h"

#define mTB_ALLOC_TABLE_MY(id_sm)       (mATs[EPOCH_MY(id_sm)])
#define mTB_OFFSET_TABLE_MY(id_sm)	(mOTs[EPOCH_MY(id_sm)])

#define BRK_INDEX_MY(id_sm)	mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define mTB_OFFSET_TB_MY(id_sm)	mTB_OFFSET_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

__device__ static fedkern_info_t	*d_fkinfo;
__device__ BOOL	static_sched;

__device__ static unsigned char		**mATs;
__device__ static unsigned short	**mOTs;
__device__ static unsigned	*mtb_epochs;

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

	if (EPOCH_MY(id_sm) >= d_fkinfo->u.sta.n_mATs)
		return 0;
	brid = BRK_INDEX_MY(id_sm);
	if (brid != 0 && pis_primary_mtb != NULL) {
		if (get_offset_TB_static() == 0)
			*pis_primary_mtb = TRUE;
		else
			*pis_primary_mtb = FALSE;
	}
	return brid;
}

__device__ void
advance_epoch_static(void)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD()) {
		EPOCH_MY(id_sm)++;
	}
	__syncwarp();
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
	int	i;

	if (blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}

	d_fkinfo = _fkinfo;
	static_sched = TRUE;
	mATs = _fkinfo->u.sta.mATs;
	mOTs = _fkinfo->u.sta.mOTs;

        mtb_epochs = (unsigned *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned));
	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}

	d_fkinfo->initialized = TRUE;
}
