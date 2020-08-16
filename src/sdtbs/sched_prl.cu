#include "../../config.h"

#include "sdtbs_cu.h"
#include "sched_cu.h"

#define EPOCH_MAX		64

#define mTB_OFFSET_TB_MY(id_sm)	mOT[mTB_INDEX_MY(id_sm) - 1]

#define BRK_INDEX_MY(id_sm)	mAT[mTB_INDEX_MY(id_sm) - 1]
#define mTB_RR_INDEX(id_sm)	EPOCH_MY(id_sm) * mTB_TOTAL_COUNT() + (d_fkinfo->n_sm_count * (threadIdx.x / N_THREADS_PER_mTB) + id_sm + blockIdx.y * d_fkinfo->n_max_mtbs_per_MTB * d_fkinfo->n_sm_count)

#define IS_SCHEDULE_DONE()	(n_mtbs_assignable == n_total_mtbs)

extern __device__ BOOL	going_to_shutdown;

__device__ static fedkern_info_t	*d_fkinfo;

/* mtb current brids */
__device__ static unsigned char	*mAT;
/* mtb current offsets */
__device__ static unsigned short	*mOT;
/* To calculate mtb offset easily, maintain mTB counts for all skrun's */
__device__ static unsigned *cnts_skrun_mtbs;

/* all mtb brids for submitted bruns */
__device__ static unsigned char	*brids_submitted;
__device__ static unsigned	n_total_mtbs;
__device__ static volatile unsigned	n_mtbs_assignable;

__device__ static unsigned	*mtb_epochs;

__device__ unsigned char cu_get_mtb_brid_rrp(fedkern_info_t *fkinfo, unsigned short *poffset);

static __device__ void
run_parallel_schedule_in_kernel(unsigned id_sm)
{
	unsigned char	brid;
	unsigned short	offset;

	switch (d_fkinfo->sched_id) {
	case 6:
		brid = cu_get_mtb_brid_rrp(d_fkinfo, &offset);
		break;
	default:
		break;
	}

	if (brid != 0) {
		BRK_INDEX_MY(id_sm) = brid;
		atomicInc((unsigned *)&n_mtbs_assignable, n_total_mtbs);
		mTB_OFFSET_TB_MY(id_sm) = offset;
	}
}

__device__ unsigned char
get_skrid_prl(BOOL *pis_primary_mtb)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		unsigned char	brid;

		brid = BRK_INDEX_MY(id_sm);
		if (brid != 0) {
			if (IS_LEADER_THREAD() && mTB_OFFSET_TB_MY(id_sm) == 0)
				*pis_primary_mtb = TRUE;
			else
				*pis_primary_mtb = FALSE;
			return brid;
		}

		if (IS_SCHEDULE_DONE())
			break;

		if (IS_LEADER_THREAD()) {
			run_parallel_schedule_in_kernel(id_sm);
		}
		SYNCWARP();
	}

	return 0;
}

__device__ void
advance_epoch_prl(void)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD()) {
		EPOCH_MY(id_sm)++;
	}
	SYNCWARP();
	if (IS_LEADER_THREAD()) {
		BRK_INDEX_MY(id_sm) = 0;
	}
	SYNCWARP();
}

__device__ skrun_t *
get_skr_prl(void)
{
	unsigned	id_sm;
	unsigned char   skrid;

	id_sm = get_smid() + 1;
	skrid = BRK_INDEX_MY(id_sm);
	if (skrid == 0)
		return NULL;
	return &d_skruns[skrid - 1];
}

__device__ unsigned
get_offset_TB_prl(void)
{
	unsigned	id_sm = get_smid() + 1;

	return mTB_OFFSET_TB_MY(id_sm);
}

__device__ unsigned
prl_get_rr_index(void)
{
	unsigned	id_sm = get_smid() + 1;
	unsigned	idx = mTB_RR_INDEX(id_sm);

	if (idx > n_total_mtbs)
		return 0;
	return idx;
}

__device__ unsigned char
prl_get_brid(unsigned idx)
{
	return brids_submitted[idx - 1];
}

__device__ unsigned short
prl_get_offset(unsigned idx)
{
	int	i;

	for (i = 0; i < d_fkinfo->n_qks; i++) {
		if (idx <= cnts_skrun_mtbs[i])
			break;
		idx -= cnts_skrun_mtbs[i];
	}
	return (unsigned short)(idx - 1);
}

static __device__ void
init_prl_sched_by_master(fedkern_info_t *_fkinfo)
{
	skrun_t	*skr;
	int	i, j, k;

	d_fkinfo = _fkinfo;

	cnts_skrun_mtbs = (unsigned *)malloc(d_fkinfo->n_qks * sizeof(unsigned));

	skr = d_skruns;
	for (i = 0; i < d_fkinfo->n_qks; i++, skr++) {
		cnts_skrun_mtbs[i] = skr->n_mtbs_per_tb * skr->dimGrid.x * skr->dimGrid.y;
		n_total_mtbs += cnts_skrun_mtbs[i];
	}

	brids_submitted = (unsigned char *)malloc(n_total_mtbs);

	skr = d_skruns;
	k = 0;
	for (i = 0; i < d_fkinfo->n_qks; i++, skr++) {
		unsigned	n_mtbs_skr = skr->n_mtbs_per_tb * skr->dimGrid.x * skr->dimGrid.y;
		for (j = 0; j < n_mtbs_skr; j++, k++) {
			brids_submitted[k] = i + 1;
		}
	}

	mAT = (unsigned char *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned char));
	mOT = (unsigned short *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned short));
	mtb_epochs = (unsigned *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned));
	if (mtb_epochs == NULL) {
		printf("out of memory: epochs table cannot be allocated\n");
		going_to_shutdown = TRUE;
		return;
	}
	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mAT[i] = 0;
		mOT[i] = 0;
		mtb_epochs[i] = 0;
	}
}

__device__ void
setup_prl_sched(fedkern_info_t *_fkinfo)
{
	if (blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}

	init_prl_sched_by_master(_fkinfo);
	d_fkinfo->initialized = TRUE;
}
