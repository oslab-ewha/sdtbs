#include "../../config.h"

#include "sdtbs_cu.h"
#include "sched_cu.h"

#define EPOCH_MAX		64

#define mTB_ALLOC_TABLE_EPOCH(epch)	(mATs + mTB_TOTAL_COUNT() * (epch))
#define mTB_ALLOC_TABLE(id_sm, idx)	(mATs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_ALLOC_TABLE_MY(id_sm)	(mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))
#define BRK_INDEX(id_sm, idx)	mTB_ALLOC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define BRK_INDEX_MY(id_sm)	mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]
#define BRK_INDEX_EPOCH(id_sm, idx, epch)	mTB_ALLOC_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]

#define BRK_N_MTBS_ASSIGNABLE(brid)	brk_n_mtbs_assignable[brid - 1]

#define mTB_OFFSET_TABLE_EPOCH(epch)	(mOTs + mTB_TOTAL_COUNT() * (epch))
#define mTB_OFFSET_TABLE(id_sm, idx)	(mOTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_OFFSET_TABLE_MY(id_sm)	(mOTs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))

#define mTB_OFFSET_TB_EPOCH(id_sm, idx, epch)	mTB_OFFSET_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_OFFSET_TB(id_sm, idx)	mTB_OFFSET_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_OFFSET_TB_MY(id_sm)		mTB_OFFSET_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define mTB_SYNC_TABLE_EPOCH(epch)    (mSTs + mTB_TOTAL_COUNT() * (epch))
#define mTB_SYNC_TABLE(id_sm, idx)    (mSTs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))

#define mTB_SYNC_EPOCH(id_sm, idx, epch)	mTB_SYNC_TABLE_EPOCH(epch)[mTB_INDEX(id_sm, idx) - 1]
#define mTB_SYNC(id_sm, idx)	mTB_SYNC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]

#define IS_SCHEDULE_DONE()	(n_tbs_assignable == d_fkinfo->n_tbs)

extern __device__ BOOL	going_to_shutdown;

__device__ static volatile int	in_scheduling;
__device__ static fedkern_info_t	*d_fkinfo;

/* epoch directory for mTB allocation table */
__device__ static volatile unsigned char	*mATs;
__device__ static volatile unsigned	*mtb_epochs;

/* offset in TB per mTB */
__device__ static volatile unsigned short	*mOTs;
/* sync counter per mTB */
__device__ static volatile unsigned short	*mSTs;

/* number of assignable mtbs per brun kernel */
__device__ static volatile unsigned	brk_n_mtbs_assignable[MAX_BENCHES];

__device__ static volatile unsigned	n_tbs_assignable;

__device__ unsigned cu_get_tb_sm_rr(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_rrf(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_fca(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_rrm(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);

static __device__ int
lock_scheduling(void)
{
	if (threadIdx.x % N_THREADS_PER_mTB != 0)
		return -1;
	if (atomicCAS((int *)&in_scheduling, 0, 1) != 0)
		return -1;
	return 0;
}

static __device__ void
unlock_scheduling(void)
{
	in_scheduling = 0;
}

static __device__ unsigned char
get_sched_brid(void)
{
	if (d_fkinfo->tbs_type == TBS_TYPE_DYNAMIC) {
		while (!IS_SCHEDULE_DONE()) {
			unsigned char	brid;
			brid = *(volatile unsigned char *)(d_fkinfo->u.dyn.brids_submitted + n_tbs_assignable);
			if (brid != 0)
				return brid;
			sleep_in_kernel();
		}
		return 0;
	}
	else {
		return d_fkinfo->u.dyn.brids_submitted[n_tbs_assignable];
	}
}

static __device__ BOOL
assign_tb(void)
{
	benchrun_k_t	*brk;
	unsigned	id_sm_sched;
	unsigned	idx_mtb_start;
	unsigned char	brid;
	int	i;

	brid = get_sched_brid();
	if (brid == 0)
		return FALSE;

	brk = &d_fkinfo->bruns[brid - 1];

	switch (d_fkinfo->sched_id) {
	case 2:
		id_sm_sched = cu_get_tb_sm_rr(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 3:
		id_sm_sched = cu_get_tb_sm_rrf(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 4:
		id_sm_sched = cu_get_tb_sm_fca(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 5:
		id_sm_sched = cu_get_tb_sm_rrm(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	default:
		break;
	}

	if (id_sm_sched == 0)
		return FALSE;

	for (i = 0; i < brk->n_mtbs_per_tb; i++) {
		if (BRK_INDEX(id_sm_sched, idx_mtb_start + i) == 0) {
			BRK_INDEX(id_sm_sched, idx_mtb_start + i) = brid;
			mTB_OFFSET_TB(id_sm_sched, idx_mtb_start + i) = BRK_N_MTBS_ASSIGNABLE(brid) + i;
			mTB_SYNC(id_sm_sched, idx_mtb_start + i) = 0;
		}
		else {
			int	epoch_next = (EPOCH(id_sm_sched, idx_mtb_start + i) + 1) % EPOCH_MAX;
			BRK_INDEX_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = brid;
			mTB_OFFSET_TB_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = BRK_N_MTBS_ASSIGNABLE(brid) + i;
			mTB_SYNC_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = 0;
		}
	}

	BRK_N_MTBS_ASSIGNABLE(brid) += brk->n_mtbs_per_tb;
	n_tbs_assignable++;
	return TRUE;
}

static __device__ void
run_schedule_in_kernel(void)
{
	if (d_fkinfo->tbs_type == TBS_TYPE_SOLO) {
		sleep_in_kernel();
		return;
	}
	if (lock_scheduling() < 0)
		return;

	if (IS_SCHEDULE_DONE()) {
		unlock_scheduling();
		return;
	}

	assign_tb();

	unlock_scheduling();
}

__device__ unsigned
find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs)
{
	int	i;

	for (i = idx_mtb_start; i <= d_fkinfo->n_max_mtbs_per_sm; i++) {
		if (BRK_INDEX(id_sm, i) == 0) {
			if (n_mtbs == 1)
				return i;
			if (i + n_mtbs - 1 <= d_fkinfo->n_max_mtbs_per_sm) {
				int	j;
				for (j = 1; j < n_mtbs; j++) {
					if (BRK_INDEX(id_sm, i + j) != 0)
						break;
				}
				if (j == n_mtbs)
					return i;
			}
		}
	}
	return 0;
}

__device__ unsigned
get_n_active_mtbs(unsigned id_sm)
{
	unsigned	count = 0;
	int	i;

	for (i = 1; i <= d_fkinfo->n_max_mtbs_per_sm; i++) {
		if (BRK_INDEX(id_sm, i) != 0)
			count++;
	}
	return count;
}

__device__ unsigned char
get_brid_dyn(BOOL *pis_primary_mtb)
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
			run_schedule_in_kernel();
		}
		SYNCWARP();
	}

	return 0;
}

__device__ void
advance_epoch_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD()) {
		EPOCH_MY(id_sm) = (EPOCH_MY(id_sm) + 1) % EPOCH_MAX;
	}
	SYNCWARP();

	/* clean up brk index if epoch is recycled */
	if (IS_LEADER_THREAD() && EPOCH_MY(id_sm) == 0) {
		BRK_INDEX_MY(id_sm) = 0;
	}
	SYNCWARP();
}

__device__ benchrun_k_t *
get_brk_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;
	unsigned	brid = BRK_INDEX_MY(id_sm);

	return &d_fkinfo->bruns[brid - 1];
}

__device__ unsigned
get_offset_TB_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;

	return mTB_OFFSET_TB_MY(id_sm);
}

__device__ void
sync_TB_threads_dyn(void)
{
	if (IS_LEADER_THREAD()) {
		benchrun_k_t	*brk = get_brk_dyn();

		if (brk->n_mtbs_per_tb > 1) {
			unsigned	id_sm = get_smid() + 1;
			unsigned	offset = get_offset_TB_dyn();
			int		idx_sync = mTB_INDEX_MY(id_sm) - offset;

			atomicInc((unsigned *)&mTB_SYNC(id_sm, idx_sync), brk->n_mtbs_per_tb - 1);
			while (mTB_SYNC(id_sm, idx_sync) > 0) {
				printf("%d\n", mTB_SYNC(id_sm, idx_sync));
			}
		}
	}
	SYNCWARP();
}

static __device__ void
setup_dyn_sched(fedkern_info_t *_fkinfo)
{
	int	size;
	int	i;

	d_fkinfo = _fkinfo;

	size = EPOCH_MAX * mTB_TOTAL_COUNT();
	mATs = (volatile unsigned char *)malloc(size);
	for (i = 0; i < size; i++) {
		mATs[i] = 0;
	}

	mOTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	mSTs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
	if (mOTs == NULL || mSTs == NULL) {
		printf("out of memory: offset or sync table cannot be allocated\n");
		going_to_shutdown = TRUE;
		return;
	}
	for (i = 0; i < size; i++) {
		mOTs[i] = 0;
		mSTs[i] = 0;
	}

	mtb_epochs = (volatile unsigned *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned));
	if (mtb_epochs == NULL) {
		printf("out of memory: epochs table cannot be allocated\n");
		going_to_shutdown = TRUE;
		return;
	}
	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}
}

__device__ void
try_setup_dyn_sched(fedkern_info_t *_fkinfo)
{
	if (_fkinfo->tbs_type == TBS_TYPE_SOLO || blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}
	setup_dyn_sched(_fkinfo);
	d_fkinfo->initialized = TRUE;
}

__device__ void
run_schedule_as_solo(fedkern_info_t *fkinfo)
{
	setup_dyn_sched(fkinfo);
	d_fkinfo->initialized = TRUE;
	while (!IS_SCHEDULE_DONE()) {
		while (TRUE) {
			if (!assign_tb())
				break;
		}
		sleep_in_kernel();
	}
}
