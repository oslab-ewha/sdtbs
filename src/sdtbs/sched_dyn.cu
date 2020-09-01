#include "../../config.h"

#include "sdtbs_cu.h"
#include "sched_cu.h"

#define EPOCH_MAX		64

#define mTB_ALLOC_TABLE_EPOCH(epch)	(mATs + mTB_TOTAL_COUNT() * (epch))
#define mTB_ALLOC_TABLE(id_sm, idx)	(mATs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_ALLOC_TABLE_MY(id_sm)	(mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))
#define SKRID(id_sm, idx)	mTB_ALLOC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define SKRID_MY(id_sm)		mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define SKR_N_TBS_SCHED(skrid)	skr_n_tbs_sched[skrid - 1]

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

extern __device__ BOOL	going_to_shutdown;

__device__ static volatile int	in_scheduling;
__device__ static fedkern_info_t	*d_fkinfo;

/* epoch directory for mTB allocation table */
__device__ static volatile unsigned short	*mATs;
__device__ static volatile unsigned	*mtb_epochs;

/* offset in TB per mTB */
__device__ static volatile unsigned short	*mOTs;
/* sync counter per mTB */
__device__ static volatile unsigned short	*mSTs;

/* number of scheduled mtbs per skr */
__device__ static volatile unsigned short	*skr_n_tbs_sched;

__device__ static volatile unsigned	cur_skrid;

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

static __device__ skrid_t
get_sched_skrid(void)
{
	while (!going_to_shutdown) {
		skrun_t	*skr = &d_skruns[cur_skrid];
		skid_t	skid;
		skid = *(volatile skid_t *)(&skr->skid);
		if (skid != 0) {
			skrid_t	skrid = cur_skrid + 1;

			SKR_N_TBS_SCHED(skrid)++;
			if (SKR_N_TBS_SCHED(skrid) == skr->n_tbs)
				cur_skrid++;
			return skrid;
		}
		if (d_fkinfo->sched_done)
			return 0;
		sleep_in_kernel();
	}
	return 0;
}

static __device__ BOOL
assign_tb(void)
{
	skrun_t		*skr;
	unsigned	id_sm_sched;
	unsigned	idx_mtb_start;
	unsigned	off_tb_base;
	skrid_t		skrid;
	int		i;

	skrid = get_sched_skrid();
	if (skrid == 0)
		return FALSE;

	skr = &d_skruns[skrid - 1];

	switch (d_fkinfo->sched_id) {
	case 2:
		id_sm_sched = cu_get_tb_sm_rr(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 3:
		id_sm_sched = cu_get_tb_sm_rrf(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 4:
		id_sm_sched = cu_get_tb_sm_fca(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 5:
		id_sm_sched = cu_get_tb_sm_rrm(d_fkinfo, skr->n_mtbs_per_tb, &idx_mtb_start);
		break;
	default:
		break;
	}

	if (id_sm_sched == 0)
		return FALSE;

	off_tb_base = SKR_N_TBS_SCHED(skrid) * skr->n_mtbs_per_tb;
	for (i = 0; i < skr->n_mtbs_per_tb; i++) {
		if (SKRID(id_sm_sched, idx_mtb_start + i) == 0) {
			SKRID(id_sm_sched, idx_mtb_start + i) = skrid;
			mTB_OFFSET_TB(id_sm_sched, idx_mtb_start + i) = off_tb_base + i;
			mTB_SYNC(id_sm_sched, idx_mtb_start + i) = 0;
		}
		else {
			int	epoch_next = (EPOCH(id_sm_sched, idx_mtb_start + i) + 1) % EPOCH_MAX;
			BRK_INDEX_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = skrid;
			mTB_OFFSET_TB_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = off_tb_base + i;
			mTB_SYNC_EPOCH(id_sm_sched, idx_mtb_start + i, epoch_next) = 0;
		}
	}
	SKR_N_TBS_SCHED(skrid)++;
	return TRUE;
}

static __device__ void
run_schedule_in_kernel(void)
{
	if (lock_scheduling() < 0)
		return;

	if (going_to_shutdown) {
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
		if (SKRID(id_sm, i) == 0) {
			if (n_mtbs == 1)
				return i;
			if (i + n_mtbs - 1 <= d_fkinfo->n_max_mtbs_per_sm) {
				int	j;
				for (j = 1; j < n_mtbs; j++) {
					if (SKRID(id_sm, i + j) != 0)
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
		if (SKRID(id_sm, i) != 0)
			count++;
	}
	return count;
}

__device__ skrid_t
get_skrid_dyn(void)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		skrid_t	skrid;

		skrid = SKRID_MY(id_sm);
		if (skrid != 0)
			return skrid;

		if (going_to_shutdown || d_fkinfo->sched_done)
			break;

		if (IS_LEADER_THREAD()) {
			run_schedule_in_kernel();
		}
		SYNCWARP();
	}

	return 0;
}

__device__ void
advance_epoch_dyn(skrid_t skrid)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD()) {
		EPOCH_MY(id_sm) = (EPOCH_MY(id_sm) + 1) % EPOCH_MAX;
		atomicAdd(d_mtbs_done_cnts + skrid - 1, 1);
	}
	SYNCWARP();

	/* clear out skrun id if epoch is recycled */
	if (IS_LEADER_THREAD() && EPOCH_MY(id_sm) == 0) {
		SKRID_MY(id_sm) = 0;
	}
	SYNCWARP();
}

__device__ skrun_t *
get_skr_dyn(void)
{
	unsigned	id_sm = get_smid() + 1;
	skrid_t		skrid = SKRID_MY(id_sm);

	return &d_skruns[skrid - 1];
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
		skrun_t	*skr = get_skr_dyn();

		if (skr->n_mtbs_per_tb > 1) {
			unsigned	id_sm = get_smid() + 1;
			unsigned	offset = get_offset_TB_dyn();
			int		idx_sync = mTB_INDEX_MY(id_sm) - offset;

			atomicInc((unsigned *)&mTB_SYNC(id_sm, idx_sync), skr->n_mtbs_per_tb - 1);
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
	mATs = (volatile unsigned short *)malloc(size * sizeof(unsigned short));
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
	skr_n_tbs_sched = (unsigned short *)malloc(MAX_QUEUED_KERNELS * sizeof(unsigned short));
	for (i = 0; i < MAX_QUEUED_KERNELS; i++) {
		skr_n_tbs_sched[i] = 0;
	}
}

__device__ void
try_setup_dyn_sched(fedkern_info_t *_fkinfo)
{
	if (blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}
	setup_dyn_sched(_fkinfo);
	d_fkinfo->initialized = TRUE;
}
