#include "sdtbs_cu.h"

#define mTB_TOTAL_COUNT()	(d_fkinfo->n_max_mtbs_per_sm * d_fkinfo->n_sm_count)
#define mTB_INDEX(id_sm, idx)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + idx)
#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)
#define EPOCH(id_sm, idx)	mtb_epochs[mTB_INDEX(id_sm, idx) - 1]
#define EPOCH_MY(id_sm)		mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define mTB_ALLOC_TABLE(id_sm, idx)	(mATs + mTB_TOTAL_COUNT() * EPOCH(id_sm, idx))
#define mTB_ALLOC_TABLE_MY(id_sm)	(mATs + mTB_TOTAL_COUNT() * EPOCH_MY(id_sm))
#define BRK_INDEX(id_sm, idx)	mTB_ALLOC_TABLE(id_sm, idx)[mTB_INDEX(id_sm, idx) - 1]
#define BRK_INDEX_MY(id_sm)	mTB_ALLOC_TABLE_MY(id_sm)[mTB_INDEX_MY(id_sm) - 1]

#define IS_LEADER_THREAD()	(threadIdx.x % N_THREADS_PER_mTB == 0)

#define IS_SCHEDULE_DONE()	(n_tbs_assignable == d_fkinfo->n_tbs)

extern __device__ BOOL	going_to_shutdown;

__device__ static volatile int	initialized;
__device__ static volatile int	in_scheduling;
__device__ static fedkern_info_t	*d_fkinfo;

/* epoch directory for mTB allocation table */
__device__ static volatile unsigned char	*mATs;
__device__ static volatile unsigned	*mtb_epochs;

__device__ static volatile unsigned	n_tbs_assignable;

__device__ unsigned cu_get_tb_sm_rr(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);
__device__ unsigned cu_get_tb_sm_rrf(fedkern_info_t *fkinfo, unsigned n_mtbs, unsigned *pidx_mtb_start);

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

static __device__ void
run_schedule_in_kernel(void)
{
	benchrun_k_t	*brk;
	unsigned	id_sm_sched;
	unsigned	idx_mtb_start;
	unsigned char	brid;
	int	i;

	if (lock_scheduling() < 0)
		return;

	if (IS_SCHEDULE_DONE()) {
		unlock_scheduling();
		return;
	}

	brid = d_fkinfo->brids[n_tbs_assignable];
	brk = &d_fkinfo->bruns[brid - 1];

	switch (d_fkinfo->sched_id) {
	case 1:
		id_sm_sched = cu_get_tb_sm_rr(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	case 2:
		id_sm_sched = cu_get_tb_sm_rrf(d_fkinfo, brk->n_mtbs_per_tb, &idx_mtb_start);
		break;
	default:
		break;
	}

	if (id_sm_sched > 0) {
		if (brk->primary_mtb_idx == 0)
			brk->primary_mtb_idx = idx_mtb_start;
		for (i = 0; i < brk->n_mtbs_per_tb; i++) {
			BRK_INDEX(id_sm_sched, idx_mtb_start + i) = brid;
		}
		n_tbs_assignable++;
	}

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

__device__ unsigned char
get_brid_dyn(BOOL *pis_primary_mtb)
{
	unsigned	id_sm;

	id_sm = get_smid() + 1;

	for (;;) {
		unsigned char	brid;

		brid = BRK_INDEX_MY(id_sm);
		if (brid != 0) {
			if (IS_LEADER_THREAD() && d_fkinfo->bruns[brid - 1].primary_mtb_idx == mTB_INDEX_MY(id_sm))
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
		__syncwarp();
	}

	return 0;
}

__device__ void
advance_epoch(void)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD())
		EPOCH_MY(id_sm)++;

	__syncwarp();
}

__device__ void
setup_dyn_sched(fedkern_info_t *_fkinfo)
{
	unsigned	mATs_size;
	int	i;

	if (blockIdx.x != 0 || blockIdx.y != 0) {
		for (i = 0;; i++) {
			for (int j = 0; j < 10000000; j++)
				if (initialized)
					return;
		}
	}

	d_fkinfo = _fkinfo;

	mATs_size = d_fkinfo->n_tbs * mTB_TOTAL_COUNT();
	mATs = (volatile unsigned char *)malloc(mATs_size);
	if (mATs == NULL) {
		printf("too big mAT: %d\n", mATs_size);
		going_to_shutdown = TRUE;
		goto out;
	}
	for (i = 0; i < mATs_size; i++) {
		mATs[i] = 0;
	}

	mtb_epochs = (volatile unsigned *)malloc(mTB_TOTAL_COUNT() * sizeof(unsigned));
	if (mtb_epochs == NULL) {
		printf("out of memory: epochs cannot be allocated\n");
		going_to_shutdown = TRUE;
		goto out;
	}
	for (i = 0; i < mTB_TOTAL_COUNT(); i++) {
		mtb_epochs[i] = 0;
	}
out:
	initialized = 1;
}
