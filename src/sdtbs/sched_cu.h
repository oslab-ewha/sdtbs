#ifndef _SCHED_CU_H_
#define _SCHED_CU_H_

#if CUDA_COMPUTE >= 60
#define SYNCWARP()	__syncwarp()
#else
#define SYNCWARP()	do {} while (0)
#endif

#define mTB_TOTAL_COUNT()	(d_fkinfo->n_max_mtbs_per_sm * d_fkinfo->n_sm_count)

#define mTB_INDEX(id_sm, idx)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + idx)
#define mTB_INDEX_MY(id_sm)	((id_sm - 1) * d_fkinfo->n_max_mtbs_per_sm + d_fkinfo->n_max_mtbs_per_MTB * blockIdx.y + (threadIdx.x / N_THREADS_PER_mTB) + 1)

#define EPOCH(id_sm, idx)	mtb_epochs[mTB_INDEX(id_sm, idx) - 1]
#define EPOCH_MY(id_sm)		mtb_epochs[mTB_INDEX_MY(id_sm) - 1]

#define IS_LEADER_THREAD()	(threadIdx.x % N_THREADS_PER_mTB == 0)

#endif
