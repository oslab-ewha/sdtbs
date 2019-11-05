#include "../../config.h"

#include "sdtbs_cu.h"
#include "sched_cu.h"

#include <pthread.h>

static pthread_t	thread;

#define IS_SCHED_DONE()	(*d_fkinfo->u.host.pdone)

#define EPOCH_HOST(id_sm)	(epochs_host[mTB_INDEX_MY(id_sm) - 1])
#define EPOCH_DEV(id_sm)	(epochs_dev[mTB_INDEX_MY(id_sm) - 1])

#define BRK_INDEX_MY(id_sm)	mAT[mTB_INDEX_MY(id_sm) - 1]

extern unsigned	n_max_mtbs;

__device__ static fedkern_info_t	*d_fkinfo;

__device__ static unsigned char	*mAT;
#if 0
__device__ static unsigned short	*mOT;
#endif
__device__ static unsigned char	*epochs_dev;
__device__ static unsigned char	*epochs_host;

extern BOOL assign_fedkern_brid_host(cudaStream_t strm, fkinfo_host_t *d_fkhost, fkinfo_host_t *fkhost, unsigned id_sm, unsigned char brid);
extern void fini_fedkern_brid_host(cudaStream_t strm, fkinfo_host_t *d_fkhost);

static __device__ void
wait_alloc_status(unsigned id_sm)
{
	if (IS_LEADER_THREAD()) {
		unsigned	brid;

		while (!IS_SCHED_DONE() && EPOCH_HOST(id_sm) == EPOCH_DEV(id_sm)) {
			unsigned	epoch;
			unsigned	done;

			sleep_in_kernel();
			asm("ld.volatile.u8 %0, [%1];": "=r"(epoch): "l"(&EPOCH_HOST(id_sm)));

			EPOCH_HOST(id_sm) = (unsigned char)epoch;
			asm("ld.volatile.u8 %0, [%1];": "=r"(done): "l"(d_fkinfo->u.host.pdone));
			*d_fkinfo->u.host.pdone = done;
		}
		asm("ld.volatile.u8 %0, [%1];": "=r"(brid): "l"(&BRK_INDEX_MY(id_sm)));
		BRK_INDEX_MY(id_sm) = (unsigned char)brid;
	}
	SYNCWARP();
}

__device__ unsigned char
get_brid_host(BOOL *pis_primary_mtb)
{
	unsigned	id_sm;
	unsigned char	brid;

	id_sm = get_smid() + 1;

	wait_alloc_status(id_sm);
	if (IS_SCHED_DONE())
		return 0;

	brid = BRK_INDEX_MY(id_sm);
	if (brid != 0 && pis_primary_mtb != NULL) {
#if 0 ///TODO
		if (pmst->offset == 0)
			*pis_primary_mtb = TRUE;
		else
			*pis_primary_mtb = FALSE;
#endif
	}
	return brid;
}

__device__ void
advance_epoch_host(void)
{
	unsigned	id_sm = get_smid() + 1;

	if (IS_LEADER_THREAD()) {
		EPOCH_DEV(id_sm)++;
	}
	SYNCWARP();
}

__device__ benchrun_k_t *
get_brk_host(void)
{
	unsigned char	brid;

	brid = get_brid_host(NULL);
	if (brid == 0)
		return NULL;
	return &d_fkinfo->bruns[brid - 1];
}

__device__ void
setup_host_sched(fedkern_info_t *_fkinfo)
{
	if (blockIdx.x != 0 || blockIdx.y != 0) {
		while (TRUE) {
			if (*(volatile BOOL *)&_fkinfo->initialized)
				return;
			sleep_in_kernel();
		}
	}

	d_fkinfo = _fkinfo;
	mAT = _fkinfo->u.host.mAT;
#if 0
	mOT = _fkinfo->u.host.mOT;
#endif
	epochs_dev = _fkinfo->u.host.epochs_dev;
	epochs_host = _fkinfo->u.host.epochs_host;

	d_fkinfo->initialized = TRUE;
}

static void *
host_scheduler(void *arg)
{
	fedkern_info_t *fkinfo = (fedkern_info_t *)arg;
	fkinfo_host_t	fkHost;
	benchrun_t	*brun;
	cudaStream_t	strm;
	int	i;

	fkHost.mAT = (unsigned char *)calloc(n_max_mtbs, sizeof(unsigned char));
#if 0
	fkHost.mOT = (unsigned short *)calloc(n_max_mtbs, sizeof(unsigned short));
#endif
	fkHost.epochs_host = (unsigned char *)calloc(n_max_mtbs, sizeof(unsigned char));
	fkHost.epochs_dev = (unsigned char *)calloc(n_max_mtbs, sizeof(unsigned char));

	cudaStreamCreate(&strm);
	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		int	j, k;
		for (j = 0; j < brun->dimGrid.y; j++) {
			for (k = 0; k < brun->dimGrid.x; k++) {
				unsigned	id_sm = sched->get_tb_sm(brun->dimBlock, k, j);
				assign_fedkern_brid_host(strm, &fkinfo->u.host, &fkHost, id_sm, i + 1);
			}
		}
	}

	fini_fedkern_brid_host(strm, &fkinfo->u.host);
	cudaStreamSynchronize(strm);
	cudaStreamDestroy(strm);

	return NULL;
}

void
run_schedule_host(fedkern_info_t *fkinfo)
{
	pthread_create(&thread, NULL, host_scheduler, fkinfo);
}
