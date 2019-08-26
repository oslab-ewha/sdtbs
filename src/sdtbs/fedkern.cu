#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

static unsigned	n_cur_mtbs_fedkern;

fedkern_info_t *
setup_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	unsigned	size_fkinfo;
	unsigned	n_max_mtbs;

	n_max_mtbs = n_sm_count * n_max_mtbs_per_sm;

	if (sched->use_static_sched)
		size_fkinfo = sizeof(fedkern_info_t) + n_max_mtbs * sizeof(unsigned);
	else
		size_fkinfo = sizeof(fedkern_info_t) + n_tbs_submitted * sizeof(unsigned);
	fkinfo = (fedkern_info_t *)calloc(1, size_fkinfo);

	fkinfo->size = size_fkinfo;
	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = sched->use_static_sched ? 0: sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->n_tbs = n_tbs_submitted;

	return fkinfo;
}

void
assign_fedkern_brun(fedkern_info_t *fkinfo,  benchrun_t *brun, unsigned char brid)
{
	benchrun_k_t    *brk;

	brk = &fkinfo->bruns[brid - 1];
	brk->skid = brun->info->skid;
	memcpy(brk->args, brun->args, sizeof(void *) * MAX_ARGS);
	brk->dimGrid = brun->dimGrid;
	brk->dimBlock = brun->dimBlock;
	brk->n_mtbs_per_tb = brun->dimBlock.x * brun->dimBlock.y / N_THREADS_PER_mTB;
}

void
assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid)
{
	fkinfo->brids[n_cur_mtbs_fedkern] = brid;
	n_cur_mtbs_fedkern++;
}

void
assign_fedkern_brid_dyn(fedkern_info_t *d_fkinfo, unsigned char brid)
{
	cudaStream_t	strm;

	cudaStreamCreate(&strm);

	cudaMemcpyAsync(d_fkinfo->brids + n_cur_mtbs_fedkern, &brid, 1, cudaMemcpyHostToDevice, strm);
	cudaStreamSynchronize(strm);
	n_cur_mtbs_fedkern++;

	cudaStreamDestroy(strm);
}

void
wait_fedkern_initialized(fedkern_info_t *d_fkinfo)
{
	cudaStream_t	strm;

	cudaStreamCreate(&strm);

	while (TRUE) {
		BOOL	initialized = FALSE;

		cudaMemcpyAsync(&initialized, &d_fkinfo->initialized, sizeof(BOOL), cudaMemcpyDeviceToHost, strm);
		cudaStreamSynchronize(strm);
		if (initialized)
			break;
	}
	cudaStreamDestroy(strm);
}
