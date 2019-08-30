#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

extern void setup_fedkern_info_static(fedkern_info_t *fkinfo);
extern void setup_fedkern_info_dyn(fedkern_info_t *fkinfo);
extern void setup_fedkern_info_kernel_static(fedkern_info_t *fkinfo);
extern void setup_fedkern_info_kernel_dyn(fedkern_info_t *fkinfo);
extern void free_fedkern_info_static(fedkern_info_t *fkinfo);
extern void free_fedkern_info_dyn(fedkern_info_t *fkinfo);

fedkern_info_t *
create_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;

	fkinfo = (fedkern_info_t *)calloc(1, sizeof(fedkern_info_t));

	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = sched->use_static_sched ? 0: sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->n_tbs = n_tbs_submitted;

	if (sched->use_static_sched)
		setup_fedkern_info_static(fkinfo);
	else
		setup_fedkern_info_dyn(fkinfo);

	return fkinfo;
}

fedkern_info_t *
create_fedkern_info_kernel(fedkern_info_t *fkinfo)
{
	fedkern_info_t	*d_fkinfo;

	if (sched->use_static_sched)
		setup_fedkern_info_kernel_static(fkinfo);
	else
		setup_fedkern_info_kernel_dyn(fkinfo);

	cudaMalloc(&d_fkinfo, sizeof(fedkern_info_t));
	cudaMemcpy(d_fkinfo, fkinfo, sizeof(fedkern_info_t), cudaMemcpyHostToDevice);

	return d_fkinfo;
}

void
free_fedkern_info(fedkern_info_t *fkinfo)
{
	if (sched->use_static_sched)
		free_fedkern_info_static(fkinfo);
	else
		free_fedkern_info_dyn(fkinfo);	
	free(fkinfo);
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
