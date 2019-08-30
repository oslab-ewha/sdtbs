#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

static unsigned	n_cur_mtbs;

static unsigned	n_cur_mtbs_fedkern;

void
setup_fedkern_info_dyn(fedkern_info_t *fkinfo)
{
	fkinfo->u.dyn.brids_submitted = (unsigned char *)calloc(n_tbs_submitted, 1);
}

void
setup_fedkern_info_kernel_dyn(fedkern_info_t *fkinfo)
{
	unsigned char	*d_brids;

	cudaMalloc(&d_brids, n_tbs_submitted);
	cudaMemcpy(d_brids, fkinfo->u.dyn.brids_submitted, n_tbs_submitted, cudaMemcpyHostToDevice);
	free(fkinfo->u.dyn.brids_submitted);
	fkinfo->u.dyn.brids_submitted = d_brids;
}

void
free_fedkern_info_dyn(fedkern_info_t *fkinfo)
{
	cudaFree(fkinfo->u.dyn.brids_submitted);
}

void
assign_fedkern_brid(fedkern_info_t *fkinfo, unsigned char brid)
{
	fkinfo->u.dyn.brids_submitted[n_cur_mtbs_fedkern] = brid;
	n_cur_mtbs_fedkern++;
}

void
assign_fedkern_brid_dyn(fedkern_info_t *d_fkinfo, unsigned char brid)
{
	cudaStream_t	strm;

	cudaStreamCreate(&strm);

	cudaMemcpyAsync(d_fkinfo->u.dyn.brids_submitted + n_cur_mtbs_fedkern, &brid, 1, cudaMemcpyHostToDevice, strm);
	cudaStreamSynchronize(strm);
	n_cur_mtbs_fedkern++;

	cudaStreamDestroy(strm);
}

void
assign_brid_dyn(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid)
{
	fkinfo->u.dyn.brids_submitted[n_cur_mtbs] = brid;
	n_cur_mtbs++;
}

void
init_sched_dyn(void)
{
}
