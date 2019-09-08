#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs;
extern unsigned	n_max_mtbs_per_sm;

void
setup_fedkern_info_host(fedkern_info_t *fkinfo)
{
	fkinfo->u.host.pdone = NULL;
	fkinfo->u.host.epochs_dev = NULL;
	fkinfo->u.host.epochs_host = NULL;
	fkinfo->u.host.mAT = NULL;
	fkinfo->u.host.mOT = NULL;
}

void
setup_fedkern_info_kernel_host(fedkern_info_t *fkinfo)
{
	cudaMalloc(&fkinfo->u.host.pdone, sizeof(BOOL));
	cudaMalloc(&fkinfo->u.host.epochs_dev, n_max_mtbs * sizeof(unsigned char));
	cudaMalloc(&fkinfo->u.host.epochs_host, n_max_mtbs * sizeof(unsigned char));
	cudaMalloc(&fkinfo->u.host.mAT, n_max_mtbs * sizeof(unsigned char));
	cudaMalloc(&fkinfo->u.host.mOT, n_max_mtbs * sizeof(unsigned short));

	cudaMemset(fkinfo->u.host.pdone, 0, sizeof(BOOL));
	cudaMemset(fkinfo->u.host.epochs_dev, 0, n_max_mtbs * sizeof(unsigned char));
	cudaMemset(fkinfo->u.host.epochs_host, 0, n_max_mtbs * sizeof(unsigned char));
	cudaMemset(fkinfo->u.host.mAT, 0, n_max_mtbs * sizeof(unsigned char));
	cudaMemset(fkinfo->u.host.mOT, 0, n_max_mtbs * sizeof(unsigned short));
}

void
free_fedkern_info_host(fedkern_info_t *fkinfo)
{
	cudaFree(fkinfo->u.host.pdone);
	cudaFree(fkinfo->u.host.epochs_dev);
	cudaFree(fkinfo->u.host.epochs_host);
	cudaFree(fkinfo->u.host.mAT);
	cudaFree(fkinfo->u.host.mOT);
}

BOOL
assign_fedkern_brid_host(cudaStream_t strm, fkinfo_host_t *d_fkhost, fkinfo_host_t *fkhost, unsigned id_sm, unsigned char brid)
{
	int	i;

	for (i = 0; i < n_max_mtbs_per_sm; i++) {
		if (fkhost->epochs_dev[i] == fkhost->epochs_host[i]) {
			fkhost->mAT[i] = brid;
			cudaMemcpyAsync(d_fkhost->mAT + i, &brid, 1, cudaMemcpyHostToDevice, strm);
			fkhost->epochs_host[i]++;
			cudaMemcpyAsync(d_fkhost->epochs_host + i, fkhost->epochs_host + i, 1, cudaMemcpyHostToDevice, strm);
			return TRUE;
		}
	}
	return FALSE;
}

void
fini_fedkern_brid_host(cudaStream_t strm, fkinfo_host_t *d_fkhost)
{
	BOOL	done = TRUE;

	cudaMemcpyAsync(d_fkhost->pdone, &done, sizeof(BOOL), cudaMemcpyHostToDevice, strm);
}
