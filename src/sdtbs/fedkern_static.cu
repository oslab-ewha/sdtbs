#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs;
extern unsigned	n_max_mtbs_per_sm;

static unsigned	*n_cur_mtbs_per_sm;
static unsigned	*sm_epochs;

static unsigned	n_mATs_cur;
static fedkern_info_t	*fkinfo_saved;

void
setup_fedkern_info_static(fedkern_info_t *fkinfo)
{
	sm_epochs = (unsigned *)calloc(n_sm_count, sizeof(unsigned));
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned ));

	n_mATs_cur = 1;

	fkinfo->u.sta.mATs = (unsigned char **)calloc(1, sizeof(unsigned char *));
	fkinfo->u.sta.mATs[0] = (unsigned char *)calloc(n_max_mtbs,  sizeof(unsigned char));
	fkinfo->u.sta.mOTs = (unsigned short **)calloc(1, sizeof(unsigned short *));
	fkinfo->u.sta.mOTs[0] = (unsigned short *)calloc(n_max_mtbs, sizeof(unsigned short));

	fkinfo_saved = fkinfo;
}

void
setup_fedkern_info_kernel_static(fedkern_info_t *fkinfo)
{
	unsigned short	**d_mOTs;
	unsigned char	**d_mATs;
	int	i;

	cudaMalloc(&d_mATs, n_mATs_cur * sizeof(unsigned char *));
	cudaMalloc(&d_mOTs, n_mATs_cur * sizeof(unsigned short *));

	for (i = 0; i < n_mATs_cur; i++) {
		unsigned char	*d_mAT;
		unsigned short	*d_mOT;

		cudaMalloc(&d_mAT, n_max_mtbs * sizeof(unsigned char));
		cudaMalloc(&d_mOT, n_max_mtbs * sizeof(unsigned short));
		cudaMemcpy(d_mAT, fkinfo->u.sta.mATs[i], n_max_mtbs * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_mOT, fkinfo->u.sta.mOTs[i], n_max_mtbs * sizeof(unsigned short), cudaMemcpyHostToDevice);
		free(fkinfo->u.sta.mATs[i]);
		free(fkinfo->u.sta.mOTs[i]);
		fkinfo->u.sta.mATs[i] = d_mAT;
		fkinfo->u.sta.mOTs[i] = d_mOT;
	}
	cudaMemcpy(d_mATs, fkinfo->u.sta.mATs, n_mATs_cur * sizeof(unsigned char *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mOTs, fkinfo->u.sta.mOTs, n_mATs_cur * sizeof(unsigned short *), cudaMemcpyHostToDevice);

	fkinfo->u.sta.n_mATs = n_mATs_cur;

	free(fkinfo->u.sta.mATs);
	free(fkinfo->u.sta.mOTs);

	fkinfo->u.sta.mATs = d_mATs;
	fkinfo->u.sta.mOTs = d_mOTs;
}

void
free_fedkern_info_static(fedkern_info_t *fkinfo)
{
	unsigned char	**mATs;
	unsigned short	**mOTs;
	int	i;

	mATs = (unsigned char **)malloc(n_mATs_cur * sizeof(unsigned char *));
	mOTs = (unsigned short **)malloc(n_mATs_cur * sizeof(unsigned short *));
	cudaMemcpy(mATs, fkinfo->u.sta.mATs, n_mATs_cur * sizeof(unsigned char *), cudaMemcpyDeviceToHost);
	cudaMemcpy(mATs, fkinfo->u.sta.mOTs, n_mATs_cur * sizeof(unsigned short *), cudaMemcpyDeviceToHost);

	for (i = 0; i < fkinfo->u.sta.n_mATs; i++) {
		cudaFree(mATs[i]);
		cudaFree(mOTs[i]);
	}
	free(mATs);
	free(mOTs);
	cudaFree(fkinfo->u.sta.mATs);
	cudaFree(fkinfo->u.sta.mOTs);
}

static void
setup_next_mAT(fedkern_info_t *fkinfo, unsigned id_sm)
{
	unsigned	epoch;

	epoch = ++sm_epochs[id_sm - 1];
	if (epoch == n_mATs_cur) {
		n_mATs_cur++;
		fkinfo->u.sta.mATs = (unsigned char **)realloc(fkinfo->u.sta.mATs, n_mATs_cur * sizeof(unsigned char *));
		fkinfo->u.sta.mOTs = (unsigned short **)realloc(fkinfo->u.sta.mOTs, n_mATs_cur * sizeof(unsigned short *));
		fkinfo->u.sta.mATs[n_mATs_cur - 1] = (unsigned char *)calloc(n_max_mtbs, sizeof(unsigned char));
		fkinfo->u.sta.mOTs[n_mATs_cur - 1] = (unsigned short *)calloc(n_max_mtbs, sizeof(unsigned short));
	}
	n_cur_mtbs_per_sm[id_sm - 1] = 0;
}

static void
assign_brid_static(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid, unsigned short offset)
{
	unsigned	epoch;
	unsigned	idx;

	epoch = sm_epochs[id_sm - 1];
	idx = (id_sm - 1) * n_max_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm - 1];

	fkinfo->u.sta.mATs[epoch][idx] = brid;
	fkinfo->u.sta.mOTs[epoch][idx] = offset;
	n_cur_mtbs_per_sm[id_sm - 1]++;
}

void
assign_fedkern_brid_static(fedkern_info_t *fkinfo, benchrun_t *brun, unsigned char brid, unsigned id_sm, unsigned short *poffset)
{
	unsigned	n_mtbs_per_tb;
	int	i;

	n_mtbs_per_tb = brun->dimBlock.x * brun->dimBlock.y / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm - 1] + n_mtbs_per_tb > n_max_mtbs_per_sm) {
		setup_next_mAT(fkinfo, id_sm);
	}

	for (i = 0; i < n_mtbs_per_tb; i++, (*poffset)++) {
		assign_brid_static(fkinfo, id_sm, brid, *poffset);
	}
}

unsigned
get_n_mTBs_for_threads(unsigned n_threads)
{
	return (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;
}

BOOL
is_sm_avail(int id_sm, unsigned n_mTBs_new)
{
	if (n_cur_mtbs_per_sm[id_sm - 1] + n_mTBs_new <= n_max_mtbs_per_sm)
		return TRUE;
	return FALSE;
}

unsigned
get_sm_n_sched_mTBs(int id_sm)
{
	return n_cur_mtbs_per_sm[id_sm - 1];
}

void
use_next_mAT(int id_sm)
{
	setup_next_mAT(fkinfo_saved, id_sm);
}
