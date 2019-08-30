#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs;
extern unsigned	n_max_mtbs_per_sm;

static unsigned n_cur_mtbs;

static unsigned	*n_cur_mtbs_per_sm;

void
setup_fedkern_info_static(fedkern_info_t *fkinfo)
{
	fkinfo->u.sta.offsets = (unsigned short *)calloc(n_max_mtbs, sizeof(unsigned short));
	fkinfo->u.sta.mAT = (unsigned char *)calloc(1, n_max_mtbs);
}

void
setup_fedkern_info_kernel_static(fedkern_info_t *fkinfo)
{
	unsigned short	*d_offsets;
	unsigned char	*d_mAT;

	cudaMalloc(&d_offsets, n_max_mtbs * sizeof(unsigned short));
	cudaMemcpy(d_offsets, fkinfo->u.sta.offsets, n_max_mtbs * sizeof(unsigned short), cudaMemcpyHostToDevice);
	free(fkinfo->u.sta.offsets);
	fkinfo->u.sta.offsets = d_offsets;

	cudaMalloc(&d_mAT, n_max_mtbs);
	cudaMemcpy(d_mAT, fkinfo->u.sta.mAT, n_max_mtbs, cudaMemcpyHostToDevice);
	free(fkinfo->u.sta.mAT);
	fkinfo->u.sta.mAT = d_mAT;
}

void
free_fedkern_info_static(fedkern_info_t *fkinfo)
{
	cudaFree(fkinfo->u.sta.offsets);
	cudaFree(fkinfo->u.sta.mAT);
}

void
assign_brid_static(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid, unsigned short offset)
{
	unsigned	idx;

	if (n_cur_mtbs_per_sm[id_sm - 1] == n_max_mtbs_per_sm) {
		FATAL(3, "no micro tb avaiable in SM[%u]", id_sm);
	}

	idx = (id_sm - 1) * n_max_mtbs_per_sm + n_cur_mtbs_per_sm[id_sm - 1];
	fkinfo->u.sta.mAT[idx] = brid;
	fkinfo->u.sta.offsets[idx] = offset;
	n_cur_mtbs_per_sm[id_sm - 1]++;
	if (fkinfo->bruns[brid - 1].primary_mtb_idx == 0)
		fkinfo->bruns[brid - 1].primary_mtb_idx = n_cur_mtbs + 1;
	n_cur_mtbs++;
}

BOOL
is_sm_avail(int id_sm, unsigned n_threads)
{
	unsigned	n_mtbs_new = (n_threads + N_THREADS_PER_mTB - 1) / N_THREADS_PER_mTB;

	if (n_cur_mtbs_per_sm[id_sm - 1] + n_mtbs_new <= n_max_mtbs_per_sm)
		return TRUE;
	return FALSE;
}

unsigned
get_sm_n_sched_mtbs(int id_sm)
{
	return n_cur_mtbs_per_sm[id_sm - 1];
}

void
init_sched_static(void)
{
	n_cur_mtbs_per_sm = (unsigned *)calloc(n_sm_count, sizeof(unsigned));
}
