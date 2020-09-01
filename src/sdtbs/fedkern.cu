#include "sdtbs_cu.h"

extern unsigned	n_max_mtbs_per_sm;

fedkern_info_t *
create_fedkern_info(void)
{
	fedkern_info_t	*fkinfo;
	fedkern_info_t	*g_fkinfo;

	fkinfo = (fedkern_info_t *)calloc(1, sizeof(fedkern_info_t));

	fkinfo->n_sm_count = n_sm_count;
	fkinfo->sched_id = sched_id;
	fkinfo->n_mtbs = n_mtbs_submitted;
	fkinfo->n_max_mtbs_per_sm = n_max_mtbs_per_sm;
	fkinfo->n_max_mtbs_per_MTB = n_max_mtbs_per_sm / n_MTBs_per_sm;
	fkinfo->sched_done = FALSE;

	cudaMalloc(&g_fkinfo, sizeof(fedkern_info_t));
	cudaMemcpy(g_fkinfo, fkinfo, sizeof(fedkern_info_t), cudaMemcpyHostToDevice);

	return g_fkinfo;
}

void
free_fedkern_info(fedkern_info_t *g_fkinfo)
{
	cudaFree(g_fkinfo);
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
