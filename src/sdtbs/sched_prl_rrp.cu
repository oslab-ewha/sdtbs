#include "sdtbs_cu.h"

extern __device__ unsigned prl_get_rr_index(void);
extern __device__ unsigned char prl_get_brid(unsigned idx);
extern __device__ unsigned short prl_get_offset(unsigned idx);

__device__ unsigned
cu_get_mtb_brid_rrp(fedkern_info_t *fkinfo, unsigned short *poffset)
{
	unsigned	idx;

	idx = prl_get_rr_index();
	if (idx == 0)
		return 0;
	*poffset = prl_get_offset(idx);
	return prl_get_brid(idx);
}

sched_t	sched_rrp = {
	"rrp",
	TBS_TYPE_PARALLEL,
	NULL,
	NULL
};
