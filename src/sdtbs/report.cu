#include "sdtbs_cu.h"

static const char	*tbs_type_desc[] = {
	"native",
	"native(relocatable)",
	"sd(static)",
	"sd(dynamic)",
	"sd(semi-dynamic)",
	"solo",
	"host",
	"parallel"
};

extern "C" void
report(unsigned elapsed)
{
	benchrun_t	*brun;
	int	i;

	printf("tbs type: %s\n", tbs_type_desc[sched->type - 1]);
	printf("policy: %s\n", sched->name);
	if (sched->type != TBS_TYPE_HW && sched->type != TBS_TYPE_HW_RELOC) {
		printf("sm count: %u\n", n_sm_count);
		printf("n threads per MTB: %u\n", n_threads_per_MTB);
		printf("n MTBs per SM: %u\n", n_MTBs_per_sm);
	}
	printf("elapsed time: %.6lf\n", elapsed / 1000000.0);
	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		printf("%s: %d\n", brun->info->code, brun->res);
	}
}
