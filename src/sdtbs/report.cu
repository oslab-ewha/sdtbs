#include "sdtbs_cu.h"

extern "C" void
report(unsigned elapsed)
{
	benchrun_t	*brun;
	int	i;

	printf("tbs type: %s\n", sched->direct_mode ? (sched->use_relocatable ? "native(relocatable)": "native") : sched->use_static_sched ? "sd(static)": "sd(dynamic)");
	printf("policy: %s\n", sched->name);
	if (!sched->direct_mode) {
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
