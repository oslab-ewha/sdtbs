#include "sdtbs_cu.h"

extern "C" void
report(unsigned elapsed)
{
	benchrun_t	*brun;
	int	i;

	printf("tbs type: %s\n", direct_mode ? (use_relocatable ? "native(relocatable)": "native") : use_static_sched ? "sd(static)": "sd(dynamic)");
	if (!direct_mode) {
		printf("sm count: %u\n", n_sm_count);
		printf("n threads per MTB: %u\n", n_threads_per_MTB);
		printf("n MTBs per SM: %u\n", n_MTBs_per_sm);
	}
	printf("elapsed time: %.3lf\n", elapsed / 1000.0);
	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		printf("%s: %d\n", brun->info->code, brun->res);
	}
}
