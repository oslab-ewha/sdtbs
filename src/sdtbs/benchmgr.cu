#include "sdtbs_cu.h"

int bench_loopcalc(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, void *args[]);
int bench_gma(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, void *args[]);
int cookarg_gma(void *args[]);

benchrun_t	benchruns[MAX_BENCHES];
int	n_benches;

static benchinfo_t	benchinfos[] = {
	{ "lc", 1, NULL, bench_loopcalc },
	{ "gma", 2, cookarg_gma, bench_gma },
	{ NULL, 0, NULL }
};

static benchinfo_t *
find_benchinfo(const char *code)
{
	int	i;

	for (i = 0; benchinfos[i].code != NULL; i++) {
		if (strcmp(benchinfos[i].code, code) == 0)
			return benchinfos + i;
	}
	return NULL;
}

static BOOL
parse_int(const char **pc_args, int *pval)
{
	const char	*p;
	int	val = 0;

	if (**pc_args == '\0')
		return FALSE;
	for (p = *pc_args; *p && *p != ','; p++) {
		if (p - *pc_args > 31)
			return FALSE;
		if (*p < '0' || *p > '9')
			return FALSE;
		val *= 10;
		val += (*p - '0');
	}
	if (*p == ',')
		p++;
	*pc_args = p;
	*pval = val;
	return TRUE;
}

static BOOL
parse_args(const char *c_args, benchrun_t *brun)
{
	int	i;

	if (c_args == NULL)
		return FALSE;
	if (!parse_int(&c_args, &brun->n_grid_width))
		return FALSE;
	if (!parse_int(&c_args, &brun->n_grid_height))
		return FALSE;
	if (!parse_int(&c_args, &brun->n_tb_width))
		return FALSE;
	if (!parse_int(&c_args, &brun->n_tb_height))
		return FALSE;

	for (i = 0; i < MAX_ARGS; i++) {
		int	arg;

		if (*c_args == '\0')
			return TRUE;
		if (!parse_int(&c_args, &arg))
			return FALSE;
		brun->args[i] = (void *)(long long)arg;
	}
	return TRUE;
}

extern "C" BOOL
add_bench(const char *code, const char *args)
{
	benchrun_t	*brun = benchruns + n_benches;
	benchinfo_t	*info;

	info = find_benchinfo(code);
	if (info == NULL)
		return FALSE;
	brun->info = info;
	if (!parse_args(args, brun))
		return FALSE;
	if (info->cookarg_func != NULL) {
		if (info->cookarg_func(brun->args) < 0) {
			error("failed to cook arguments");
			return FALSE;
		}
	}
	n_benches++;
	return TRUE;
}

extern "C" void
report_bench_result(void)
{
	benchrun_t	*brun;
	int	i;

	brun = benchruns;
	for (i = 0; i < n_benches; i++, brun++) {
		printf("%s: %d\n", brun->info->code, brun->res);
	}
}
