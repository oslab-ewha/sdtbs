#include "sdtbs.h"
#include "benchmgr.h"

int bench_native_loopcalc(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, int args[]);

benchrun_t	benchruns[MAX_BENCHES];
int	n_benches;

static benchinfo_t	benchinfos[] = {
	{ "lc", 1, bench_native_loopcalc },
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
parse_args(const char *c_args, benchrun_t *pbrun)
{
	int	i;

	if (!parse_int(&c_args, &pbrun->n_tbs_x))
		return FALSE;
	if (!parse_int(&c_args, &pbrun->n_tbs_y))
		return FALSE;
	if (!parse_int(&c_args, &pbrun->n_threads_x))
		return FALSE;
	if (!parse_int(&c_args, &pbrun->n_threads_y))
		return FALSE;

	for (i = 0; i < MAX_ARGS; i++) {
		if (*c_args == '\0')
			return TRUE;
		if (!parse_int(&c_args, &pbrun->args[i]))
			return FALSE;
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
	n_benches++;
	return TRUE;
}
