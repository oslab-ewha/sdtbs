#include "sdtbs.h"

static void
usage(void)
{
	printf(
"sdtbs <options> <benchmark spec>...\n"
"<options>:\n"
"  -d <device no>: select GPU device\n"
"  -p <policy:optional arg>: scheduling policy\n"
"     optional policy argument can be provided with a colon\n"
"     supported policies: hw(hardware scheduling)\n"
"                         rr(round-robin, default)\n"
"                         rrf(round-robin fully)\n"
"                         fca(first come allocation)\n"
"                         rrm(round-robin max):max mtbs per sm\n"
"     hw policy suffix: R(relocatable)\n"
"     other policy suffix: D(semi-dynamic), S(static)\n"
"  -M <MTB count per sm>\n"
"  -T <thread count per MTB>\n"
"  -h: help\n"
"<benchmark spec>: <code>:<arg string>\n"
" <code>:\n"
"  lc: repetitive calculation(in-house)\n"
"  gma: global memory access(in-house)\n"
"  lma: localized memory access(in-house)\n"
"  kmeans: kmeans\n"
" <arg string>:\n"
"   NOTE: First 4 arguments are <grid width>,<grid height>,<tb width>,<tb height>\n"
"   lc: <calculation type>,<iterations>,<# iterations for calculation type greater than 3>\n"
"        calculation type: 1(int),2(float),3(double),default:empty\n"
"                          4(float/double),5(int/float),6(int/double)\n"
"                          7(float/double tb),8(int/float tb),9(int/double tb)\n"
"   gma: <global mem in KB>,<iterations>\n"
"   lma: <chunk size in byte>,<reference span>,<iterations>\n"
"   kmeans: <# of points per thread>,<# of clusters>,<# of features>,<iterations>\n"
		);
}

unsigned	devno;
unsigned	arg_n_MTBs_per_sm;
unsigned	arg_n_threads_per_MTB;

static int
parse_benchargs(int argc, char *argv[])
{
	int	i;

	if (argc == 0) {
		error("no benchmark provided");
		return -1;
	}
	for (i = 0; i < argc; i++) {
		char	*colon, *args = NULL;

		colon = strchr(argv[i], ':');
		if (colon != NULL) {
			*colon = '\0';
			args = strdup(colon + 1);
		}
		if (!add_bench(argv[i], args)) {
			error("invalid benchmark code or arguments: %s", argv[i]);
			return -1;
		}
	}
	return 0;
}

static void
select_device(const char *str_devno)
{
	if (sscanf(str_devno, "%u", &devno) != 1) {
		error("%s: invalid GPU device number", str_devno);
		return;
	}
	if (!select_gpu_device(devno)) {
		error("failed to set GPU device: %s", str_devno);
	}
}

static void
setup_n_MTBs(const char *str_n_MTBs)
{
	if (sscanf(str_n_MTBs, "%u", &arg_n_MTBs_per_sm) != 1 || arg_n_MTBs_per_sm == 0) {
		error("%s: invalid number of MTBs per SM", str_n_MTBs);
	}
}

static void
setup_n_threads(const char *str_n_threads)
{
	if (sscanf(str_n_threads, "%u", &arg_n_threads_per_MTB) != 1 || arg_n_threads_per_MTB == 0) {
		error("%s: invalid number of threads per MTB", str_n_threads);
	}
}

static int
parse_options(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "d:p:M:T:h")) != -1) {
		switch (c) {
		case 'd':
			select_device(optarg);
			break;
		case 'p':
			setup_sched(optarg);
			break;
		case 'M':
			setup_n_MTBs(optarg);
			break;
		case 'T':
			setup_n_threads(optarg);
			break;
		case 'h':
			usage();
			return -100;
		default:
			usage();
			return -1;
		}
	}
	return 0;
}

int
main(int argc, char *argv[])
{
	BOOL	res;
	unsigned	elapsed;

	if (parse_options(argc, argv) < 0) {
		return 1;
	}

	if (parse_benchargs(argc - optind, argv + optind) < 0) {
		return 2;
	}

	if (run_tbs(&elapsed)) {
		report(elapsed);
		return 0;
	}
	return 4;
}
