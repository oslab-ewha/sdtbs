#include "sdtbs.h"

static void
usage(void)
{
	printf(
"sdtbs <options> <benchmark spec>...\n"
"<options>:\n"
"  -d <device no>: select GPU device\n"
"  -x: run direct mode\n"
"  -h: help\n"
"<benchmark spec>: <code>:<arg string>\n"
" <code>:\n"
"  lc: repetitive calculation(in-house)\n"
" <arg string>:\n"
"   NOTE: First 4 arguments are <grid width>,<grid height>,<tb width>,<tb heigh>\n"
"   lc: <calculation type>,<# iterations>,<# iterations for calculation type greater than 3>\n"
"        calculation type: 1(int),2(float),3(double),default:empty\n"
"                          4(float/double),5(int/float),6(int/double)\n"
"                          7(float/double tb),8(int/float tb),9(int/double tb)\n"
		);
}

static BOOL	direct_mode;

unsigned	devno;

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

static int
parse_options(int argc, char *argv[])
{
	int	c;

	while ((c = getopt(argc, argv, "d:xh")) != -1) {
		switch (c) {
		case 'd':
			select_device(optarg);
			break;
		case 'x':
			direct_mode = TRUE;
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
	if (parse_options(argc, argv) < 0) {
		return 1;
	}

	if (parse_benchargs(argc - optind, argv + optind) < 0) {
		return 2;
	}

	if (direct_mode)
		run_native_tbs();
	else
		run_sd_tbs();
	return 0;
}
