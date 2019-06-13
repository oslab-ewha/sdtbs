#include "sdtbs.h"

extern void run_macrotb(void);

static void
usage(void)
{
	printf(
"sdtbs <options> <benchmark spec>...\n"
"<options>:\n"
"  -d <device no>: select GPU device\n"
"  -h: help\n"
"<benchmark spec>: <code>:<arg string>\n"
" <code>:\n"
		);
}

static void
error(const char *fmt, ...)
{
	char	*msg;
	va_list	ap;
	int	n;
	
	va_start(ap, fmt);
	n = vasprintf(&msg, fmt, ap);
	va_end(ap);
	if (n >= 0) {
		fprintf(stderr, "error: %s\n", msg);
		free(msg);
	}
}

static void
select_device(const char *str_devno)
{
	unsigned	devno;

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

	while ((c = getopt(argc, argv, "d:h")) != -1) {
		switch (c) {
		case 'd':
			select_device(optarg);
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

	run_macrotb();
	return 0;
}
