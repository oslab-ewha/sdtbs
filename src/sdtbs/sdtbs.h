#ifndef _SDTBS_H_
#define _SDTBS_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#define TRUE	1
#define FALSE	0

typedef int	BOOL;

#ifdef __cplusplus
extern "C" {
#endif

#define ASSERT(cond)                    do { assert(cond); } while (0)
#define FATAL(exitcode, fmt, ...)       do { error(fmt, ## __VA_ARGS__); exit(exitcode); } while (0)

extern BOOL	direct_mode;
extern BOOL	use_relocatable;
extern BOOL	use_static_sched;
extern unsigned	sched_id;
extern unsigned	devno;

void error(const char *fmt, ...);
BOOL add_bench(const char *code, const char *args);

void report(unsigned elapsed);

BOOL select_gpu_device(unsigned devno);

BOOL run_native_tbs(unsigned *pticks);
BOOL run_sd_tbs(unsigned *pticks);

void setup_sched(const char *strpol);

#ifdef __cplusplus
}
#endif

#endif
