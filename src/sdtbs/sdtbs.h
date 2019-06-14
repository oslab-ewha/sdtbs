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

extern unsigned	devno;

void error(const char *fmt, ...);
BOOL add_bench(const char *code, const char *args);
BOOL select_gpu_device(unsigned devno);

void run_native_tbs(void);
void run_sd_tbs(void);

#ifdef __cplusplus
}
#endif

#endif
