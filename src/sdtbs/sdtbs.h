#ifndef _SDTBS_H_
#define _SDTBS_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>

#define TRUE	1
#define FALSE	0

typedef int	BOOL;

#ifdef __cplusplus
extern "C" {
#endif

BOOL select_gpu_device(unsigned devno);

#ifdef __cplusplus
}
#endif

#endif
