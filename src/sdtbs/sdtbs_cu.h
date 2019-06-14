#ifndef _SDTBS_CU_H_
#define _SDTBS_CU_H_

#include "sdtbs.h"

#define MAX_BENCHES	20
#define MAX_ARGS	5
#define N_THREADS_PER_mTB	32

extern unsigned n_sm_count;
extern unsigned n_threads_per_tb;
extern unsigned n_mtbs;
extern unsigned n_mtbs_per_sm;
extern int	n_benches;

typedef int (*bench_native_t)(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, int args[]);

typedef struct {
	const char	*code;
	int	skid;
	bench_native_t	bench_native;
} benchinfo_t;

typedef struct {
	benchinfo_t	*info;
	int	n_tbs_x, n_tbs_y;
	int	n_threads_x, n_threads_y;
	int	args[MAX_ARGS];
} benchrun_t;

typedef struct {
	int	skid;
	int	args[MAX_ARGS];
} micro_tb_t;

extern benchrun_t	benchruns[MAX_BENCHES];

extern micro_tb_t	*mtbs;

BOOL setup_gpu_devinfo(void);
void setup_micro_tbs(void);
micro_tb_t *get_mtb(int id_sm);

void sched_rr(void);

#endif
