#ifndef _SDTBS_CU_H_
#define _SDTBS_CU_H_

#include "sdtbs.h"

#define MAX_BENCHES	20
#define MAX_ARGS	5
#define N_THREADS_PER_mTB	32

extern unsigned n_sm_count;
extern unsigned n_threads_per_MTB;
extern unsigned n_mtbs;
extern unsigned n_mtbs_per_sm;
extern int	n_benches;

typedef int (*cookarg_func_t)(void *args[]);
typedef int (*bench_func_t)(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, void *args[]);

typedef struct {
	const char	*code;
	int	skid;
	cookarg_func_t	cookarg_func;
	bench_func_t	bench_func;
} benchinfo_t;

typedef struct {
	int	skid;
	void *	args[MAX_ARGS];
} micro_tb_t;

typedef struct {
	benchinfo_t	*info;
	int	n_grid_width, n_grid_height;
	int	n_tb_width, n_tb_height;
	void *	args[MAX_ARGS];
	micro_tb_t	*mtb_first;
	int	res;
} benchrun_t;

typedef struct {
	const char	*name;
	unsigned (*get_tb_sm)(unsigned n_tbs_x, unsigned n_tbs_y);
} sched_t;

extern benchrun_t	benchruns[MAX_BENCHES];

extern micro_tb_t	*mtbs;

extern unsigned n_grid_width, n_grid_height;
extern unsigned n_tb_width, n_tb_height;

BOOL setup_gpu_devinfo(void);
void setup_micro_tbs(void);

micro_tb_t *get_mtb(unsigned id_sm);
BOOL is_sm_avail(int id_sm, unsigned n_threads);

void run_schedule(void);
void collect_mtb_result(void);

#endif
