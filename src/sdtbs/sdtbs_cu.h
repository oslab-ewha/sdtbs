#ifndef _SDTBS_CU_H_
#define _SDTBS_CU_H_

#include "sdtbs.h"

#define MAX_BENCHES	20
#define MAX_ARGS	5
#define N_THREADS_PER_mTB	32

extern unsigned n_sm_count;
extern unsigned n_threads_per_MTB;
extern unsigned	n_MTBs_per_sm;
extern int	n_benches;
extern int	n_tbs_submitted;
extern int	n_mtbs_submitted;

typedef int (*cookarg_func_t)(void *args[]);
typedef int (*bench_func_t)(cudaStream_t strm, int n_tbs_x, int n_tbs_y, int n_threads_x, int n_threads_y, void *args[], int *pres);

typedef struct {
	unsigned	skid;
	void *	args[MAX_ARGS];
	int		res;
	unsigned	n_mtbs_per_tb;
	int		primary_mtb_idx;
} benchrun_k_t;

typedef struct {
	benchrun_k_t	bruns[MAX_BENCHES];
	unsigned	size;
	unsigned	sched_id;
	unsigned	n_sm_count;
	unsigned	n_max_mtbs_per_sm;
	unsigned	n_max_mtbs_per_MTB;
	unsigned	n_mtbs;
	unsigned	n_tbs;
	unsigned char	brids[0];
} fedkern_info_t;

typedef struct {
	const char	*code;
	int	skid;
	cookarg_func_t	cookarg_func;
	bench_func_t	bench_func;
} benchinfo_t;

typedef struct {
	benchinfo_t	*info;
	int	n_grid_width, n_grid_height;
	int	n_tb_width, n_tb_height;
	void *	args[MAX_ARGS];
	int	res;
} benchrun_t;

typedef struct {
	const char	*name;
	unsigned (*get_tb_sm)(unsigned n_tbs_x, unsigned n_tbs_y);
} sched_t;

extern benchrun_t	benchruns[MAX_BENCHES];

extern unsigned n_grid_width, n_grid_height;
extern unsigned n_tb_width, n_tb_height;

__device__ uint get_smid(void);
__device__ unsigned find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs);

BOOL setup_gpu_devinfo(void);
fedkern_info_t *setup_fedkern_info(void);

BOOL assign_brid(fedkern_info_t *fkinfo, unsigned id_sm, unsigned char brid);
BOOL is_sm_avail(int id_sm, unsigned n_threads);

void run_schedule(fedkern_info_t *kfinfo);

void init_tickcount(void);
unsigned long long get_tickcount(void);

#endif
