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

typedef int (*cookarg_func_t)(dim3 dimGrid, dim3 dimBlock, void *args[]);
typedef void (*bench_func_t)(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres);

typedef struct {
	unsigned	skid;
	void *	args[MAX_ARGS];
	int		res;
	dim3		dimGrid, dimBlock;
	unsigned	n_mtbs_per_tb;
	int		primary_mtb_idx;
} benchrun_k_t;

typedef struct {
	benchrun_k_t	bruns[MAX_BENCHES];
	unsigned	size;
	unsigned	sched_id;
	void *		sched_arg;
	unsigned	n_sm_count;
	unsigned	n_max_mtbs_per_sm;
	unsigned	n_max_mtbs_per_MTB;
	unsigned	n_mtbs;
	unsigned	n_tbs;
	BOOL		initialized;
	BOOL		fully_dynamic;
	unsigned char	brids[0];
} fedkern_info_t;

typedef struct {
	const char	*code;
	int	skid;
	cookarg_func_t	cookarg_func;
	bench_func_t	bench_func;
	bench_func_t	bench_func_noreloc;
} benchinfo_t;

typedef struct {
	benchinfo_t	*info;
	dim3	dimGrid, dimBlock;
	void *	args[MAX_ARGS];
	int	res;
} benchrun_t;

typedef struct {
	const char	*name;
	BOOL		direct_mode, use_semi_dynamic_sched, use_static_sched, use_relocatable;
	void *(*parse_arg)(const char *argstr);
	unsigned (*get_tb_sm)(dim3 dimBlock, unsigned n_tbs_x, unsigned n_tbs_y);
} sched_t;

extern sched_t		*sched;
extern benchrun_t	benchruns[MAX_BENCHES];

__device__ uint get_smid(void);
__device__ void sleep_in_kernel(void);
__device__ unsigned find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs);
__device__ unsigned get_n_active_mtbs(unsigned id_sm);

BOOL setup_gpu_devinfo(void);
fedkern_info_t *setup_fedkern_info(void);

BOOL is_sm_avail(int id_sm, unsigned n_threads);
unsigned get_sm_n_sched_mtbs(int id_sm);

BOOL run_schedule(fedkern_info_t *kfinfo);
void run_schedule_dyn(fedkern_info_t *d_kfinfo);

void init_tickcount(void);
unsigned long long get_tickcount(void);

#endif
