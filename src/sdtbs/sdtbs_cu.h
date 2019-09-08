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
} benchrun_k_t;

typedef struct {
	/* number of mAT */
	unsigned	n_mATs;
	/* mtb allocation tables */
	unsigned char	**mATs;
	/* mtb offset tables */
	unsigned short	**mOTs;
} fkinfo_static_t;

typedef struct {
	unsigned char	*brids_submitted;
} fkinfo_dyn_t;

typedef enum {
	TBS_TYPE_HW = 1,
	TBS_TYPE_HW_RELOC,
	TBS_TYPE_STATIC,
	TBS_TYPE_DYNAMIC,
	TBS_TYPE_SEMI_DYNAMIC,
	TBS_TYPE_SOLO,
	TBS_TYPE_HOST
} tbs_type_t;

typedef struct {
	BOOL		*pdone;
	unsigned char	*epochs_dev;
	unsigned char	*epochs_host;
	unsigned char	*mAT;
	unsigned short	*mOT;
} fkinfo_host_t;

typedef struct {
	benchrun_k_t	bruns[MAX_BENCHES];
	unsigned	sched_id;
	tbs_type_t	tbs_type;
	void *		sched_arg;
	unsigned	n_sm_count;
	unsigned	n_max_mtbs_per_sm;
	unsigned	n_max_mtbs_per_MTB;
	unsigned	n_mtbs;
	unsigned	n_tbs;
	BOOL		initialized;

	union {
		fkinfo_static_t	sta;
		fkinfo_dyn_t	dyn;
		fkinfo_host_t	host;
	} u;
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
	tbs_type_t	type;
	void *(*parse_arg)(const char *argstr);
	unsigned (*get_tb_sm)(dim3 dimBlock, unsigned n_tbs_x, unsigned n_tbs_y);
} sched_t;

extern sched_t		*sched;
extern benchrun_t	benchruns[MAX_BENCHES];

__device__ uint get_smid(void);
__device__ void sleep_in_kernel(void);
__device__ unsigned find_mtb_start(unsigned id_sm, unsigned idx_mtb_start, unsigned n_mtbs);
__device__ unsigned get_n_active_mtbs(unsigned id_sm);

fedkern_info_t *create_fedkern_info(void);
fedkern_info_t *create_fedkern_info_kernel(fedkern_info_t *fkinfo);
void free_fedkern_info(fedkern_info_t *fkinfo);

BOOL setup_gpu_devinfo(void);

unsigned get_n_mTBs_for_threads(unsigned n_threads);
BOOL is_sm_avail(int id_sm, unsigned n_mTBs);
unsigned get_sm_n_sched_mTBs(int id_sm);
void use_next_mAT(int id_sm);

BOOL run_schedule(fedkern_info_t *fkinfo);
void run_schedule_dyn(fedkern_info_t *fkinfo);
void run_schedule_host(fedkern_info_t *fkinfo);

void init_tickcount(void);
unsigned get_tickcount(void);

#endif
