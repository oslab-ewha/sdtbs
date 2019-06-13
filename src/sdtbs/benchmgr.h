#ifndef _BENCH_H_
#define _BENCH_H_

#define MAX_BENCHES	20	
#define MAX_ARGS	5

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

extern benchrun_t	benchruns[MAX_BENCHES];
extern int	n_benches;

#endif
