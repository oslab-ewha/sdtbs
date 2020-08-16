#include "sdtbs_cu.h"

#include <pthread.h>
#include <unistd.h>

__device__ tbs_type_t	d_tbs_type;
__device__ skrun_t	*d_skruns;
__device__ unsigned	*d_mtbs_done_cnts;
static skrun_t	*g_skruns;
static unsigned	*g_mtbs_done_cnts;

static unsigned	*info_n_mtbs;

static BOOL	*skrun_dones;
static unsigned	skrid_done_min;
static unsigned	cur_skrid_host;

static BOOL	checker_done;
static pthread_t	checker;
static pthread_mutex_t	mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	cond = PTHREAD_COND_INITIALIZER;

static cudaStream_t	strm_submit;

#define SK_PROTO(name)	__device__ int name(void *args[])
#define SK_FUNCS(base)	SK_PROTO(base); SK_PROTO(base##_reloc);

SK_FUNCS(loopcalc)
SK_FUNCS(gma)
SK_FUNCS(lma)
SK_FUNCS(kmeans)

__device__ int
run_sub_kernel(skid_t skid, void *args[])
{
	switch (skid) {
	case LOOPCALC:
		return loopcalc(args);
	case LOOPCALC_RELOC:
		return loopcalc_reloc(args);
	case GMA:
		return gma(args);
	case GMA_RELOC:
		return gma_reloc(args);
	case LMA:
		return lma(args);
	case LMA_RELOC:
		return lma_reloc(args);
	case KMEANS:
		return kmeans(args);
	case KMEANS_RELOC:
		return kmeans_reloc(args);
	default:
		return 0;
	}
}

skrid_t
submit_skrun(skid_t skid, dim3 dimGrid, dim3 dimBlock, void *d_args[], int *d_pres)
{
	skrid_t	skrid;
	skrun_t	skrun;

	skrun.skid = skid;
	skrun.dimGrid = dimGrid;
	skrun.dimBlock = dimBlock;
	skrun.d_args = d_args;
	skrun.d_pres = d_pres;
	skrun.n_tbs = dimGrid.x * dimGrid.y;
	skrun.n_mtbs_per_tb = dimBlock.x * dimBlock.y / N_THREADS_PER_mTB;

	pthread_mutex_lock(&mutex);

	skrid = cur_skrid_host + 1;
	info_n_mtbs[skrid - 1] = skrun.n_tbs * skrun.n_mtbs_per_tb;

	cudaMemcpyAsync(g_skruns + cur_skrid_host, &skrun, sizeof(skrun_t), cudaMemcpyHostToDevice, strm_submit);
	cudaStreamSynchronize(strm_submit);

	cur_skrid_host++;

	pthread_mutex_unlock(&mutex);

	return skrid;
}

void
wait_skrun(skrid_t skrid)
{
	pthread_mutex_lock(&mutex);

	while (!checker_done && !skrun_dones[skrid - 1])
		pthread_cond_wait(&cond, &mutex);

	pthread_mutex_unlock(&mutex);
}

static void
notify_done_skruns(unsigned *mtbs_done_cnts, unsigned n_checks)
{
	unsigned	min_new = skrid_done_min;
	BOOL		notify = FALSE;
	unsigned	i;

	pthread_mutex_lock(&mutex);

	for (i = 0; i < n_checks; i++) {
		if (skrun_dones[i + skrid_done_min])
			continue;
		if (mtbs_done_cnts[i] == info_n_mtbs[i + skrid_done_min]) {
			notify = TRUE;
			skrun_dones[i + skrid_done_min] = TRUE;
			if (min_new == i + skrid_done_min) {
				min_new++;
			}
		}
	}
	skrid_done_min = min_new;
	if (notify)
		pthread_cond_broadcast(&cond);
	pthread_mutex_unlock(&mutex);
}

static void *
skruns_checkfunc(void *arg)
{
	cudaStream_t	strm;

	cudaStreamCreate(&strm);

	while (!checker_done) {
		unsigned	n_checks = cur_skrid_host - skrid_done_min;
		if (n_checks > 0) {
			unsigned	*mtbs_done_cnts = (unsigned *)malloc(sizeof(unsigned) * n_checks);
			cudaMemcpyAsync(mtbs_done_cnts, g_mtbs_done_cnts + skrid_done_min, sizeof(unsigned) * n_checks, cudaMemcpyDeviceToHost, strm);
			cudaStreamSynchronize(strm);
			notify_done_skruns(mtbs_done_cnts, n_checks);
			free(mtbs_done_cnts);
		}
		usleep(100000);
	}

	cudaStreamDestroy(strm);
	return NULL;
}

__global__ void
kernel_init_skrun(tbs_type_t type, skrun_t *skruns, unsigned *mtbs_done_cnts)
{
	int	i;

	d_tbs_type = type;
	d_skruns = skruns;
	d_mtbs_done_cnts = mtbs_done_cnts;
	for (i = 0; i < MAX_QUEUED_KERNELS; i++) {
		skruns[i].skid = 0;
		mtbs_done_cnts[i] = 0;
	}
}

void
init_skrun(void)
{
	cudaStreamCreate(&strm_submit);

	cudaMalloc(&g_skruns, sizeof(skrun_t) * MAX_QUEUED_KERNELS);
	cudaMalloc(&g_mtbs_done_cnts, sizeof(unsigned) * MAX_QUEUED_KERNELS);

	info_n_mtbs = (unsigned *)calloc(MAX_QUEUED_KERNELS, sizeof(unsigned));
	skrun_dones = (BOOL *)calloc(MAX_QUEUED_KERNELS, sizeof(BOOL));

	pthread_create(&checker, NULL, skruns_checkfunc, NULL);

	dim3 dimGrid(1,1), dimBlock(1,1);
	kernel_init_skrun<<<dimGrid, dimBlock>>>(sched->type, g_skruns, g_mtbs_done_cnts);
	cudaDeviceSynchronize();
}

void
fini_skrun(void)
{
}
