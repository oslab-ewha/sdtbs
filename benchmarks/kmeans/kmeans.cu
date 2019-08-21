#include <stdio.h>

#include "../benchapi.h"

#define FLT_MAX 3.40282347e+38

typedef struct {
	int	npoints;
	int	*d_membership;
	float	*d_clusters;
	float	*d_features;
	float	*d_features_flipped;
} kmeans_conf_t;

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;

#define ASSUMED_NR_CLUSTERS	32

/* constant memory for cluster centers */
__constant__ float c_clusters[ASSUMED_NR_CLUSTERS * 34];

__device__ static void
kmeansPoints(float *features, int nfeatures, int npoints, int nclusters, int *membership, float *clusters)
{
	const unsigned	block_id = get_gridDimX() * get_blockIdxY() + get_blockIdxX();
	// point/thread ID
	const unsigned int	point_id = block_id * get_blockDimX() * get_blockDimY() + get_threadIdxX();

	int index = -1;

	if (point_id < npoints) {
		int	i, j;
		float	min_dist = FLT_MAX;
		float	dist; /* distance square between a point to cluster center */

		/* find the cluster center id with min distance to pt */
		for (i = 0; i < nclusters; i++) {
			 /* base index of cluster centers for inverted array */
			int	cluster_base_index = i * nfeatures;
			 /* Euclidean distance sqaure */
			float	ans = 0.0;

			for (j = 0; j < nfeatures; j++) {
				 /* appropriate index of data point */
				int addr = point_id + j * npoints;
				 /* distance between a data point to cluster centers */
				float diff = (tex1Dfetch(t_features, addr) - c_clusters[cluster_base_index + j]);
				/* sum of squares */
				ans += diff * diff;
			}
			dist = ans;

			/* see if distance is smaller than previous ones:
			 * if so, change minimum distance and save index of cluster center
			 */
			if (dist < min_dist) {
				min_dist = dist;
				index = i;
			}
		}
	}

	if (point_id < npoints) {
		/* assign the membership to object point_id */
		membership[point_id] = index;
	}
}

__device__ int
kmeans(void *args[])
{
	int     nclusters = (int)(long long)args[0];
        int     nfeatures = (int)(long long)args[1];
	kmeans_conf_t	*pkmc = (kmeans_conf_t *)args[2];

	kmeansPoints(pkmc->d_features, nfeatures, pkmc->npoints, nclusters, pkmc->d_membership, pkmc->d_clusters);
	return 0;
}

static void
init_membership(kmeans_conf_t *pkmc)
{
	int	*membership;
	int	i;

	membership = (int *)malloc(pkmc->npoints * sizeof(int));
	for (i = 0; i < pkmc->npoints; i++)
		membership[i] = -1;
	cudaMalloc(&pkmc->d_membership, pkmc->npoints * sizeof(int));
	cudaMemcpy(pkmc->d_membership, membership, pkmc->npoints * sizeof(int), cudaMemcpyHostToDevice);
	free(membership);
}

static void
invert_mapping(float *output, float *input, int npoints, int nfeatures)
{
	int     i;

	for (i = 0; i < npoints; i++) {
		int     j;
		for (j = 0; j < nfeatures; j++)
			output[i + npoints * j] = input[i * nfeatures + j];
	}
}

static float *
setup_features(kmeans_conf_t *pkmc, int nfeatures)
{
	float	*features, *features_inverted;
	int	i;

	features = (float *)malloc(pkmc->npoints * nfeatures * sizeof(float));
	for (i = 0; i < pkmc->npoints * nfeatures; i++) {
		features[i] = rand() / rand();
	}

	features_inverted = (float *)malloc(pkmc->npoints * nfeatures * sizeof(float));
	invert_mapping(features_inverted, features, pkmc->npoints, nfeatures);

	cudaMalloc(&pkmc->d_features, pkmc->npoints * nfeatures * sizeof(float));
	cudaMemcpy(pkmc->d_features, features_inverted, pkmc->npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&pkmc->d_features_flipped, pkmc->npoints * nfeatures * sizeof(float));
	cudaMemcpy(pkmc->d_features_flipped, features, pkmc->npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	free(features_inverted);
	return features;
}

static void
setup_clusters(kmeans_conf_t *pkmc, float *features, int nclusters, int nfeatures)
{
	float	*clusters;
	int	i;

	clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));

	cudaMalloc(&pkmc->d_clusters, nclusters * nfeatures * sizeof(float));
	/* copy clusters (host to device) */
	cudaMemcpy(pkmc->d_clusters, clusters, nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	/* randomly pick cluster centers */
	for (i = 0; i < nclusters; i++) {
		int	n = (int)rand() % pkmc->npoints;
		int	j;
		for (j = 0; j < nfeatures; j++) {
			clusters[i * nfeatures + j] = features[n * nfeatures + j]; // remapped
		}
	}

	free(clusters);
}

int
cookarg_kmeans(dim3 dimGrid, dim3 dimBlock, void *args[])
{
	int	nclusters = (int)(long long)args[0];
	int	nfeatures = (int)(long long)args[1];
	kmeans_conf_t	kmc, *d_pkmc;
	int	npoints = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y;
	float	*features;

	kmc.npoints = npoints;
	init_membership(&kmc);

	features = setup_features(&kmc, nfeatures);
	setup_clusters(&kmc, features, nclusters, nfeatures);

	/* set up texture */
	cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
	t_features.filterMode = cudaFilterModePoint;
	t_features.normalized = false;
	t_features.channelDesc = chDesc0;

	cudaBindTexture(NULL, &t_features, kmc.d_features, &chDesc0, npoints * nfeatures * sizeof(float));

	cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
	t_features_flipped.filterMode = cudaFilterModePoint;
	t_features_flipped.normalized = false;
	t_features_flipped.channelDesc = chDesc1;

	cudaBindTexture(NULL, &t_features_flipped, kmc.d_features_flipped, &chDesc1, npoints * nfeatures * sizeof(float));

	cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
	t_clusters.filterMode = cudaFilterModePoint;
	t_clusters.normalized = false;
	t_clusters.channelDesc = chDesc2;

	cudaBindTexture(NULL, &t_clusters, kmc.d_clusters, &chDesc2, nclusters * nfeatures * sizeof(float));

	free(features);

	cudaMalloc(&d_pkmc, sizeof(kmeans_conf_t));
	cudaMemcpy(d_pkmc, &kmc, sizeof(kmeans_conf_t), cudaMemcpyHostToDevice);
	args[2] = d_pkmc;

	return 0;
}

__global__ static void
kernel_kmeans(void *args[], int *pres)
{
	native_mode = 1;
	*pres = kmeans(args);
}

void
bench_kmeans(cudaStream_t strm, dim3 dimGrid, dim3 dimBlock, void *args[], int *pres)
{
	kernel_kmeans<<<dimGrid, dimBlock, 0, strm>>>(args, pres);
}