__global__ void calcPi(float *rndX, float *rndY, int *blocks_counts, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	__shared__ int counts[512];
	int count = 0;
	for (int i = idx; i < n; i += offset) {
		if (rndX[i] * rndX[i] + rndY[i] * rndY[i] < 1.0f) {
			count++;
		}
	}
	counts[threadIdx.x] = count;

	__syncthreads();

	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += counts[j];
		}
		blocks_counts[blockIdx.x] = total;
	}
}

#define kernel calcPi
#include "main.h"