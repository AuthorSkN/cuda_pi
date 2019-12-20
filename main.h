#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "ctime"

#define n 1024 * 1024 * 32

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "\nCUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "\n";
		cudaDeviceReset();
		exit(99);
	}
}
float calcPICPU(float *rndX, float *rndY, unsigned size);
void gpu_fillRand(float *a, float *b, unsigned int size);
void piWithCuda(float *dev_rndX, float *dev_rndY, unsigned int size);

int main()
{
	float *rndX, *rndY;
	float *dev_rndX = 0, *dev_rndY = 0;

	rndX = (float *)calloc(n, sizeof(float));
	rndY = (float *)calloc(n, sizeof(float));

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void **)&dev_rndX, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&dev_rndY, n * sizeof(float)));

	gpu_fillRand(dev_rndX, dev_rndY, n);

	checkCudaErrors(cudaMemcpy(rndX, dev_rndX, n * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rndY, dev_rndY, n * sizeof(float), cudaMemcpyDeviceToHost));

	clock_t startCPU;
	startCPU = clock();
	float cpu_result = calcPICPU(rndX, rndY, n);
	printf("\nCPU's time for pi: %f", (clock() - startCPU) / (double)CLOCKS_PER_SEC);
	printf("\nCPU's result: %f", cpu_result);


	int *dev_blocks_counts = 0, *blocks_counts = 0;
	float gpuTime = 0.0f;
	cudaEvent_t start, stop;

	int threads = 512;
	int block_num = n / (128 * threads);

	blocks_counts = (int *)calloc(block_num, sizeof(int));

	checkCudaErrors(cudaMalloc((void **)&dev_blocks_counts, 512 * sizeof(int)));
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	checkCudaErrors(cudaMemset(dev_blocks_counts, 0, sizeof(int)));

	calcPi<<<block_num, threads>>>(dev_rndX, dev_rndY, dev_blocks_counts, n);

	checkCudaErrors(cudaMemcpy(blocks_counts, dev_blocks_counts, block_num * sizeof(int), cudaMemcpyDeviceToHost));
	int count = 0;
	for (int i = 0; i < block_num; i++)
	{
		count += blocks_counts[i];
	};

	float gpu_result = float(count) * 4 / float(n);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&gpuTime, start, stop));
	printf("\nGPU's time spent executing %s: %f seconds", "kernel", gpuTime / 1000);
	printf("\nGPU's result: %f\n", gpu_result);
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	cudaFree(dev_rndX);
	cudaFree(dev_rndY);
	cudaFree(dev_blocks_counts);
	return 0;
}

void gpu_fillRand(float *a, float *b, unsigned int size)
{
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, a, size);
	curandGenerateUniform(prng, b, size);
}

float calcPICPU(float *rndX, float *rndY, unsigned size)
{
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		if (rndX[i] * rndX[i] + rndY[i] * rndY[i] < 1.0f)
		{
			count++;
		}
	}
	return float(count) * 4.0 / size;
}
