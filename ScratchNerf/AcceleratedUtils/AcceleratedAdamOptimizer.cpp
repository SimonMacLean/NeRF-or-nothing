#include "AcceleratedAdamOptimizer.h"

#include <corecrt_math.h>
#include <cuda_runtime.h>

AcceleratedAdamOptimizer::AcceleratedAdamOptimizer(int* layer_sizes, int num_layers)
{
	this->num_layers = num_layers;
	this->layer_sizes = layer_sizes;
	this->dev_m = new float*[num_layers];
	this->dev_v = new float*[num_layers];
	for(int i = 0; i < num_layers; i++)
	{
		cudaMalloc(&dev_m[i], layer_sizes[i] * sizeof(float));
		cudaMalloc(&dev_v[i], layer_sizes[i] * sizeof(float));
	}
}

void AcceleratedAdamOptimizer::step(float** params, float** grads, float learning_rate)
{
	extern void adam_optimizer_step(float* variables, const float* gradients, float* m, float* v, float learning_rate, float beta1, float beta2, float inv_1_minus_beta1_pow, float inv_1_minus_beta2_pow, int size);
	iteration++;
	float inv_1_minus_beta1_pow = 1/(1 - powf(beta1, iteration));
	float inv_1_minus_beta2_pow = 1/(1 - powf(beta2, iteration));
	float beta1 = this->beta1;
	float beta2 = this->beta2;
	dim3 block(1024);
	for(int i = 0; i < num_layers; i++)
	{
		dim3 grid((layer_sizes[i] + block.x - 1) / block.x);
		void* aos_args[10] = {params + i, grads + i, dev_m + i, dev_v + i, &learning_rate, &beta1, &beta2, &inv_1_minus_beta1_pow, &inv_1_minus_beta2_pow, layer_sizes + i};
		cudaLaunchKernel((void*)adam_optimizer_step, grid, block, aos_args);
	}
	cudaDeviceSynchronize();
}
