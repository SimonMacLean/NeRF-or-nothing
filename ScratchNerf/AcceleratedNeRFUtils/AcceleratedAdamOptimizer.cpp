#include "AcceleratedAdamOptimizer.h"
#include <corecrt_math.h>
#include "helpers.h"

namespace AcceleratedNeRFUtils {
	AcceleratedAdamOptimizer::AcceleratedAdamOptimizer(array<Int32>^ layer_sizes)
	{
		this->num_layers = layer_sizes->Length;
		this->layer_sizes = layer_sizes;
		this->dev_m = gcnew array<float*>(num_layers);
		this->dev_v = gcnew array<float*>(num_layers);
		pin_ptr<float*> dev_m_ptr = &dev_m[0];
		pin_ptr<float*> dev_v_ptr = &dev_v[0];
		for (int i = 0; i < num_layers; i++)
		{
			float** m = dev_m_ptr;
			float** v = dev_v_ptr;
			cudaMalloc(m + i, layer_sizes[i] * sizeof(float));
			cudaMalloc(v + i, layer_sizes[i] * sizeof(float));
		}
	}

	void AcceleratedAdamOptimizer::step(array<float*>^ params, array<float*>^ grads, float learning_rate)
	{
		extern void adam_optimizer_step(float*, const float*, float*, float*, const float, const float, const float, const float, const float, const int);
		iteration++;
		float inv_1_minus_beta1_pow = 1 / (1 - powf(beta1, iteration));
		float inv_1_minus_beta2_pow = 1 / (1 - powf(beta2, iteration));
		float beta1 = this->beta1;
		float beta2 = this->beta2;
		for (int i = 0; i < num_layers; i++)
		{
			int layer_size = layer_sizes[i];
			void* aos_args[10] = {
				get_head_ptr(params) + i, get_head_ptr(grads) + i, get_head_ptr(dev_m) + i, get_head_ptr(dev_v) + i,
				&learning_rate, &beta1, &beta2, &inv_1_minus_beta1_pow, &inv_1_minus_beta2_pow, &layer_size
			};
			cudaLaunchKernel((void*)adam_optimizer_step, dim3(layer_size) / block_1d, block_1d, aos_args);
		}
		cudaDeviceSynchronize();
	}

	AcceleratedAdamOptimizer::~AcceleratedAdamOptimizer()
	{
		for (int i = 0; i < num_layers; i++)
		{
			cudaFree(dev_m[i]);
			cudaFree(dev_v[i]);
		}
	}
}
