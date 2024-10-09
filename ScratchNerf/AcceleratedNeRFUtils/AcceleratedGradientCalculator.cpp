#include "AcceleratedGradientCalculator.h"

#include <cstdio>
#include <cuda_runtime.h>

#include "helpers.h"

namespace AcceleratedNeRFUtils {
	AcceleratedGradientCalculator::AcceleratedGradientCalculator(int batch_size)
	{
		float3* grad_ptr;
		cudaMalloc(&grad_ptr, batch_size * sizeof(float3));
		float3* pixels_ptr;
		cudaMalloc(&pixels_ptr, batch_size * sizeof(float3));
		gradient = grad_ptr;
		pixels = pixels_ptr;
	}
	uint64_t AcceleratedGradientCalculator::get_output_gradient(uint64_t input, array<Vector3>^ pixels, uint64_t loss_mults, float loss_mult_sum, int level)
	{
		extern void get_output_gradient(const float3*, const float3*, const float*, float3*, const float, const int);
		float3* input_ptr = (float3*)input;
		float* loss_mults_ptr = (float*)loss_mults;
		cudaMemcpy(get_head_ptr(pixels), this->pixels, pixels->Length * sizeof(float3), cudaMemcpyHostToDevice);
		float3* grad_ptr = gradient;
		float3* pixels_ptr = this->pixels;
		void* gog_args[6] = { &input_ptr, &pixels_ptr, &loss_mults_ptr, &grad_ptr, &loss_mult_sum, &level};
		cudaLaunchKernel((void*)get_output_gradient, dim3(pixels->Length) / block_1d, block_1d, gog_args);
		cudaDeviceSynchronize();
		return (uint64_t)grad_ptr;
	}

	AcceleratedGradientCalculator::~AcceleratedGradientCalculator()
	{
		cudaFree(gradient);
		cudaFree(pixels);
	}
}
