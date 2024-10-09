#pragma once
#include <cuda_runtime.h>
namespace AcceleratedNeRFUtils {
	using namespace System;
	public ref class AcceleratedAdamOptimizer
	{
	public:
		AcceleratedAdamOptimizer(array<Int32>^ layer_sizes);
		void step(array<float*>^ params, array<float*>^ grads, float learning_rate);
		~AcceleratedAdamOptimizer();
	private:
		int num_layers;
		array<Int32>^ layer_sizes;
		array<float*>^ dev_m;
		array<float*>^ dev_v;
		static constexpr float beta1 = 0.9f;
		static constexpr float beta2 = 0.999f;
		int iteration = 0;
		static constexpr float epsilon = 1e-8f;
	};
}

