#pragma once
#include <cstdint>
#include <vector_types.h>

namespace AcceleratedNeRFUtils {
	using namespace System;
	using namespace System::Numerics;
	public ref class AcceleratedGradientCalculator
	{
	public:
		AcceleratedGradientCalculator(int batch_size);
		uint64_t get_output_gradient(uint64_t input, array<Vector3>^ pixels, uint64_t loss_mults, float loss_mult_sum, int level);
		~AcceleratedGradientCalculator();
	private:
		float3* gradient;
		float3* pixels;
	};
}

