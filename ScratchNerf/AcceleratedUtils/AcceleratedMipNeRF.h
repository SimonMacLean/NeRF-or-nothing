#pragma once
#include <vector_types.h>

#include "AcceleratedMLP.h"
using namespace System;
using namespace System::Numerics;
public ref class AcceleratedMipNeRF
{
public:
	AcceleratedMipNeRF();
	array<Single>^ GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
		array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<array<Vector3>^, array<Vector3>^> getOutputGradient);
private:
	static constexpr int num_levels = 2;
	static constexpr int num_samples = 128;
	static constexpr int num_rays = 4096;
	accelerated_mlp^ mlp_;
	float3* dev_origins_;
	float3* dev_directions_;
	float* dev_radii_;
	float* dev_nears_;
	float* dev_fars_;
	float* dev_loss_mults_;
	float* dev_t_vals_[num_levels];
	float3* dev_means_[num_levels];
	float* dev_weights_[num_levels];
	float* dev_transmittance_[num_levels];
	float* dev_alpha_[num_levels];
	float3* dev_output_rgb_[num_levels];
	float* dev_output_density_[num_levels];
	curandState* dev_states_;
};

