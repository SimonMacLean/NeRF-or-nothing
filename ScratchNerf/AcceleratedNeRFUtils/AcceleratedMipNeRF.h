#pragma once
#include <cstdint>

#include "AcceleratedMLP.h"
#include "helpers.h"

namespace AcceleratedNeRFUtils {
	using namespace System;
	using namespace System::Numerics;
	public ref class AcceleratedMipNeRF
	{
	public:
		AcceleratedMipNeRF();
		array<float*>^ GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
		                           array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<uint64_t, int, float, uint64_t, uint64_t>^
		                           getOutputGradient);
		array<int>^ GetLayerSizes();
		AcceleratedMLP^ mlp = gcnew AcceleratedMLP(16, 4);
		~AcceleratedMipNeRF();
	private:
		float3* dev_origins_;
		float3* dev_directions_;
		float* dev_radii_;
		float* dev_nears_;
		float* dev_fars_;
		float* dev_loss_mults_;
		array<float*>^ dev_t_vals_ = gcnew array<float*>(num_levels);
		array <float3*>^ dev_means_ = gcnew array<float3*>(num_levels);
		array <float3*>^ dev_covs_ = gcnew array<float3*>(num_levels);
		array<float*>^ dev_weights_ = gcnew array<float*>(num_levels);
		array<float*>^ dev_transmittance_ = gcnew array<float*>(num_levels);
		array<float*>^ dev_alpha_ = gcnew array<float*>(num_levels);
		array < float3*>^ dev_output_rgb_ = gcnew array<float3*>(num_levels);
		array<float*>^ dev_output_density_ = gcnew array<float*>(num_levels);
		array<float*>^ dev_encoded_position_data_ = gcnew array<float*>(num_levels);
		array<float*>^ dev_encoded_direction_data_ = gcnew array<float*>(num_levels);
		array < float3*>^ dev_composite_rgb = gcnew array<float3*>(num_levels);
		array < float3*>^ dev_color_grad_ = gcnew array<float3*>(num_levels);
		array<float*>^ dev_density_grad_ = gcnew array<float*>(num_levels);
		curandState* dev_states_;
	};
}

