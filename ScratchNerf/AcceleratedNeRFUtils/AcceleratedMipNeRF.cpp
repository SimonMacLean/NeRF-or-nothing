#include "AcceleratedMipNeRF.h"

#include <ctime>
namespace AcceleratedNeRFUtils
{
	AcceleratedMipNeRF::AcceleratedMipNeRF()
	{
		extern void initialize_curand(curandState * states, const unsigned long long seed, const int size);
		float3* dev_origins_ptr, * dev_directions_ptr;
		float* dev_radii_ptr, * dev_nears_ptr, * dev_fars_ptr, * dev_loss_mults_ptr;
		curandState* dev_states_ptr;
		cudaMalloc(&dev_origins_ptr, num_rays * sizeof(float3));
		cudaMalloc(&dev_directions_ptr, num_rays * sizeof(float3));
		cudaMalloc(&dev_radii_ptr, num_rays * sizeof(float));
		cudaMalloc(&dev_nears_ptr, num_rays * sizeof(float));
		cudaMalloc(&dev_fars_ptr, num_rays * sizeof(float));
		cudaMalloc(&dev_loss_mults_ptr, num_rays * sizeof(float));
		cudaMalloc(&dev_states_ptr, num_rays * num_samples * sizeof(curandState));
		for (int i = 0; i < num_levels; i++)
		{
			cudaMalloc(get_head_ptr(dev_t_vals_) + i, num_rays * (num_samples + 1) * sizeof(float));
			cudaMalloc(get_head_ptr(dev_means_) + i, num_rays * num_samples * sizeof(float3));
			cudaMalloc(get_head_ptr(dev_covs_) + i, num_rays * num_samples * sizeof(float3));
			cudaMalloc(get_head_ptr(dev_weights_) + i, num_rays * num_samples * sizeof(float));
			cudaMalloc(get_head_ptr(dev_transmittance_) + i, num_rays * num_samples * sizeof(float));
			cudaMalloc(get_head_ptr(dev_alpha_) + i, num_rays * num_samples * sizeof(float));
			cudaMalloc(get_head_ptr(dev_output_rgb_) + i, num_rays * sizeof(float3));
			cudaMalloc(get_head_ptr(dev_output_density_) + i, num_rays * sizeof(float));
			cudaMalloc(get_head_ptr(dev_encoded_position_data_) + i, num_rays * num_samples * num_positional_frequencies * 3 * sizeof(float));
			cudaMalloc(get_head_ptr(dev_encoded_direction_data_) + i, num_rays * num_samples * (num_directional_frequencies + 1) * 3 * sizeof(float));
			cudaMalloc(get_head_ptr(dev_composite_rgb) + i, num_rays * sizeof(float3));
			cudaMalloc(get_head_ptr(dev_color_grad_) + i, num_rays * num_samples * sizeof(float3));
			cudaMalloc(get_head_ptr(dev_density_grad_) + i, num_rays * num_samples * sizeof(float));
		}
		dev_origins_ = dev_origins_ptr;
		dev_directions_ = dev_directions_ptr;
		dev_radii_ = dev_radii_ptr;
		dev_nears_ = dev_nears_ptr;
		dev_fars_ = dev_fars_ptr;
		dev_loss_mults_ = dev_loss_mults_ptr;
		dev_states_ = dev_states_ptr;
		unsigned long long seed = time(nullptr);
		int size = num_rays * num_samples;
		void* ic_args[3] = { &dev_states_ptr, &seed, &size };
		cudaLaunchKernel((void*)initialize_curand, dim3(size) / block_1d, block_1d, ic_args);
		cudaDeviceSynchronize();
	}

	array<float*>^ AcceleratedMipNeRF::GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
		array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<uint64_t, int, float, uint64_t, uint64_t>^ getOutputGradient)
	{
		extern void get_sample_t_vals(const curandState*, const float*, const float*, float*);
		extern void get_resampled_t_vals(const curandState*, const float*, const float*, float*);
		extern void cast_rays(const float*, const float3*, const float3*, float3*, float3*, const float*);
		extern void encode_input_data(const float3*, const float3*, const float3*, float*, float*);
		extern void volumetric_rendering(const float3*, const float*, const float*, const float3*, float3*, float*, float*, float*);
		extern void volumetric_rendering_gradient(const float3*, const float*, const float*, const float*, const float3*, const float*, const float3*, float3*, float*);
		int lossMultiplierSum = 0;
		for(int i = 0; i < lossMultipliers->Length; i++)
		{
			lossMultiplierSum += lossMultipliers[i];
		}
		pin_ptr<Vector3> origins_ptr = &origins[0];
		pin_ptr<Vector3> directions_ptr = &directions[0];
		pin_ptr<float> radii_ptr = &radii[0];
		pin_ptr<float> nears_ptr = &nears[0];
		pin_ptr<float> fars_ptr = &fars[0];
		pin_ptr<float> lossMultipliers_ptr = &lossMultipliers[0];
		cudaMemcpy(dev_origins_, origins_ptr, num_rays * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_directions_, directions_ptr, num_rays * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_radii_, radii_ptr, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_nears_, nears_ptr, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fars_, fars_ptr, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_loss_mults_, lossMultipliers_ptr, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		float3* dev_origins = dev_origins_;
		float3* dev_directions = dev_directions_;
		float* dev_radii = dev_radii_;
		float* dev_nears = dev_nears_;
		float* dev_fars = dev_fars_;
		float* dev_loss_mults = dev_loss_mults_;
		curandState* dev_states = dev_states_;
		int ray_num = num_rays;
		int sample_num = num_samples;
		for (int iLevel = 0; iLevel < num_levels; iLevel++)
		{
			if (iLevel == 0)
			{
				void* gstv_args[4] = { &dev_states, &dev_nears, &dev_fars, get_head_ptr(dev_t_vals_) + iLevel };
				cudaLaunchKernel((void*)get_sample_t_vals, dim3(num_rays) / block_1d, block_1d, gstv_args);
				cudaDeviceSynchronize();
			}
			else
			{
				void* grtv_args[4] = { &dev_states, get_head_ptr(dev_t_vals_) + iLevel - 1, get_head_ptr(dev_weights_) + iLevel - 1, get_head_ptr(dev_t_vals_) + iLevel };
				cudaLaunchKernel((void*)get_resampled_t_vals, dim3(num_rays) / block_1d, block_1d, grtv_args);
				cudaDeviceSynchronize();
			}
			void* cs_args[6] = { get_head_ptr(dev_t_vals_) + iLevel, &dev_origins, &dev_directions, get_head_ptr(dev_means_) + iLevel, get_head_ptr(dev_covs_) + iLevel, &dev_radii };
			cudaLaunchKernel((void*)cast_rays, dim3(num_rays, num_samples) / block_2d, block_2d, cs_args);
			cudaDeviceSynchronize();
			void* eid_args[5] = { get_head_ptr(dev_means_) + iLevel, get_head_ptr(dev_covs_) + iLevel, &dev_directions, get_head_ptr(dev_encoded_position_data_) + iLevel, get_head_ptr(dev_encoded_direction_data_) + iLevel };
			cudaLaunchKernel((void*)encode_input_data, dim3(num_rays, num_samples, num_positional_frequencies) / block_3d, block_3d, eid_args);
			cudaDeviceSynchronize();
			ValueTuple<uint64_t, uint64_t>^ output = mlp->get_output(dev_encoded_position_data_[iLevel], dev_encoded_direction_data_[iLevel], iLevel);
			dev_output_rgb_[iLevel] = (float3*)output->Item1;
			dev_output_density_[iLevel] = (float*)output->Item2;
			void* vr_args[8] = { get_head_ptr(dev_output_rgb_) + iLevel, get_head_ptr(dev_output_density_) + iLevel, get_head_ptr(dev_t_vals_) + iLevel, dev_directions, get_head_ptr(dev_composite_rgb) + iLevel, get_head_ptr(dev_alpha_) + iLevel, get_head_ptr(dev_transmittance_) + iLevel, get_head_ptr(dev_weights_) + iLevel };
			cudaLaunchKernel((void*)volumetric_rendering, dim3(num_rays) / block_1d, block_1d, vr_args);
			cudaDeviceSynchronize();
		}
		float3* dev_color_gradients[num_levels];
		for (int i = 0; i < num_levels; i++)
		{
			dev_color_gradients[i] = reinterpret_cast<float3*>(getOutputGradient(reinterpret_cast<uint64_t>(dev_composite_rgb[i]), i, lossMultiplierSum, reinterpret_cast<uint64_t>(dev_loss_mults)));
			void* vrg_args[9] = { dev_color_gradients + i, get_head_ptr(dev_alpha_) + i, get_head_ptr(dev_transmittance_) + i, get_head_ptr(dev_weights_) + i, get_head_ptr(dev_output_rgb_) + i, get_head_ptr(dev_t_vals_) + i, dev_directions + i, get_head_ptr(dev_color_grad_) + i, get_head_ptr(dev_density_grad_) + i };
			cudaLaunchKernel((void*)volumetric_rendering_gradient, dim3(num_rays) / block_1d, block_1d, vrg_args);
		}
		cudaDeviceSynchronize();
		mlp->reset_gradients(0);
		for (int i = 0; i < num_levels; i++)
		{
			mlp->get_gradient((float*)dev_color_grad_[i], dev_density_grad_[i], i);
		}
		cudaDeviceSynchronize();
		return mlp->allGradients;
	}

	array<int>^ AcceleratedMipNeRF::GetLayerSizes()
	{
		return mlp->get_layer_sizes();
	}

	AcceleratedMipNeRF::~AcceleratedMipNeRF()
	{
		cudaFree(dev_origins_);
		cudaFree(dev_directions_);
		cudaFree(dev_radii_);
		cudaFree(dev_nears_);
		cudaFree(dev_fars_);
		cudaFree(dev_loss_mults_);
		for (int i = 0; i < num_levels; i++)
		{
			cudaFree(get_head_ptr(dev_t_vals_) + i);
			cudaFree(get_head_ptr(dev_means_) + i);
			cudaFree(get_head_ptr(dev_covs_) + i);
			cudaFree(get_head_ptr(dev_weights_) + i);
			cudaFree(get_head_ptr(dev_transmittance_) + i);
			cudaFree(get_head_ptr(dev_alpha_) + i);
			cudaFree(get_head_ptr(dev_output_rgb_) + i);
			cudaFree(get_head_ptr(dev_output_density_) + i);
			cudaFree(get_head_ptr(dev_encoded_position_data_) + i);
			cudaFree(get_head_ptr(dev_encoded_direction_data_) + i);
			cudaFree(get_head_ptr(dev_composite_rgb) + i);
			cudaFree(get_head_ptr(dev_color_grad_) + i);
			cudaFree(get_head_ptr(dev_density_grad_) + i);
		}
	}
}
