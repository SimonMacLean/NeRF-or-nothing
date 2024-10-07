#include "AcceleratedMipNeRF.h"

#include <ctime>

AcceleratedMipNeRF::AcceleratedMipNeRF()
{
	extern void initialize_curand(curandState * states, const unsigned long long seed, const int size);
	float3** dev_origins_ptr = nullptr, ** dev_directions_ptr = nullptr;
	float** dev_radii_ptr = nullptr, ** dev_nears_ptr = nullptr, ** dev_fars_ptr = nullptr, ** dev_loss_mults_ptr = nullptr, ** dev_t_vals_level_0_ptr = nullptr, ** dev_t_vals_level_1_ptr = nullptr;
	curandState** dev_states_ptr = nullptr;
	cudaMalloc(dev_origins_ptr, num_rays * sizeof(float3));
	cudaMalloc(dev_directions_ptr, num_rays * sizeof(float3));
	cudaMalloc(dev_radii_ptr, num_rays * sizeof(float));
	cudaMalloc(dev_nears_ptr, num_rays * sizeof(float));
	cudaMalloc(dev_fars_ptr, num_rays * sizeof(float));
	cudaMalloc(dev_loss_mults_ptr, num_rays * sizeof(float));
	cudaMalloc(dev_states_ptr, num_rays * num_samples * sizeof(curandState));
	cudaMalloc(dev_t_vals_level_0_ptr, num_rays * (num_samples + 1) * sizeof(float));
	cudaMalloc(dev_t_vals_level_1_ptr, num_rays * num_samples * sizeof(float));
	dev_origins_  = *dev_origins_ptr;
	dev_directions_ = *dev_directions_ptr;
	dev_radii_ = *dev_radii_ptr;
	dev_nears_ = *dev_nears_ptr;
	dev_fars_ = *dev_fars_ptr;
	dev_loss_mults_ = *dev_loss_mults_ptr;
	dev_states_ = *dev_states_ptr;
	dev_t_vals_level_0_ = *dev_t_vals_level_0_ptr;
	dev_t_vals_level_1_ = *dev_t_vals_level_1_ptr;
	mlp_ = gcnew accelerated_mlp(16, 4);
	unsigned long long seed = time(nullptr);
	int size = num_rays * num_samples;
	curandState* dev_states = dev_states_;
	void* ic_args[3] = {&dev_states, &seed, &size};
	dim3 block(1024);
	dim3 grid((size + block.x - 1) / block.x);
	cudaLaunchKernel((void*)initialize_curand, grid, block, ic_args);
	cudaDeviceSynchronize();
}

array<float>^ AcceleratedMipNeRF::GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
                                              array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<array<Vector3>^, array<Vector3>^> getOutputGradient)
{
	extern void get_sample_t_vals(const curandState *, const float*, const float*, float*);
	extern void cast_rays(const float*, const float3 *, const float3 *, float3 *, float3 *, const float*, int);
	cudaMemcpy(dev_origins_, (pin_ptr<Vector3>) & origins[0], origins->Length * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_directions_, (pin_ptr<Vector3>) & directions[0], directions->Length * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_radii_, (pin_ptr<float>) & radii[0], radii->Length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nears_, (pin_ptr<float>) & nears[0], nears->Length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fars_, (pin_ptr<float>) & fars[0], fars->Length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_loss_mults_, (pin_ptr<float>) & lossMultipliers[0], lossMultipliers->Length * sizeof(float), cudaMemcpyHostToDevice);
	for (int iLevel = 0; iLevel < num_levels; iLevel++)
	{
		if(iLevel == 0)
		{
			float** tVals_ptr = nullptr;

		}
	}

}
