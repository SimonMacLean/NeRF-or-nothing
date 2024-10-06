#pragma once
#include <vector_types.h>

#include "AcceleratedMLP.h"
using namespace System;
using namespace System::Numerics;
public ref class AcceleratedMipNeRF
{
public:
	array<Single>^ GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
		array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<array<Vector3>^, array<Vector3>^> getOutputGradient);
private:
	accelerated_mlp mlp;
	float3* dev_origins;
	float3* dev_directions;
	float* dev_radii;
	float* dev_nears;
	float* dev_fars;
	float* dev_loss_mults;
};

