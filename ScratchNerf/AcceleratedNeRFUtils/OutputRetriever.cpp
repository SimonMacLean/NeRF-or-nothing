#include "OutputRetriever.h"

#include <cstdio>
#include <cuda_runtime_api.h>
namespace AcceleratedNeRFUtils {
	array<Vector3>^ OutputRetriever::RetrieveOutput(uint64_t dev_output, int size)
	{
		float3* output_dev_ptr = (float3*)dev_output;
		array<Vector3>^ output = gcnew array<Vector3>(size);
		pin_ptr<Vector3> output_host_ptr = &output[0];
		cudaMemcpy(output_host_ptr, output_dev_ptr, size * sizeof(float3), cudaMemcpyDeviceToHost);
		
		return output;
	}
}
