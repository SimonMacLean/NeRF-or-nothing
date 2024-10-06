#include "AcceleratedMipNeRF.h"

array<float>^ AcceleratedMipNeRF::GetGradient(array<Vector3>^ origins, array<Vector3>^ directions, array<float>^ radii,
	array<float>^ nears, array<float>^ fars, array<float>^ lossMultipliers, Func<array<Vector3>^, array<Vector3>^> getOutputGradient)
{
}
