#pragma once
#include <vector_types.h>
extern dim3 block_1d;
extern dim3 block_2d;
extern dim3 block_3d;
template <typename T>
inline T* get_head_ptr(array<T>^ arr)
{
	pin_ptr<T> ptr = &arr[0];
	return ptr;
}
inline dim3 operator /(const dim3& a, const dim3& b)
{
	return dim3((a.x + b.x - 1) / b.x, (a.y + b.y - 1) / b.y, (a.z + b.z - 1) / b.z);
}
constexpr int num_levels = 2;
constexpr int num_samples = 128;
constexpr int num_rays = 1024;
constexpr int num_positional_frequencies = 16;
constexpr int num_directional_frequencies = 4;