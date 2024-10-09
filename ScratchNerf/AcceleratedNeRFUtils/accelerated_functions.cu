#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <curand_kernel.h>
extern __device__ float atomicAdd(float* address, float val);
extern __device__ float rsqrtf(float x);
__device__ float relu(const float x) { return (x > 0) * x; }
__device__ float relu_derivative(const float x) { return x > 0; }
__device__ float sigmoid(const float x) { return 1 / (1 + expf(-x)); }
__device__ float sigmoid_derivative(const float x) {
	const float sigmoid = 1 / (1 + expf(-x));
	return sigmoid * (1 - sigmoid);
}
__device__ float soft_plus(const float x) { return logf(1 + expf(x)); }
constexpr int num_samples = 128;
constexpr int num_rays = 1024;
__global__ void initialize_curand(curandState* states, const unsigned long long seed, const int size)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	curand_init(seed, idx, 0, states + idx);
}
__global__ void initialize_glorot(const curandState* states, float* weights, float* biases, const int num_neurons, const int input_size)
{
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int input_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (neuron_idx >= num_neurons || input_idx >= input_size)
		return;
	curandState state = states[neuron_idx * input_size + input_idx];
	const float glorot = sqrtf(6.f / (num_neurons + input_size));
	weights[neuron_idx * input_size + input_idx] = (curand_uniform(&state) * 2 - 1) * glorot;
	if (input_idx == 0)
		biases[neuron_idx] = curand_uniform(&state) * glorot;
}
__global__ void get_neuron_output(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	float weighted_sum = 0.f;
	for (int i = 0; i < input_size; i++)
		weighted_sum += inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i] * weights[neuron_idx * input_size + i];
	weighted_sum += biases[neuron_idx];
	weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = weighted_sum;
	outputs[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = relu(weighted_sum);
}
__global__ void get_neuron_output_sigmoid(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	float weighted_sum = 0.f;
	for (int i = 0; i < input_size; i++)
		weighted_sum += inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i] * weights[neuron_idx * input_size + i];
	weighted_sum += biases[neuron_idx];
	weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = weighted_sum;
	outputs[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = sigmoid(weighted_sum);
}
__global__ void get_neuron_output_soft_plus(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	float weighted_sum = 0.f;
	for (int i = 0; i < input_size; i++)
		weighted_sum += inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i] * weights[neuron_idx * input_size + i];
	weighted_sum += biases[neuron_idx];
	weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = weighted_sum;
	outputs[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = soft_plus(weighted_sum);
}
__global__ void get_neuron_output_conjoined_inputs(const float* inputs_a, const float* inputs_b, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_a_size, const int input_b_size)
{
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	float weighted_sum = 0.f;
	for (int i = 0; i < input_a_size; i++)
		weighted_sum += inputs_a[ray_idx * input_a_size * num_samples + sample_idx * input_a_size + i] * weights[neuron_idx * (input_a_size + input_b_size) + i];
	for (int i = 0; i < input_b_size; i++)
		weighted_sum += inputs_b[ray_idx * input_b_size * num_samples + sample_idx * input_b_size + i] * weights[neuron_idx * (input_a_size + input_b_size) + input_a_size + i];
	weighted_sum += biases[neuron_idx];
	weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = weighted_sum;
	outputs[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx] = relu(weighted_sum);
}
__global__ void backpropagate_neuron(const float* inputs, const float* weights, const float* weighted_sums, const float* output_grads, float* input_grads, float* weight_grads, float* bias_grads, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	const float output_grad = output_grads[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float weighted_sum = weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float activation_derivative = relu_derivative(weighted_sum);
	const int start = (ray_idx * num_samples + sample_idx) % input_size;
	for (int offset = 0; offset < input_size; offset++) {
		const int i = (start + offset) % input_size;
		const float input_val = inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i];
		const float weight_val = weights[neuron_idx * input_size + i];
		atomicAdd(&weight_grads[neuron_idx * input_size + i],
			input_val * output_grad * activation_derivative);
		atomicAdd(&input_grads[ray_idx * input_size * num_samples + sample_idx * input_size + i],
			weight_val * output_grad * activation_derivative);
	}
	atomicAdd(&bias_grads[neuron_idx], output_grad * activation_derivative);
}
__global__ void backpropagate_neuron_sigmoid(const float* inputs, const float* weights, const float* weighted_sums, const float* output_grads, float* input_grads, float* weight_grads, float* bias_grads, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	const float output_grad = output_grads[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float weighted_sum = weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float activation_derivative = sigmoid_derivative(weighted_sum);
	const int start = (ray_idx * num_samples + sample_idx) % input_size;
	for (int offset = 0; offset < input_size; offset++) {
		const int i = (start + offset) % input_size;
		const float input_val = inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i];
		const float weight_val = weights[neuron_idx * input_size + i];
		atomicAdd(&weight_grads[neuron_idx * input_size + i],
			input_val * output_grad * activation_derivative);
		atomicAdd(&input_grads[ray_idx * input_size * num_samples + sample_idx * input_size + i],
			weight_val * output_grad * activation_derivative);
	}
	atomicAdd(&bias_grads[neuron_idx], output_grad * activation_derivative);
}
__global__ void backpropagate_neuron_soft_plus(const float* inputs, const float* weights, const float* weighted_sums, const float* output_grads, float* input_grads, float* weight_grads, float* bias_grads, const int num_neurons, const int input_size) {
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	const float output_grad = output_grads[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float weighted_sum = weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float activation_derivative = sigmoid(weighted_sum);
	const int start = (ray_idx * num_samples + sample_idx) % input_size;
	for (int offset = 0; offset < input_size; offset++) {
		const int i = (start + offset) % input_size;
		const float input_val = inputs[ray_idx * input_size * num_samples + sample_idx * input_size + i];
		const float weight_val = weights[neuron_idx * input_size + i];
		atomicAdd(&weight_grads[neuron_idx * input_size + i],
			input_val * output_grad * activation_derivative);
		atomicAdd(&input_grads[ray_idx * input_size * num_samples + sample_idx * input_size + i],
			weight_val * output_grad * activation_derivative);
	}
	atomicAdd(&bias_grads[neuron_idx], output_grad * activation_derivative);
}
__global__ void backpropagate_neuron_partial_conjoined(const float* inputs_a, const float* inputs_b, const float* weights, const float* weighted_sums, const float* output_grads, float* input_a_grads, float* weight_grads, float* bias_grads, const int num_neurons, const int input_a_size, const int input_b_size)
{
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ray_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int sample_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (neuron_idx >= num_neurons || ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	const float output_grad = output_grads[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float weighted_sum = weighted_sums[ray_idx * num_neurons * num_samples + sample_idx * num_neurons + neuron_idx];
	const float activation_derivative = relu_derivative(weighted_sum);
	const int start = ray_idx * num_samples + sample_idx;
	for (int offset = 0; offset < input_a_size; offset++) {
		const int i = (start + offset) % input_a_size;
		const float input_val = inputs_a[ray_idx * input_a_size * num_samples + sample_idx * input_a_size + i];
		const float weight_val = weights[neuron_idx * (input_a_size + input_b_size) + i];
		atomicAdd(&weight_grads[neuron_idx * (input_a_size + input_b_size) + i],
			input_val * output_grad * activation_derivative);
		atomicAdd(&input_a_grads[ray_idx * input_a_size * num_samples + sample_idx * input_a_size + i],
			weight_val * output_grad * activation_derivative);
	}
	for (int offset = 0; offset < input_b_size; offset++) {
		const int i = (start + offset) % input_b_size;
		const float input_val = inputs_b[ray_idx * input_b_size * num_samples + sample_idx * input_b_size + i];
		const float weight_val = weights[neuron_idx * (input_a_size + input_b_size) + i + input_a_size];
		atomicAdd(&weight_grads[neuron_idx * (input_a_size + input_b_size) + i + input_a_size],
			input_val * output_grad * activation_derivative);
	}
	atomicAdd(&bias_grads[neuron_idx], output_grad * activation_derivative);
}
constexpr int num_frequencies = 16;
constexpr int deg_frequencies = 4;
__device__ float expected_sin_mean(const float x, const float xVar) { return expf(-.5f * xVar) * sinf(x); }
__device__ float expected_cos_mean(const float x, const float xVar) { return expf(-.5f * xVar) * cosf(x); }
__global__ void encode_input_data(const float3* location_means, const float3* location_variance_diagonals, const float3* direction_data, float* encoded_location_data, float* encoded_direction_data)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int frequency_idx = blockIdx.z * blockDim.z + threadIdx.z;
	if (ray_idx >= num_rays || sample_idx >= num_samples || frequency_idx >= num_frequencies)
		return;
	const float3 xMean = location_means[ray_idx * num_samples + sample_idx];
	const float3 xVar = location_variance_diagonals[ray_idx * num_samples + sample_idx];
	const float scale = 1 << frequency_idx;
	const float3 x = make_float3(xMean.x * scale, xMean.y * scale, xMean.z * scale);
	const float3 yVar = make_float3(xVar.x * scale * scale, xVar.y * scale * scale, xVar.z * scale * scale);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6] = expected_sin_mean(x.x, yVar.x);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6 + 1] = expected_sin_mean(x.y, yVar.y);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6 + 2] = expected_sin_mean(x.z, yVar.z);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6 + 3] = expected_cos_mean(x.x, yVar.x);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6 + 4] = expected_cos_mean(x.y, yVar.y);
	encoded_location_data[((ray_idx * num_samples + sample_idx) * num_frequencies + frequency_idx) * 6 + 5] = expected_cos_mean(x.z, yVar.z);
	if (frequency_idx >= deg_frequencies)
		return;
	constexpr int resultNumCols = deg_frequencies * 2 + 1;
	const float3 xDir = direction_data[ray_idx * num_samples + sample_idx];
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 1)) * 3] = sinf(xDir.x);
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 1)) * 3 + 1] = sinf(xDir.y);
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 1)) * 3 + 2] = sinf(xDir.z);
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 2)) * 3] = cosf(xDir.x);
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 2)) * 3 + 1] = cosf(xDir.y);
	encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + (frequency_idx + 2)) * 3 + 2] = cosf(xDir.z);
	if (frequency_idx == 0)
	{
		encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + frequency_idx) * 3] = xDir.x;
		encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + frequency_idx) * 3 + 1] = xDir.y;
		encoded_direction_data[((ray_idx * num_samples + sample_idx) * resultNumCols + frequency_idx) * 3 + 2] = xDir.z;
	}
}
__global__ void get_sample_t_vals(const curandState* states, const float* near, const float* far, float* tVals)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (ray_idx >= num_rays || sample_idx > num_samples)
		return;
	curandState state = states[ray_idx * (num_samples + 1) + sample_idx];
	const int tValIndex = ray_idx * (num_samples + 1) + sample_idx;
	float tVal = sample_idx / (float)num_samples;
	float near_val = near[ray_idx];
	float far_val = far[ray_idx];
	tVal = near_val * (1 - tVal) + far_val * tVal;
	float prevTVal = sample_idx > 0 ? ((sample_idx - 1) / (float)num_samples) : tVal;
	prevTVal = near_val * (1 - prevTVal) + far_val * prevTVal;
	float nextTVal = sample_idx <= num_samples ? ((sample_idx + 1) / (float)num_samples) : tVal;
	nextTVal = near_val * (1 - nextTVal) + far_val * nextTVal;
	const float prevMidpoint = 0.5f * (tVal + prevTVal);
	const float nextMidpoint = 0.5f * (tVal + nextTVal);
	const float rand = curand_uniform(&state);
	tVals[tValIndex] = prevMidpoint + rand * (nextMidpoint - prevMidpoint);
}
constexpr float resample_padding = 0.01f;
constexpr float eps = 1e-5f;
constexpr float machine_eps = 1e-7f;
__global__ void get_resampled_t_vals(const curandState* states, const float* t_vals, const float* weights, float* new_t_vals)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;
	curandState state = states[ray_idx];
	float weights_blur[num_samples];
	float weightSum = 0;
	for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
	{
		const int sampleIndex = ray_idx * num_samples + sample_idx;
		weights_blur[sample_idx] = 0.5f * (sample_idx == 0 ? weights[sampleIndex] : fmaxf(weights[sampleIndex - 1], weights[sampleIndex]) + sample_idx == num_samples - 1 ? weights[sampleIndex] : fmaxf(weights[sampleIndex], weights[sampleIndex + 1])) + resample_padding;
		weightSum += weights_blur[sample_idx];
	}
	const float padding = fmaxf(0.f, eps - weightSum);
	for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
		weights_blur[sample_idx] += padding / num_samples;
	weightSum += padding;
	float cdf[num_samples + 1];
	for (int sample_idx = 0; sample_idx < num_samples; sample_idx++)
		cdf[sample_idx + 1] = fminf(1.f, cdf[sample_idx] + weights_blur[sample_idx] / weightSum);
	cdf[num_samples] = 1.f;
	float u[num_samples + 1];
	for (int sample_idx = 0; sample_idx <= num_samples; sample_idx++)
		u[sample_idx] = fminf((sample_idx + curand_uniform(&state)) / (num_samples + 1), 1 - machine_eps);
	int u_index = 0;
	for (int cdf_index = 1; cdf_index <= num_samples; cdf_index++)
	{
		if (u_index > num_samples)
			return;
		if (u[u_index] < cdf[cdf_index])
			continue;
		const int new_t_val_index = ray_idx * num_samples + u_index;
		const int t_val_index = ray_idx * (num_samples + 1) + cdf_index;
		const float binsG0 = t_vals[t_val_index - 1];
		const float binsG1 = t_vals[t_val_index];
		const float cdfG0 = cdf[cdf_index - 1];
		const float cdfG1 = cdf[cdf_index];
		float t = (u[u_index] - cdfG0) / (cdfG1 - cdfG0);
		if (t == NAN) t = 0;
		t = fmaxf(0.f, fminf(1.f, t));
		new_t_vals[new_t_val_index] = binsG0 + t * (binsG1 - binsG0);
		cdf_index--;
		u_index++;
	}
}
__global__ void cast_rays(const float* t_vals, const float3* origins, const float3* directions, float3* sample_means, float3* sample_covs, const float* radii)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (ray_idx >= num_rays || sample_idx >= num_samples)
		return;
	const float t0 = t_vals[ray_idx * (num_samples + 1) + sample_idx];
	const float t1 = t_vals[ray_idx * (num_samples + 1) + sample_idx + 1];
	const float radius = radii[ray_idx];
	const float meanDistance = (t0 + t1) / 2;
	const float halfWidth = (t1 - t0) / 2;
	const float meanDistanceSquared = meanDistance * meanDistance;
	const float halfWidthSquared = halfWidth * halfWidth;
	const float denominator = 3 * meanDistanceSquared + halfWidthSquared;
	const float distanceMean = meanDistance + 2 * meanDistance * halfWidthSquared / denominator;
	const float distanceVariance = halfWidthSquared / 3 - 4.f / 15.f * (halfWidthSquared * halfWidthSquared * (12 * meanDistanceSquared - halfWidthSquared)) / (denominator * denominator);
	const float radiusVariance = radius * radius * (meanDistanceSquared / 4 + 5.f / 12.f * halfWidthSquared - 4.f / 15.f * (halfWidthSquared * halfWidthSquared) / denominator);
	const float3 direction = directions[ray_idx];
	sample_means[ray_idx * num_samples + sample_idx] = make_float3(direction.x * distanceMean + origins[ray_idx].x, direction.y * distanceMean + origins[ray_idx].y, direction.z * distanceMean + origins[ray_idx].z);
	const float directionMagnitudeSquared = fmaxf(1e-10f, direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
	const float3 dOuterDiagonal = make_float3(direction.x * direction.x, direction.y * direction.y, direction.z * direction.z);
	const float3 nullOuterDiagonal = make_float3(1 - dOuterDiagonal.x / directionMagnitudeSquared, 1 - dOuterDiagonal.y / directionMagnitudeSquared, 1 - dOuterDiagonal.z / directionMagnitudeSquared);
	const float3 tCovarianceDiagonal = make_float3(distanceVariance * dOuterDiagonal.x, distanceVariance * dOuterDiagonal.y, distanceVariance * dOuterDiagonal.z);
	const float3 xyCovarianceDiagonal = make_float3(radiusVariance * nullOuterDiagonal.x, radiusVariance * nullOuterDiagonal.y, radiusVariance * nullOuterDiagonal.z);
	sample_covs[ray_idx * num_samples + sample_idx] = make_float3(tCovarianceDiagonal.x + xyCovarianceDiagonal.x, tCovarianceDiagonal.y + xyCovarianceDiagonal.y, tCovarianceDiagonal.z + xyCovarianceDiagonal.z);
}
__global__ void volumetric_rendering(const float3* samples_rgb, const float* samples_density, const float* t_vals, const float3* directions, float3* rgb, float* alpha, float* transmittance, float* weights)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;
	float3 compRgb = make_float3(0.f, 0.f, 0.f);
	float accVal = 0.f;
	float weightedDistanceSum = 0.f;
	for (int i = 0; i < num_samples; i++)
	{
		const int index = ray_idx * num_samples + i;
		const int tValIndex = ray_idx * (num_samples + 1) + i;
		alpha[index] = 1 - expf(-samples_density[index] * (t_vals[tValIndex + 1] - t_vals[tValIndex]) * sqrtf(directions[ray_idx].x * directions[ray_idx].x + directions[ray_idx].y * directions[ray_idx].y + directions[ray_idx].z * directions[ray_idx].z));
		transmittance[index] = i == 0 ? 1.f : transmittance[index - 1] * (1 - alpha[index - 1]);
		weights[index] = alpha[index] * transmittance[index];
		compRgb = make_float3(compRgb.x + weights[index] * samples_rgb[index].x, compRgb.y + weights[index] * samples_rgb[index].y, compRgb.z + weights[index] * samples_rgb[index].z);
		accVal += weights[index];
		weightedDistanceSum += weights[index] * (t_vals[index] + t_vals[index + 1]) / 2;
	}
	//const float distanceVal = fminf(fmaxf(accVal > 0 ? weightedDistanceSum / accVal : FLT_MAX, t_vals[0]), t_vals[num_samples]);
	compRgb.x += 1.f - accVal;
	compRgb.y += 1.f - accVal;
	compRgb.z += 1.f - accVal;
	rgb[ray_idx] = compRgb;
	//distance[ray_idx] = distanceVal;
	//acc[ray_idx] = accVal;
}
constexpr float coarse_loss_mult = 0.1f;
constexpr int num_levels = 2;
__global__ void get_output_gradient(const float3* composite_rgb, const float3* pixels, const float* loss_mults, float3* comp_rgb_grad, const float loss_mult_sum, const int level)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;
	float3 compRgb = composite_rgb[ray_idx];
	float3 pixel = pixels[ray_idx];
	float loss_mult = loss_mults[ray_idx];
	float3 compositeRgbGradient = level == 0 ? make_float3(0, 0, 0) : comp_rgb_grad[ray_idx];
	compositeRgbGradient.x += 2 * loss_mult / loss_mult_sum * (compRgb.x - pixel.x) * (level < num_levels - 1 ? coarse_loss_mult : 1);
	compositeRgbGradient.y += 2 * loss_mult / loss_mult_sum * (compRgb.y - pixel.y) * (level < num_levels - 1 ? coarse_loss_mult : 1);
	compositeRgbGradient.z += 2 * loss_mult / loss_mult_sum * (compRgb.z - pixel.z) * (level < num_levels - 1 ? coarse_loss_mult : 1);
	comp_rgb_grad[ray_idx] = compositeRgbGradient;

}
__global__ void volumetric_rendering_gradient(const float3* comp_rgb_grad, const float* alpha, const float* transmittance, const float* weights, const float3* color_samples, const float* t_vals, const float3* directions, float3* color_grad, float* density_grad)
{
	const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (ray_idx >= num_rays)
		return;
	float3 compositeRgbGradient = comp_rgb_grad[ray_idx];
	float dLdWeights[num_samples - 1];
	// dL/dAcc = - (dL/dCompRgb dot 1)
	float dLdAcc = -(compositeRgbGradient.x + compositeRgbGradient.y + compositeRgbGradient.z);
	// Initialize gradients with respect to transmittance and alpha
	float dLdTransmittance[num_samples];
	float dLdAlpha[num_samples - 1];
	// Compute gradients with respect to weights and colors
	dLdTransmittance[num_samples - 1] = 0;
	// Compute gradient with respect to density
	float3 direction = directions[ray_idx];
	float directionLength = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
	for (int i = num_samples - 2; i >= 0; i--)
	{
		float3 sampleColor = color_samples[ray_idx * num_samples + i];
		float returnedTransmittance = transmittance[ray_idx * num_samples + i];
		float returnedAlpha = alpha[ray_idx * num_samples + i];
		// dL/dWeights[i] = compositeRgbGradient dot color[i]
		dLdWeights[i] = compositeRgbGradient.x * sampleColor.x + compositeRgbGradient.y * sampleColor.y + compositeRgbGradient.z * sampleColor.z + dLdAcc;
		float returnedWeight = weights[ray_idx * num_samples + i];
		// dL/dColor[i] = compositeRgbGradient * weights[i]
		color_grad[ray_idx * num_samples + i] = make_float3(compositeRgbGradient.x * returnedWeight, compositeRgbGradient.y * returnedWeight, compositeRgbGradient.z * returnedWeight);
		// weights[i] = alpha[i] * transmittance[i]
		dLdAlpha[i] = dLdWeights[i] * returnedTransmittance - dLdTransmittance[i + 1] * returnedTransmittance;
		dLdTransmittance[i] = dLdWeights[i] * returnedAlpha + dLdTransmittance[i + 1] * (1 - returnedAlpha);
		float t0 = t_vals[ray_idx * (num_samples + 1) + i];
		float t1 = t_vals[ray_idx * (num_samples + 1) + i + 1];
		float deltaT = t1 - t0;
		// Compute exp(-s)
		float expNegS = 1 - alpha[ray_idx * num_samples + i]; // Since alpha[i] = 1 - exp(-s)
		// Compute dAlpha/dDensity
		float d_alpha_d_density = expNegS * deltaT * directionLength;
		// Compute dL/dDensity[i] = dL/dAlpha[i] * dAlpha/dDensity
		density_grad[ray_idx * num_samples + i] = dLdAlpha[i] * d_alpha_d_density;
	}
}
__global__ void adam_optimizer_step(float* variables, const float* gradients, float* m, float* v, const float learning_rate, const float beta1, const float beta2, const float inv_1_minus_beta1_pow, const float inv_1_minus_beta2_pow, const int size)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	float gradient = gradients[idx];
	float m_idx = beta1 * m[idx] + (1 - beta1) * gradient;
	float v_idx = beta2 * v[idx] + (1 - beta2) * gradient * gradient;
	m[idx] = m_idx;
	v[idx] = v_idx;
	float mHat = m_idx * inv_1_minus_beta1_pow;
	float vHat = v_idx * inv_1_minus_beta2_pow;
	variables[idx] -= learning_rate * mHat * rsqrtf(vHat + 1e-8f);
}
