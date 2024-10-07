#include "AcceleratedMLP.h"

#include <ctime>
#include <cuda_runtime.h>
dim3 divup(dim3 num, dim3 denom)
{
    return dim3((num.x + denom.x - 1) / denom.x, (num.y + denom.y - 1) / denom.y, (num.y + denom.y - 1) / denom.y);
}
void initialize_curand(curandState* states, int size)
{
    extern void initialize_curand(curandState * states, const unsigned long long seed, const int size);
    unsigned long long timeSeed = time(nullptr);
    void* ic_args[3] = {&states, &timeSeed, &size};
    dim3 block_size(1024);
    dim3 grid_size = divup(dim3(size), block_size);
    cudaLaunchKernel((void*)initialize_curand, grid_size, block_size, ic_args);
}
void initialize_weights(curandState* states, float* weights, float* biases, int num_rows, int num_cols)
{
    extern void initialize_glorot(curandState* states, float* weights, float* biases, const int num_neurons, const int input_size);
    void* ig_args[5] = {&states, &weights, &biases, &num_rows, &num_cols};
    dim3 block_size(32, 32);
    dim3 grid_size = divup(dim3(num_cols, num_rows), block_size);
    cudaLaunchKernel((void*)initialize_glorot, grid_size, block_size, ig_args);
}
constexpr int sample_num = 128;
constexpr int ray_num = 4096;
void apply_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size);
    void* gno_args[7] = {&inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size};
    dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
    cudaLaunchKernel((void*)get_neuron_output, grid_size, block_size, gno_args);
}
void apply_sigmoid_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output_sigmoid(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size);
    void* gnos_args[7] = { &inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size };
    dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
    cudaLaunchKernel((void*)get_neuron_output_sigmoid, grid_size, block_size, gnos_args);
}
void apply_soft_plus_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output_soft_plus(const float* inputs, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_size);
    void* gnosp_args[7] = { &inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size };
    dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
    cudaLaunchKernel((void*)get_neuron_output_soft_plus, grid_size, block_size, gnosp_args);
}
void apply_layer_conjoined_inputs(float* inputs_a, float* inputs_b, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size_a, int input_size_b)
{
    extern void get_neuron_output_conjoined_inputs(const float* inputs_a, const float* inputs_b, const float* weights, const float* biases, float* outputs, float* weighted_sums, const int num_neurons, const int input_a_size, const int input_b_size);
	void* gnoci_args[9] = {&inputs_a, &inputs_b, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size_a, &input_size_b};
    dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
    cudaLaunchKernel((void*)get_neuron_output_conjoined_inputs, grid_size, block_size, gnoci_args);
}
void backpropagate_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
                         float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
                         int input_size)
{
    extern void backpropagate_neuron(const float*, const float*,
        const float*, const float*,
        float*, float*, float*,
        const int, const int);
	void* bn_args[9] = {&inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size};
	dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
	cudaLaunchKernel((void*)backpropagate_neuron, grid_size, block_size, bn_args);
}
void backpropagate_sigmoid_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
    float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
    int input_size)
{
	extern void backpropagate_neuron_sigmoid(const float*, const float*,
		const float*, const float*,
		float*, float*, float*,
		const int, const int);
	void* bns_args[9] = {&inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size};
	dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
	cudaLaunchKernel((void*)backpropagate_neuron_sigmoid, grid_size, block_size, bns_args);
}
void backpropagate_soft_plus_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
	float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
	int input_size)
{
	extern void backpropagate_neuron_soft_plus(const float*, const float*,
		const float*, const float*,
		float*, float*, float*,
		const int, const int);
	void* bnsp_args[9] = {&inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size};
	dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
	cudaLaunchKernel((void*)backpropagate_neuron_soft_plus, grid_size, block_size, bnsp_args);
}
void backpropagate_layer_partial_conjoined(float* inputs_a, float* inputs_b, float* weights, float* weighted_sums, float* output_gradient,
    float* input_a_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
    int input_a_size, int input_b_size)
{
	extern void backpropagate_neuron_partial_conjoined(const float*, const float*, const float*,
        const float*, const float*,
        float*, float*, float*,
        const int, const int, const int);
    void* bnpc_args[11] = { &inputs_a, &inputs_b, &weights, &weighted_sums, &output_gradient, &input_a_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_a_size, &input_b_size };
    dim3 block_size(16, 8, 8);
    dim3 grid_size = divup(dim3(num_neurons, ray_num, sample_num), block_size);
    cudaLaunchKernel((void*)backpropagate_neuron_partial_conjoined, grid_size, block_size, bnpc_args);
}
accelerated_mlp::accelerated_mlp(int deg_point, int deg_view)
{
    int max_size = 0;
    location_encodings = 2 * deg_point;
    direction_encodings = 2 * deg_view + 1;
    cudaMalloc(weights_, net_width * location_dimension * location_encodings * sizeof(float));
    cudaMalloc(biases_, net_width * sizeof(float));
    cudaMalloc(outputs_, net_width * sizeof(float));
    cudaMalloc(weighted_sums_, net_width * sizeof(float));
    cudaMalloc(input_grads_, location_dimension * location_encodings * sizeof(float));
    cudaMalloc(weight_grads_, net_width * location_dimension * location_encodings * sizeof(float));
    cudaMalloc(bias_grads_, net_width * sizeof(float));
    max_size = std::max(max_size, net_width * location_dimension * location_encodings);
    for (int i = 1; i < net_depth; i++)
    {
	    cudaMalloc(weights_ + i, net_width * (i % skip_layer == 0 ? net_width + location_dimension * location_encodings : net_width) * sizeof(float));
        cudaMalloc(biases_ + i, net_width * sizeof(float));
        cudaMalloc(outputs_ + i, net_width * sizeof(float));
        cudaMalloc(weighted_sums_ + i, net_width * sizeof(float));
        cudaMalloc(input_grads_ + i, net_width * sizeof(float));
        cudaMalloc(weight_grads_ + i, net_width * (i % skip_layer == 0 ? net_width + location_dimension * location_encodings : net_width) * sizeof(float));
        cudaMalloc(bias_grads_ + i, net_width * sizeof(float));
        max_size = std::max(max_size, net_width * (i % skip_layer == 0 ? net_width + location_dimension * location_encodings : net_width));
    }
    cudaMalloc(weights_ + net_depth, net_width * num_density_channels * sizeof(float));
    cudaMalloc(biases_ + net_depth, net_width * sizeof(float));
    cudaMalloc(outputs_ + net_depth, num_density_channels * sizeof(float));
	cudaMalloc(weighted_sums_ + net_depth, num_density_channels * sizeof(float));
    cudaMalloc(input_grads_ + net_depth, net_width * sizeof(float));
    cudaMalloc(weight_grads_ + net_depth, net_width * num_density_channels * sizeof(float));
    cudaMalloc(bias_grads_ + net_depth, net_width * sizeof(float));
    max_size = std::max(max_size, net_width * num_density_channels);
    cudaMalloc(weights_ + net_depth + 1, net_width_condition * (net_width + direction_dimension * direction_encodings) * sizeof(float));
    cudaMalloc(biases_ + net_depth + 1, net_width_condition * sizeof(float));
    cudaMalloc(outputs_ + net_depth + 1, net_width_condition * sizeof(float));
	cudaMalloc(weighted_sums_ + net_depth + 1, net_width_condition * sizeof(float));
    cudaMalloc(weight_grads_ + net_depth + 1, net_width_condition * (net_width + direction_dimension * direction_encodings) * sizeof(float));
    cudaMalloc(bias_grads_ + net_depth + 1, net_width_condition * sizeof(float));
    max_size = std::max(max_size, net_width_condition * (net_width + direction_dimension * direction_encodings));
    for (int i = 1; i < net_depth_condition; i++)
    {
	    cudaMalloc(weights_ + net_depth + 1 + i, net_width_condition * net_width_condition * sizeof(float));
        cudaMalloc(biases_ + net_depth + 1 + i, net_width_condition * sizeof(float));
        cudaMalloc(outputs_ + net_depth + 1 + i, net_width_condition * sizeof(float));
	    cudaMalloc(weighted_sums_ + net_depth + 1 + i, net_width_condition * sizeof(float));
        cudaMalloc(input_grads_ + net_depth + 1 + i, net_width_condition * sizeof(float));
        cudaMalloc(weight_grads_ + net_depth + 1 + i, net_width_condition * net_width_condition * sizeof(float));
        cudaMalloc(bias_grads_ + net_depth + 1 + i, net_width_condition * sizeof(float));
        max_size = std::max(max_size, net_width_condition * net_width_condition);
    }
    cudaMalloc(weights_ + net_depth + 1 + net_depth_condition, num_rgb_channels * net_width_condition * sizeof(float));
    cudaMalloc(biases_ + net_depth + 1 + net_depth_condition, num_rgb_channels * sizeof(float));
    cudaMalloc(outputs_ + net_depth + 1 + net_depth_condition, num_rgb_channels * sizeof(float));
	cudaMalloc(weighted_sums_ + net_depth + 1 + net_depth_condition, num_rgb_channels * sizeof(float));
    cudaMalloc(input_grads_ + net_depth + 1 + net_depth_condition, net_width_condition * sizeof(float));
    cudaMalloc(weight_grads_ + net_depth + 1 + net_depth_condition, num_rgb_channels * net_width_condition * sizeof(float));
    cudaMalloc(bias_grads_ + net_depth + 1 + net_depth_condition, net_width_condition * sizeof(float));
    max_size = std::max(max_size, num_rgb_channels * net_width_condition);
    curandState** states = nullptr;
    cudaMalloc(states, max_size * sizeof(curandState));
    states_ = *states;
    initialize_curand(states_, max_size);
    cudaDeviceSynchronize();
    initialize_weights(states_, weights_[0], biases_[0], location_encodings, location_dimension);
    for (int i = 1; i < net_depth; i++)
	    initialize_weights(states_, weights_[i], biases_[i], net_width, i % skip_layer == 0 ? net_width + location_dimension * location_encodings : net_width);
    initialize_weights(states_, weights_[net_depth], biases_[net_depth], net_width, num_density_channels);
    initialize_weights(states_, weights_[net_depth + 1], biases_[net_depth + 1], net_width_condition, net_width + direction_dimension * direction_encodings);
    for (int i = 1; i < net_depth_condition; i++)
	    initialize_weights(states_, weights_[net_depth + 1 + i], biases_[net_depth + 1 + i], net_width_condition, net_width_condition);
    initialize_weights(states_, weights_[net_depth + 1 + net_depth_condition], biases_[net_depth + 1 + net_depth_condition], num_rgb_channels, net_width_condition);
    cudaDeviceSynchronize();
}

std::pair<float*, float*> accelerated_mlp::get_output(float* dev_encoded_position, float* dev_encoded_direction)
{
    this->encoded_position = dev_encoded_position;
    this->encoded_direction = dev_encoded_direction;
    apply_layer(dev_encoded_position, weights_[0], biases_[0], outputs_[0], weighted_sums_[0], net_width, location_dimension * location_encodings);
    cudaDeviceSynchronize();
    for(int i = 1; i < net_depth; i++)
	{
        if (i % skip_layer == 0)
	        apply_layer_conjoined_inputs(outputs_[i-1], dev_encoded_position, weights_[i], biases_[i], outputs_[i], weighted_sums_[i], net_width, net_width, location_dimension * location_encodings);
        else
			apply_layer(outputs_[i - 1], weights_[i], biases_[i], outputs_[i], weighted_sums_[i], net_width, net_width);
        cudaDeviceSynchronize();
	}
    apply_soft_plus_layer (outputs_[net_depth - 1], weights_[net_depth], biases_[net_depth], outputs_[net_depth], weighted_sums_[net_depth], net_width, num_density_channels);
    apply_layer_conjoined_inputs(outputs_[net_depth - 1], dev_encoded_direction, weights_[net_depth + 1], biases_[net_depth + 1], outputs_[net_depth + 1], weighted_sums_[net_depth + 1], net_width_condition, net_width, direction_dimension * direction_encodings);
    cudaDeviceSynchronize();
    for (int i = 1; i < net_depth_condition; i++)
    {
	    apply_layer(outputs_[net_depth + i], weights_[net_depth + 1 + i], biases_[net_depth + 1 + i], outputs_[net_depth + 1 + i], weighted_sums_[net_depth + 1 + i], net_width_condition, net_width_condition);
        cudaDeviceSynchronize();
    }
    apply_sigmoid_layer (outputs_[net_depth + net_depth_condition], weights_[net_depth + 1 + net_depth_condition], biases_[net_depth + 1 + net_depth_condition], outputs_[net_depth + 1 + net_depth_condition], weighted_sums_[net_depth + 1 + net_depth_condition], num_rgb_channels, net_width_condition);
    cudaDeviceSynchronize();
    return std::make_pair(outputs_[net_depth], outputs_[net_depth + 1 + net_depth_condition]);
}

std::pair<float*[11], float*[11]> accelerated_mlp::get_gradient(float* color_gradient, float* density_gradient)
{
    backpropagate_sigmoid_layer(outputs_[net_depth + net_depth_condition],
                                weights_[net_depth + 1 + net_depth_condition],
                                weighted_sums_[net_depth + 1 + net_depth_condition], color_gradient,
                                input_grads_[net_depth + 1 + net_depth_condition],
                                weight_grads_[net_depth + 1 + net_depth_condition],
                                bias_grads_[net_depth + 1 + net_depth_condition], num_rgb_channels, net_width_condition);
    cudaDeviceSynchronize();
    for(int i = net_depth_condition - 1; i >= 1; i--)
	{
		backpropagate_layer(outputs_[net_depth + i], weights_[net_depth + 1 + i], weighted_sums_[net_depth + 1 + i],
		                    input_grads_[net_depth + i + 1], input_grads_[net_depth + i], weight_grads_[net_depth + 1 + i],
		                    bias_grads_[net_depth + 1 + i], net_width_condition, net_width_condition);
		cudaDeviceSynchronize();
	}
    backpropagate_layer_partial_conjoined(outputs_[net_depth - 1], encoded_direction, weights_[net_depth + 1],
        weighted_sums_[net_depth + 1], input_grads_[net_depth + 1],
        input_grads_[net_depth - 1], weight_grads_[net_depth + 1],
        bias_grads_[net_depth + 1], net_width_condition, net_width, direction_dimension * direction_encodings);
    backpropagate_soft_plus_layer(outputs_[net_depth - 1], weights_[net_depth], weighted_sums_[net_depth],
                                  input_grads_[net_depth], input_grads_[net_depth - 1], weight_grads_[net_depth],
                                  bias_grads_[net_depth], net_width, num_density_channels);
    for (int i = net_depth - 1; i >= 1; i--)
    {
	    if (i % skip_layer == 0)
		    backpropagate_layer_partial_conjoined(outputs_[i - 1], encoded_position, weights_[i],
				weighted_sums_[i], input_grads_[i + 1],
				input_grads_[i - 1], weight_grads_[i],
				bias_grads_[i], net_width, net_width, location_dimension * location_encodings);
		else
            backpropagate_layer(outputs_[i - 1], weights_[i], weighted_sums_[i], input_grads_[i + 1], input_grads_[i],
                                weight_grads_[i], bias_grads_[i], net_width, net_width);
    }
    backpropagate_layer(encoded_position, weights_[0], weighted_sums_[0], input_grads_[1], input_grads_[0],
                        weight_grads_[0], bias_grads_[0], net_width, location_dimension * location_encodings);
    return std::make_pair(weight_grads_, bias_grads_);
}
