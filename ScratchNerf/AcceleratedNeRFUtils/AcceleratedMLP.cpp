// ReSharper disable IdentifierTypo
#include "AcceleratedMLP.h"

#include <ctime>

#include "helpers.h"

void initialize_curand(curandState* states, int size)
{
    extern void initialize_curand(curandState *, const unsigned long long, const int);
    unsigned long long timeSeed = time(nullptr);
    void* ic_args[3] = { &states, &timeSeed, &size };
    cudaLaunchKernel((void*)initialize_curand, dim3(size) / block_1d, block_1d, ic_args);
}
void initialize_weights(curandState* states, float* weights, float* biases, int num_rows, int num_cols)
{
    extern void initialize_glorot(const curandState *, float*, float*, const int, const int);
    void* ig_args[5] = { &states, &weights, &biases, &num_rows, &num_cols };
    cudaLaunchKernel((void*)initialize_glorot, dim3(num_cols, num_rows) / block_2d, block_2d, ig_args);
}
constexpr int sample_num = 128;
constexpr int ray_num = 1024;
void apply_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output(const float*, const float*, const float*, float*, float*, const int, const int);
    void* gno_args[7] = { &inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size };
    cudaLaunchKernel((void*)get_neuron_output, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, gno_args);
}
void apply_sigmoid_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output_sigmoid(const float*, const float*, const float*, float*, float*, const int, const int);
    void* gnos_args[7] = { &inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size };
    cudaLaunchKernel((void*)get_neuron_output_sigmoid, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, gnos_args);
}
void apply_soft_plus_layer(float* inputs, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size)
{
    extern void get_neuron_output_soft_plus(const float*, const float*, const float*, float*, float*, const int, const int);
    void* gnosp_args[7] = { &inputs, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size };
    cudaLaunchKernel((void*)get_neuron_output_soft_plus, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, gnosp_args);
}
void apply_layer_conjoined_inputs(float* inputs_a, float* inputs_b, float* weights, float* biases, float* outputs, float* weighted_sums, int num_neurons, int input_size_a, int input_size_b)
{
    extern void get_neuron_output_conjoined_inputs(const float*, const float*, const float*, const float*, float*, float*, const int, const int, const int);
    void* gnoci_args[9] = { &inputs_a, &inputs_b, &weights, &biases, &outputs, &weighted_sums, &num_neurons, &input_size_a, &input_size_b };
    cudaLaunchKernel((void*)get_neuron_output_conjoined_inputs, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, gnoci_args);
}
void backpropagate_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
    float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
    int input_size)
{
    extern void backpropagate_neuron(const float*, const float*,
        const float*, const float*,
        float*, float*, float*,
        const int, const int);
    void* bn_args[9] = { &inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size };
    cudaLaunchKernel((void*)backpropagate_neuron, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, bn_args);
}
void backpropagate_sigmoid_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
    float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
    int input_size)
{
    extern void backpropagate_neuron_sigmoid(const float*, const float*,
        const float*, const float*,
        float*, float*, float*,
        const int, const int);
    void* bns_args[9] = { &inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size };
    cudaLaunchKernel((void*)backpropagate_neuron_sigmoid, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, bns_args);
}
void backpropagate_soft_plus_layer(float* inputs, float* weights, float* weighted_sums, float* output_gradient,
    float* input_gradient, float* weight_gradient, float* bias_gradient, int num_neurons,
    int input_size)
{
    extern void backpropagate_neuron_soft_plus(const float*, const float*,
        const float*, const float*,
        float*, float*, float*,
        const int, const int);
    void* bnsp_args[9] = { &inputs, &weights, &weighted_sums, &output_gradient, &input_gradient, &weight_gradient, &bias_gradient, &num_neurons, &input_size };
    cudaLaunchKernel((void*)backpropagate_neuron_soft_plus, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, bnsp_args);
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
    cudaLaunchKernel((void*)backpropagate_neuron_partial_conjoined, dim3(num_neurons, ray_num, sample_num) / block_3d, block_3d, bnpc_args);
}
namespace AcceleratedNeRFUtils {
    void AcceleratedMLP::do_malloc_step(int num_neurons, int num_neurons_prev, int index)
    {
        cudaMalloc(get_head_ptr(weights_) + index, num_neurons * num_neurons_prev * sizeof(float));
        cudaMalloc(get_head_ptr(biases_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(outputs_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(outputs_) + net_depth + net_depth_condition + 2 + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(weighted_sums_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(weighted_sums_) + net_depth + net_depth_condition + 2 + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(input_grads_) + index, num_neurons_prev * sizeof(float));
        cudaMalloc(get_head_ptr(weight_grads_) + index, num_neurons * num_neurons_prev * sizeof(float));
        cudaMalloc(get_head_ptr(bias_grads_) + index, num_neurons * sizeof(float));
    }
    void AcceleratedMLP::reset_layer_gradients(int num_neurons, int num_neurons_prev, int index)
    {
        cudaMemset(get_head_ptr(input_grads_) + index, 0, num_neurons_prev * sizeof(float));
        cudaMemset(get_head_ptr(weight_grads_) + index, 0, num_neurons * num_neurons_prev * sizeof(float));
        cudaMemset(get_head_ptr(bias_grads_) + index, 0, num_neurons * sizeof(float));
    }
    void AcceleratedMLP::reset_conjoined_layer_gradients(int num_neurons, int num_neurons_prev, int num_added_inputs, int index)
    {
        cudaMemset(get_head_ptr(input_grads_) + index, 0, num_neurons_prev * sizeof(float));
        cudaMemset(get_head_ptr(weight_grads_) + index, 0, num_neurons * (num_neurons_prev + num_added_inputs) * sizeof(float));
        cudaMemset(get_head_ptr(bias_grads_) + index, 0, num_neurons * sizeof(float));
    }
    void AcceleratedMLP::reset_gradients(int layer)
    {
        int offset = layer * (net_depth + net_depth_condition + 2);
        reset_layer_gradients(net_width, location_dimension * location_encodings, offset);
        for (int i = 1; i < net_depth; i++)
        {
            if (i % skip_layer == 0)
                reset_conjoined_layer_gradients(net_width, net_width, location_dimension * location_encodings, offset + i);
            else
                reset_layer_gradients(net_width, net_width, offset + i);
        }
        reset_layer_gradients(num_density_channels, net_width, offset + net_depth);
        reset_conjoined_layer_gradients(net_width_condition, net_width, direction_dimension * direction_encodings, offset + net_depth + 1);
        for (int i = 1; i < net_depth_condition; i++)
            reset_layer_gradients(net_width_condition, net_width_condition, offset + net_depth + 1 + i);
        reset_layer_gradients(num_rgb_channels, net_width_condition, offset + net_depth + 1 + net_depth_condition);
    }

    array<int>^ AcceleratedMLP::get_layer_sizes()
    {
        array<int>^ layer_sizes = gcnew array<int>(2 * net_depth + 2 * net_depth_condition + 4);
        int biasOffset = net_depth + net_depth_condition + 2;
        layer_sizes[0] = location_dimension * location_encodings * net_width;
        layer_sizes[biasOffset] = net_width;
        for (int i = 1; i < net_depth; i++)
        {
            layer_sizes[i] = i % skip_layer == 0 ? (net_width + location_dimension * location_encodings) * net_width : net_width * net_width;
            layer_sizes[biasOffset + i] = net_width;
        }
        layer_sizes[net_depth] = net_width * num_density_channels;
        layer_sizes[biasOffset + net_depth] = num_density_channels;
        layer_sizes[net_depth + 1] = (net_width + direction_dimension * direction_encodings) * net_width_condition;
        layer_sizes[biasOffset + net_depth + 1] = net_width_condition;
        for (int i = 1; i < net_depth_condition; i++)
        {
            layer_sizes[net_depth + 1 + i] = net_width_condition * net_width_condition;
            layer_sizes[biasOffset + net_depth + 1 + i] = net_width_condition;
        }
        layer_sizes[net_depth + 1 + net_depth_condition] = net_width_condition * num_rgb_channels;
        layer_sizes[biasOffset + net_depth + 1 + net_depth_condition] = num_rgb_channels;
        return layer_sizes;
    }

    void AcceleratedMLP::do_malloc_conjoined(int num_neurons, int num_neurons_prev, int additional_inputs, int index)
    {
        cudaMalloc(get_head_ptr(weights_) + index, num_neurons * (num_neurons_prev + additional_inputs) * sizeof(float));
        cudaMalloc(get_head_ptr(biases_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(outputs_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(outputs_) + net_depth + net_depth_condition + 2 + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(weighted_sums_) + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(weighted_sums_) + net_depth + net_depth_condition + 2 + index, num_neurons * sizeof(float));
        cudaMalloc(get_head_ptr(input_grads_) + index, num_neurons_prev * sizeof(float));
        cudaMalloc(get_head_ptr(weight_grads_) + index, num_neurons * (num_neurons_prev + additional_inputs) * sizeof(float));
        cudaMalloc(get_head_ptr(bias_grads_) + index, num_neurons * sizeof(float));
    }
    AcceleratedMLP::AcceleratedMLP(int deg_point, int deg_view)
    {
        int max_size = 0;
        location_encodings = 2 * deg_point;
        direction_encodings = 2 * deg_view + 1;
        do_malloc_step(net_width, location_dimension * location_encodings, 0);
        max_size = std::max(max_size, net_width * location_dimension * location_encodings);
        for (int i = 1; i < net_depth; i++)
        {
            if (i % skip_layer == 0)
                do_malloc_conjoined(net_width, net_width, location_dimension * location_encodings, i);
            else
                do_malloc_step(net_width, net_width, i);
            max_size = std::max(max_size, net_width * (i % skip_layer == 0 ? net_width + location_dimension * location_encodings : net_width));
        }
        do_malloc_step(num_density_channels, net_width, net_depth);
        max_size = std::max(max_size, net_width * num_density_channels);
        do_malloc_conjoined(net_width_condition, net_width, direction_dimension * direction_encodings, net_depth + 1);
        max_size = std::max(max_size, net_width_condition * (net_width + direction_dimension * direction_encodings));
        for (int i = 1; i < net_depth_condition; i++)
        {
            do_malloc_step(net_width_condition, net_width_condition, net_depth + 1 + i);
            max_size = std::max(max_size, net_width_condition * net_width_condition);
        }
        do_malloc_step(num_rgb_channels, net_width_condition, net_depth + 1 + net_depth_condition);
        max_size = std::max(max_size, num_rgb_channels * net_width_condition);
        curandState* states;
        cudaMalloc(&states, max_size * sizeof(curandState));
        states_ = states;
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
        for (int i = 0; i < net_depth + net_depth_condition + 2; i++)
        {
            allParams[i] = weights_[i];
            allParams[i + net_depth + net_depth_condition + 2] = biases_[i];
        }
    }
    ValueTuple<uint64_t, uint64_t>^ AcceleratedMLP::get_output(float* dev_encoded_position, float* dev_encoded_direction, int level)
    {
        this->encoded_position = dev_encoded_position;
        this->encoded_direction = dev_encoded_direction;
        int offset = level * (net_depth + net_depth_condition + 2);
        apply_layer(dev_encoded_position, weights_[0], biases_[0], outputs_[offset], weighted_sums_[offset], net_width,
            location_dimension * location_encodings);
        cudaDeviceSynchronize();
        for (int i = 1; i < net_depth; i++)
        {
            if (i % skip_layer == 0)
                apply_layer_conjoined_inputs(outputs_[offset + i - 1], dev_encoded_position, weights_[i], biases_[i],
                    outputs_[offset + i], weighted_sums_[offset + i], net_width, net_width,
                    location_dimension * location_encodings);
            else
                apply_layer(outputs_[offset + i - 1], weights_[i], biases_[i], outputs_[offset + i],
                    weighted_sums_[offset + i], net_width, net_width);
            cudaDeviceSynchronize();
        }
        apply_soft_plus_layer(outputs_[offset + net_depth - 1], weights_[net_depth], biases_[net_depth],
            outputs_[offset + net_depth], weighted_sums_[offset + net_depth], net_width,
            num_density_channels);
        apply_layer_conjoined_inputs(outputs_[offset + net_depth - 1], dev_encoded_direction, weights_[net_depth + 1],
            biases_[net_depth + 1], outputs_[offset + net_depth + 1],
            weighted_sums_[offset + net_depth + 1], net_width_condition, net_width,
            direction_dimension * direction_encodings);
        cudaDeviceSynchronize();
        for (int i = 1; i < net_depth_condition; i++)
        {
            apply_layer(outputs_[offset + net_depth + i], weights_[net_depth + 1 + i], biases_[net_depth + 1 + i],
                outputs_[offset + net_depth + 1 + i], weighted_sums_[offset + net_depth + 1 + i],
                net_width_condition, net_width_condition);
            cudaDeviceSynchronize();
        }
        apply_sigmoid_layer(outputs_[offset + net_depth + net_depth_condition],
            weights_[net_depth + 1 + net_depth_condition], biases_[net_depth + 1 + net_depth_condition],
            outputs_[offset + net_depth + 1 + net_depth_condition],
            weighted_sums_[offset + net_depth + 1 + net_depth_condition], num_rgb_channels,
            net_width_condition);
        cudaDeviceSynchronize();
        return gcnew ValueTuple<uint64_t, uint64_t>((uint64_t)outputs_[offset + net_depth], (uint64_t)outputs_[offset + net_depth + 1 + net_depth_condition]);
    }
    array<float*>^ AcceleratedMLP::get_gradient(float* color_gradient, float* density_gradient, int layer)
    {
        reset_gradients(layer);
        backpropagate_sigmoid_layer(outputs_[net_depth + net_depth_condition],
            weights_[net_depth + 1 + net_depth_condition],
            weighted_sums_[net_depth + 1 + net_depth_condition], color_gradient,
            input_grads_[net_depth + 1 + net_depth_condition],
            weight_grads_[net_depth + 1 + net_depth_condition],
            bias_grads_[net_depth + 1 + net_depth_condition], num_rgb_channels, net_width_condition);
        cudaDeviceSynchronize();
        for (int i = net_depth_condition - 1; i >= 1; i--)
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
            density_gradient, input_grads_[net_depth - 1], weight_grads_[net_depth],
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
        for (int i = 0; i < net_depth + net_depth_condition + 2; i++)
        {
            allGradients[i] = weight_grads_[i];
            allGradients[i + net_depth + net_depth_condition + 2] = bias_grads_[i];
        }
        return allGradients;
    }

    AcceleratedMLP::~AcceleratedMLP()
    {
        for (int i = 0; i < net_depth + net_depth_condition + 2; i++)
		{
			cudaFree(weights_[i]);
			cudaFree(biases_[i]);
			cudaFree(outputs_[i]);
			cudaFree(outputs_[net_depth + net_depth_condition + 2 + i]);
			cudaFree(weighted_sums_[i]);
			cudaFree(weighted_sums_[net_depth + net_depth_condition + 2 + i]);
			cudaFree(input_grads_[i]);
			cudaFree(weight_grads_[i]);
			cudaFree(bias_grads_[i]);
		}
		cudaFree(states_);
    }
}
