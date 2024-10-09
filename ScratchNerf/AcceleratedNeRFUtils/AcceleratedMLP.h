#pragma once
#include <cstdint>
#include <curand_kernel.h>

namespace AcceleratedNeRFUtils {
    using namespace System;
	public ref class AcceleratedMLP
	{
    public:
        static constexpr int num_levels = 2;
        static constexpr int net_depth = 8;          // The depth of the first part of MLP.
        static constexpr int net_width = 256;        // The width of the first part of MLP.
        static constexpr int net_depth_condition = 1; // The depth of the second part of MLP.
        static constexpr int net_width_condition = 128; // The width of the second part of MLP.
        static constexpr int num_rgb_channels = 3;    // Number of RGB channels.
        static constexpr int num_density_channels = 1; // Number of density channels.
        static constexpr int location_dimension = 3; // Location dimension.
        static constexpr int direction_dimension = 3; // Direction dimension.
        static constexpr int skip_layer = 4;         // Skip connection every N layers.
        void reset_gradients(int layer);
        array<int>^ get_layer_sizes();
        void do_malloc_conjoined(int num_neurons, int num_neurons_prev, int additional_inputs, int index);
        AcceleratedMLP(int deg_point, int deg_view);
        array<float*>^ allGradients = gcnew array<float*>(2 * (net_depth + net_depth_condition + 2));
        array<float*>^ allParams = gcnew array<float*>(2 * (net_depth + net_depth_condition + 2));
        ValueTuple<uint64_t, uint64_t>^ get_output(float* dev_encoded_position, float* dev_encoded_direction, int level);
        array<float*>^ get_gradient(float* color_gradient, float* density_gradient, int layer);
        ~AcceleratedMLP();
    private:
        void do_malloc_step(int num_neurons, int num_neurons_prev, int index);
        void reset_layer_gradients(int num_neurons, int num_neurons_prev, int index);
        void reset_conjoined_layer_gradients(int num_neurons, int num_neurons_prev, int num_added_inputs, int index);
        int location_encodings;
        int direction_encodings;
        array<float*>^ weights_ = gcnew array<float*>(net_depth + net_depth_condition + 2);
        array<float*>^ biases_ = gcnew array<float*>(net_depth + net_depth_condition + 2);
        array<float*>^ outputs_ = gcnew array<float*>(num_levels * (net_depth + net_depth_condition + 2));
        array<float*>^ weighted_sums_ = gcnew array<float*>(num_levels * (net_depth + net_depth_condition + 2));
        float* encoded_position;
        float* encoded_direction;
        array<float*>^ input_grads_ = gcnew array<float*>(net_depth + net_depth_condition + 2);
        array<float*>^ weight_grads_ = gcnew array<float*>(net_depth + net_depth_condition + 2);
        array<float*>^ bias_grads_ = gcnew array<float*>(net_depth + net_depth_condition + 2);
        curandState* states_;
	};
}

