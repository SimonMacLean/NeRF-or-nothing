#pragma once
#include <curand_kernel.h>
#include <utility>
using namespace System;
using namespace System::Numerics;
ref class accelerated_mlp sealed
{
public:
    accelerated_mlp(int deg_point, int deg_view);
	std::pair<float*, float*> get_output(float* encoded_position, float* encoded_direction);
    float* get_gradient(float* color_gradient, float* density_gradient);
private:
    static constexpr int net_depth = 8;          // The depth of the first part of MLP.
    static constexpr int net_width = 256;        // The width of the first part of MLP.
    static constexpr int net_depth_condition = 1; // The depth of the second part of MLP.
    static constexpr int net_width_condition = 128; // The width of the second part of MLP.
    static constexpr int num_rgb_channels = 3;    // Number of RGB channels.
    static constexpr int num_density_channels = 1; // Number of density channels.
    static constexpr int location_dimension = 3; // Location dimension.
    static constexpr int direction_dimension = 3; // Direction dimension.
    int location_encodings;
    int direction_encodings = 1;
    static constexpr int skip_layer = 4;         // Skip connection every N layers.
    float* weights_[net_depth + net_depth_condition + 2];
    float* biases_[net_depth + net_depth_condition + 2];
    float* outputs_[net_depth + net_depth_condition + 2];
    float* weighted_sums_[net_depth + net_depth_condition + 2];
    curandState* states_;
};

