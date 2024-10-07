#pragma once
ref class AcceleratedAdamOptimizer
{
public:
	AcceleratedAdamOptimizer(int* layer_sizes, int num_layers);
	void step(float** params, float** grads, float learning_rate);
private:
	int num_layers;
	int* layer_sizes;
	float** dev_m;
	float** dev_v;
	constexpr float beta1 = 0.9f;
	constexpr float beta2 = 0.999f;
	int iteration = 0;
	constexpr float epsilon = 1e-8f;
};

