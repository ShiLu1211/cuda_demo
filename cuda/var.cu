#include <curand_kernel.h>

extern "C" __global__
void var_kernel(const float* returns, int n, float confidence, float holding, float* out_var) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState state;
    curand_init(42, i, 0, &state);
    float Z = curand_normal(&state);
    float r = returns[i] * (1.0f + Z * 0.01f); // 模拟波动后的收益

    float loss = holding - r;
    out_var[i] = loss;
}

