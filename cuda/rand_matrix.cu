#include <curand_kernel.h>

extern "C" __global__
void generate_random_matrix_kernel(double* random_matrix, int num_paths, int num_steps, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        // 初始化 curand
        curandState localState;
        curand_init(seed, idx, 0, &localState);

        // 生成随机数
        for (int j = 0; j < num_steps; ++j) {
            random_matrix[idx * num_steps + j] = curand_normal_double(&localState);
        }
    }
}

extern "C"
__global__ void simulate_prices_kernel(double* random_matrix, double* price_matrix, int num_paths, int num_steps, double S0, double r, double q, double v, double T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        double dt = T / num_steps;
        double S = S0;
        for (int j = 0; j < num_steps; ++j) {
            double Z = random_matrix[idx * num_steps + j];
            S = S * exp((r - q - 0.5 * v * v) * dt + v * sqrt(dt) * Z);
            price_matrix[idx * num_steps + j] = S;
        }
    }
}