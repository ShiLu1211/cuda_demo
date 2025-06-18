#include <curand_kernel.h>

extern "C" __global__
void monte_carlo(float* results, int num_paths,
    float S0, float K, float r, float sigma, float T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_paths) {
        curandState state;
        curand_init(1234, i, 0, &state);

        float Z = curand_normal(&state);
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * Z);
        float payoff = fmaxf(ST - K, 0.0f);

        results[i] = payoff;
    }
}
