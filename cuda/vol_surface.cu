#include <math.h>

__device__ float normpdf(float x) {
    return 0.3989422804014327f * expf(-0.5f * x * x);
}

extern "C" __global__
void vol_surface_kernel(const float* S, const float* K, const float* T,
    const float* r, const float* P, float* out, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    float s = S[idx], k = K[idx], t = T[idx], rr = r[idx], price = P[idx];
    float sigma = 0.2f;

    for (int i = 0; i < 20; i++) {
        float d1 = (logf(s / k) + (rr + 0.5f * sigma * sigma) * t) / (sigma * sqrtf(t));
        float d2 = d1 - sigma * sqrtf(t);
        float call = s * normcdf(d1) - k * expf(-rr * t) * normcdf(d2);
        float vega = s * sqrtf(t) * normpdf(d1);
        float diff = call - price;
        if (fabsf(diff) < 1e-6) break;
        sigma -= diff / vega;
    }

    out[idx] = sigma;
}