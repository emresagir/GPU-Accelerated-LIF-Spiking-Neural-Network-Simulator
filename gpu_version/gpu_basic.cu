#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gpu_basic.h"

#define TEST 1

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/*
 * Synaptic update:
 *   g[i] = alpha * g[i] + ext_spikes[t*N + i] * ext_weight
 *                        + sum_j( W[i*N + j] * s_prev[j] )
 *
 * Ext spikes are folded in here so the alpha decay applies uniformly
 * to all inputs, matching the CPU model.
 */
__global__ void synapse_kernel(
    const float*   __restrict__ W,
    const uint8_t* __restrict__ s_prev,
    const uint8_t* __restrict__ ext_step,
    float* g,
    float alpha,
    float ext_w,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc = (float)ext_step[i] * ext_w;
    int row = i * n;
    for (int j = 0; j < n; ++j)
        acc += W[row + j] * (float)s_prev[j];

    g[i] = alpha * g[i] + acc;
}

/*
 * Neuron update:
 *   soft-reset: u[i] -= s_prev[i] * theta
 *   integrate:  u[i]  = beta * u[i] + (1 - beta) * g[i]
 *   spike:      s[i]  = u[i] > theta
 */
__global__ void neuron_kernel(
    float*   u,
    const float*   __restrict__ g,
    const uint8_t* __restrict__ s_prev,
    uint8_t* s,
    float*   u_trace,
    float*   g_trace,
    uint8_t* s_trace,
    float beta,
    float thresh,
    int n,
    int t_idx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ui = u[i] - (float)s_prev[i] * thresh;
    ui = beta * ui + (1.0f - beta) * g[i];

    uint8_t si = (ui > thresh) ? 1 : 0;
    u[i] = ui;
    s[i] = si;

    int idx = t_idx * n + i;
    u_trace[idx] = ui;
    g_trace[idx] = g[i];
    s_trace[idx] = si;
}

static void load_f32_file(const char* path, float* buf, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(EXIT_FAILURE); }
    size_t r = fread(buf, sizeof(float), count, f);
    if (r != count) { fprintf(stderr, "Short read %s\n", path); fclose(f); exit(EXIT_FAILURE); }
    fclose(f);
}

static void load_u8_file(const char* path, uint8_t* buf, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(EXIT_FAILURE); }
    size_t r = fread(buf, sizeof(uint8_t), count, f);
    if (r != count) { fprintf(stderr, "Short read %s\n", path); fclose(f); exit(EXIT_FAILURE); }
    fclose(f);
}

int main(void) {
    const float beta  = expf(-dt / tau_m);
    const float alpha = expf(-dt / tau_s);

    size_t W_sz  = (size_t)N * N;
    size_t TN_sz = (size_t)T * N;

    /* Host buffers */
    float*   h_W   = (float*)  malloc(W_sz  * sizeof(float));
    uint8_t* h_ext = (uint8_t*)malloc(TN_sz * sizeof(uint8_t));

    load_f32_file("../torch/W_post_pre.f32", h_W,   W_sz);
    load_u8_file ("../torch/ext_spikes.u8",  h_ext, TN_sz);

    /* Device buffers: weights, state, full ext-spike array */
    float*   d_W;
    float*   d_u, *d_g;
    uint8_t* d_s_a, *d_s_b;   /* ping-pong buffers for s / s_prev */
    uint8_t* d_ext;            /* all T*N ext spikes, uploaded once */

    CUDA_CHECK(cudaMalloc((void**)&d_W,   W_sz  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_u,   N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_g,   N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_s_a, N     * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_s_b, N     * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_ext, TN_sz * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemcpy(d_W,   h_W,   W_sz  * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ext, h_ext, TN_sz * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u,   0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g,   0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_s_a, 0, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_s_b, 0, N * sizeof(uint8_t)));

    /* Trace buffers */
    float*   d_u_trace, *d_g_trace;
    uint8_t* d_s_trace;
    CUDA_CHECK(cudaMalloc((void**)&d_u_trace, TN_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_g_trace, TN_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_s_trace, TN_sz * sizeof(uint8_t)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    /* s_prev = d_s_a (zeros), s = d_s_b; swap each step */
    uint8_t* d_s_prev = d_s_a;
    uint8_t* d_s      = d_s_b;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    for (int t = 0; t < T; ++t) {
        uint8_t* d_ext_step = d_ext + (size_t)t * N;

        synapse_kernel<<<blocks, threads>>>(d_W, d_s_prev, d_ext_step,
                                            d_g, alpha, ext_weight, N);
        neuron_kernel <<<blocks, threads>>>(d_u, d_g, d_s_prev, d_s,
                                            d_u_trace, d_g_trace, d_s_trace,
                                            beta, theta, N, t);

        /* swap ping-pong pointers */
        uint8_t* tmp = d_s_prev;
        d_s_prev = d_s;
        d_s      = tmp;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    printf("GPU simulation time: %.3f ms\n", elapsed_ms);

#if TEST == 1
    float*   h_u_trace = (float*)  malloc(TN_sz * sizeof(float));
    float*   h_g_trace = (float*)  malloc(TN_sz * sizeof(float));
    uint8_t* h_s_trace = (uint8_t*)malloc(TN_sz * sizeof(uint8_t));

    CUDA_CHECK(cudaMemcpy(h_u_trace, d_u_trace, TN_sz * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g_trace, d_g_trace, TN_sz * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s_trace, d_s_trace, TN_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    FILE* f_u = fopen("u_gpu.bin", "wb");
    FILE* f_g = fopen("g_gpu.bin", "wb");
    FILE* f_s = fopen("s_gpu.bin", "wb");
    fwrite(h_u_trace, sizeof(float),   TN_sz, f_u);
    fwrite(h_g_trace, sizeof(float),   TN_sz, f_g);
    fwrite(h_s_trace, sizeof(uint8_t), TN_sz, f_s);
    fclose(f_u); fclose(f_g); fclose(f_s);

    free(h_u_trace); free(h_g_trace); free(h_s_trace);
#endif

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_u));   CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_s_a)); CUDA_CHECK(cudaFree(d_s_b));
    CUDA_CHECK(cudaFree(d_ext));
    CUDA_CHECK(cudaFree(d_u_trace)); CUDA_CHECK(cudaFree(d_g_trace)); CUDA_CHECK(cudaFree(d_s_trace));

    free(h_W); free(h_ext);
    return 0;
}
