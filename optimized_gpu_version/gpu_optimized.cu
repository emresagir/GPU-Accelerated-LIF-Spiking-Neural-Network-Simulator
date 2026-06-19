#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gpu_optimized.h"

#define TEST 0

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

// Must divide evenly into blockDim.x (both are 256 here). 
// If 512, the threads need to do 2 loads, if 128 half of the threads would be idle.
#define TILE_W 256


__global__ void fused_snn_kernel(
    const float*   __restrict__ W,
    const uint8_t* __restrict__ s_prev,
    const uint8_t* __restrict__ ext_step,
    float*   u,
    float*   g,
    uint8_t* s,
    float*   u_trace,
    float*   g_trace,
    uint8_t* s_trace,
    float alpha, float ext_w,
    float beta,  float thresh,
    int n, int t_idx
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocating the shared memory.
    __shared__ uint8_t s_tile[TILE_W];

    // Calculating the external contribution with the external spikes and the weights.
    float ext_contrib = (i < n) ? (float)ext_step[i] * ext_w : 0.0f;

    // With acc, the g[i] is now not written to the global memory, this will be used later for calculating decayed g[i].
    float acc = ext_contrib;

    // Calculating the base of the current neuron, for the recurrent calculation.
    const float* W_row = (i < n) ? W + (size_t)i * n : NULL;

    // Shared memory tiling for the s_prev[]
    for (int tile_start = 0; tile_start < n; tile_start += TILE_W) {
        // Cooperative load, each thread loads one element to the s_tile from the s_prev array.
        int j_load = tile_start + threadIdx.x;
        s_tile[threadIdx.x] = (j_load < n) ? s_prev[j_load] : 0;
        __syncthreads();

        if (i < n) {
            int tile_end = min(TILE_W, n - tile_start);
            // Accumulation into the acc of the weights if there was a spike previously.
            // --- float4 vectorized inner loop ---
            int k = 0;
            int tile_end4 = (tile_end / 4) * 4;
            for (; k < tile_end4; k += 4) {
                float4 w4 = *reinterpret_cast<const float4*>(W_row + tile_start + k);
                acc += w4.x * (float)s_tile[k    ];
                acc += w4.y * (float)s_tile[k + 1];
                acc += w4.z * (float)s_tile[k + 2];
                acc += w4.w * (float)s_tile[k + 3];
            }
        }
        __syncthreads();
    }

    if (i >= n) return;

    // Synaptic decay
    float gi = alpha * g[i] + acc;
    g[i] = gi; 

    // Membrane update with soft reset if there is a spike
    float ui = u[i] - (float)s_prev[i] * thresh;
    // Membrane decay
    ui = beta * ui + (1.0f - beta) * gi;

    // Spike decision
    uint8_t si = (ui > thresh) ? 1 : 0;
    u[i] = ui;
    s[i] = si;

#if TEST == 1
    int idx = t_idx * n + i;
    u_trace[idx] = ui;
    g_trace[idx] = gi;
    s_trace[idx] = si;
#endif
}

/* ── file helpers ── */

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

    size_t W_sz  = (size_t)GRID_N * GRID_N;
    size_t TN_sz = (size_t)STEPS_T * GRID_N;

    /* Host buffers */
    float*   h_W   = (float*)  malloc(W_sz  * sizeof(float));
    uint8_t* h_ext = (uint8_t*)malloc(TN_sz * sizeof(uint8_t));

    load_f32_file("../torch/W_post_pre.f32", h_W,   W_sz);
    load_u8_file ("../torch/ext_spikes.u8",  h_ext, TN_sz);

    /* Device buffers */
    float*   d_W;
    float*   d_u, *d_g;
    uint8_t* d_s_a, *d_s_b;
    uint8_t* d_ext;

    float*   d_u_trace = NULL;
    float*   d_g_trace = NULL;
    uint8_t* d_s_trace = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_W,   W_sz  * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_u,   GRID_N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_g,   GRID_N     * sizeof(float)));  
    CUDA_CHECK(cudaMalloc((void**)&d_s_a, GRID_N     * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_s_b, GRID_N     * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_ext, TN_sz * sizeof(uint8_t)));

    CUDA_CHECK(cudaMemcpy(d_W,   h_W,   W_sz  * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ext, h_ext, TN_sz * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u,   0, GRID_N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g,   0, GRID_N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_s_a, 0, GRID_N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_s_b, 0, GRID_N * sizeof(uint8_t)));

#if TEST == 1
    CUDA_CHECK(cudaMalloc((void**)&d_u_trace, TN_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_g_trace, TN_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_s_trace, TN_sz * sizeof(uint8_t)));
#endif

    /*
     * Block size must equal TILE_W so that each thread loads exactly one
     * element of s_prev into the shared-memory tile.
     */
    int threads = TILE_W;   /* == 256 */
    // Dividing the neuron size into block of threads of 256.
    int blocks  = (GRID_N + threads - 1) / threads;

    uint8_t* d_s_prev = d_s_a;
    uint8_t* d_s      = d_s_b;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    for (int t = 0; t < STEPS_T; ++t) {
        uint8_t* d_ext_step = d_ext + (size_t)t * GRID_N;


        fused_snn_kernel<<<blocks, threads>>>(
            d_W, d_s_prev, d_ext_step,
            d_u, d_g, d_s,
            d_u_trace, d_g_trace, d_s_trace,
            alpha, ext_weight, beta, theta, GRID_N, t
        );

        /* Swap ping-pong pointers */
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

    FILE* f_u = fopen("u_gpu_opt.bin", "wb");
    FILE* f_g = fopen("g_gpu_opt.bin", "wb");
    FILE* f_s = fopen("s_gpu_opt.bin", "wb");
    fwrite(h_u_trace, sizeof(float),   TN_sz, f_u);
    fwrite(h_g_trace, sizeof(float),   TN_sz, f_g);
    fwrite(h_s_trace, sizeof(uint8_t), TN_sz, f_s);
    fclose(f_u); fclose(f_g); fclose(f_s);

    free(h_u_trace); free(h_g_trace); free(h_s_trace);
#endif

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_s_a));
    CUDA_CHECK(cudaFree(d_s_b));
    CUDA_CHECK(cudaFree(d_ext));

#if TEST == 1
    CUDA_CHECK(cudaFree(d_u_trace));
    CUDA_CHECK(cudaFree(d_g_trace));
    CUDA_CHECK(cudaFree(d_s_trace));
#endif

    free(h_W); free(h_ext);
    return 0;
}
