#include "cpu_model.h"

float u[N];          // membrane potential
float g[N];          // synaptic current
float Wflat[N*N];       // synaptic weights (dense for now) and they are being stored in a flat manner.
// The connection is recurrent, 
// all the neurons are connected to themselves and to eachother.
// there is no layers.

uint8_t ext_spikes[T * N];
uint8_t s[N];
uint8_t s_prev[N];

// For test
float u_trace[T * N];
float g_trace[T * N];
uint8_t s_trace[T * N];

void init(){
    /* Access Wflat[i*N + j] */
    loadw_f32("../torch/W_post_pre.f32", (size_t)N * (size_t)N, Wflat);
    load_ext_spikes("../torch/ext_spikes.u8", ext_spikes);

    for (int i = 0; i < N; ++i) {
        u[i] = 0.0f;
        g[i] = 0.0f;
        s[i] = 0;
        s_prev[i] = 0;
    }

    //Test 
    float sum_ext_spikes = 0.0f;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            sum_ext_spikes += ext_spikes[t*N + i];
        }
    }
    printf("ext spike mean = %f\n",
       (float)sum_ext_spikes / (T*N));
}


int main(void)
{
    const float beta  = expf(-dt / tau_m);
    const float alpha = expf(-dt / tau_s);

    init();

    /* Time loop */
    for (int t = 0; t < T; ++t) {

        /* Synapse update */
        for (int i = 0; i < N; ++i) {
            float input_current = 0.0f;
            
            input_current += ext_spikes[t*N + i] * ext_weight;

            //sum (wji * sj[n])
            for (int j = 0; j < N; ++j) {
                input_current += Wflat[i*N + j] * s_prev[j];
            }
            // gi[t] = α gi[t − 1] + sum
            g[i] = alpha * g[i] + input_current;
        }

        // Neuron Update
        for (int i = 0; i < N; ++i) {

            /* Soft reset */
            // u[t] = u[t-1] - s[t-1] * θ
            u[i] -= s_prev[i] * theta;

            /* Integration */
            // u[t] = β u[t] + (1 - β) i[t]
            u[i] = beta * u[i] + (1.0f - beta) * g[i];

            /* Spike */
            // s[t] = Θ(u[t] - θ)
            s[i] = (u[i] > theta);
        }

        /* Copy spikes */
        for (int i = 0; i < N; ++i) {
            s_prev[i] = s[i];
        }

        // For test
        for (int i = 0; i < N; ++i) {
            u_trace[t * N + i] = u[i];
            g_trace[t * N + i] = g[i];
            s_trace[t * N + i] = s[i];
}
    }

    printf("Simulation finished.\n");
    // Test
    FILE *f_u = fopen("u_c.bin", "wb");
    FILE *f_g = fopen("g_c.bin", "wb");
    FILE *f_s = fopen("s_c.bin", "wb");
    fwrite(u_trace, sizeof(float), T * N, f_u);
    fwrite(g_trace, sizeof(float), T * N, f_g);
    fwrite(s_trace, sizeof(uint8_t), T * N, f_s);
    fclose(f_u);
    fclose(f_g);
    fclose(f_s);
    return 0;
}
