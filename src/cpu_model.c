#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#define N 1024      // number of neurons
#define T 1000      // timesteps

const float dt      = 1e-3f;
const float tau_m   = 20e-3f;   // membrane time constant
const float tau_s   = 5e-3f;    // synaptic time constant
const float theta   = 1.0f;

float u[N];          // membrane potential
float g[N];          // synaptic current
float W[N][N];       // synaptic weights (dense for now)
// The connection is recurrent, 
// all the neurons are connected to themselves and to eachother.
// there is no layers.

uint8_t s[N];
uint8_t s_prev[N];


void print_neuron_states(float u[]){
    for (int i = 0; i < N; ++i) {
        printf("u[%d] = %f \n", i, u[i]);
    }
}

void init(){
    for (int i = 0; i < N; ++i) {
        u[i] = 0.0f;
        g[i] = 0.0f;
        s[i] = 0;
        s_prev[i] = 0;

        for (int j = 0; j < N; ++j) {
            // simple random weights
            W[j][i] = ((float)rand() / RAND_MAX) * 0.05f;
        }
    }
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

            //sum (wji * sj[n])
            for (int j = 0; j < N; ++j) {
                input_current += W[j][i] * s_prev[j];
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
    }

    print_neuron_states(u);
    printf("Simulation finished.\n");
    // In win, for not the terminal to close.
    int a;
    scanf( "%d", &a);
    return 0;
}
