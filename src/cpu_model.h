#ifndef CPU_MODEL_H
#define CPU_MODEL_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>


// #define N 1024      // number of neurons
// #define T 1000      // timesteps

#define N 8      // number of neurons
#define T 20      // timesteps

#define dt          1e-3f
#define tau_m       20e-3f   // membrane time constant
#define tau_s       5e-3f    // synaptic time constant
#define theta       1.0f
#define ext_weight  10.2f 


// Prototypes
void loadw_f32(const char *fn, size_t count, float arr[]);
void load_ext_spikes(const char *fn, uint8_t arr[]);

#endif /* CPU_MODEL_H */