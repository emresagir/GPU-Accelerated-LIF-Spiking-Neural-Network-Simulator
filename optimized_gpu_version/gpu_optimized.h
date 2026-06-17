#ifndef GPU_BASIC_H
#define GPU_BASIC_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// #define N 1024
// #define T 3000

#define N 8
#define T 20

#define dt          1e-3f
#define tau_m       20e-3f
#define tau_s       5e-3f
#define theta       1.0f
#define ext_weight  10.2f

#endif /* GPU_BASIC_H */