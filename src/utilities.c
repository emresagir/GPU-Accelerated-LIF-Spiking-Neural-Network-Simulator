#include <stdio.h>
#include <stdlib.h>
#include "cpu_model.h"


void loadw_f32(const char *fn, size_t count, float arr[]) {
    FILE *f = fopen(fn,"rb");
    float *buf = malloc(sizeof(float) * count);
    size_t r = fread(buf, sizeof(float), count, f);
    if(r != count) fprintf(stderr,"read %zu of %zu elements\n", r, count);
    fclose(f);
    memcpy(arr, buf, sizeof(float) * N * N);
    printf("Weights are loaded \n");
}

void load_ext_spikes(const char *fn, uint8_t arr[]){
    uint8_t *buf = malloc(T * N);
    FILE *f = fopen(fn,"rb");
    fread(buf, 1, T * N, f);
    fclose(f);
    memcpy(arr, buf, sizeof(uint8_t) * T * N);
    printf("Ext spikes are loaded \n");
}
