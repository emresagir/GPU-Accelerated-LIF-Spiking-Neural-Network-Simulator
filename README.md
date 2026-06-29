# GPU-Accelerated LIF Spiking Neural Network Simulator

Check `report_ieee_v/GPU_Accelerated_SNN_Simulator_Emre_Sagir.pdf` report for the upmost detail. 

A discrete-time Leaky Integrate-and-Fire (LIF) spiking neural network simulator with a current-based exponential synapse model, implemented in four versions for direct performance comparison:

- **Torch** — Python reference implementation, used for correctness validation
- **C (CPU)** — baseline single-threaded implementation
- **GPU Basic** — naive CUDA port (separate synapse/neuron kernels)
- **GPU Optimized** — fused kernel with shared-memory tiling and `float4` vectorized loads

## Results

- **3.3x** speed-up: Optimized GPU vs. Basic GPU
- **112.7x** speed-up: Optimized GPU vs. CPU (at N=16384, T=1000)
- At large neuron counts, Basic and Optimized GPU versions converge as the workload becomes memory-bandwidth bound rather than compute bound

## Repo structure

```
src/                    C (CPU) model + Makefile + benchmark script
torch/                  PyTorch reference model used for correctness validation
gpu_version/            Basic CUDA implementation
optimized_gpu_version/  Optimized CUDA implementation (fused kernel, tiling, vectorized loads)
profiling_basic/        Nsight Systems profiling output (basic GPU)
profiling_optimized/    Nsight Systems profiling output (optimized GPU)
plotting/               Scripts for correctness and performance plots
report_ieee_v/          Final IEEE-format report, slides, and figures
```

## Building & running

**CPU model:**
```bash
cd src
make
./cpu_model
```

**GPU models** (requires CUDA toolkit + `nvcc`):
```bash
cd gpu_version
nvcc -O3 gpu_basic.cu -o gpu_basic
./gpu_basic

cd ../optimized_gpu_version
nvcc -O3 gpu_optimized.cu -o optimized_gpu
./optimized_gpu
```

Neuron count (`N`) and timestep count (`T`) are set via `-D` flags (e.g. `-DGRID_N=4096 -DSTEPS_T=1000`); see the `.sbatch` launcher scripts for batch sweeps.

**Reference / correctness check:**
```bash
cd torch
python snn_reference.py
cd ../src
python test_C_vs_pytorch.py
```

## Validation

Each implementation is checked against the Torch reference by comparing membrane potential and spike traces timestep-by-timestep on identical input spikes and weights. See `report_ieee_v/` for the full write-up, plots, and benchmark methodology.

## Author

Emre Sagir — Politecnico di Torino, GPU Programming course project.
