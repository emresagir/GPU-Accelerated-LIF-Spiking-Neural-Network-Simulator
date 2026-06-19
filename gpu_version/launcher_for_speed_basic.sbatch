#!/bin/bash
#SBATCH --job-name=GPU_snn_benchmark
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s339049@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:15:00          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=GPU_benchmark.log
#SBATCH --mem=8G                 
#SBATCH --gres=gpu:1

# Load modules
module load gcc/12.4.0
module load hpcx
module load nvhpc-hpcx-2.20-cuda12/25.1 

echo "Loaded NVIDIA SDK !!!"
nvidia-smi
nvcc --version

echo "Starting Automated Benchmarks..."

N_values=(256 512 1024 2048 4096 8192 16384)
T_values=(100 500 1000)

CSV_FILE="benchmark_results_basic.csv"
echo "N,T,Time_ms" > $CSV_FILE

# Ensure the expected data directory exists
mkdir -p ../torch

for t in "${T_values[@]}"; do
    for n in "${N_values[@]}"; do
        echo "--------------------------------------------"
        echo "Generating mock data, compiling, and running for N=$n, T=$t..."
        
        # 1. Dynamically generate binary files with matching sizes
        python3 -c "
import numpy as np
# Match the data distribution from your PyTorch reference script
W = (np.random.rand($n, $n).astype(np.float32) * 0.05)
W.tofile('../torch/W_post_pre.f32')

# Generate input Poisson spike matrix (rate=20Hz, dt=0.001 -> prob=0.02)
ext = (np.random.rand($t, $n) < 0.02).astype(np.uint8)
ext.tofile('../torch/ext_spikes.u8')
"
        
        # 2. Compile with custom macros
        nvcc -O3 \
             -DGRID_N=$n -DSTEPS_T=$t \
             gpu_basic.cu -o snn_benchmark.out
        
        if [ $? -ne 0 ]; then
            echo "Compilation failed for N=$n, T=$t"
            continue
        fi

        # 3. Run and parse results
        output=$(./snn_benchmark.out)
        time_ms=$(echo "$output" | grep "GPU simulation time" | awk '{print $4}')
        
        if [ ! -z "$time_ms" ]; then
            echo "$n,$t,$time_ms" >> $CSV_FILE
            echo "Success! Saved: N=$n, T=$t, Time=$time_ms ms"
        else
            echo "Failed to parse execution time. Output was:"
            echo "$output"
        fi
    done
done

# Clean up temporary execution artifact
rm -f snn_benchmark.out
echo "Done! All results are saved in $CSV_FILE"