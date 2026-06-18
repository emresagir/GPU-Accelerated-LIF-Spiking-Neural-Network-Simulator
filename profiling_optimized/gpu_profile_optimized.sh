#!/bin/bash
#SBATCH --job-name=GPU_profile_Optimized
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=profile_optimized.log

PROFILE_TOOL="nsys" # nsys | ncu
APP="gpu_optimized" # gpu_optimized

NSYS_TRACE_COMMON="cuda,osrt,nvtx"
NSYS_TRACE_MPI="cuda,osrt,mpi,nvtx"

NCU_SECTIONS_VADD="SpeedOfLight,MemoryWorkloadAnalysis,LaunchStats,Occupancy"
NCU_SECTIONS_SGEMM="SpeedOfLight,LaunchStats,Occupancy,InstructionStats"

module purge
module load gcc/12.4.0
module load nvhpc/25.1
module load openmpi/5.0.7_gcc12

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -n "${NVHPC:-}" ]; then
        CUDA_HOME="$NVHPC/Linux_x86_64/25.1/cuda"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname $(dirname $(command -v nvcc)))"
    fi
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc -O3 ../optimized_gpu_version/gpu_optimized.cu -o gpu_optimized

if [ "$PROFILE_TOOL" = "nsys" ]; then
    nsys profile --force-overwrite=true \
        -o "${APP}_nsys" --trace="$NSYS_TRACE_COMMON" --sample=cpu \
        ./gpu_optimized

    nsys stats "${APP}_nsys.nsys-rep" --force-export=true > "${APP}_nsys_summary.txt"

elif [ "$PROFILE_TOOL" = "ncu" ]; then
    ncu --set full \
        --export "${APP}_ncu" ./gpu_optimized
fi

