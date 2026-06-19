#!/bin/bash
#SBATCH --job-name=GPU_basic_snn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s339049@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:00:10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=GPU_basic.log
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1

# ************************************************
# command to launch the job in the scheduler:
# sbatch launcher.sbatch

# Command to check the job status:
# squeue -u jrodriguez
# ************************************************

# load cuda module
# module load nvidia/cudasdk/10.1

module load gcc/12.4.0
module load hpcx
#module load nvhpc-hpcx-cuda12/25.1
module load nvhpc-hpcx-2.20-cuda12/25.1 

echo "Loaded NVIDIA SDK !!!"
nvidia-smi

# check the current version of the compiler:
#nvc --version              #for C
#nvc++ --version            #for C++
#nvfortran --version        #for Fortran

nvcc --version

echo "Compiling!"

nvcc gpu_basic.cu -o gpu_basic.o

./gpu_basic.o

echo "Done!"





# module avail  # check all the available packages in the system.













# nvc++ -cuda CUDA_add/add.cu -o CUDA_add/add.o


# Copia dei file da storage HOME a storage locale del nodo

#cp -r /home/myusername/data $TMPDIR/data

# myexecutable -in $TMPDIR/data -out $TMPDIR/results

# Copia dei risultati qualora generati nella cartella temporanea
#cp -r $TMPDIR/results /home/myusername/results

