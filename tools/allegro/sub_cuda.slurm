#!/bin/bash
#SBATCH -J Nequip
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
#SBATCH --exclusive

ulimit -s unlimited

export OMP_NUM_THREADS=1
module load compiler/cuda/11.4
nvidia-smi
source /public/home/majianzheng/anaconda3/bin/activate
conda activate nequip
nequip-train base.yaml --warn-unused | tee log000
