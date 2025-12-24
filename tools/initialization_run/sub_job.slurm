#!/bin/bash
#SBATCH -J mlmd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p queue3-1
#SBATCH -t 10-00:00:00

init=5
traj=5

export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
export LD_LIBRARY_PATH=/public/home/majianzheng/softwares/intel64:$LD_LIBRARY_PATH
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
ulimit -s unlimited
module load compiler/gcc/7.3.1
source /public/home/majianzheng/anaconda3/bin/activate
conda activate nequip

echo "============================================================"
env | grep "MKLROOT="
echo "============================================================"
echo "Job ID: $SLURM_JOB_NAME"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of processors: $SLURM_NTASKS"
echo "Task is running on the following nodes:"
echo $SLURM_JOB_NODELIST
echo "OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK"
echo "============================================================"
echo


for ((m=init; m<=traj; m++))
do
    dir_name=$(printf "run%04d" "$m")
    cp ./example_config.yaml "./$dir_name/" 
    (cd "$dir_name" && python -m znmd.md_main --config example_config.yaml > main.out)
done
