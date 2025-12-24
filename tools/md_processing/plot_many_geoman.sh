#!/bin/bash
#SBATCH -J mlmd
#SBATCH -p queue1-1
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 144:00:00

init=1
traj=100

source /public/home/majianzheng/anaconda3/bin/activate
conda activate nequip

for ((m=init; m<=traj; m++))
do
    dir_name=$(printf "run%04d" "$m")
    echo "$dir_name"
    cp geoman.py plot_geoman.py "./$dir_name/"
    cd "./$dir_name"
    #python geoman.py -r 1 2 -n 1
    python geoman.py -d 5 2 1 3 -n 1
    python plot_geoman.py 
    cd ../
done
