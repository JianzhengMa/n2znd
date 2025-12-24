#!/bin/bash
#SBATCH -J gamess
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -p queue3-1
#SBATCH -t 10-00:00:00

init=1
traj=16
prefix="./run_S0"

export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
export LD_LIBRARY_PATH=/public/home/majianzheng/softwares/intel64:$LD_LIBRARY_PATH
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
ulimit -s unlimited
module load compiler/gcc/7.3.1

EXEC=/public/home/majianzheng/softwares/gamess

ppwd=`pwd`
echo  "`date`      Job Start" >>${ppwd}/log
for ((m=init; m<=traj; m++))
do
    TmpRunDir="input$(tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24)"

    mkdir $EXEC/restart/$TmpRunDir
    cp $EXEC/rungms $EXEC/restart/$TmpRunDir/rungms-tmp
    sed -i '1a set TmpRunDir='$TmpRunDir $EXEC/restart/$TmpRunDir/rungms-tmp

    all=$(ipcs -s | sed -n '4,$p' | cut -d" " -f2)
    echo $all
    for i in $all ;do ipcrm -s $i ; done

    cd "./$prefix/$m"
    $EXEC/restart/$TmpRunDir/rungms-tmp input.inp 00 64 > input.out
    rm -r $EXEC/restart/$TmpRunDir

    if grep -q "GRADIENT OF THE ENERGY" input.out; then
        echo  "`date`      ${m} scf write successfully" >>${ppwd}/log
    else
        echo  "`date`      Fail in scf writing ${m}" >>${ppwd}/log
    fi

    cd ../../
done
