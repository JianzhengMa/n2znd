#!/bin/bash

CHECK_LIST=(10 132 1333 1388 1592 1950 2034 2047 2036 275 287 3 319 387 468 46 810 971 955)

SLURM_SCRIPT_CONTENT=$(cat << 'EOF'
#!/bin/bash
#SBATCH -J gamess
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -p queue3-1
#SBATCH -t 10-00:00:00

prefix="./run_S1"
m=REPLACE_ME  

export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
export LD_LIBRARY_PATH=/public/home/majianzheng/softwares/intel64:$LD_LIBRARY_PATH
export UCX_TLS=dc,self
export OMP_NUM_THREADS=1
ulimit -s unlimited
module load compiler/gcc/7.3.1

EXEC=/public/home/majianzheng/softwares/gamess

ppwd=$(pwd)
echo  "$(date)      Job Start" >>${ppwd}/log

TmpRunDir="input$(tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24)"

mkdir -p $EXEC/restart/$TmpRunDir
cp $EXEC/rungms $EXEC/restart/$TmpRunDir/rungms-tmp
sed -i '1a set TmpRunDir='"$TmpRunDir" $EXEC/restart/$TmpRunDir/rungms-tmp

all=$(ipcs -s | sed -n '4,$p' | cut -d" " -f2)
for i in $all ;do ipcrm -s $i ; done

cd "./$prefix/$m" || { echo "Directory $prefix/$m not found!" >>${ppwd}/log; exit 1; }
$EXEC/restart/$TmpRunDir/rungms-tmp input.inp 00 64 > input.out
rm -r $EXEC/restart/$TmpRunDir

if grep -q "GRADIENT OF THE ENERGY" input.out; then
    echo  "$(date)      $m scf write successfully" >>${ppwd}/log
else
    echo  "$(date)      Fail in scf writing $m" >>${ppwd}/log
fi
EOF
)

TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

for dir in "${CHECK_LIST[@]}"; do
    SLURM_FILE="$TEMP_DIR/sub_gamess_$dir.slurm"
    
    echo "${SLURM_SCRIPT_CONTENT//REPLACE_ME/$dir}" > "$SLURM_FILE"
    
    sbatch "$SLURM_FILE"
    echo "Submitted job for directory $dir"
done

echo "All jobs have been submitted. Temporary scripts are in $TEMP_DIR"
