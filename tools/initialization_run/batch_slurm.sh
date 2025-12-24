#!/bin/bash

SLURM_FILE="sub_job.slurm"
START=1
STOP=10
PAR=2
GROUP=$(( ($STOP - $START + 1) / $PAR ))

for i in $(seq 1 $(( $GROUP + 1 )) ); do
  begin=$(( $START + ($i-1) * $PAR ))
  end=$(($START + $i * $PAR - 1 ))
  
  if [ "$end" -gt "$STOP"  ]; then
    end=$STOP
  fi

  if [ $begin -eq $(( $STOP + 1  )) ]; then
     break 
  fi

  sed -i "s/ *init=[0-9]*/init=$begin/" "$SLURM_FILE"
  sed -i "s/ *traj=[0-9]*/traj=$end/" "$SLURM_FILE"

  sbatch "$SLURM_FILE"
done

