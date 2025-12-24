#!/bin/bash
#SBATCH -J gamess
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -p queue3-1
#SBATCH -t 10-00:00:00

base_dir=./run_S1
error_file="error_dirs.txt"

> "$error_file"

check_dir() {
    dir="$1"
    m=$(basename "$dir")
    if [ ! -f "$dir/input.out" ]; then
        echo "Missing input.out in folder: $m" >&2
        echo -n "$m " >> "$error_file"
        return 1
    elif ! grep -q "GRADIENT OF THE ENERGY" "$dir/input.out"; then
        echo "Fail in scf writing: $m" >&2
        echo -n "$m " >> "$error_file"
        return 1
    fi
    return 0
}

export -f check_dir
export error_file

find "$base_dir" -maxdepth 1 -type d -name '[0-9]*' -print0 | \
    sort -z -n | \
    xargs -0 -P $SLURM_NTASKS -I {} bash -c 'check_dir "{}"' _

if [ -s "$error_file" ]; then
    echo -e "\nError directories list:"
    cat "$error_file"
else
    echo -e "\nAll directories processed successfully."
fi
