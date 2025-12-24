#!/bin/bash

CHECK_LIST=(1333 1388 2047 2036 287 319 387 468 971 955)
prefix="./run_S0"

# Iterate through directories (directory traversal method)[1](@ref)
for dir in "${CHECK_LIST[@]}"; do
    target_file="$prefix/${dir}/input.inp"
    
    # Verify file existence (file check condition)[1](@ref)
    if [ -f "$target_file" ]; then
        # Perform in-place substitution using sed
#        sed -i 's/maxit=100/maxit=200/g' "$target_file"
        sed -i 's/diis=.f. SOSCF=.t./diis=.t. SOSCF=.f./g' "$target_file"
        echo "Modified: $target_file"
    else
        echo "File not found: $target_file" >&2
    fi
done

echo "Batch modification completed!"
