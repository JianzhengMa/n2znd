#!/usr/bin/env python3
"""
Usage:
    python batch_interpolate.py -i1 file1.xyz -i2 file2.xyz -n 5 -o result.xyz
    python batch_interpolate.py -i1 file1.xyz -i2 file2.xyz -n 3 -m internal -o result.xyz
    python batch_interpolate.py -i1 file1.xyz -i2 file2.xyz -n 4 -p 20 --include-end -o result.xyz

Parameters:
    -i1, --input1: First XYZ file
    -i2, --input2: Second XYZ file
    -n, --num-pairs: Number of structure pairs to interpolate
    -m, --method: Interpolation method (cartesian or internal, default: cartesian)
    -p, --interp-points: Number of interpolation points per pair (default: 10)
    -o, --output: Output filename
    --include-end: Include endpoint structures
    --keep-temp: Keep temporary files
"""

import os
import sys
import random
import subprocess
import argparse
from pathlib import Path

def read_xyz_structures(filename):
    """Read all molecular structures from XYZ file"""
    structures = []
    
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            try:
                atom_count = int(line)
            except ValueError:
                i += 1
                continue
            
            i += 1
            if i >= len(lines):
                break
                
            title_line = lines[i].strip()
            i += 1
            
            structure = [title_line]
            atoms_read = 0
            
            while atoms_read < atom_count and i < len(lines):
                if lines[i].strip():
                    structure.append(lines[i].rstrip())
                    atoms_read += 1
                i += 1
            
            structures.append(structure)
    
    return structures

def check_structure_match(struct1, struct2):
    """Check if two structures have same atom count and types"""
    def get_symbols(structure):
        symbols = []
        for line in structure[1:]:
            if line.strip():
                symbols.append(line.split()[0])
        return symbols
    
    symbols1 = get_symbols(struct1)
    symbols2 = get_symbols(struct2)
    
    if len(symbols1) != len(symbols2):
        return False, f"Atom count mismatch: {len(symbols1)} vs {len(symbols2)}"
    
    for i, (s1, s2) in enumerate(zip(symbols1, symbols2)):
        if s1 != s2:
            return False, f"Atom type mismatch at position {i}: {s1} vs {s2}"
    
    return True, "Structures match"

def write_single_xyz(filename, structure):
    """Write single structure to XYZ file"""
    with open(filename, 'w') as f:
        f.write(f"{len(structure)-1}\n")
        f.write(f"{structure[0]}\n")
        for i in range(1, len(structure)):
            f.write(f"{structure[i]}\n")

def run_interpolation(script, start_file, end_file, n_points, include_end):
    """Run interpolation script"""
    cmd = [sys.executable, script, '-i', start_file, '-o', end_file, '-n', str(n_points)]
    if include_end:
        cmd.append('-a')
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find output file
        start_base = Path(start_file).stem
        end_base = Path(end_file).stem
        output_file = f"interpol_{start_base}2{end_base}.xyz"
        
        if os.path.exists(output_file):
            return output_file
        else:
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch molecular structure interpolation')
    parser.add_argument('-i1', '--input1', required=True, help='First XYZ file')
    parser.add_argument('-i2', '--input2', required=True, help='Second XYZ file')
    parser.add_argument('-n', '--num-pairs', type=int, required=True, help='Number of structure pairs to interpolate')
    parser.add_argument('-m', '--method', choices=['cartesian', 'internal'], default='cartesian', help='Interpolation method')
    parser.add_argument('-p', '--interp-points', type=int, default=10, help='Interpolation points per pair')
    parser.add_argument('-o', '--output', required=True, help='Output filename')
    parser.add_argument('--include-end', action='store_true', help='Include endpoint structures')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.input1):
        print(f"Error: File not found {args.input1}")
        return
    if not os.path.exists(args.input2):
        print(f"Error: File not found {args.input2}")
        return
    
    # Read structures
    print("Reading structure files...")
    structures1 = read_xyz_structures(args.input1)
    structures2 = read_xyz_structures(args.input2)
    
    print(f"File 1: {len(structures1)} structures")
    print(f"File 2: {len(structures2)} structures")
    
    # Check if enough structures
    if len(structures1) < args.num_pairs or len(structures2) < args.num_pairs:
        print("Error: Not enough structures in input files")
        return
    
    # Select interpolation script
    if args.method == 'cartesian':
        script = 'geom_interpol_cartesian.py'
    else:
        script = 'geom_interpol_internal.py'
    
    if not os.path.exists(script):
        print(f"Error: Interpolation script not found {script}")
        return
    
    # Randomly select structure pairs
    indices1 = random.sample(range(len(structures1)), args.num_pairs)
    indices2 = random.sample(range(len(structures2)), args.num_pairs)
    
    all_structures = []
    success_count = 0
    
    # Process each structure pair
    for i, (idx1, idx2) in enumerate(zip(indices1, indices2)):
        print(f"Processing pair {i+1}/{args.num_pairs}...")
        
        # Check structure compatibility
        match, msg = check_structure_match(structures1[idx1], structures2[idx2])
        if not match:
            print(f"  Skip: {msg}")
            continue
        
        # Create file names
        start_file = f"temp_start_{i}.xyz"
        end_file = f"temp_end_{i}.xyz"
        
        # Write temporary files
        write_single_xyz(start_file, structures1[idx1])
        write_single_xyz(end_file, structures2[idx2])
        
        # Run interpolation
        interp_file = run_interpolation(
            script, start_file, end_file, 
            args.interp_points, args.include_end
        )
        
        # Clean up temporary input files
        if os.path.exists(start_file):
            os.remove(start_file)
        if os.path.exists(end_file):
            os.remove(end_file)
        
        if interp_file:
            # Read interpolation results
            interp_structures = read_xyz_structures(interp_file)
            all_structures.extend(interp_structures)
            success_count += 1
            
            # Remove interpolation output file
            if os.path.exists(interp_file):
                os.remove(interp_file)
    
    # Clean up all temporary files at the end
    if not args.keep_temp:
        # Remove any remaining temporary files
        for filename in os.listdir('.'):
            if filename.startswith('temp_start_') and filename.endswith('.xyz'):
                os.remove(filename)
            elif filename.startswith('temp_end_') and filename.endswith('.xyz'):
                os.remove(filename)
            elif filename.startswith('interpol_') and filename.endswith('.xyz'):
                os.remove(filename)
            elif filename.endswith('.zmatrix'):
                os.remove(filename)
    
    # Write final output
    if all_structures:
        print(f"Writing results to {args.output}...")
        with open(args.output, 'w') as f:
            for structure in all_structures:
                f.write(f"{len(structure)-1}\n")
                for line in structure:
                    f.write(f"{line}\n")
        
        print(f"Done! Successfully processed {success_count} pairs, generated {len(all_structures)} structures")
    else:
        print("Error: No structures generated")

if __name__ == '__main__':
    main()
