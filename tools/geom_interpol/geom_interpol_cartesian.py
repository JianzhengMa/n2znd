#!/usr/bin/env python3
import re
import math
import sys
import os
import numpy as np
from optparse import OptionParser 

# Written by Jianzheng Ma, Fudan University, Shanghai, China, 2024-04-02.
# Geometrical coordinate interpolation code (Cartesian version)

# Usage examples:
# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz
# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz -n 10
# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz -a

parser = OptionParser()
parser.add_option("-i", dest="start", action="store", type=str, nargs=1, \
                  help="Interpolation start point file (xyz format file)") 
parser.add_option("-o", dest="end", action="store", type=str, nargs=1, \
                  help="Interpolation end point file (xyz format file)") 
parser.add_option("-n", dest="fsav", action="store", type=int, nargs=1, \
                  help="Interpolation point number", default=10) 
parser.add_option("-a", dest="end_geo", action="store_true", \
                  help="Include end point coordinate in interpolation results", default=False) 
(options, args) = parser.parse_args()
start_file = options.start
end_file = options.end
fsav = options.fsav
end_geo = options.end_geo

def atom_symbol(filename):
    """Extract atom symbols from xyz file"""
    with open(filename) as f:
        atom_symbol = []
        for i in f:
            result = re.search('([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}', i)
            if result is not None:
                atom_symbol.append(result.group().split()[0])
        return atom_symbol

# Validate input structures
atom_n_start = len(atom_symbol(start_file))
atom_n_end = len(atom_symbol(end_file))
if atom_n_start != atom_n_end:
    print(f"Error: Different atom counts in {start_file} ({atom_n_start}) and {end_file} ({atom_n_end})")
    exit()
    
atom_symb_start = atom_symbol(start_file)
atom_symb_end = atom_symbol(end_file)
if atom_symb_start != atom_symb_end:
    print(f"Error: Atom types don't match between {start_file} and {end_file}")
    exit()

def get_coordinate(filename):
    """Extract coordinates and symbols from xyz file"""
    coordinate = []
    symbols = []
    with open(filename) as f:
        for i in f:
            result = re.search('([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}', i)
            if result is not None:
                parts = result.group().split()
                symbols.append(parts[0])
                coordinate.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coordinate

def align_structures(start_coords, end_coords):
    """Align two structures using Kabsch algorithm to minimize RMSD"""
    # Center both structures at origin
    start_center = np.mean(start_coords, axis=0)
    end_center = np.mean(end_coords, axis=0)
    
    centered_start = start_coords - start_center
    centered_end = end_coords - end_center
    
    # Compute covariance matrix
    H = np.dot(centered_start.T, centered_end)
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    d = np.linalg.det(np.dot(Vt.T, U.T))
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.sign(d)]])
    R = np.dot(Vt.T, np.dot(E, U.T))
    
    # Rotate end structure to align with start structure
    aligned_end = np.dot(centered_end, R) + start_center
    
    return centered_start + start_center, aligned_end

def cartesian_interpolation(start_coords, end_coords, fsav, end_geo_flag):
    """Perform linear interpolation in Cartesian space"""
    all_interpolated = []
    n_atoms = len(start_coords)
    
    # Calculate step vectors for each atom
    step_vectors = [(end_coords[i] - start_coords[i]) / fsav for i in range(n_atoms)]
    
    # Generate interpolated structures
    for i in range(fsav + 1):
        interpolated = []
        for j in range(n_atoms):
            # Calculate position for this step
            pos = start_coords[j] + i * step_vectors[j]
            interpolated.append(pos)
        all_interpolated.append(interpolated)
    
    # Add end point if requested
    if end_geo_flag:
        all_interpolated.append(end_coords)
    
    return all_interpolated

def write_xyz_file(filename, symbols, coords, comment):
    """Write coordinates to XYZ file"""
    with open(filename, 'a') as f:
        f.write(f"{len(coords)}\n")
        f.write(f"{comment}\n")
        for symbol, coord in zip(symbols, coords):
            f.write(f"{symbol}\t{coord[0]:.8f}\t{coord[1]:.8f}\t{coord[2]:.8f}\n")

def check_structure_distortion(coords1, coords2, threshold=0.5):
    """Check if two structures are distorted beyond threshold"""
    max_dist = 0.0
    for a1, a2 in zip(coords1, coords2):
        dist = math.sqrt(sum((a1[i] - a2[i])**2 for i in range(3)))
        if dist > max_dist:
            max_dist = dist
    return max_dist > threshold

def main():
    # Read start and end structures
    start_symbols, start_coords = get_coordinate(start_file)
    end_symbols, end_coords = get_coordinate(end_file)
    
    # Convert to numpy arrays for easier manipulation
    start_coords_np = np.array(start_coords)
    end_coords_np = np.array(end_coords)
    
    # Align structures to minimize RMSD
    aligned_start, aligned_end = align_structures(start_coords_np, end_coords_np)
    
    # Perform Cartesian interpolation
    interpolated_coords = cartesian_interpolation(aligned_start, aligned_end, fsav, end_geo)
    
    # Prepare output file
    atom_n = len(start_symbols)
    interpol_file = f"interpol_{start_file.split('.')[0]}2{end_file.split('.')[0]}.xyz"
    
    if os.path.exists(interpol_file):
        os.remove(interpol_file)
    
    # Generate interpolated structures
    prev_coords = aligned_start
    for counter, coords in enumerate(interpolated_coords):
        # Check for significant distortion
        if counter > 0 and check_structure_distortion(prev_coords, coords):
            print(f"Warning: Significant distortion detected at interpolation point {counter}")
            print("Consider reducing the number of interpolation points or using a different method.")
        
        prev_coords = coords
        
        # Write to file
        write_xyz_file(interpol_file, start_symbols, coords, f"Interpolation point: {counter}")

if __name__ == '__main__':
    main()
