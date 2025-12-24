#!/usr/bin/env python3.9
import os
import re
import numpy as np
import ase
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
import sys
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
import time

# Configuration parameters
start = 1
end = 100
nproc = 32  # Number of parallel processes
scfout_paths = ["./run_S0/{:d}/".format(i) for i in range(start, end+1)]
mlff_dat = "mlff_dat.extxyz"

# Conversion constants
au_to_ang = 0.529177208
au_to_ev = 27.2113956555
Ha_Bohr_to_ev_ang = 51.42208619083232

def read_atoms(filename):
    """Read atomic structure from input file."""
    elements = []
    positions = []
    in_data_block = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('$data'):
                in_data_block = True
                next(f)  # Skip title line
                next(f)  # Skip symmetry line
                continue
            if in_data_block and line and not line.lower().startswith('$end'):
                parts = line.split()
                if len(parts) >= 4:
                    element = parts[0]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4]) if len(parts) > 4 else 0.0
                    elements.append(element)
                    positions.append([x, y, z])
    
    return ase.Atoms(elements, positions=positions, pbc=False)

def read_force(filename, atom_n):
    """Read force information from output file."""
    gradient = []
    command = "grep -A " + str(atom_n) + " " + "'UNITS ARE HARTREE/BOHR'" + " " + filename
    data = os.popen(command).read().splitlines()
    for i in data:
        result = re.search('(\s+\d+\s+)([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}', i)
        if result is not None:
            gradient.append([float(i) for i in result.group().split()[2:5]])
    force = -np.array(gradient) * Ha_Bohr_to_ev_ang
    return force

def get_state(filename):
    """Get excited state information from output file."""
    command = "grep 'SELECTING EXCITED STATE IROOT='" + " " + filename
    data = os.popen(command).read()
    if data:
        state = data.split()[4]
    return state

def state_energy(filename, state):
    """Read state energies from output file."""
    data_1 = []  # State symbols
    data_2 = []  # State energies
    command = "sed -n '/SUMMARY OF MRSF-DFT RESULTS/,/SELECTING EXCITED STATE IROOT/p'" + " " + filename
    data = os.popen(command).read().splitlines()
    for i in data:
        result = re.search('(\s+\d+\s+)([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){7}', i)
        if result is not None:
            state_symbol = result.group().split()[0]
            state_energy = result.group().split()[2]
            data_1.append(state_symbol)
            data_2.append(float(state_energy))
    energy_dict = dict(zip(data_1, data_2))
    if state:
        return energy_dict[str(state)]
    elif state is False:
        return energy_dict

def check_out(filename):
    """Check if output file contains force information."""
    with open(filename) as f:
        lines = f.readlines()
        reverse_lines = reversed(lines)
        for i in reverse_lines:
            result = re.search("UNITS ARE HARTREE/BOHR", i)
            if result is not None:
                return result.group()
                break

def process_scf(filepath):
    """Process a single calculation directory."""
    input_path = os.path.join(filepath, "input.inp")
    output_path = os.path.join(filepath, "input.out")
    
    try:
        checkout = check_out(output_path)
        if not checkout:
            print(f"Skipping {filepath} - no force information found")
            return None
            
        atoms = read_atoms(input_path)
        atom_n = len(atoms)
        force = read_force(output_path, atom_n)
        state = get_state(output_path)
        energy = state_energy(output_path, state) * au_to_ev 
        
        atoms.calc = SinglePointCalculator(
            energy=energy,
            forces=force,
            atoms=atoms,
        )
        print(f"{filepath} read successfully!")
        return atoms
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def process_scf_with_index(args):
    idx, filepath = args
    return (idx, process_scf(filepath))

if __name__ == "__main__":
    # Create list of (index, filepath) pairs to maintain order
    indexed_paths = list(enumerate(scfout_paths))
    
    # Process in parallel while maintaining order
    with Pool(processes=nproc) as pool:
        results = pool.map(process_scf_with_index, indexed_paths)
    
    # Sort results by original index and filter out None values
    results.sort(key=lambda x: x[0])
    atoms_list = [result[1] for result in results if result[1] is not None]
    
    # Write output file
    write(mlff_dat, atoms_list, format="extxyz")
    count = len(atoms_list)
    print(f"Successfully processed {count} structures out of {len(scfout_paths)}")
