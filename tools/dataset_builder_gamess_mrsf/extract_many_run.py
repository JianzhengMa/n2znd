#!/usr/bin/env python3.9
"""
Author: Jianzheng Ma, Fudan University, Shanghai, China, 2025-03-20
"""
from ase.io import read, write
import os

BASE_DIR = './'
OUTPUT_FILE = 'output.xyz'
START_RUN = 1
END_RUN = 23
NSCF = 2000  # Number of Self-Consistent Field iterations
NSTEP = 50   # Steps to skip between frames

def clear_output_file(file_path):
    """Remove the output file if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)

def generate_output_file(base_dir, start_run, end_run, nscf, nstep, output_file):
    """Read coordinates from multiple folders and write them to a single output file."""
    # Create folder names based on the specified range
    folder_names = [f"run{i:04d}" for i in range(start_run, end_run + 1)]
    
    for folder in folder_names:
        folder_path = os.path.join(base_dir, folder)

        trajectory_file = os.path.join(folder_path, "coordinate.traj")
        coord_file_path = os.path.join(folder_path, "coordinate.xyz")
        if not os.path.isfile(trajectory_file):
            atoms_all = read(coord_file_path, index=":")
            write(trajectory_file, atoms_all)

        # Check if the coordinate file exists
        if os.path.isfile(trajectory_file):
            # Read the coordinate data and write to the output file
            atoms_list = read(trajectory_file, index=slice(0, nscf, nstep))
            for i,atoms in enumerate(atoms_list):
                write(output_file, atoms, format="xyz", append=True, comment=f"{folder} Step={nstep * i}")

def main():
    """Main function to execute the script."""
    clear_output_file(OUTPUT_FILE)
    generate_output_file(BASE_DIR, START_RUN, END_RUN, NSCF, NSTEP, OUTPUT_FILE)

if __name__ == "__main__":
    main()
