#!/usr/bin/env python3
"""
Molecular Structure Optimization using Neural Network Potential

This program optimizes molecular structures using a trained NequIP neural network potential.
It reads an input structure, sets up the calculator, and performs geometry optimization.

Features:
- Uses ASE framework for molecular dynamics
- Implements LBFGS optimization with adjustable convergence criteria
- Supports multiple input formats (XYZ, CIF, etc.)
- Saves optimized structure and optimization trajectory in XYZ format
"""

import os
import time
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from nequip.ase import NequIPCalculator
import numpy as np

class XYZTrajectory:
    """
    Custom trajectory writer that saves in XYZ format
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        self.step = 0
        
    def write(self, atoms):
        """Write current structure to XYZ file"""
        write(self.file, atoms, format='xyz')
        self.step += 1
        
    def close(self):
        """Close the trajectory file"""
        self.file.close()

def optimize_structure(input_file, net_path, output_file="optimized.xyz", 
                      trajectory_file="optimization.xyz", log_file="optimization.log",
                      fmax=0.05, max_steps=200):
    """
    Optimize a molecular structure using neural network potential
    
    Parameters:
    input_file: Path to input structure file
    net_path: Path to trained NequIP model
    output_file: Output file for optimized structure
    trajectory_file: Trajectory file for optimization steps (XYZ format)
    log_file: Log file for optimization progress
    fmax: Convergence criterion (max force)
    max_steps: Maximum optimization steps
    
    Returns:
    optimized_atoms: Optimized structure
    """
    # 1. Read input structure
    atoms = read(input_file)
    n_atoms = len(atoms)
    print(f"System contains {n_atoms} atoms")
    
    # 2. Set up neural network potential calculator
    species_name = {s: s for s in atoms.get_chemical_symbols()}
    
    calc = NequIPCalculator.from_deployed_model(
        model_path=net_path,
        set_global_options=True,
        species_to_type_name=species_name
    )
    
    atoms.calc = calc
    
    # 3. Create XYZ trajectory writer
    xyz_traj = XYZTrajectory(trajectory_file)
    
    # Define callback function to write trajectory
    def write_trajectory():
        """Callback function to write current structure to XYZ trajectory"""
        xyz_traj.write(atoms)
    
    # 4. Perform optimization
    print(f"Starting optimization (fmax={fmax}, max_steps={max_steps})")
    start_time = time.time()
    
    # Use LBFGS optimizer with XYZ trajectory
    dyn = LBFGS(atoms, logfile=log_file)
    dyn.attach(write_trajectory, interval=1)
    dyn.run(fmax=fmax, steps=max_steps)
    
    # Close trajectory file
    xyz_traj.close()
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Optimization completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Total steps: {dyn.nsteps}")
    
    # 5. Save optimized structure
    write(output_file, atoms)
    print(f"Optimized structure saved to {output_file}")
    
    # 6. Calculate final energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))
    
    print(f"Final energy: {energy:.6f} eV")
    print(f"Maximum force: {max_force:.6f} eV/Å")
    
    return atoms

def format_time(seconds):
    """
    Format time in seconds to human-readable format
    
    Parameters:
    seconds: Time in seconds
    
    Returns:
    Formatted time string
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

if __name__ == "__main__":
    # ===== USER-CONFIGURABLE PARAMETERS =====
    # Input structure file
    input_file = "S0_E_MRSF.xyz"  # Replace with your input file
    
    # Neural network potential path
    net_path = "/public/home/majianzheng/LDMRM_ML/1-Templete_Mole_ML-2/5-ML_namd/net/1.pth"
    
    # Output files
    output_file = "optimized_structure.xyz"
    trajectory_file = "optimization_path.xyz"  # Now in XYZ format
    log_file = "optimization.log"
    
    # Optimization parameters
    fmax_value = 0.01    # Convergence criterion (eV/Å)
    max_steps_value = 300 # Maximum optimization steps
    
    # ===== OPTIMIZATION EXECUTION =====
    try:
        optimized_atoms = optimize_structure(
            input_file, 
            net_path,
            output_file=output_file,
            trajectory_file=trajectory_file,
            log_file=log_file,
            fmax=fmax_value,
            max_steps=max_steps_value
        )
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        exit(1)
    except RuntimeError as e:
        print(f"Runtime error during optimization: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)
    
    # Save summary
    with open('optimization_summary.txt', 'w') as f:
        f.write(f"Structure Optimization Summary\n")
        f.write(f"=============================\n")
        f.write(f"Input structure: {input_file}\n")
        f.write(f"Neural network potential: {net_path}\n")
        f.write(f"Convergence criterion (fmax): {fmax_value} eV/Å\n")
        f.write(f"Maximum steps: {max_steps_value}\n")
        f.write(f"Optimized structure: {output_file}\n")
        f.write(f"Optimization trajectory: {trajectory_file} (XYZ format)\n")
        f.write(f"Optimization log: {log_file}\n")
        f.write(f"\nFinal energy: {optimized_atoms.get_potential_energy():.6f} eV\n")
        
        forces = optimized_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        f.write(f"Maximum force: {max_force:.6f} eV/Å\n")
        
        # Calculate RMSD if reference structure provided
        # Uncomment if you have a reference structure
        # reference = read("reference.xyz")
        # rmsd = calculate_rmsd(optimized_atoms, reference)
        # f.write(f"RMSD from reference: {rmsd:.4f} Å\n")
        
        f.write("\nResults saved in:\n")
        f.write(f"- {output_file}: Optimized structure\n")
        f.write(f"- {trajectory_file}: Optimization trajectory (XYZ format)\n")
        f.write(f"- {log_file}: Optimization log\n")
        f.write(f"- optimization_summary.txt: This summary\n")
    
    print("\nOptimization completed successfully!")
