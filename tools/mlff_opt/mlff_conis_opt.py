#!/usr/bin/env python3
"""
Conical Intersection Optimization using Neural Network Potentials

This program optimizes conical intersection (CI) geometries between two electronic states
using neural network potentials for both states.

Features:
- Uses ASE framework for molecular dynamics
- Implements penalty function method for CI optimization
- Supports two different neural network potentials (one for each electronic state)
- Saves optimized CI geometry and energy information
- Allows direct specification of initial CI geometry
- Saves trajectory in XYZ format
"""

import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from nequip.ase import NequIPCalculator
from ase.calculators.calculator import Calculator, all_changes
import copy

class CI_Calculator(Calculator):
    """
    Custom calculator for conical intersection optimization
    
    This calculator combines two neural network potentials (for two electronic states)
    and implements a penalty function to optimize towards the conical intersection.
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calc_s0, calc_s1, penalty_factor=1.0, **kwargs):
        """
        Initialize CI calculator
        
        Parameters:
        calc_s0: Calculator for ground state (S0)
        calc_s1: Calculator for excited state (S1)
        penalty_factor: Weight for the energy difference penalty
        """
        Calculator.__init__(self, **kwargs)
        self.calc_s0 = calc_s0
        self.calc_s1 = calc_s1
        self.penalty_factor = penalty_factor
        self.energy_s0 = None
        self.energy_s1 = None
        self.forces_s0 = None
        self.forces_s1 = None
        self.energy_diff = None
        
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """
        Calculate energy and forces for CI optimization
        """
        # Calculate S0 energy and forces
        atoms_s0 = atoms.copy()
        atoms_s0.calc = self.calc_s0
        self.energy_s0 = atoms_s0.get_potential_energy()
        self.forces_s0 = atoms_s0.get_forces()
        
        # Calculate S1 energy and forces
        atoms_s1 = atoms.copy()
        atoms_s1.calc = self.calc_s1
        self.energy_s1 = atoms_s1.get_potential_energy()
        self.forces_s1 = atoms_s1.get_forces()
        
        # Energy difference
        self.energy_diff = self.energy_s1 - self.energy_s0
        
        # Total CI energy (penalty function)
        ci_energy = self.energy_s0 + self.penalty_factor * (self.energy_diff)**2
        
        # Forces (gradient of penalty function)
        ci_forces = self.forces_s0 + 2 * self.penalty_factor * self.energy_diff * (self.forces_s1 - self.forces_s0)
        
        # Store results
        self.results = {
            'energy': ci_energy,
            'forces': ci_forces,
            'energy_s0': self.energy_s0,
            'energy_s1': self.energy_s1,
            'energy_diff': self.energy_diff
        }

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

def optimize_conical_intersection(initial_ci_file, net_path_s0, net_path_s1, 
                                 penalty_factor=1.0, fmax=0.05, max_steps=200,
                                 output_prefix="ci"):
    """
    Optimize conical intersection geometry starting from a specified initial structure
    
    Parameters:
    initial_ci_file: Path to initial CI geometry file
    net_path_s0: Path to trained NequIP model for ground state (S0)
    net_path_s1: Path to trained NequIP model for excited state (S1)
    penalty_factor: Weight for the energy difference penalty
    fmax: Convergence criterion (max force)
    max_steps: Maximum optimization steps
    output_prefix: Prefix for output files
    
    Returns:
    optimized_atoms: Optimized CI geometry
    energy_s0: S0 energy at CI
    energy_s1: S1 energy at CI
    energy_diff: Energy difference at CI
    """
    # Load initial CI geometry
    atoms = read(initial_ci_file)
    n_atoms = len(atoms)
    print(f"System contains {n_atoms} atoms")
    
    # Set up calculators for both states
    species_name = {s: s for s in atoms.get_chemical_symbols()}
    
    calc_s0 = NequIPCalculator.from_deployed_model(
        model_path=net_path_s0,
        set_global_options=True,
        species_to_type_name=species_name
    )
    
    calc_s1 = NequIPCalculator.from_deployed_model(
        model_path=net_path_s1,
        set_global_options=True,
        species_to_type_name=species_name
    )
    
    # Calculate initial energies
    atoms.calc = calc_s0
    initial_energy_s0 = atoms.get_potential_energy()
    
    atoms.calc = calc_s1
    initial_energy_s1 = atoms.get_potential_energy()
    initial_energy_diff = initial_energy_s1 - initial_energy_s0
    
    print(f"Initial S0 energy: {initial_energy_s0:.6f} eV")
    print(f"Initial S1 energy: {initial_energy_s1:.6f} eV")
    print(f"Initial energy difference: {initial_energy_diff:.6f} eV")
    
    # Create CI calculator
    ci_calc = CI_Calculator(calc_s0, calc_s1, penalty_factor=penalty_factor)
    atoms.calc = ci_calc
    
    # Create XYZ trajectory writer
    xyz_traj = XYZTrajectory(f'{output_prefix}_opt.xyz')
    
    # Define callback function to write trajectory
    def write_trajectory():
        """Callback function to write current structure to XYZ trajectory"""
        xyz_traj.write(atoms)
    
    # Perform optimization
    print(f"Starting CI optimization (penalty_factor={penalty_factor}, fmax={fmax})")
    start_time = time.time()
    
    # Use LBFGS optimizer with XYZ trajectory
    dyn = LBFGS(atoms, logfile=f'{output_prefix}_opt.log')
    dyn.attach(write_trajectory, interval=1)
    dyn.run(fmax=fmax, steps=max_steps)
    
    # Close trajectory file
    xyz_traj.close()
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Optimization completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Total steps: {dyn.nsteps}")
    
    # Get final results
    energy_s0 = ci_calc.energy_s0
    energy_s1 = ci_calc.energy_s1
    energy_diff = ci_calc.energy_diff
    
    print(f"Final S0 energy: {energy_s0:.6f} eV")
    print(f"Final S1 energy: {energy_s1:.6f} eV")
    print(f"Final energy difference: {energy_diff:.6f} eV")
    
    # Save optimized structure
    write(f'{output_prefix}_optimized.xyz', atoms)
    print(f"Optimized CI geometry saved to {output_prefix}_optimized.xyz")
    
    # Save energy information
    with open(f'{output_prefix}_energies.dat', 'w') as f:
        f.write(f"S0_energy (eV)\tS1_energy (eV)\tEnergy_difference (eV)\n")
        f.write(f"{initial_energy_s0:.8f}\t{initial_energy_s1:.8f}\t{initial_energy_diff:.8f}\n")
        f.write(f"{energy_s0:.8f}\t{energy_s1:.8f}\t{energy_diff:.8f}\n")
    
    return atoms, energy_s0, energy_s1, energy_diff

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
    # Record start time
    start_time = time.time()
    
    # ===== USER-CONFIGURABLE PARAMETERS =====
    # Initial CI geometry file
    initial_ci_file = "CI1_MRSF.xyz"  # Replace with your initial CI structure
    
    # Neural network potential paths
    net_path_s0 = "/public/home/majianzheng/LDMRM_ML/1-Templete_Mole_ML-2/5-ML_namd/net/1.pth"  # Ground state model
    net_path_s1 = "/public/home/majianzheng/LDMRM_ML/1-Templete_Mole_ML-2/5-ML_namd/net/2.pth"  # Excited state model
    
    # Optimization parameters
    penalty_factor = 20      # Weight for energy difference penalty
    fmax_value = 0.01         # Convergence criterion (eV/Å)
    max_steps_value = 300      # Maximum optimization steps
    
    # Output prefix
    output_prefix = "ci_optimization"
    
    # ===== OPTIMIZE CONICAL INTERSECTION =====
    print("\n===== Optimizing Conical Intersection =====")
    optimized_atoms, energy_s0, energy_s1, energy_diff = optimize_conical_intersection(
        initial_ci_file,
        net_path_s0,
        net_path_s1,
        penalty_factor=penalty_factor,
        fmax=fmax_value,
        max_steps=max_steps_value,
        output_prefix=output_prefix
    )
    
    # ===== SAVE SUMMARY =====
    elapsed = time.time() - start_time
    print(f"\nTotal calculation time: {format_time(elapsed)}")
    
    with open(f'{output_prefix}_summary.txt', 'w') as f:
        f.write(f"Conical Intersection Optimization Summary\n")
        f.write(f"========================================\n")
        f.write(f"Initial CI geometry: {initial_ci_file}\n")
        f.write(f"S0 neural network potential: {net_path_s0}\n")
        f.write(f"S1 neural network potential: {net_path_s1}\n")
        f.write(f"Penalty factor: {penalty_factor}\n")
        f.write(f"Convergence criterion (fmax): {fmax_value} eV/Å\n")
        f.write(f"Maximum steps: {max_steps_value}\n")
        f.write(f"Calculation time: {format_time(elapsed)}\n")
        f.write(f"\nFinal S0 energy: {energy_s0:.6f} eV\n")
        f.write(f"Final S1 energy: {energy_s1:.6f} eV\n")
        f.write(f"Final energy difference: {energy_diff:.6f} eV\n")
        f.write(f"\nResults saved in:\n")
        f.write(f"- {output_prefix}_optimized.xyz: Optimized CI geometry\n")
        f.write(f"- {output_prefix}_opt.xyz: Optimization trajectory (XYZ format)\n")
        f.write(f"- {output_prefix}_energies.dat: Energy information\n")
        f.write(f"- {output_prefix}_opt.log: Optimization log\n")
        f.write(f"- {output_prefix}_summary.txt: This summary\n")
    
    print("\nConical intersection optimization completed successfully!")
    print(f"Final energy difference: {energy_diff:.6f} eV")
