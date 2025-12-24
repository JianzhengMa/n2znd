#!/usr/bin/env python3
"""
Transition State Optimization using Neural Network Potential
Transition state optimization program supporting Dimer and FIRE methods
"""

import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import LBFGS, FIRE
from nequip.ase import NequIPCalculator
from ase.io import read, write
from ase.parallel import world, parprint
import copy
import traceback

def setup_calculator(atoms, net_path):
    """
    Set up neural network potential calculator
    """
    species_name = {s: s for s in atoms.get_chemical_symbols()}
    calculator = NequIPCalculator.from_deployed_model(
        model_path=net_path,
        set_global_options=True,
        species_to_type_name=species_name
    )
    return calculator

def optimize_ts_dimer(initial_guess, net_path, fmax=0.05, max_steps=200):
    """
    Optimize transition state using Dimer method 
    Dimer method finds saddle points on potential energy surface through rotation and translation operations
    """
    parprint("Starting transition state optimization using Dimer method...")
    
    # Set up calculator
    calculator = setup_calculator(initial_guess, net_path)
    initial_guess.calc = calculator
    
    try:
        from ase.mep import MinModeAtoms, MinModeTranslate
        
        # Create MinModeAtoms object
        minmode_atoms = MinModeAtoms(initial_guess)
        
        # Set displacement vector - This is a key parameter of the Dimer method 
        pos = minmode_atoms.get_positions()
        center = np.mean(pos, axis=0)
        displacement = pos - center
        displacement = displacement / np.linalg.norm(displacement) * 0.01
        minmode_atoms.displacement_vector = displacement
        
        # Create Dimer optimizer
        dimer = MinModeTranslate(minmode_atoms)
        
        # Run Dimer optimization
        dimer.run(fmax=fmax, steps=max_steps)
        
        ts_optimized = minmode_atoms.copy()
        return ts_optimized, dimer
        
    except Exception as e:
        parprint(f"Dimer method failed: {e}")
        parprint("Falling back to FIRE method...")
        return optimize_ts_fire(initial_guess, net_path, fmax, max_steps)

def optimize_ts_fire(initial_guess, net_path, fmax=0.05, max_steps=200):
    """
    Optimize transition state using FIRE method 
    FIRE is an efficient optimization algorithm suitable for potential energy surface optimization
    """
    parprint("Starting transition state optimization using FIRE method...")
    
    # Set up calculator
    calculator = setup_calculator(initial_guess, net_path)
    initial_guess.calc = calculator
    
    # Use FIRE optimizer
    optimizer = FIRE(initial_guess, 
                    trajectory='ts_fire_optimization.traj',
                    logfile='ts_fire_optimization.log')
    
    # Run optimization
    optimizer.run(fmax=fmax, steps=max_steps)
    
    return initial_guess, optimizer

def pre_optimize_structure(atoms, net_path, fmax_preopt=0.1, steps_preopt=50):
    """
    Pre-optimize structure to improve initial guess quality
    """
    parprint("Pre-optimizing structure to improve initial guess...")
    
    calculator = setup_calculator(atoms, net_path)
    atoms.calc = calculator
    
    # Pre-optimize with loose convergence criteria
    dyn = FIRE(atoms, trajectory='pre_optimization.traj')
    dyn.run(fmax=fmax_preopt, steps=steps_preopt)
    
    parprint(f"Pre-optimization completed. Energy: {atoms.get_potential_energy():.3f} eV")
    return atoms

def verify_transition_state(atoms, net_path, displacement=0.01):
    """
    Verify transition state through vibrational frequency analysis 
    """
    from ase.vibrations import Vibrations
    
    parprint("Verifying transition state with vibrational frequency analysis...")
    
    # Set up calculator
    calculator = setup_calculator(atoms, net_path)
    atoms.calc = calculator
    
    # Calculate vibrational frequencies
    vib = Vibrations(atoms, delta=displacement)
    vib.run()
    
    # Get frequencies
    frequencies = vib.get_frequencies()
    
    # Count imaginary frequencies (negative values indicate imaginary frequencies)
    imaginary_count = sum(1 for f in frequencies if f < 0)
    
    parprint(f"Found {imaginary_count} imaginary frequency(ies)")
    
    # Save vibrational analysis results
    vib.summary(log='vibrational_summary.txt')
    if imaginary_count > 0:
        vib.write_mode(-1)  # Save vibration mode corresponding to the largest imaginary frequency
    
    # A true transition state should have only one imaginary frequency 
    return imaginary_count == 1, frequencies

def calculate_energy_profile(ts_structure, reactant, product, net_path):
    """
    Calculate reaction barrier and energy change
    """
    parprint("Calculating reaction energy profile...")
    
    # Set up calculator
    calculator = setup_calculator(ts_structure, net_path)
    
    ts_structure.calc = calculator
    reactant.calc = setup_calculator(reactant, net_path)
    product.calc = setup_calculator(product, net_path)
    
    # Calculate energies
    e_ts = ts_structure.get_potential_energy()
    e_reactant = reactant.get_potential_energy()
    e_product = product.get_potential_energy()
    
    # Calculate barriers and energy changes
    forward_barrier = e_ts - e_reactant
    backward_barrier = e_ts - e_product
    enthalpy_change = e_product - e_reactant
    
    parprint(f"Forward barrier: {forward_barrier:.3f} eV")
    parprint(f"Backward barrier: {backward_barrier:.3f} eV")
    parprint(f"Reaction enthalpy: {enthalpy_change:.3f} eV")
    
    return forward_barrier, backward_barrier, enthalpy_change

def format_time(seconds):
    """Format time display"""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def generate_optimization_report(ts_optimized, optimizer, is_ts, frequencies, elapsed,
                               initial_guess, net_path, method_used, energy_profile=None):
    """Generate optimization report"""
    
    report = f"""
TRANSITION STATE OPTIMIZATION REPORT
====================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Optimization Method: {method_used}
Potential Function: {os.path.basename(net_path)}

OPTIMIZATION SUMMARY
-------------------
Total time: {format_time(elapsed)}
Optimization steps: {getattr(optimizer, 'nsteps', 'N/A')}
Final force convergence: {getattr(optimizer, 'fmax', 'N/A'):.3f} eV/Å
Atoms in optimized structure: {len(ts_optimized)}

TRANSITION STATE VALIDATION
--------------------------
Transition state confirmed: {'YES' if is_ts else 'NO'}
Number of imaginary frequencies: {sum(1 for f in frequencies if f < 0)}
"""
    
    if energy_profile:
        forward_barrier, backward_barrier, enthalpy_change = energy_profile
        report += f"""
ENERGY PROFILE
-------------
Forward reaction barrier: {forward_barrier:.3f} eV
Backward reaction barrier: {backward_barrier:.3f} eV
Reaction enthalpy: {enthalpy_change:.3f} eV
"""
    
    imaginary_freqs = [f for f in frequencies if f < 0]
    if imaginary_freqs:
        report += f"Largest imaginary frequency: {imaginary_freqs[0]:.2f} cm⁻¹\n"
    
    report += f"""
FILES GENERATED
---------------
- optimized_transition_state.xyz : Optimized TS structure
- ts_{method_used}_optimization.traj : Optimization trajectory
- ts_{method_used}_optimization.log : Optimization log
- vibrational_summary.txt : Frequency analysis results
"""
    
    with open('ts_optimization_report.txt', 'w') as f:
        f.write(report)

def main():
    start_time = time.time()
    
    # ===== User-configurable parameters =====
    initial_guess_file = "transition_state.xyz"  # Initial guess for transition state
    net_path = "/public/home/majianzheng/LDMRM_ML/2-Oxindole_Without_OME-2/4-MLMD/net/1.pth"
    
    # Optimization method selection: 'dimer' or 'fire'
    optimization_method = "fire"
    
    # Convergence criteria
    fmax = 0.05  # Force convergence criterion (eV/Å)
    max_steps = 200  # Maximum optimization steps
    
    # Pre-optimization parameters
    pre_optimize = True  # Whether to perform pre-optimization
    fmax_preopt = 0.1    # Pre-optimization convergence criterion
    
    # ===== Execute optimization =====
    parprint("=== Transition State Optimization ===")
    parprint(f"Method: {optimization_method}")
    parprint(f"Potential: {os.path.basename(net_path)}")
    
    # Read initial structure
    try:
        initial_guess = read(initial_guess_file)
        parprint(f"Initial structure: {len(initial_guess)} atoms from {initial_guess_file}")
    except FileNotFoundError:
        parprint(f"Error: File {initial_guess_file} not found")
        return
    
    # Pre-optimization (optional)
    if pre_optimize:
        try:
            initial_guess = pre_optimize_structure(initial_guess, net_path, fmax_preopt)
        except Exception as e:
            parprint(f"Pre-optimization warning: {e}")
            parprint("Continuing with original structure...")
    
    # Transition state optimization
    try:
        if optimization_method == "dimer":
            parprint("Using Dimer method for transition state search...")
            ts_optimized, optimizer = optimize_ts_dimer(
                initial_guess, net_path, fmax=fmax, max_steps=max_steps
            )
        elif optimization_method == "fire":
            parprint("Using FIRE method for transition state search...")
            ts_optimized, optimizer = optimize_ts_fire(
                initial_guess, net_path, fmax=fmax, max_steps=max_steps
            )
        else:
            parprint(f"Error: Unknown optimization method '{optimization_method}'")
            parprint("Please choose 'dimer' or 'fire'")
            return
    except Exception as e:
        parprint(f"Optimization failed: {e}")
        traceback.print_exc()
        return
    
    # Save optimized structure
    write('optimized_transition_state.xyz', ts_optimized)
    parprint("Optimized structure saved as 'optimized_transition_state.xyz'")
    
    # Verify transition state
    is_ts, frequencies = verify_transition_state(ts_optimized, net_path)
    
    # Calculate energy profile (if reactant and product structures are available)
    energy_profile = None
    reactant_file = "EM.xyz"
    product_file = "EP.xyz"
    
    if os.path.exists(reactant_file) and os.path.exists(product_file):
        try:
            reactant = read(reactant_file)
            product = read(product_file)
            energy_profile = calculate_energy_profile(ts_optimized, reactant, product, net_path)
        except Exception as e:
            parprint(f"Energy profile calculation skipped: {e}")
    
    # Generate report
    elapsed = time.time() - start_time
    generate_optimization_report(
        ts_optimized, optimizer, is_ts, frequencies, elapsed,
        initial_guess, net_path, optimization_method, energy_profile
    )
    
    # Output final results
    parprint(f"\n=== Optimization Completed in {format_time(elapsed)} ===")
    parprint(f"Transition state confirmed: {is_ts}")
    
    imaginary_freqs = [f for f in frequencies if f < 0]
    if imaginary_freqs:
        parprint(f"Imaginary frequency: {imaginary_freqs[0]:.2f} cm⁻¹")
    
    if energy_profile:
        forward_barrier, backward_barrier, enthalpy_change = energy_profile
        parprint(f"Forward barrier: {forward_barrier:.3f} eV")
        parprint(f"Reaction enthalpy: {enthalpy_change:.3f} eV")
    
    parprint("Detailed report saved in 'ts_optimization_report.txt'")

if __name__ == "__main__":
    main()
