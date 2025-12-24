#!/usr/bin/env python3
"""
Reaction Path Calculation for Large Molecules using Neural Network Potential and NEB Method

This program calculates the minimum energy path (MEP) between two molecular configurations
using the Nudged Elastic Band (NEB) method with a neural network potential (NequIP).

Features:
- Uses ASE framework for molecular dynamics and NEB implementation
- Implements two-stage optimization with adjustable convergence criteria
- Generates energy profile visualization
- Supports parallel computation for efficiency
- All results are saved directly to files without displaying
- Handles different coordinate origins through structure alignment
- Added control for transition state output
- Added explicit control for endpoint optimization
- Ensures optimized endpoints are included in final reaction_path.xyz
"""

import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.mep import NEB
from ase.optimize import LBFGS
from nequip.ase import NequIPCalculator
from ase.io import read, write
from ase.parallel import world, parprint
import copy

def align_structures(start_coords, end_coords):
    """
    Align two structures using Kabsch algorithm to minimize RMSD
    
    Parameters:
    start_coords: Coordinates of starting structure (n_atoms x 3)
    end_coords: Coordinates of ending structure (n_atoms x 3)
    
    Returns:
    aligned_start: Centered starting structure
    aligned_end: Aligned and centered ending structure
    """
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
    
    # Calculate RMSD after alignment
    rmsd = np.sqrt(np.mean(np.square(centered_start - np.dot(centered_end, R))))
    parprint(f"Structures aligned with RMSD: {rmsd:.4f} Å")
    
    return centered_start + start_center, aligned_end

def optimize_endpoint(atoms, fmax=0.05, label="endpoint"):
    """
    Optimize a single endpoint structure
    
    Parameters:
    atoms: ASE Atoms object to optimize
    fmax: Convergence criterion (max force)
    label: Label for output files
    
    Returns:
    optimized: Optimized structure
    """
    parprint(f"Optimizing {label} structure...")
    dyn = LBFGS(atoms, trajectory=f'opt_{label}.traj', logfile=f'opt_{label}.log')
    dyn.run(fmax=fmax)
    return atoms.copy()

def calculate_reaction_path_large(start_file, end_file, net_path, num_images=7, max_steps=300, 
                                 output_ts=True, fmax_coarse=0.2, fmax_fine=0.05,
                                 optimize_endpoints=False, endpoint_fmax=0.05):
    """
    Calculate reaction path for molecules using NEB method with neural network potential
    
    Parameters:
    start_file: XYZ file containing starting structure
    end_file: XYZ file containing ending structure
    net_path: Path to trained NequIP model
    num_images: Number of images in the path (including endpoints)
    max_steps: Maximum optimization steps
    output_ts: Whether to output transition state structure
    fmax_coarse: Convergence criterion for coarse optimization (max force)
    fmax_fine: Convergence criterion for fine optimization (max force)
    optimize_endpoints: Whether to optimize endpoint structures before NEB
    endpoint_fmax: Convergence criterion for endpoint optimization
    
    Returns:
    images: List of optimized structures along the path
    energies: Potential energies of each image
    optimizer1: First stage optimizer object
    optimizer2: Second stage optimizer object
    """
    # 1. Read starting and ending structures
    start_atoms = read(start_file)
    end_atoms = read(end_file)
    n_atoms = len(start_atoms)
    parprint(f"System contains {n_atoms} atoms")
    
    # Check if structures have same number of atoms
    if len(start_atoms) != len(end_atoms):
        raise ValueError("Start and end structures have different number of atoms")
    
    # Get coordinates as numpy arrays
    start_coords = start_atoms.positions
    end_coords = end_atoms.positions
    
    # 2. Align structures to common coordinate system
    aligned_start, aligned_end = align_structures(start_coords, end_coords)
    
    # Update atoms objects with aligned coordinates
    start_atoms_aligned = start_atoms.copy()
    start_atoms_aligned.positions = aligned_start
    end_atoms_aligned = end_atoms.copy()
    end_atoms_aligned.positions = aligned_end
    
    # 3. Optimize endpoints if requested
    if optimize_endpoints:
        # Set up calculator for endpoint optimization
        species_name = {s: s for s in start_atoms.get_chemical_symbols()}
        endpoint_calc = NequIPCalculator.from_deployed_model(
            model_path=net_path,
            set_global_options=True,
            species_to_type_name=species_name
        )
        
        # Optimize starting structure
        start_atoms_aligned.calc = endpoint_calc
        start_atoms_aligned = optimize_endpoint(start_atoms_aligned, 
                                               fmax=endpoint_fmax, 
                                               label="start")
        write('optimized_start.xyz', start_atoms_aligned)
        
        # Optimize ending structure
        end_atoms_aligned.calc = endpoint_calc
        end_atoms_aligned = optimize_endpoint(end_atoms_aligned, 
                                            fmax=endpoint_fmax, 
                                            label="end")
        write('optimized_end.xyz', end_atoms_aligned)
        
        # Update coordinates after optimization
        aligned_start = start_atoms_aligned.positions
        aligned_end = end_atoms_aligned.positions
    else:
        parprint("Skipping endpoint optimization as requested")
    
    # 4. Create initial path with linear interpolation
    images = [start_atoms_aligned.copy()]
    for i in range(1, num_images-1):
        fraction = i / (num_images-1)
        new_atoms = start_atoms_aligned.copy()
        new_pos = aligned_start + fraction * (aligned_end - aligned_start)
        new_atoms.positions = new_pos
        images.append(new_atoms)
    images.append(end_atoms_aligned.copy())
    
    # 5. Set up neural network potential calculator
    species_name = {s: s for s in start_atoms.get_chemical_symbols()}
    
    # Create master calculator and clones for each image
    master_calc = NequIPCalculator.from_deployed_model(
        model_path=net_path,
        set_global_options=True,
        species_to_type_name=species_name
    )
    
    # Create deep copies of the calculator for each image
    calculators = []
    for _ in images:
        # Create a deep copy of the calculator
        calc = copy.deepcopy(master_calc)
        calculators.append(calc)
    
    # 6. Assign calculators to images
    for i, image in enumerate(images):
        image.calc = calculators[i]
    
    # 7. Create NEB object with parallel support
    neb = NEB(images, climb=True, parallel=(world.size > 1))
    
    # 8. Interpolate initial path using IDPP for better initial guess
    neb.interpolate(method='idpp')
    
    # 9. Two-stage optimization with adjustable convergence criteria
    parprint(f"Starting stage 1 optimization (coarse, fmax={fmax_coarse})")
    optimizer1 = LBFGS(neb, trajectory='neb_coarse.traj', logfile='neb_coarse.log')
    optimizer1.run(fmax=fmax_coarse, steps=max_steps//2)
    
    parprint(f"Starting stage 2 optimization (fine, fmax={fmax_fine})")
    optimizer2 = LBFGS(neb, trajectory='neb_fine.traj', logfile='neb_fine.log')
    optimizer2.run(fmax=fmax_fine, steps=max_steps//2)
    
    # 10. Calculate path energies
    energies = [image.get_potential_energy() for image in images]
    
    # 11. Save energy data
    with open('reaction_energy.dat', 'w') as f:
        for i, energy in enumerate(energies):
            f.write(f"{i}\t{energy:.8f}\n")
    
    # 12. Output transition state if requested
    if output_ts:
        ts_index = np.argmax(energies)
        ts_atoms = images[ts_index]
        write('transition_state.xyz', ts_atoms)
        parprint(f"Transition state saved as 'transition_state.xyz' (image #{ts_index})")
    
    # 13. Save optimized path - ensure optimized endpoints are included
    write('reaction_path.xyz', images)
    parprint("Reaction path saved with optimized endpoints")
    
    return images, energies, optimizer1, optimizer2

def save_energy_profile(energies):
    """
    Save energy profile plot without displaying it
    
    Parameters:
    energies: List of potential energies for each image
    """
    # Create reaction coordinate
    reaction_coordinate = np.linspace(0, 1, len(energies))
    
    # Create energy profile plot
    plt.figure(figsize=(10, 6))
    plt.plot(reaction_coordinate, energies, 'o-', linewidth=2)
    plt.xlabel('Reaction Coordinate')
    plt.ylabel('Energy (eV)')
    plt.title('Reaction Energy Profile')
    plt.grid(True)
    plt.savefig('energy_profile.png', dpi=300)
    plt.close()

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
    # Input files and model path
    start_file = "trans.xyz"
    end_file = "CI1.xyz"
    net_path = "/public/home/majianzheng/LDMRM_ML/3-Retinal_LDMRM/6-ml-znmd/net/1.pth"
    
    # Path configuration
    num_images_value = 25
    max_steps_value = 1000
    
    # Convergence criteria (adjust these values as needed)
    fmax_coarse = 0.3  # Convergence criterion for coarse optimization (eV/Å)
    fmax_fine = 0.15   # Convergence criterion for fine optimization (eV/Å)
    
    # Endpoint optimization control
    optimize_endpoints = False  # Set to True to optimize endpoints, False to skip
    endpoint_fmax = 0.05       # Convergence for endpoint optimization
    
    # Output control
    output_transition_state = False
    
    # ===== CALCULATION EXECUTION =====
    # Calculate reaction path
    try:
        images, energies, opt1, opt2 = calculate_reaction_path_large(
            start_file, end_file, net_path, 
            num_images=num_images_value,
            max_steps=max_steps_value,
            output_ts=output_transition_state,
            fmax_coarse=fmax_coarse,
            fmax_fine=fmax_fine,
            optimize_endpoints=optimize_endpoints,
            endpoint_fmax=endpoint_fmax
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error calculating reaction path: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)
    
    # Save energy profile
    save_energy_profile(energies)
    
    # Calculate and print execution time
    elapsed = time.time() - start_time
    print(f"\nCalculation completed! Total time: {format_time(elapsed)}")
    
    # Calculate energy metrics
    barrier_forward = max(energies) - energies[0]
    barrier_reverse = max(energies) - energies[-1]
    reaction_energy = energies[-1] - energies[0]
    ts_index = np.argmax(energies)
    total_steps = opt1.nsteps + opt2.nsteps
    
    # Print key results
    print(f"Forward energy barrier: {barrier_forward:.3f} eV")
    print(f"Reverse energy barrier: {barrier_reverse:.3f} eV")
    print(f"Reaction energy: {reaction_energy:.3f} eV")
    print(f"Transition state at image #{ts_index}")
    print(f"Total optimization steps: {total_steps}")
    
    # Save final summary
    with open('calculation_summary.txt', 'w') as f:
        f.write(f"Reaction Path Calculation Summary\n")
        f.write(f"================================\n")
        f.write(f"Start structure: {start_file}\n")
        f.write(f"End structure: {end_file}\n")
        f.write(f"Neural network potential: {net_path}\n")
        f.write(f"Number of images: {len(images)}\n")
        f.write(f"Total optimization steps: {total_steps}\n")
        f.write(f"Calculation time: {format_time(elapsed)}\n")
        f.write(f"Coarse convergence (fmax): {fmax_coarse} eV/Å\n")
        f.write(f"Fine convergence (fmax): {fmax_fine} eV/Å\n")
        if optimize_endpoints:
            f.write(f"Endpoints optimized with fmax: {endpoint_fmax} eV/Å\n")
            f.write(f"Optimized start structure saved as: optimized_start.xyz\n")
            f.write(f"Optimized end structure saved as: optimized_end.xyz\n")
        else:
            f.write(f"Endpoint optimization: Skipped\n")
        f.write(f"Forward energy barrier: {barrier_forward:.3f} eV\n")
        f.write(f"Reverse energy barrier: {barrier_reverse:.3f} eV\n")
        f.write(f"Reaction energy: {reaction_energy:.3f} eV\n")
        f.write(f"Transition state at image #{ts_index}\n")
        if output_transition_state:
            f.write(f"Transition state saved as 'transition_state.xyz'\n")
        f.write(f"\nResults saved in:\n")
        f.write(f"- reaction_path.xyz: Optimized structures including endpoints\n")
        f.write(f"- reaction_energy.dat: Energy values\n")
        f.write(f"- energy_profile.png: Energy profile plot\n")
        f.write(f"- neb_coarse.traj: Coarse optimization trajectory\n")
        f.write(f"- neb_fine.traj: Fine optimization trajectory\n")
        if optimize_endpoints:
            f.write(f"- optimized_start.xyz: Optimized starting structure\n")
            f.write(f"- optimized_end.xyz: Optimized ending structure\n")
        if output_transition_state:
            f.write(f"- transition_state.xyz: Transition state structure\n")
