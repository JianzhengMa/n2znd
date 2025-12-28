import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from tqdm import tqdm
import matplotlib
import sys
import os
from joblib import Parallel, delayed

matplotlib.use('Agg')  # Non-interactive backend

# --- Core Calculation Functions ---

def calculate_structure_similarity(struct1, struct2):
    """
    Calculate the similarity between two structures.
    """
    # Quick check for atom count mismatch
    if len(struct1['elements']) != len(struct2['elements']):
        return 0.0
    
    # Procrustes Analysis
    try:
        # Centering
        centroid1 = struct1['coordinates'].mean(axis=0)
        centroid2 = struct2['coordinates'].mean(axis=0)
        coords1 = struct1['coordinates'] - centroid1
        coords2 = struct2['coordinates'] - centroid2
        
        # Calculate best overlap
        mtx1, mtx2, disparity = procrustes(coords1, coords2)
        
        # Similarity = 1 - normalized distance
        similarity = max(0.0, 1.0 - np.sqrt(disparity))
    except Exception:
        similarity = 0.0
    return similarity

def calculate_matrix_row(row_idx, all_structures):
    """
    Parallel task function: Calculates one row of the distance matrix.
    
    Args:
        row_idx: Index of the current row (i)
        all_structures: List of all structures
        
    Returns: 
        (row_idx, list_of_results) where result is (col_idx, distance)
    """
    target_struct = all_structures[row_idx]
    n_total = len(all_structures)
    results = []
    
    # Only calculate the upper triangle (j > i)
    # This avoids redundant calculations and the diagonal
    for j in range(row_idx + 1, n_total):
        sim = calculate_structure_similarity(target_struct, all_structures[j])
        dist = 1.0 - sim
        results.append((j, dist))
        
    return row_idx, results

# --- File Reading ---

def read_xyz_structures(xyz_file):
    """Read all structures from an XYZ file."""
    structures = []
    if not os.path.exists(xyz_file):
        print(f"Error: File {xyz_file} not found.")
        return []
        
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        try:
            line = lines[i].strip()
            if not line: 
                i += 1
                continue
            n_atoms = int(line)
            i += 2 # Skip atom count and comment line
            coordinates = []
            for j in range(n_atoms):
                if i + j >= len(lines): break
                tokens = lines[i + j].strip().split()
                if len(tokens) < 4: continue
                # We only need coordinates; element symbols are not used for calculation
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                coordinates.append([x, y, z])
            if coordinates:
                structures.append({
                    'elements': ['X']*n_atoms, # Placeholder
                    'coordinates': np.array(coordinates)
                })
            i += n_atoms
        except:
            i += 1
    return structures

# --- Main Logic ---

def run_multi_file_analysis(ref_file, target_files, output_prefix):
    # 1. Read all files
    print(f"--- Reading Reference File: {ref_file} ---")
    ref_structures = read_xyz_structures(ref_file)
    if not ref_structures: return
    n_ref = len(ref_structures)

    all_structures = ref_structures[:]
    # Store ranges to identify which structure belongs to which file later
    ranges = [(0, n_ref, "Reference: " + os.path.basename(ref_file))]
    current_idx = n_ref

    for t_file in target_files:
        print(f"--- Reading Target File: {t_file} ---")
        t_structs = read_xyz_structures(t_file)
        if t_structs:
            n_t = len(t_structs)
            all_structures.extend(t_structs)
            ranges.append((current_idx, current_idx + n_t, os.path.basename(t_file)))
            current_idx += n_t

    total_structs = len(all_structures)
    print(f"\nTotal structures: {total_structs}")
    print(f"Parallel processing using all available CPU cores...")

    # 2. Parallel calculation of the distance matrix
    # Using joblib's Parallel and delayed
    # n_jobs=-1 means use all available CPU cores
    # verbose=5 provides progress updates in the terminal
    
    row_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(calculate_matrix_row)(i, all_structures) 
        for i in range(total_structs)
    )

    # 3. Assemble the matrix
    print("Assembling distance matrix...")
    full_dist_matrix = np.zeros((total_structs, total_structs))
    
    for row_idx, col_data in row_results:
        for col_idx, dist in col_data:
            full_dist_matrix[row_idx, col_idx] = dist
            full_dist_matrix[col_idx, row_idx] = dist # Symmetric matrix

    # 4. Global MDS (also enabled for parallel processing)
    print("Performing Global MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1, max_iter=300)
    coords_global = mds.fit_transform(full_dist_matrix)

    # 5. Generate outputs (Plots and Data files)
    print("Generating outputs...")
    ref_start, ref_end, ref_name = ranges[0]
    ref_coords = coords_global[ref_start:ref_end]

    # Determine global limits for consistent plotting
    x_min, x_max = coords_global[:, 0].min(), coords_global[:, 0].max()
    y_min, y_max = coords_global[:, 1].min(), coords_global[:, 1].max()
    margin = 0.05
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)

    # Loop to generate comparison plots
    for i in range(1, len(ranges)):
        start, end, name = ranges[i]
        target_coords = coords_global[start:end]
        
        # Calculate average similarity
        sub_dist = full_dist_matrix[ref_start:ref_end, start:end]
        avg_sim = np.mean(1.0 - sub_dist)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(ref_coords[:, 0], ref_coords[:, 1], c='blue', alpha=0.6, label=ref_name, s=30)
        plt.scatter(target_coords[:, 0], target_coords[:, 1], c='red', alpha=0.6, label=name, s=30)
        plt.title(f"Comparison: {ref_name} vs {name}\nAvg Sim: {avg_sim:.3f}")
        plt.xlim(x_lim); plt.ylim(y_lim)
        plt.legend()
        plt.xlabel("MDS Dimension 1"); plt.ylabel("MDS Dimension 2")
        
        out_name = f"{output_prefix}_vs_{name.replace('.xyz', '')}.png"
        plt.savefig(out_name, dpi=300)
        plt.close()
        
        # Save .dat data for Origin
        dat_name = f"{output_prefix}_vs_{name.replace('.xyz', '')}.dat"
        max_len = max(len(ref_coords), len(target_coords))
        data_mat = np.full((max_len, 4), np.nan)
        data_mat[:len(ref_coords), 0:2] = ref_coords
        data_mat[:len(target_coords), 2:4] = target_coords
        header = f"{ref_name}_X\t{ref_name}_Y\t{name}_X\t{name}_Y"
        np.savetxt(dat_name, data_mat, fmt='%.6f', delimiter='\t', header=header, comments='')

    # Generate Summary Plot (All in one)
    plt.figure(figsize=(12, 10))
    plt.scatter(ref_coords[:, 0], ref_coords[:, 1], c='black', alpha=0.3, label=ref_name, s=40, zorder=1)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ranges)-1))
    for i in range(1, len(ranges)):
        start, end, name = ranges[i]
        plt.scatter(coords_global[start:end, 0], coords_global[start:end, 1], 
                   color=colors[i-1], alpha=0.7, label=name, s=20, zorder=2)
    plt.xlim(x_lim); plt.ylim(y_lim)
    plt.legend()
    plt.title(f"Global Comparison (Reference: {ref_name})")
    plt.savefig(f"{output_prefix}_ALL_combined.png", dpi=300)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python xyz_global_mds_parallel.py <ref.xyz> <target1.xyz> ... <output_prefix>")
    else:
        run_multi_file_analysis(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
