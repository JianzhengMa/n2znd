#!/usr/bin/env python3
"""
Parallel Coordinate Aggregator with Ordered Output
Author: Jianzheng Ma, Fudan University, Shanghai, China, 2025-03-20
"""
import os
import argparse
from multiprocessing import Pool

def process_run(run_id: int) -> tuple:
    """Process single run directory and return results with sequence marker"""
    run_dir = f"./run{run_id:04d}"
    coord_file = os.path.join(run_dir, "coordinate.xyz")
    
    try:
        with open(coord_file, 'r') as f:
            lines = f.readlines()
    except IOError:
        return (run_id, None)
    
    try:
        atom_count = int(lines[0].strip())
        if len(lines) < 2 + atom_count:
            return (run_id, None)
    except ValueError:
        return (run_id, None)
    
    return (run_id, [
        f"{atom_count}\n",
        f"run{run_id:04d}\n",
        *lines[-atom_count:]
    ])

def main():
    # Configure command-line interface
    parser = argparse.ArgumentParser(
        description="Aggregate molecular coordinates from parallel simulations")
    parser.add_argument("-n", "--num-run", type=int, required=True,
                        help="Total number of simulation runs")
    args = parser.parse_args()
    
    # Initialize output file
    output_file = "traj_last_coord.xyz"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Parallel processing with order preservation
    with Pool(processes=os.cpu_count()) as pool:
        # Generate parallel tasks
        task_ids = range(1, args.num_run + 1)
        # Map-reduce processing with order control
        results = pool.map(process_run, task_ids)
    
    # Sequential ordered write operation
    with open(output_file, 'w') as f:
        for run_id, content in sorted(results, key=lambda x: x[0]):
            if content:
                f.writelines(content)

if __name__ == "__main__":
    main()
