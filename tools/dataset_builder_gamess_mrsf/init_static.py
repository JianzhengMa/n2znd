import os
from ase.io import read

# Constants
STATIC_DIR = "./run"
NSTEP = 1
TRAJ_FILE = "output_2100.xyz"
TEMPLATE = "templete.inp"

# Read all structures from XYZ file
structures = read(TRAJ_FILE, index=slice(0,None,NSTEP))

# Read template file content
with open(TEMPLATE, 'r') as f:
    template_lines = f.readlines()

# Find positions of $data and $end markers
data_start = None
data_end = None
for i, line in enumerate(template_lines):
    stripped = line.strip().lower()
    if stripped == '$data':
        data_start = i
    elif stripped == '$end' and data_start is not None and data_end is None:
        data_end = i

if data_start is None or data_end is None:
    raise ValueError("Missing $data or $end markers in template file.")

# Split template into three parts: before coordinates, title lines, and after coordinates
pre_data = template_lines[:data_start + 1]  # Includes $data line
post_data = template_lines[data_end:]       # Includes $end and following content
data_block = template_lines[data_start + 1 : data_end]
title_lines = data_block[:2]                # Title lines (e.g., "LDMRM\nC1\n")

# Process each structure
for idx, atoms in enumerate(structures, start=1):
    # Create directory for this structure
    dir_name = os.path.join(STATIC_DIR, str(idx) )
    os.makedirs(dir_name, exist_ok=True)
    
    # Generate new coordinate lines
    new_atom_lines = []
    for atom in atoms:
        symbol = atom.symbol
        atomic_number = atom.number
        x, y, z = atom.position
        # Format coordinate line with fixed-width alignment
        line = f"   {symbol}   {atomic_number}     {x:14.10f}      {y:14.10f}      {z:14.10f}\n"
        new_atom_lines.append(line)
    
    # Construct new input file content
    new_content = []
    new_content.extend(pre_data)      # Header before coordinates
    new_content.extend(title_lines)   # Title lines
    new_content.extend(new_atom_lines) # New coordinates
    new_content.extend(post_data)     # Footer after coordinates
    
    # Write to input file
    input_file = os.path.join(dir_name, 'input.inp')
    with open(input_file, 'w') as f_out:
        f_out.writelines(new_content)

print("All input files generated successfully.")
