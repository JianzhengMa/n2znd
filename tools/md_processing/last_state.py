#!/usr/bin/env python3
import os
import re

parent_directory = './'
output_file = 'last_state.dat'

results = []

folder_pattern = re.compile(r'^run0\d+$')

for folder_name in os.listdir(parent_directory):
    if folder_pattern.match(folder_name):
        folder_path = os.path.join(parent_directory, folder_name)
        file_path = os.path.join(folder_path, 'state.log')

        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    results.append(f"{folder_name}: {last_line}")

with open(output_file, 'w') as f:
    for result in results:
        f.write(result + '\n')

print(f"The results have been output to file {output_file}")
