#!/usr/bin/env python3
def process_file(file_path):
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        processed_lines = []
        for i, line in enumerate(lines, 1):
            # Remove leading/trailing whitespace and split columns
            parts = line.strip().split()
            
            # Validate column count
            if len(parts) < 2:
                print(f"Warning: Line {i} has less than 2 columns, skipped")
                processed_lines.append(line)  # Keep original line
                continue
            
            # Process first column data
            try:
                col1_value = float(parts[1])  # Modified to process first column
                if col1_value < 0:
                    col1_value += 360
                    # Format optimization: keep integer form for integers, 6 decimal places for decimals
                    parts[1] = str(int(col1_value)) if col1_value.is_integer() else f"{col1_value:.6f}"
            except ValueError:
                print(f"Error: Line {i} first column is not a valid number: '{parts[0]}'")
            
            # Recombine line data (maintain original delimiter structure)
            processed_lines.append(" ".join(parts) + "\n")
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)
        print(f"File processing completed: {file_path}")
    
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
    except Exception as e:
        print(f"Unknown error: {str(e)}")

# Usage example
process_file("mlmd-5213-5214.dat")
