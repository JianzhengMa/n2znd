#!/usr/bin/env python3
"""
Author: Jianzheng Ma, Fudan University, Shanghai, China, 2025-03-20
"""
import os
from optparse import OptionParser

def parse_arguments():
    """Parse command line arguments"""
    parser = OptionParser()
    parser.add_option("-n", dest="num_run", type="int", default=1000,
                     help="Number of run directories, default: %default")
    options, _ = parser.parse_args()
    return options.num_run

def get_valid_directories(num_run):
    """Find valid directories containing state.log files"""
    return [f"./run{str(i).zfill(4)}" for i in range(1, num_run+1) 
           if os.path.exists(f"./run{str(i).zfill(4)}/state.log")]

def process_state_logs(run_dirs):
    """Process state.log files and collect valid directories"""
    valid_dirs = []
    steps = []
    
    for dir_path in run_dirs:
        try:
            temp_path = os.path.join(dir_path, "temp_state.dat")
            # Clean previous temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Read and validate state.log
            with open(os.path.join(dir_path, "state.log")) as f:
                lines = f.readlines()
                if lines:
                    steps.append(len(lines))
                    valid_dirs.append(dir_path)
                    
        except Exception as e:
            print(f"Error processing {dir_path}: {str(e)}")
    
    if not valid_dirs:
        raise ValueError("No valid data directories found")
    
    return valid_dirs, min(steps)

def generate_time_series(valid_dirs, step_min):
    """Generate time series data for valid directories"""
    try:
        for dir_path in valid_dirs:
            input_file = os.path.join(dir_path, "state.log")
            output_file = os.path.join(dir_path, "temp_state.dat")
            
            with open(input_file) as f_in, open(output_file, "w") as f_out:
                lines = f_in.readlines()[:step_min]
                states = [int(line.split()[2]) for line in lines]
                times = [round(float(line.split()[1]), 1) for line in lines]
                
                for idx in range(step_min-1):
                    current_time = times[idx]
                    next_time = times[idx+1]
                    state = states[idx]
                    
                    t = current_time
                    while t < next_time:
                        f_out.write(f"{t:<10.1f}{state:<5}\n")
                        t = round(t + 0.1, 1)
    except Exception as e:
        raise RuntimeError(f"Time series generation failed: {str(e)}")

def calculate_statistics(valid_dirs):
    """Calculate state occupation statistics"""
    temp_files = [os.path.join(d, "temp_state.dat") for d in valid_dirs]
    
    # Determine minimum time steps
    with open(temp_files[0]) as f:
        step_new_min = len(f.readlines())
    
    # Initialize statistics
    stats = [{"S1": 0, "S2": 0} for _ in range(step_new_min)]
    total_runs = len(temp_files)
    
    # Aggregate data
    for file_path in temp_files:
        with open(file_path) as f:
            lines = f.readlines()[:step_new_min]
            for idx, line in enumerate(lines):
                state = int(line.split()[1])
                if state == 1:
                    stats[idx]["S1"] += 1
                elif state == 2:
                    stats[idx]["S2"] += 1
                    
    return stats, total_runs

def write_output(stats, total_runs):
    """Write final output file"""
    with open("mean_occup.dat", "w") as f_out:
        # Header
        f_out.write(f"{'Time(fs)':<10}{'S2':^10}{'S1':^10}\n")
        
        # Data rows
        for idx, data in enumerate(stats):
            time = idx * 0.1
            s2 = data["S2"] / total_runs
            s1 = data["S1"] / total_runs
            f_out.write(f"{time:^10.1f}{s2:^10.5f}{s1:^10.5f}\n")

def main():
    """Main workflow controller"""
    try:
        # Configuration phase
        num_run = parse_arguments()
        run_dirs = get_valid_directories(num_run)
        
        # Data preparation phase
        valid_dirs, step_min = process_state_logs(run_dirs)
        generate_time_series(valid_dirs, step_min)
        
        # Analysis phase
        stats, total_runs = calculate_statistics(valid_dirs)
        write_output(stats, total_runs)
        
    except Exception as e:
        print(f"Program terminated with error: {str(e)}")
        return

if __name__ == "__main__":
    main()
