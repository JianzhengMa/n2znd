import os
import re

start = 1
end = 120
state = 2 # 2: S1; 1: S0
scfout_paths = ["./run_S1/{:d}/".format(i) for i in range(start, end+1)]

def state_energy(filename,state):
    data_1 = []#State symbol
    data_2 = []#State energy
    command = "sed -n '/SUMMARY OF MRSF-DFT RESULTS/,/SELECTING EXCITED STATE IROOT/p'" + " " + filename
    data = os.popen(command).read().splitlines()
    for i in data:
        result = re.search('(\s+\d+\s+)([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){7}',i)
        if result != None:
            state_symbol = result.group().split()[0]
            state_energy =  result.group().split()[2]
            data_1.append(state_symbol)
            data_2.append(float(state_energy))
    energy_dict = dict(zip(data_1,data_2))
    if state:
        return energy_dict[str(state)]
    elif state == False:
        return energy_dict

with open('energy_ab.dat', 'w') as f:
    for dir_path in scfout_paths:
        dir_path = os.path.normpath(dir_path)
        folder_name = os.path.basename(dir_path)
        result_file = os.path.join(dir_path,"input.out")
        try:
            energy = state_energy(result_file, state)  
            if energy is not None:
                f.write(f"{folder_name}\t{energy:.8f}\n")
            else:
                f.write(f"{folder_name}\tSTATE_NOT_FOUND\n")
        except Exception as e:
            print(f"Error processing {dir_path}: {str(e)}")
            f.write(f"{folder_name}\tERROR\n")
