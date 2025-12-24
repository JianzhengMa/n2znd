#!/usr/bin/env python3
import os
from optparse import OptionParser 

# Written by Jianzheng Ma, Fudan University, Shanghai, China, 2024-05-03.

parser = OptionParser() 
parser.add_option("-n", dest="max_index", action="store", type=int, nargs=1, \
                  help = "The max index number of runs.", default = 1000) 
(options, args) = parser.parse_args()
max_index = options.max_index

run_list = []
for i in range(1, int(max_index + 1)):
    run_index = str(i).zfill(4)
    run_dir = "./run" + run_index
    if os.path.exists(run_dir):
        run_list.append(i)

for j,element in enumerate(run_list):
    if int(j + 1) != element:
        initial_run_index = str(element).zfill(4)
        initial_run_dir = "run" + initial_run_index
        new_run_index = str(j + 1).zfill(4)
        new_run_dir = "run" + new_run_index
        Command = "mv " + initial_run_dir + " " + new_run_dir
        os.system(Command)