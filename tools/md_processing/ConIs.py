#!/usr/bin/env python3
import os
from optparse import OptionParser 

# Written by Jianzheng Ma, Fudan University, Shanghai, China, 2024-03-18.
#./ConIs.py -n 3

parser = OptionParser()
parser.add_option("-n", dest="num_run", action="store", type=int, nargs=1, \
                  help = "Number of run directory", default = "1000") 

(options, args) = parser.parse_args()
num_run = options.num_run

f1 = open("ConIs_hop.xyz", 'w')
for i in range(1,int(num_run + 1)):
    run_index = str(i).zfill(4)
    run_dir = "./run" + run_index
    if os.path.exists(run_dir):
        file_name = run_dir + "/ConIs.xyz"
        if os.path.exists(file_name):
            lines = open(file_name, 'r').readlines()
            f1.write(lines[0])
            lines_2 = lines[1].split()[0] + "   " + lines[1].split()[1] + "   " + "run=" + str(i).zfill(4) + "\n"
            f1.write(lines_2)
            atom_n = int(lines[0])
            for j in range(atom_n):
                f1.write(lines[j + 2])
f1.close()
