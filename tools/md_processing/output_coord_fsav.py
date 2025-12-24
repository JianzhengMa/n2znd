#!/usr/bin/env python3
import os
from optparse import OptionParser 

# Edited by Jianzheng Ma, Fudan University, Shanghai, China, 2024-04-12.

# This code is used to output a result every fsav steps.
#./output_coord_fsav.py -i [coordinate filename] -n [fsav]


au_to_ang = 0.529177208
au_to_ev = 27.2113956555
Ha_Bohr_to_ev_ang = 51.42208619083232

parser = OptionParser() 
parser.add_option("-i", dest="input_file", action="store", type=str, nargs=1, \
                  help = "Output coordinate file: coordinate.xyz", default = "coordinate.xyz") 
parser.add_option("-n", dest="fsav", action="store", type=int, nargs=1, \
                  help = "Output a result every fsav steps.", default = 1) 
parser.add_option("-e", dest="end_step", action="store", type=int, nargs=1, \
                  help = "Dynamics simulation end step.", default = 2000) 
(options, args) = parser.parse_args()
input_file = options.input_file
fsav = options.fsav
end_step = options.end_step

try:
    lines = open(input_file, 'r').readlines()
except IOError:
    print("Could not open %s." % input_file)
    exit()

atom_n = int(lines[0])
struc_num = int((len(lines) / (atom_n + 2)))
line_group = int(struc_num / fsav)

extxyz = False
if len(lines[2].split()) == 7:
    extxyz = True

if not extxyz:
    file_name = input_file.split(".")[0] + "_every_" + str(fsav) + ".xyz"
if extxyz:
    file_name = input_file.split(".")[0] + "_every_" + str(fsav) + ".extxyz"
if os.path.exists(file_name):
    os.remove(file_name)

for i in range(line_group):
    out_file = open(file_name, 'a+')
    if int(i * fsav) <= end_step:
        out_file.write(str(atom_n) + "\n")
        out_file.write(lines[int(i * (atom_n + 2) * fsav) + 1])
        for j in range(atom_n + 2):
            if j > 1:
                label = lines[int(i * (atom_n + 2) * fsav) + j].split()[0]
                coord_x = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[1]) , '>18.10f') 
                coord_y = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[2]) , '>18.10f') 
                coord_z = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[3]) , '>18.10f')
                if extxyz:
                    force_x = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[4]) , '>18.10f') 
                    force_y = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[5]) , '>18.10f') 
                    force_z = format(float(lines[int(i * (atom_n + 2) * fsav) + j].split()[6]) , '>18.10f') 
                    write_result = label + "\t" + coord_x + coord_y + coord_z + force_x + force_y + force_z + "\n" 
                if not extxyz:
                    write_result = label + "\t" + coord_x + coord_y + coord_z + "\n"
                out_file.write(write_result)
    out_file.close()
