#!/usr/bin/env python3

# Edit by Jianzheng Ma, Fudan University, Shanghai, China, 2024-04-09.
# Script for the conversion of
# gamess-us output files to frequency files
# for molden. Only the last occurring
# instances of coordinates, frequencies
# and normal modes are used. 
# Unit: FREQ (CM**-1), FR-COORD (a.u.), INT (km/mol)

# For nonlinear molecule:
# ./GAMESS_freq.py -f [frequency calculation output gamess-us file] 

# For linear molecule:
# ./GAMESS_freq.py -f [frequency calculation output gamess-us file] -l

import os
import re
from optparse import OptionParser 

parser = OptionParser() 
parser.add_option("-f", dest="file", action="store", type=str, nargs=1, \
                  help = "The frequency calculation output gamess-us file.") 
parser.add_option("-l", dest="is_liner", action="store_true", \
                  help = "Is linear molecule?",default = False) 
(options, args) = parser.parse_args()

gamess_file = options.file
is_liner = options.is_liner

out_file = gamess_file.split(".")[0] + '.molden'
out = open(out_file, 'w')

out_temp_file = gamess_file.split(".")[0] + '.temp'
out_temp = open(out_temp_file, 'w')

try:
    lines = open(gamess_file, 'r').readlines()
except IOError:
    print("Could not open %s." % gamess_file)
    exit()

# check if file is sucessfully completed orca file:
is_gamess = False
finished = False
if lines[3].strip() == "----- GAMESS execution script 'rungms' -----":
    is_gamess = True

string = "'ddikick.x: exited gracefully.'"    
command = "grep" + " " + string + " " + gamess_file
result = os.popen(command).read()
if str(result).strip() == "ddikick.x: exited gracefully.":
    finished = True

if not is_gamess:
    print("File %s is not in gamess-us output format (probably)!" % gamess_file)
    exit()
elif is_gamess and not finished:
    print("The job either has crashed or has not finished yet.")
    exit()
elif is_gamess and finished:
    print("Reading data from file %s..." % gamess_file)

# set falgs
read = False
geom = False
freq = False
intensity = False
nmode = False
counter = 0

# parse input
for line in lines:
    if line.startswith(" ATOM      ATOMIC                      COORDINATES (BOHR)"):
        read = True
        geom = True
        coords = []

    elif line.startswith(" ANALYZING SYMMETRY OF NORMAL MODES..."):
        read = True
        nmode = True

    elif line.startswith("  MODE FREQ(CM**-1)  SYMMETRY  RED. MASS  IR INTENS."):
        read = True
        freq = True
        freqs = []
        intensity = True
        intdict = {}

    elif read and geom:
        line = line.strip()
        # print(line)
        if line.startswith("CHARGE"):
            continue
        if line:
            line = line.split()
            coords.append("%s %s %s %s" % (line[0], line[2], line[3], line[4]))
        else:
            read = False
            geom = False

    elif read and nmode:
        n_atoms = len(coords)
        fredom = n_atoms * 3
        cart = fredom + 6
        if "REFERENCE ON SAYVETZ CONDITIONS" in line:
            read = False
            nmode = False
            counter = 0
        else:
            result = re.search('\s+(\d+)?\s+([a-zA-Z]{0,2})\s+[a-zA-Z]{1}((\s+)?(-?)\d+\.\d+){1,5}',line)
            if result != None:
                counter = counter + 1
                if counter % cart == fredom + 1 or counter % cart == fredom + 2 or counter % cart == fredom + 3 \
                    or counter % cart == fredom + 4 or counter % cart == fredom + 5 or counter % cart == 0:
                    pass
                else:
                    out_temp.write(result.group() + "\n")

    elif read and freq and intensity:
        line = line.strip()
        if line:
            line = line.split()
            freqs.append(line[1])
            intdict[int(line[0])] = float(line[4]) * 42.2561 # from DEBYE**2/AMU-ANGSTROM**2 to km/mol
        else:
            read = False
            freq = False
            intensity = False

# check coords
if len(coords) > 0:
    print("Found system of %d atoms." % len(coords))
else:
    print("No cartesian coordinates found.")
    exit()

# check freqs
if len(freqs) > 0:
    print("Found %d vibrational frequencies." % len(freqs))
else:
    print("No vibrational frequencies found.")

# check intensities
if len(intdict) == 0:
    print('No intensities available.')
else:
    print('IR intensities available.')

counter = 0
modes = []
out_temp.close()
lines_temp = open(out_temp_file, 'r').readlines()
for l_t in lines_temp:
    l_t = l_t.strip()
    counter = counter + 1
    if counter % 3 == 1:
        l_t = l_t.split()[3:]
        modes.append(l_t)
    else:
        l_t = l_t.split()[1:]
        modes.append(l_t)

# check modes
if len(modes) == 0:
    print("No normal modes found.")
    exit()

# combine modes
n_atoms = len(coords)
cart = n_atoms * 3
for i in range(cart, len(modes)):
    modes[i % cart] = modes[i % cart] + modes[i]

# sort modes
n_modes = len(modes[0])
print("Found %d normal modes." % n_modes)
modes = modes[:n_modes]
cmodes = [[] for i in range(n_modes)]
for line in modes:
    for i in range(n_modes):
        cmodes[i].append(line[i])
modes = cmodes

# check for consistency:
if n_modes != len(freqs):
    print("Mismatch between frequencies and vibrational normal modes detected.")
    exit()

out.write("[MOLDEN FORMAT]\n")

# write frequencies
# n: atomic number
# Fredom of linear molecule: 3n-5 (Discard the translational freedom and rotational freedom).
# Fredom of nonlinear molecule: 3n-6 (Discard the translational freedom and rotational freedom).
out.write("[FREQ]\n")
for i,freq in enumerate(freqs): 
    if not is_liner:
        if i < 6: # nonlinear molecule
            out.write(str(0.0) + '\n') 
        else:
            out.write(freq + '\n')
    else:
        if i < 5: # linear molecule
            out.write(str(0.0) + '\n') 
        else:
            out.write(freq + '\n')


# write coordinates block (A.U.)
out.write("[FR-COORD]\n")
for coord in coords:
    out.write(coord + '\n')

# write normal modes:
out.write("[FR-NORM-COORD]\n")
for i in range(n_modes):
    out.write("vibration %d\n" % (i + 1))
    for j in range(len(modes[i])):
        out.write(modes[i][j] + ' ')
        if (j + 1) % 3 == 0:
            out.write('\n')
out.write('[INT]\n')
for i in range(n_modes):
    if i in intdict:
        f = intdict[i]
    else:
        f = 0.
    out.write('%16.9f\n' % f)
out.close()
print("Molden output written to %s" % out_file)

if  os.path.exists(out_temp_file):
    command = "rm" + " " + out_temp_file
    os.system(command)