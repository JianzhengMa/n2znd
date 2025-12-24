#!/usr/bin/env python3
import os
import re
import shutil
from optparse import OptionParser 

#./set_ini_run.py -w initconds -s EP.inp -f mndo 
#-w: Wigner Sampling File Name initconds
#-s: Single Point Input File Name
#-f: mndo or gamess
#MLFF : ./set_ini_run.py -w initconds -f mlmd 
# Written by Jianzheng Ma, Fudan University, Shanghai, China, 2024-03-02.

LAB_NUM = {'H': 1, 'He': 2,
           'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
           'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
           'K': 19, 'Ca': 20,
           'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
           'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
           'Rb': 37, 'Sr': 38,
           'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
           'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
           'Cs': 55, 'Ba': 56,
           'La': 57,
           'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
           'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
           'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
           'Fr': 87, 'Ra': 88,
           'Ac': 89,
           'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
           'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
           'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
           }

au_to_ang = 0.529177257507
amu_to_au = 1822.88853006

parser = OptionParser() 
parser.add_option("-w", dest="initconds", action="store", type=str, nargs=1, \
                  help = "Wigner Sampling File Name initconds", default = "initconds") 
parser.add_option("-s", dest="inp", action="store", type=str, nargs=1, \
                  help = "Single Point Input File Name",default = "test.inp") 
parser.add_option("-f", dest="flag", action="store", type=str, nargs=1, \
                  help = "The type of electronical structural calculation input filename",\
                    default = "gamess") 
(options, args) = parser.parse_args()

initconds = options.initconds
input_file_name = options.inp
input_type = options.flag

def get_CoorAndMome(filename):
    try:
        lines = open(initconds, 'r').readlines()
    except IOError:
        print("Could not open %s." % initconds)
        exit()
    read = False

    coords = []
    momentum = []

    for line in lines:
        if line.startswith("Natom"):
            line = line.strip()
            Natom = int(line.split()[1])

        if line.startswith("Index"):
            read = True
            continue

        if read:
            line = line.strip()
            if line.startswith("Atoms"):
                continue
            elif line.startswith("States"):
                read = False
            else:
                line = line.split()
                coords.append("%s %s %s %s %s" % (line[0], line[1], line[2], line[3], line[4]))
                momentum.append("%s %s %s %s %s" % (line[0], line[5],line[6], line[7], line[8]))
    index_num = int(len(coords) / Natom)
    return (Natom,index_num,coords,momentum)

def check(Natom,index_num):
    #check coordinate
    if Natom > 0:
        print("Found system of %d atoms." % Natom)
    else:
        print("No cartesian coordinates found.")
        exit()
    #check index 
    if index_num > 0:
        print("Found system of %d phase space points.\n" % index_num)
    else:
        print("No phase space points found.")
        exit()

def get_input_atom_MNDO2020(filename):
    label_num = []
    with open(filename) as f:
        for i in f:
            result = re.search('(\s+\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}',i)
            if result != None:
                if int(result.group().split()[0]) != 0:
                    label_num.append(result.group().split()[0]) 
    input_atom_n = len(label_num)
    return int(input_atom_n)

def get_input_atom_GAMESS_MRSF(filename):
    label_num = []
    with open(filename) as f:
        for i in f:
            result = re.search('([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}',i)
            if result != None:
                label_num.append(result.group().split()[0]) 
    input_atom_n = len(label_num)
    return int(input_atom_n)

def check_atom_n(Natom,input_atom_n):
    if Natom == input_atom_n:
        print("Atomic number check suggessful!\n")
    else:
        print("Error: Please ensure the atomic number of %s and %s are the same.\n"%(initconds,input_file_name))
        exit(0)

def clear_file(index_num):
    for i in range(1, int(index_num + 1)):
        run_index = str(i).zfill(4)
        run_dir = "./run" + run_index
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
    if os.path.exists("sampling.xyz"):
        os.remove("sampling.xyz")

def write_file_Gamess_MRSF(index_num,Natom,coords,momentum,input_file):
    coor_xyz_file = open("sampling.xyz","w+")
    for i in range(1, int(index_num + 1)):
        run_index = str(i).zfill(4)
        run_dir = "./run" + run_index
        command_mkrun = "mkdir" + " " + run_dir
        os.system(command_mkrun)

        mom_file = run_dir + "/momentum.inp"
        mom_out = open(mom_file, 'w')

        file_ini = open(input_file)

        coor_file = run_dir + "/" + input_file
        coor_out = open(coor_file,"w+")

        coor_xyz_file.write(str(Natom) + "\n")
        coor_xyz_file.write("Index" + " " + str(i) + "\n")

        coor_str_list = []
        coor_xyz = []

        for j in range (1, int(Natom + 1)):
            mom_list = momentum[int((i - 1) * Natom + (j - 1))].split()
            coor_list = coords[int((i - 1) * Natom + (j - 1))].split()
            label = mom_list[0]

            amu   = float(mom_list[1].strip())
            mom_x = float(mom_list[2]) * amu_to_au * amu
            mom_y = float(mom_list[3]) * amu_to_au * amu
            mom_z = float(mom_list[4]) * amu_to_au * amu

            mommentum_1 = format(mom_x, '>18.10f') 
            mommentum_2 = format(mom_y, '>18.10f')
            mommentum_3 = format(mom_z, '>18.10f')

            mom_str = label + "  " + mommentum_1 + mommentum_2 + mommentum_3
            mom_out.write(mom_str + "\n")  

            atom_num = coor_list[1]
            coor_x = float(coor_list[2]) * au_to_ang
            coor_y = float(coor_list[3]) * au_to_ang
            coor_z = float(coor_list[4]) * au_to_ang

            coordinate_1 = format(coor_x, '>18.10f') 
            coordinate_2 = format(coor_y, '>18.10f') 
            coordinate_3 = format(coor_z, '>18.10f') 

            coor_str = label + "  " + atom_num + coordinate_1 + coordinate_2 + coordinate_3
            coor_str_list.append(coor_str)

            coor_xyz = label + "  " + coordinate_1 + coordinate_2 + coordinate_3
            coor_xyz_file.write(coor_xyz + "\n")

        n = 0
        for k in file_ini:
            result = re.search('([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}',k)
            if result == None:
                coor_out.write(k)
            else:
                reps = re.sub(result.group(), coor_str_list[n],k)
                coor_out.write(reps)
                n = n + 1

        mom_out.close()  
        coor_out.close()
        file_ini.close()
        print("Writing suggessfully " + run_dir)

    coor_xyz_file.close()

def write_file_MNDO2020(index_num,Natom,coords,momentum,input_file):
    coor_xyz_file = open("sampling.xyz","w+")
    for i in range(1, int(index_num + 1)):
        run_index = str(i).zfill(4)
        run_dir = "./run" + run_index
        command_mkrun = "mkdir" + " " + run_dir
        os.system(command_mkrun)

        mom_file = run_dir + "/momentum.inp"
        mom_out = open(mom_file, 'w')

        file_ini = open(input_file)

        coor_file = run_dir + "/" + input_file
        coor_out = open(coor_file,"w+")

        coor_xyz_file.write(str(Natom) + "\n")
        coor_xyz_file.write("Index" + " " + str(i) + "\n")

        coor_str_list = []
        coor_xyz = []

        for j in range (1, int(Natom + 1)):
            mom_list = momentum[int((i - 1) * Natom + (j - 1))].split()
            coor_list = coords[int((i - 1) * Natom + (j - 1))].split()
            label = mom_list[0]

            amu   = float(mom_list[1].strip())
            mom_x = float(mom_list[2]) * amu_to_au * amu
            mom_y = float(mom_list[3]) * amu_to_au * amu
            mom_z = float(mom_list[4]) * amu_to_au * amu

            mommentum_1 = format(mom_x, '>18.10f') 
            mommentum_2 = format(mom_y, '>18.10f')
            mommentum_3 = format(mom_z, '>18.10f')

            mom_str = label + "  " + mommentum_1 + mommentum_2 + mommentum_3
            mom_out.write(mom_str + "\n")  
            
            atom_num = str(LAB_NUM[label])
            coor_x = float(coor_list[2]) * au_to_ang
            coor_y = float(coor_list[3]) * au_to_ang
            coor_z = float(coor_list[4]) * au_to_ang

            coordinate_1 = format(coor_x, '>18.10f') 
            coordinate_2 = format(coor_y, '>18.10f') 
            coordinate_3 = format(coor_z, '>18.10f') 

            coor_str = atom_num + coordinate_1 + coordinate_2 + coordinate_3
            coor_str_list.append(coor_str)

            coor_xyz = label + "  " + coordinate_1 + coordinate_2 + coordinate_3
            coor_xyz_file.write(coor_xyz + "\n")

        n = 0
        for k in file_ini:
            result = re.search('(\s+\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}',k)
            if result == None:
                coor_out.write(k)
            elif result.group().split()[0] == str(0):
                coor_out.write(k)
            else:
                row_1 = format(coor_str_list[n].split()[0], '>5s')
                row_2 = format(float(coor_str_list[n].split()[1]), '>18.10f')
                row_3 = format(result.group().split()[2], '>2s')
                row_4 = format(float(coor_str_list[n].split()[2]), '>18.10f')
                row_5 = format(result.group().split()[4], '>2s')
                row_6 = format(float(coor_str_list[n].split()[3]), '>18.10f')
                row_7 = format(result.group().split()[6], '>2s')
                result_new = row_1 + row_2 + row_3 + row_4 + row_5 + row_6 + row_7
                reps = re.sub(result.group(), result_new, k)
                coor_out.write(reps)
                n = n + 1

        mom_out.close()  
        coor_out.close()
        file_ini.close()
        print("Writing suggessfully " + run_dir)

    coor_xyz_file.close()

def write_file_MLMD(index_num,Natom,coords,momentum):
    coor_xyz_file = open("sampling.xyz","w+")
    for i in range(1, int(index_num + 1)):
        run_index = str(i).zfill(4)
        run_dir = "./run" + run_index
        command_mkrun = "mkdir" + " " + run_dir
        os.system(command_mkrun)

        mom_file = run_dir + "/momentum.inp"
        mom_out = open(mom_file, 'w')

        coor_file = run_dir + "/coordinate.inp" 
        coor_out = open(coor_file,"w")
        coor_out.write(str(Natom) + "\n")
        coor_out.write("Index: " + str(i) + "\n")

        coor_xyz_file.write(str(Natom) + "\n")
        coor_xyz_file.write("Index" + " " + str(i) + "\n")

        coor_xyz = []

        for j in range (1, int(Natom + 1)):
            mom_list = momentum[int((i - 1) * Natom + (j - 1))].split()
            coor_list = coords[int((i - 1) * Natom + (j - 1))].split()
            label = mom_list[0]

            amu   = float(mom_list[1].strip())
            mom_x = float(mom_list[2]) * amu_to_au * amu
            mom_y = float(mom_list[3]) * amu_to_au * amu
            mom_z = float(mom_list[4]) * amu_to_au * amu

            mommentum_1 = format(mom_x, '>18.10f') 
            mommentum_2 = format(mom_y, '>18.10f')
            mommentum_3 = format(mom_z, '>18.10f')

            mom_str = label + "  " + mommentum_1 + mommentum_2 + mommentum_3
            mom_out.write(mom_str + "\n")  

            coor_x = float(coor_list[2]) * au_to_ang
            coor_y = float(coor_list[3]) * au_to_ang
            coor_z = float(coor_list[4]) * au_to_ang

            coordinate_1 = format(coor_x, '>18.10f') 
            coordinate_2 = format(coor_y, '>18.10f') 
            coordinate_3 = format(coor_z, '>18.10f') 

            coor_str = label + "  " +  coordinate_1 + coordinate_2 + coordinate_3
            coor_out.write(coor_str + "\n")

            coor_xyz = label + "  " + coordinate_1 + coordinate_2 + coordinate_3
            coor_xyz_file.write(coor_xyz + "\n")

        mom_out.close()  
        coor_out.close()
        print("Writing suggessfully " + run_dir)

    coor_xyz_file.close()

def main():
    print("\nStart set the initial run file for nonadiabatic molecular dynamics simulation!\n")
    Natom,index_num,coords,momentum = get_CoorAndMome(initconds)
    check(Natom,index_num)
    if input_type == "gamess":
        print("The type of electronical structural calculation is GAMESS-US")
        input_atom_n = get_input_atom_GAMESS_MRSF(input_file_name)
        check_atom_n(Natom,input_atom_n)
        clear_file(index_num)
        print("\nStart write the result...\n")
        write_file_Gamess_MRSF(index_num,Natom,coords,momentum,input_file_name)
        print("\nWriting the result suggessfully\n")
    elif input_type == "mndo":
        print("The type of electronical structural calculation is MNDO2020")
        input_atom_n = get_input_atom_MNDO2020(input_file_name)
        check_atom_n(Natom,input_atom_n)
        clear_file(index_num)
        print("\nStart write the result...\n")
        write_file_MNDO2020(index_num,Natom,coords,momentum,input_file_name)
        print("\nWriting the result suggessfully\n")
    elif input_type == "mlmd":
        print("The type of electronical structural calculation is MLMD")
        clear_file(index_num)
        print("\nStart write the result...\n")
        write_file_MLMD(index_num,Natom,coords,momentum)
        print("\nWriting the result suggessfully\n")
    else:
        print("Error: Please input the correct type of electronical structural calculation!")
        exit(0)
if __name__ == '__main__':
    main()
