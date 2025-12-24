#!/usr/bin/env python3
import re
import math
import sys
import os
from optparse import OptionParser 

# Writed by Jianzheng Ma, Fudan University, Shanghai, China, 2024-04-02.
# Geometrical coordinate interpolation code

# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz

# Specify the interpolation point number: 
# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz -n 10

# Specify including the end point file coordinate: 
# ./geom_interpol.py -i StartPointFile.xyz -o EndPointFile.xyz -a

parser = OptionParser()
parser.add_option("-i", dest="start", action="store", type=str, nargs=1, \
                  help = "Interpolation start point file (xyz format file)") 
parser.add_option("-o", dest="end", action="store", type=str, nargs=1, \
                  help = "Interpolation end point file (xyz format file)") 
parser.add_option("-n", dest="fsav", action="store", type=int, nargs=1, \
                  help = "Interpolation point number",default = "10") 
parser.add_option("-a", dest="end_geo", action="store_true", \
                  help = "Whether to include the end point file coordinate in the interpolation result",\
                    default = False) 
(options, args) = parser.parse_args()
start_file = options.start
end_file = options.end
fsav = options.fsav
end_geo = options.end_geo

ang_bohr = 1.88972612
Bohrs = False
Radians = False
deg2rad = math.pi / 180.0

def atom_symbol(filename):
    with open(filename) as f:
        atom_symbol = []
        for i in f:
            result = re.search('([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}',i)
            if result != None:
                atom_symbol.append(result.group().split()[0])
        return (atom_symbol)

#check the atomic number
atom_n_start = len(atom_symbol(start_file))
atom_n_end = len(atom_symbol(end_file))
if atom_n_start != atom_n_end:
    print("Error: The atomic number of file %s and %s are different!."%(start_file,end_file))
    exit()
    
#check the atom symble
atom_symb_start = atom_symbol(start_file)
atom_symb_end = atom_symbol(end_file)
bool_result = []
for i in range(atom_n_start):
    if atom_symb_start[i] == atom_symb_end[i]:
        bool_result.append(True)
    else:
        bool_result.append(False)
if False in bool_result:
    print("Error: Please ensure the atom symble of %s and %s are same."%(start_file,end_file))
    exit()

def get_coordinate(filename):
    coordinate = []
    with open(filename) as f:
        for i in f:
            result = re.search('([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}',i)
            if result != None:
                coordinate.append([float(i) for i in result.group().split()[1:4]]) 
        return coordinate

def rscalar3d(a, b):
    '''Scalar product of 3-dimensional vectors a and b.'''
    s = 0
    for i in range(3):
        s += a[i] * b[i]
    return s

def rnorm3d(a):
    '''Norm of a 3-dimensional vector.'''
    return math.sqrt(rscalar3d(a, a))

def rcross3d(a, b):
    '''Vector product (Cross Product) of 3-dimensional vectors a and b.'''
    s = [0, 0, 0]
    s[0] = a[1] * b[2] - a[2] * b[1]
    s[1] = a[2] * b[0] - a[0] * b[2]
    s[2] = a[0] * b[1] - a[1] * b[0]
    return s

def rangle3d(a, b):
    '''Angle between two 3-dimensional vectors.'''
    x = rscalar3d(a, b) / (rnorm3d(a) * rnorm3d(b))
    angle = math.acos(max(-1.0, min(1.0, x)))
    if Radians:
        return angle
    else:
        return angle / deg2rad

def get_dis(a, b):
    '''Bond distance between atoms a and b.'''
    s = [0, 0, 0]
    for i in range(3):
        s[i] = b[i] - a[i]
    if Bohrs:
        return rnorm3d(s) * ang_bohr
    else:
        return rnorm3d(s)


def get_ang(a, b, c):
    '''Bond angle of a-b and b-c.'''
    r1 = [0, 0, 0]
    r2 = [0, 0, 0]
    for i in range(3):
        r1[i] = a[i] - b[i]
        r2[i] = c[i] - b[i]
    return rangle3d(r1, r2)


def get_dih(a, b, c, d):
    '''Dihedral angle between the a-b-c and b-c-d planes.'''
    r1 = [0, 0, 0]
    r2 = [0, 0, 0]
    r3 = [0, 0, 0]
    for i in range(3):
        r1[i] = b[i] - a[i]
        r2[i] = c[i] - b[i]
        r3[i] = d[i] - c[i]
    q1 = rcross3d(r1, r2)
    q2 = rcross3d(r2, r3)
    if q1 == [0., 0., 0.] or q2 == [0., 0., 0.]:
        sys.stderr.write('Undefined dihedral angle!')
        return float('NaN')
    Q = rcross3d(q1, q2)
    if Q == [0., 0., 0.]:
        sign = 1.
    elif rangle3d(Q, r2) < 90. * [deg2rad, 1.][Radians is None]:
        sign = 1.
    else:
        sign = -1.
    return rangle3d(q1, q2) * sign

def Cartesian2Zmatrix(filename,filename_zmatrix):
    coor = get_coordinate(filename)
    string = ''
    symbols = atom_symbol(filename)
    for idx in range(len(symbols)):
        if idx == 0:
            string = '%3s\n'%symbols[idx]
        elif idx == 1:
            string += '%3s %3d %20.8f\n'%(symbols[idx],1,get_dis(coor[0],coor[idx]))
        elif idx == 2:
            string += '%3s %3d %20.8f %3d %20.8f\n'%(symbols[idx],2,get_dis(coor[1],coor[idx]),1,
                                                 get_ang(coor[0],coor[1],coor[idx]))
        else:
            string += '%3s %3d %20.8f %3d %20.8f %3d %20.8f\n'%(symbols[idx],idx,get_dis(coor[idx],coor[idx-1]),idx-1,\
                                                          get_ang(coor[idx],coor[idx-1],coor[idx-2]),\
                                                          idx-2,get_dih(coor[idx-3],coor[idx-2],coor[idx-1],coor[idx]))
    out_zmatrix = open(filename_zmatrix, 'w')
    out_zmatrix.write(string)
    out_zmatrix.close()

def replace_vars(vlist, variables):
    """ Replaces a list of variable names (vlist) with their values
        from a dictionary (variables).
    """
    for i, v in enumerate(vlist):
        if v in variables:
            vlist[i] = variables[v]
        else:
            try:
                # assume the "variable" is a number
                vlist[i] = float(v)
            except:
                print("Problem with entry " + str(v))

def read_zmatrix(filename):

    """ Reads in a z-matrix in standard format,
        returning a list of atoms and coordinates.
    """

#   if the Z matrix is:
#   C
#   H   1           1.07000000
#   H   2           1.74730248   1          35.26439865
#   H   3           1.74730301   2          59.99999023   1         -35.26438466
#   H   4           1.74730266   3          59.99999322   2          70.52879623
#   run the read_zmatrix function, we will get:
# (['C', 'H', 'H', 'H', 'H'], [1, 2, 3, 4], [1.07, 1.74730248, 1.74730301, 1.74730266], 
# [1, 2, 3], [35.26439865, 59.99999023, 59.99999322], [1, 2], [-35.26438466, 70.52879623])
    
    zmatf = open(filename, 'r')
    atom_names = []
    r_connect = []  # bond connectivity
    r_list = []     # list of bond length values
    ang_connect = []  # angle connectivity
    ang_list = []     # list of bond angle values
    dih_connect = []  # dihedral connectivity
    dih_list = []     # list of dihedral values
    variables = {} # dictionary of named variables
    
    if not zmatf.closed:
        for line in zmatf:
            words = line.split()
            eqwords = line.split('=')
            
            if len(eqwords) > 1:
                # named variable found 
                varname = str(eqwords[0]).strip()
                try:
                    varval  = float(eqwords[1])
                    variables[varname] = varval
                except:
                    print("Invalid variable definition: " + line)
            
            else:
                if len(words) > 0:
                    atom_names.append(words[0])
                if len(words) > 1:
                    r_connect.append(int(words[1]))
                if len(words) > 2:
                    r_list.append(words[2])
                if len(words) > 3:
                    ang_connect.append(int(words[3]))
                if len(words) > 4:
                    ang_list.append(words[4])
                if len(words) > 5:
                    dih_connect.append(int(words[5]))
                if len(words) > 6:
                    dih_list.append(words[6])
    
    # replace named variables with their values
    replace_vars(r_list, variables)
    replace_vars(ang_list, variables)
    replace_vars(dih_list, variables)
    
    return (atom_names, r_connect, r_list, ang_connect, ang_list, dih_connect, dih_list) 

def Zmatrix2Cartesian(atom_names, r_connect, r_list, ang_connect, ang_list, dih_connect, dih_list):
    """Prints out an xyz file from a decomposed z-matrix"""
    n_atom = len(atom_names)
    
    # put the first atom at the origin
    xyz_arr = []
    for i in range(n_atom):
        xyz_arr.append([0.0, 0.0, 0.0])

    if (n_atom > 1):
        # second atom at [r01, 0, 0]
        xyz_arr[1] = [r_list[0], 0.0, 0.0]

    if (n_atom > 2):
        # third atom in the xy-plane
        # such that the angle a012 is correct 
        i = r_connect[1] - 1
        j = ang_connect[0] - 1
        r = r_list[1]
        theta = ang_list[0] * math.pi / 180.0
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        a_i = xyz_arr[i]
        b_ij = [0.0, 0.0, 0.0]
        b_ij[0] = xyz_arr[j][0] - xyz_arr[i][0]
        b_ij[1] = xyz_arr[j][1] - xyz_arr[i][1]
        b_ij[2] = xyz_arr[j][2] - xyz_arr[i][2]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyz_arr[2] = [x, y, 0.0]

    for n in range(3, n_atom):
        # back-compute the xyz coordinates
        # from the positions of the last three atoms
        r = r_list[n-1]
        theta = ang_list[n-2] * math.pi / 180.0
        phi = dih_list[n-3] * math.pi / 180.0
        
        sinTheta = math.sin(theta)
        cosTheta = math.cos(theta)
        sinPhi = math.sin(phi)
        cosPhi = math.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = r_connect[n-1] - 1
        j = ang_connect[n-2] - 1
        k = dih_connect[n-3] - 1

        a = xyz_arr[k]
        b = xyz_arr[j]
        c = xyz_arr[i]

        ab = [0.0, 0.0, 0.0]
        ab[0] = b[0] - a[0]
        ab[1] = b[1] - a[1]
        ab[2] = b[2] - a[2]

        bc = [0.0, 0.0, 0.0]
        bc[0] = c[0] - b[0]
        bc[1] = c[1] - b[1]
        bc[2] = c[2] - b[2]

        norm_bc = rnorm3d(bc)
        bc[0] = bc[0] / norm_bc
        bc[1] = bc[1] / norm_bc
        bc[2] = bc[2] / norm_bc

        nv = rcross3d(ab, bc)
        norm_nv = rnorm3d(nv)
        nv[0] = nv[0] / norm_nv
        nv[1] = nv[1] / norm_nv
        nv[2] = nv[2] / norm_nv

        ncbc = rcross3d(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyz_arr[n] = [new_x, new_y, new_z]
            
    # print results
    convert_result = ''
    for i in range(n_atom):
        convert_result +='{:<4s}\t{:>11.15f}\t{:>11.15f}\t{:>11.15f}\n'.\
            format(atom_names[i], xyz_arr[i][0] + 1.0, xyz_arr[i][1] + 1.0, xyz_arr[i][2] + 1.0)

    return convert_result

def coor_interpol(list_start, list_end, fsav):
    step = []
    all_list_inter = []
    for i, j in zip(list_start, list_end):
        step.append((j - i) / fsav) 
    for i in range(fsav):
        list_inter = []
        for j, k in zip(list_start, step):
            list_inter.append(j + i * k)
        all_list_inter.append(list_inter)
    if end_geo:
        all_list_inter.append(list_end)
    return all_list_inter

def main():
    start_zmatrix = start_file.split(".")[0] + ".zmatrix"
    end_zmatrix = end_file.split(".")[0] + ".zmatrix"
    Cartesian2Zmatrix(start_file,start_zmatrix)
    Cartesian2Zmatrix(end_file,end_zmatrix)
    atom_names_start, r_connect_start, r_list_start, \
        ang_connect_start, ang_list_start, dih_connect_start, dih_list_start \
                                                                            = read_zmatrix(start_zmatrix)
    atom_names_end, r_connect_end, r_list_end, \
        ang_connect_end, ang_list_end, dih_connect_end, dih_list_end \
                                                                        = read_zmatrix(end_zmatrix)
    r_inter_list = coor_interpol(r_list_start, r_list_end, fsav)
    ang_inter_list = coor_interpol(ang_list_start, ang_list_end, fsav)
    dih_inter_list = coor_interpol(dih_list_start, dih_list_end, fsav)
    atom_n = len(atom_names_start)
    interpol_file = "interpol_" + start_file.split(".")[0] + "2" + end_file.split(".")[0] + ".xyz"
    if os.path.exists(interpol_file):
        os.remove(interpol_file)

    counter = 0
    for r_inter, ang_inter, dih_inter in zip(r_inter_list,ang_inter_list,dih_inter_list):
        out_xyz = open(interpol_file, 'a+')
        interpol_xyz = Zmatrix2Cartesian(atom_names_start, r_connect_start, r_inter,\
                                         ang_connect_start, ang_inter, dih_connect_start, dih_inter)
        out_xyz.write(str(atom_n) + "\n")
        out_xyz.write("Interpolation point: " + str(counter) + "\n")
        out_xyz.write(interpol_xyz)
        counter = counter + 1
        out_xyz.close()

if __name__ == '__main__':
    main()
