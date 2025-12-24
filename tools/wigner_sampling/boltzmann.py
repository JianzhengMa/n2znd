import os
import numpy as np

U_TO_AMU = 1. / 5.4857990943e-4
ang = 0.529177257507
MASSES = {'H': 1.007825 * U_TO_AMU,'He': 4.002603 * U_TO_AMU,'Li': 7.016004 * U_TO_AMU,'Be': 9.012182 * U_TO_AMU,
          'B': 11.009305 * U_TO_AMU,'C': 12.000000 * U_TO_AMU,'N': 14.003074 * U_TO_AMU,'O': 15.994915 * U_TO_AMU,
          'F': 18.998403 * U_TO_AMU,'Ne': 19.992440 * U_TO_AMU,'Na': 22.989770 * U_TO_AMU,'Mg': 23.985042 * U_TO_AMU,
          'Al': 26.981538 * U_TO_AMU,'Si': 27.976927 * U_TO_AMU,'P': 30.973762 * U_TO_AMU,'S': 31.972071 * U_TO_AMU,
          'Cl': 34.968853 * U_TO_AMU,'Ar': 39.962383 * U_TO_AMU,'K': 38.963707 * U_TO_AMU,'Ca': 39.962591 * U_TO_AMU,
          'Sc': 44.955910 * U_TO_AMU,'Ti': 47.947947 * U_TO_AMU,'V': 50.943964 * U_TO_AMU,'Cr': 51.940512 * U_TO_AMU,
          'Mn': 54.938050 * U_TO_AMU,'Fe': 55.934942 * U_TO_AMU,'Co': 58.933200 * U_TO_AMU,'Ni': 57.935348 * U_TO_AMU,
          'Cu': 62.929601 * U_TO_AMU,'Zn': 63.929147 * U_TO_AMU,'Ga': 68.925581 * U_TO_AMU,'Ge': 73.921178 * U_TO_AMU,
          'As': 74.921596 * U_TO_AMU,'Se': 79.916522 * U_TO_AMU,'Br': 78.918338 * U_TO_AMU,'Kr': 83.911507 * U_TO_AMU,
          'Rb': 84.911789 * U_TO_AMU,'Sr': 87.905614 * U_TO_AMU,'Y': 88.905848 * U_TO_AMU,'Zr': 89.904704 * U_TO_AMU,
          'Nb': 92.906378 * U_TO_AMU,'Mo': 97.905408 * U_TO_AMU,'Tc': 98.907216 * U_TO_AMU,'Ru': 101.904350 * U_TO_AMU,
          'Rh': 102.905504 * U_TO_AMU,'Pd': 105.903483 * U_TO_AMU,'Ag': 106.905093 * U_TO_AMU,'Cd': 113.903358 * U_TO_AMU,
          'In': 114.903878 * U_TO_AMU,'Sn': 119.902197 * U_TO_AMU,'Sb': 120.903818 * U_TO_AMU,'Te': 129.906223 * U_TO_AMU,
          'I': 126.904468 * U_TO_AMU,'Xe': 131.904154 * U_TO_AMU,'Cs': 132.905447 * U_TO_AMU,'Ba': 137.905241 * U_TO_AMU,
          'La': 138.906348 * U_TO_AMU,'Ce': 139.905435 * U_TO_AMU,'Pr': 140.907648 * U_TO_AMU,'Nd': 141.907719 * U_TO_AMU,
          'Pm': 144.912744 * U_TO_AMU,'Sm': 151.919729 * U_TO_AMU,'Eu': 152.921227 * U_TO_AMU,'Gd': 157.924101 * U_TO_AMU,
          'Tb': 158.925343 * U_TO_AMU,'Dy': 163.929171 * U_TO_AMU,'Ho': 164.930319 * U_TO_AMU,'Er': 165.930290 * U_TO_AMU,
          'Tm': 168.934211 * U_TO_AMU,'Yb': 173.938858 * U_TO_AMU,'Lu': 174.940768 * U_TO_AMU,'Hf': 179.946549 * U_TO_AMU,
          'Ta': 180.947996 * U_TO_AMU,'W': 183.950933 * U_TO_AMU,'Re': 186.955751 * U_TO_AMU,'Os': 191.961479 * U_TO_AMU,
          'Ir': 192.962924 * U_TO_AMU,'Pt': 194.964774 * U_TO_AMU,'Au': 196.966552 * U_TO_AMU,'Hg': 201.970626 * U_TO_AMU,
          'Tl': 204.974412 * U_TO_AMU,'Pb': 207.976636 * U_TO_AMU,'Bi': 208.980383 * U_TO_AMU,'Po': 208.982416 * U_TO_AMU,
          'At': 209.987131 * U_TO_AMU,'Rn': 222.017570 * U_TO_AMU,'Fr': 223.019731 * U_TO_AMU,'Ra': 226.025403 * U_TO_AMU,
          'Ac': 227.027747 * U_TO_AMU,'Th': 232.038050 * U_TO_AMU,'Pa': 231.035879 * U_TO_AMU,'U': 238.050783 * U_TO_AMU,
          'Np': 237.048167 * U_TO_AMU,'Pu': 244.064198 * U_TO_AMU,'Am': 243.061373 * U_TO_AMU,'Cm': 247.070347 * U_TO_AMU,
          'Bk': 247.070299 * U_TO_AMU,'Cf': 251.079580 * U_TO_AMU,'Es': 252.082972 * U_TO_AMU,'Fm': 257.095099 * U_TO_AMU,
          'Md': 258.098425 * U_TO_AMU,'No': 259.101024 * U_TO_AMU,'Lr': 262.109692 * U_TO_AMU,'Rf': 267. * U_TO_AMU,
          'Db': 268. * U_TO_AMU,'Sg': 269. * U_TO_AMU,'Bh': 270. * U_TO_AMU,'Hs': 270. * U_TO_AMU,'Mt': 278. * U_TO_AMU,
          'Ds': 281. * U_TO_AMU,'Rg': 282. * U_TO_AMU,'Cn': 285. * U_TO_AMU,'Nh': 286. * U_TO_AMU,'Fl': 289. * U_TO_AMU,
          'Mc': 290. * U_TO_AMU,'Lv': 293. * U_TO_AMU,'Ts': 294. * U_TO_AMU,'Og': 294. * U_TO_AMU
          }


class Atom:
    def __init__( self, symbol='??', mass=0, coord=np.array([0.0, 0.0, 0.0]), veloc=np.array([0.0, 0.0, 0.0]), kine=0, mom=np.array([0.0, 0.0, 0.0])):
         self.symbol = symbol
         self.mass = mass
         self.coord = coord
         self.veloc = veloc
         self.kine = kine
         self.mom = mom

def import_geom(filename):
    molecule = []
    natom = 0
    with open(filename) as geom:
        for n, line in enumerate(geom):
            if n == 0:
                natom = int(line)
            else:
                if len(line.split()) != 0:
                    elem = line.split()[0].capitalize()
                    mass = MASSES[elem]
                    coord = np.array(line.split()[1:4])
                    molecule.append(Atom(elem, mass, coord)) 
    if len(molecule) != natom:
        print("file format(.xyz) is error,and please check it again")
    return molecule


def random_velocity(Q, P, velo):
    #Spherical random sampling
    theta = 2 * np.arccos(np.sqrt(1-Q))
    phi = 2 * np.pi * P
    Vx = velo * np.sin(theta) * np.cos(phi)
    Vy = velo * np.sin(theta) * np.sin(phi)
    Vz = velo * np.cos(theta)
    return np.array([Vx, Vy, Vz])


def boltzmann_function(atom, factor=1.0):
    R = 8.31446261853  # J.mol-1.K-1
    AU_veloc = 2.18769125293E6  # Bohr / a.u. => m/s
    v_FC = np.sqrt(8 * R * temperature /
                   ((atom.mass / U_TO_AMU) * 0.001 * np.pi))
    v_min = v_FC * np.sqrt(8 / np.pi)
    if not kinetic_fix:
        factor = np.random.rand()
    v_rand = factor * (v_FC - v_min) + v_min
    Q_random = np.random.rand()
    P_random = np.random.rand()
    veloc = random_velocity(Q_random, P_random, v_rand) / AU_veloc
    atom.veloc = veloc
    atom.kine = np.sum(atom.veloc**2) * atom.mass * 0.5
    atom.mom = atom.veloc * atom.mass
    return


def generate_veloc(ntime, molecule, filename):
    if kinetic_fix:
        rand_factor = np.random.rand()
    if os.path.exists(filename):
        print("remove %s and rewirte!!!!" % filename)
        os.remove(filename)
    for n in range(ntime):
        for i in molecule:
            if kinetic_fix:
                boltzmann_function(i, factor=rand_factor)
            else:
                boltzmann_function(i)
        print_geom(molecule, n, filename)


def print_geom(molecule, time, outputfile):
    Ekin_total = 0.0
    natom = len(molecule)
    for atom in molecule:
        Ekin_total += atom.kine
    with open(outputfile, 'a+') as f:
        f.write(str(natom)+'\n')
        f.write('Ekin  = ' + format(Ekin_total, '15.10f'))
        f.write('   Time = '+str(time+1) + '\n')
        for atom in molecule:
            elem = format(str(atom.symbol), '<5s')
            xyz = ''.join(format(float(x), '>18.10f') for x in atom.coord)
            mom = ''.join(format(float(x), '>18.10f') for x in atom.mom)
            f.write(elem + xyz + mom + '\n')


def main():
    global temperature, kinetic_fix
    temperature = 300.0
    kinetic_fix = True
    filename = 'geo'
    outputfile = 'initconds.xyz'
    ntime = 100
    molecules = import_geom(filename)
    generate_veloc(ntime, molecules, outputfile)


if __name__ == '__main__':
    main()





