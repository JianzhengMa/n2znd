#!/usr/bin/env python3
"""
Geoman - A tool for analyzing molecular geometry parameters from trajectory files.

Usage:
  ./geoman.py -f coordinate.xyz -r 1 2      # Bond distance between atoms 1 and 2
  ./geoman.py -f coordinate.xyz -a 1 2 3    # Bond angle of 1-2-3
  ./geoman.py -f coordinate.xyz -d 1 2 3 4  # Dihedral angle between 1-2-3-4
  ./geoman.py -f coordinate.xyz -p 1 2 3 4  # Pyramidalization angle
  ./geoman.py -f coordinate.xyz -r 1 2 -n 20 # Output every 20 steps
  ./geoman.py -f coordinate.xyz -a 1 2 3 -c # Apply angle unwrapping to bond angles

Author: Jianzheng Ma, Fudan University, Shanghai, China, 2024-03-08
"""

import re
import sys
import math
import os
from optparse import OptionParser 

# Constants
ANG_BOHR = 1.88972612
DEG2RAD = math.pi / 180.0

class GeometryAnalyzer:
    def __init__(self):
        self.bohrs = False
        self.radians = False
        self.coord_file = "coordinate.xyz"
        self.fsav = 20
        self.flag = None
        self.con_deg = False
        self.setup_options()

    def setup_options(self):
        """Configure command line options"""
        parser = OptionParser() 
        parser.add_option("-f", dest="coord_file", action="store", type=str,
                        help="Input coordinate file (default: coordinate.xyz)", 
                        default="coordinate.xyz") 
        parser.add_option("-n", dest="fsav", action="store", type=int,
                        help="Output interval in steps (default: 20)", 
                        default=20) 
        parser.add_option("-r", dest="dis", action="store", type=int, nargs=2,
                        help="Bond distance between atoms a and b",
                        default=(0,0)) 
        parser.add_option("-a", dest="ang", action="store", type=int, nargs=3,
                        help="Bond angle of a-b-c",
                        default=(0,0,0)) 
        parser.add_option("-d", dest="dih", action="store", type=int, nargs=4,
                        help="Dihedral angle between a-b-c-d",
                        default=(0,0,0,0)) 
        parser.add_option("-p", dest="pyr", action="store", type=int, nargs=4,
                        help="Pyramidalization angle between a-b and b-c-d plane",
                        default=(0,0,0,0)) 
        parser.add_option("-c", dest="con_deg", action="store_true",
                        help="Constrain angles to continuous values (unwrap)",
                        default=False) 

        options, _ = parser.parse_args()
        self.__dict__.update(options.__dict__)
        self.determine_analysis_type()

    def determine_analysis_type(self):
        """Set analysis flag based on command line arguments"""
        if "-r" in sys.argv:
            self.flag = "dis"
        elif "-a" in sys.argv:
            self.flag = "ang"
        elif "-d" in sys.argv:
            self.flag = "dih"
        elif "-p" in sys.argv:
            self.flag = "pyr"

    def run(self):
        """Main execution method"""
        if not self.flag:
            print("Please specify the geometric parameter type!")
            sys.exit(1)

        coord_list, time_out, step = self.read_coordinates()
        self.write_results(coord_list, time_out, step)
        
        # Apply unwrapping to angles if requested
        if self.con_deg and self.flag in ["ang", "dih", "pyr"]:
            self.unwrap_angles()

    def read_coordinates(self):
        """Read coordinate file and extract atomic positions"""
        try:
            with open(self.coord_file, 'r') as f:
                lines = f.readlines()
            atom_num = int(lines[0].strip())
        except IOError:
            print(f"Could not open {self.coord_file}.")
            sys.exit(1)

        coord_list = []
        coord_temp = []
        time_list = []
        
        for line in lines:
            coord = re.search(r'([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}', line)
            if coord:
                result = coord.group().split()
                coord_temp.append([float(result[1]), float(result[2]), float(result[3])])
                if len(coord_temp) == atom_num:
                    coord_list.append(coord_temp)
                    coord_temp = []
            if "Time" in line:
                time_list.append(float(line.split()[1].split("=")[1]))

        step = len(coord_list) // self.fsav
        time_out = [time_list[int(i * self.fsav)] for i in range(step)] if time_list else []
        
        return coord_list, time_out, step

    @staticmethod
    def scalar_product(a, b):
        """Scalar product of 3-dimensional vectors a and b"""
        return sum(ai * bi for ai, bi in zip(a, b))

    @staticmethod
    def vector_norm(a):
        """Norm of a 3-dimensional vector"""
        return math.sqrt(GeometryAnalyzer.scalar_product(a, a))

    @staticmethod
    def cross_product(a, b):
        """Vector product of 3-dimensional vectors a and b"""
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]

    def angle_between(self, a, b):
        """Angle between two 3-dimensional vectors"""
        x = self.scalar_product(a, b) / (self.vector_norm(a) * self.vector_norm(b))
        angle = math.acos(max(-1.0, min(1.0, x)))
        return angle if self.radians else angle / DEG2RAD

    def calculate_distance(self, a, b):
        """Bond distance between atoms a and b"""
        vector = [bi - ai for ai, bi in zip(a, b)]
        distance = self.vector_norm(vector)
        return distance * ANG_BOHR if self.bohrs else distance

    def calculate_angle(self, a, b, c):
        """Bond angle of a-b-c"""
        r1 = [ai - bi for ai, bi in zip(a, b)]
        r2 = [ci - bi for ci, bi in zip(c, b)]
        return self.angle_between(r1, r2)

    def calculate_dihedral(self, a, b, c, d):
        """Dihedral angle between a-b-c-d"""
        r1 = [bi - ai for ai, bi in zip(a, b)]
        r2 = [ci - bi for ci, bi in zip(c, b)]
        r3 = [di - ci for di, ci in zip(d, c)]
        
        q1 = self.cross_product(r1, r2)
        q2 = self.cross_product(r2, r3)
        
        if q1 == [0., 0., 0.] or q2 == [0., 0., 0.]:
            sys.stderr.write('Undefined dihedral angle!')
            return float('NaN')
            
        Q = self.cross_product(q1, q2)
        if Q == [0., 0., 0.]:
            sign = 1.
        elif self.angle_between(Q, r2) < 90. * (DEG2RAD if not self.radians else 1.):
            sign = 1.
        else:
            sign = -1.
            
        return self.angle_between(q1, q2) * sign

    def calculate_pyramidalization(self, a, b, c, d):
        """Pyramidalization angle between a-b bond and b-c-d plane"""
        r1 = [ai - bi for ai, bi in zip(a, b)]
        r2 = [ci - bi for ci, bi in zip(c, b)]
        r3 = [di - bi for di, bi in zip(d, b)]
        
        q1 = self.cross_product(r2, r3)
        if q1 == [0., 0., 0.]:
            sys.stderr.write('Undefined pyramidalization angle!')
            return float('NaN')
            
        angle = self.angle_between(q1, r1)
        return (90.0 * DEG2RAD - angle) if self.radians else (90.0 - angle)

    def write_results(self, coord_list, time_out, step):
        """Write analysis results to output file"""
        if self.flag == "dis":
            filename = f"distance_{self.dis[0]}_{self.dis[1]}.dat"
            header = f"# distance: {self.dis[0]} {self.dis[1]}\n#\n"
            column_header = "#Time(fs)" if time_out else "#Step"
            column_header += "          distance(ang)\n"
            
            with open(filename, 'w') as f:
                f.write(header + column_header)
                for i in range(step):
                    a = coord_list[int(i * self.fsav)][self.dis[0]-1]
                    b = coord_list[int(i * self.fsav)][self.dis[1]-1]
                    dist = self.calculate_distance(a, b)
                    
                    if time_out:
                        line = f"{time_out[i]:<20.8f}{dist:<20.8f}\n"
                    else:
                        line = f"{int(i * self.fsav):<20d}{dist:<20.8f}\n"
                    f.write(line)

        elif self.flag == "ang":
            filename = f"angle_{self.ang[0]}_{self.ang[1]}_{self.ang[2]}.dat"
            header = f"# angle: {self.ang[0]} {self.ang[1]} {self.ang[2]}\n#\n"
            column_header = "#Time(fs)" if time_out else "#Step"
            column_header += "          angle(deg)\n"
            
            with open(filename, 'w') as f:
                f.write(header + column_header)
                for i in range(step):
                    a = coord_list[int(i * self.fsav)][self.ang[0]-1]
                    b = coord_list[int(i * self.fsav)][self.ang[1]-1]
                    c = coord_list[int(i * self.fsav)][self.ang[2]-1]
                    angle = self.calculate_angle(a, b, c)
                    
                    if time_out:
                        line = f"{time_out[i]:<20.8f}{angle:<20.8f}\n"
                    else:
                        line = f"{int(i * self.fsav):<20d}{angle:<20.8f}\n"
                    f.write(line)

        elif self.flag == "dih":
            filename = f"dihedral_{self.dih[0]}_{self.dih[1]}_{self.dih[2]}_{self.dih[3]}.dat"
            header = f"# dihedral: {self.dih[0]} {self.dih[1]} {self.dih[2]} {self.dih[3]}\n#\n"
            column_header = "#Time(fs)" if time_out else "#Step"
            column_header += "          dihedral(deg)\n"
            
            with open(filename, 'w') as f:
                f.write(header + column_header)
                for i in range(step):
                    a = coord_list[int(i * self.fsav)][self.dih[0]-1]
                    b = coord_list[int(i * self.fsav)][self.dih[1]-1]
                    c = coord_list[int(i * self.fsav)][self.dih[2]-1]
                    d = coord_list[int(i * self.fsav)][self.dih[3]-1]
                    dihedral = self.calculate_dihedral(a, b, c, d)
                    
                    if time_out:
                        line = f"{time_out[i]:<20.8f}{dihedral:<20.8f}\n"
                    else:
                        line = f"{int(i * self.fsav):<20d}{dihedral:<20.8f}\n"
                    f.write(line)

        elif self.flag == "pyr":
            filename = f"pyramid_{self.pyr[0]}_{self.pyr[1]}_{self.pyr[2]}_{self.pyr[3]}.dat"
            header = f"# pyramid: {self.pyr[0]} {self.pyr[1]} {self.pyr[2]} {self.pyr[3]}\n#\n"
            column_header = "#Time(fs)" if time_out else "#Step"
            column_header += "          pyramid(deg)\n"
            
            with open(filename, 'w') as f:
                f.write(header + column_header)
                for i in range(step):
                    a = coord_list[int(i * self.fsav)][self.pyr[0]-1]
                    b = coord_list[int(i * self.fsav)][self.pyr[1]-1]
                    c = coord_list[int(i * self.fsav)][self.pyr[2]-1]
                    d = coord_list[int(i * self.fsav)][self.pyr[3]-1]
                    pyramid = self.calculate_pyramidalization(a, b, c, d)
                    
                    if time_out:
                        line = f"{time_out[i]:<20.8f}{pyramid:<20.8f}\n"
                    else:
                        line = f"{int(i * self.fsav):<20d}{pyramid:<20.8f}\n"
                    f.write(line)

    def unwrap_angles(self):
        """Adjust angles to continuous values by unwrapping 360° jumps"""
        if self.flag == "ang":
            filename = f"angle_{self.ang[0]}_{self.ang[1]}_{self.ang[2]}.dat"
        elif self.flag == "dih":
            filename = f"dihedral_{self.dih[0]}_{self.dih[1]}_{self.dih[2]}_{self.dih[3]}.dat"
        elif self.flag == "pyr":
            filename = f"pyramid_{self.pyr[0]}_{self.pyr[1]}_{self.pyr[2]}_{self.pyr[3]}.dat"
        else:
            return
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        header = lines[:3]
        data_lines = lines[3:]
        
        times, angles = [], []
        for line in data_lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    angles.append(float(parts[1]))
                except ValueError:
                    continue

        if not angles:
            return

        unwrapped = [angles[0]]
        cumulative_correction = 0
        prev = angles[0]
        
        for current in angles[1:]:
            delta = current - prev
            
            # For bond angles (0-180° range), we need special handling
            if self.flag == "ang":
                if delta > 180:
                    cumulative_correction -= 360
                elif delta < -180:
                    cumulative_correction += 360
            else:  # For dihedral and pyramidalization angles
                if delta > 180:
                    cumulative_correction -= 360
                elif delta < -180:
                    cumulative_correction += 360
            
            unwrapped.append(current + cumulative_correction)
            prev = current

        with open(filename, 'w') as f:
            f.writelines(header)
            for t, angle in zip(times, unwrapped):
                f.write(f"{t:<20.8f}{angle:<20.8f}\n")

if __name__ == '__main__':
    analyzer = GeometryAnalyzer()
    analyzer.run()
