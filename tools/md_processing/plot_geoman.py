#!/usr/bin/env python3.9
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False
# File name
filename = 'distance_1_2.dat'
#filename = "dihedral_5_2_1_3.dat"

# Get the prefix of the filename for the y-axis label
title_prefix = filename.split('.')[0]

# Initialize lists for x and y data
x_data = []
y_data = []

# Read data from the file
with open(filename, 'r') as file:
    # Skip the first three lines
    for _ in range(3):
        next(file)
    
    # Read the data from the remaining lines
    for line in file:
        values = line.split()  # Split the line into values
        x_data.append(float(values[0]))  # First column data as x-axis
        y_data.append(float(values[1]))  # Second column data as y-axis

# Create the plot
plt.plot(x_data, y_data)
#plt.title('Line Plot of Data from {}'.format(filename))  # Title of the plot
plt.xlabel('Time (fs)',fontsize=16, labelpad=5)  # X-axis label
plt.ylabel("Ti-O (Å)",fontsize=16, labelpad=5)  # Y-axis label based on filename
plt.tick_params(which='both', labelsize=12)
plt.tight_layout(pad=0.2)
plt.savefig(title_prefix + '.png', dpi=720)
