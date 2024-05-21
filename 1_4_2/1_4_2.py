import numpy as np
import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt
import scipy.io as sio
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976

# Load reflectance data
reflectance = np.array(pd.read_csv('1_4_2/HEx_2_Reflect.txt', sep='\t', header=None))

# load illuminant data
Fs = pd.read_csv('1_4_2/HEx_2_Fs.txt', sep='\t', header=None)
D65 = pd.read_csv('1_4_2/HEx_2_D65.txt', header=None)
A = pd.read_csv('1_4_2/HEx_2_A.txt', header=None)
illuminants = pd.concat([D65,A,Fs[[0,1,2]]], axis=1)
illuminants.columns = ['d65','a','f1','f2','f3']
wavelengths = Fs[3]

# load CMFs from previous exercise
mat = sio.loadmat('1_4_1/HW141_data.mat')
cmf_data = pd.DataFrame(mat['cmf'], columns=['x','y','z'])
cmf_x, cmf_y, cmf_z = cmf_data.x, cmf_data.y, cmf_data.z

# Function to compute XYZ from reflectance
def compute_xyz(reflectance, illuminant, cmf_x, cmf_y, cmf_z, wavelengths):
    k = 1 / simps(illuminant * cmf_y, wavelengths)
    X = k * simps(reflectance * illuminant * cmf_x, wavelengths)
    Y = k * simps(reflectance * illuminant * cmf_y, wavelengths)
    Z = k * simps(reflectance * illuminant * cmf_z, wavelengths)
    return X, Y, Z

# Convert XYZ to CIELAB
def xyz_to_lab(X, Y, Z, X_n, Y_n, Z_n):
    def f(t):
        delta = 6 / 29
        return t ** (1/3) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4 / 29)
    
    L = 116 * f(Y / Y_n) - 16
    a = 500 * (f(X / X_n) - f(Y / Y_n))
    b = 200 * (f(Y / Y_n) - f(Z / Z_n))
    return LabColor(L, a, b)

# Standard reference white values for D65
X_n, Y_n, Z_n = compute_xyz(np.ones_like(wavelengths), illuminants['d65'], cmf_x, cmf_y, cmf_z, wavelengths)

# Compute XYZ for each object under each illuminant
illuminants_spds = {'D65': illuminants['d65'], 'A': illuminants['a'], 'F1': illuminants['f1'], 'F2': illuminants['f2'], 'F3': illuminants['f3']}
xyz_values = {illuminant: [] for illuminant in illuminants_spds}
lab_values = {illuminant: [] for illuminant in illuminants_spds}

for illuminant, spd in illuminants_spds.items():
    for reflec in np.transpose(reflectance): # object 0,1,2
        X, Y, Z = compute_xyz(reflec, spd, cmf_x, cmf_y, cmf_z, wavelengths)
        xyz_values[illuminant].append((X, Y, Z))
        lab_values[illuminant].append(xyz_to_lab(X, Y, Z, X_n, Y_n, Z_n))

delta_e = {illuminant: {} for illuminant in illuminants_spds}
for illuminant in illuminants_spds:
    delta_e[illuminant]['0_1'] = delta_e_cie1976(lab_values[illuminant][0], lab_values[illuminant][1])
    delta_e[illuminant]['0_2'] = delta_e_cie1976(lab_values[illuminant][0], lab_values[illuminant][2])

# Compute the metamerism index
metamerism_index = {illuminant: {} for illuminant in ['A', 'F1', 'F2', 'F3']}

for illuminant in metamerism_index:
    metamerism_index[illuminant]['0_1'] = delta_e[illuminant]['0_1'] / delta_e['D65']['0_1']
    metamerism_index[illuminant]['0_2'] = delta_e[illuminant]['0_2'] / delta_e['D65']['0_2']

# check if the objects are indeed metametic under D65
given_XYZ = (42.5, 33.0, 15.1)
for i, (X, Y, Z) in enumerate(xyz_values['D65']):
    print(f"Object {i} XYZ under D65: X = {X:.2f}, Y = {Y:.2f}, Z = {Z:.2f}")
    if np.allclose([X, Y, Z], given_XYZ, atol=0.2):
        print(f"Object {i} IS metameric with given values under D65.")
    else:
        print(f"Object {i} IS NOT metameric with given values under D65.")

# Plot the color signals
plt.figure(figsize=(12, 8))
label_dict = {0: 'object 0', 1:'object 1', 2:'object 2'}
for i in range(3):
    plt.plot(wavelengths, np.transpose(reflectance)[i], label=label_dict[i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Reflectance Spectra of Munsell Colors')
plt.legend()
plt.show()

# Plot the color signals under each illuminant
for illuminant in ['A', 'F1', 'F2', 'F3']:
    plt.figure(figsize=(12, 8))
    for i, obj in enumerate(['Object0', 'Object1', 'Object2']):
        plt.plot(wavelengths, reflectance[:, i] * illuminants_spds[illuminant], label=f'{obj} under {illuminant}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Color Signal')
    plt.title(f'Color Signals under {illuminant}')
    plt.legend()
    plt.show()

# Print the metamerism indices
for illuminant in metamerism_index:
    print(f"Metamerism index for {illuminant}:")
    print(f"  Pair (0,1): {metamerism_index[illuminant]['0_1']:.4f}")
    print(f"  Pair (0,2): {metamerism_index[illuminant]['0_2']:.4f}")

