import numpy as np
import pandas as pd
from scipy.integrate import simps
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000
import scipy.io as sio
import pickle 
import matplotlib.pyplot as plt

mat = sio.loadmat('1_4_1/HW141_data.mat')

# Load data
reflectances = np.transpose(mat['reflectance'])
wavelengths = pd.Series(mat['wavelength'].astype(float).flatten())
d65_illuminant = pd.Series(mat['source'].flatten())
cmf_data = pd.DataFrame(mat['cmf'], columns=['x','y','z'])
cmf_x, cmf_y, cmf_z = cmf_data.x, cmf_data.y, cmf_data.z

# Constants for normalization
k = 100 / simps(d65_illuminant * cmf_y, wavelengths)

# Function to compute XYZ from reflectance
def compute_xyz(reflectance):
    X = k * simps(reflectance * d65_illuminant * cmf_x, wavelengths)
    Y = k * simps(reflectance * d65_illuminant * cmf_y, wavelengths)
    Z = k * simps(reflectance * d65_illuminant * cmf_z, wavelengths)
    return X, Y, Z

# Function to convert XYZ to CIELAB
def xyz_to_lab(X, Y, Z, X_n, Y_n, Z_n):
    def f(t):
        delta = 6 / 29
        return t ** (1/3) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4 / 29)
    
    L = 116 * f(Y / Y_n) - 16
    a = 500 * (f(X / X_n) - f(Y / Y_n))
    b = 200 * (f(Y / Y_n) - f(Z / Z_n))
    return L, a, b

# Reference white under D65
X_n, Y_n, Z_n = compute_xyz(np.ones_like(wavelengths))

# Compute CIELAB values for all Munsell chips
lab_values = []
for reflectance in reflectances:
    X, Y, Z = compute_xyz(reflectance)
    L, a, b = xyz_to_lab(X, Y, Z, X_n, Y_n, Z_n)
    lab_values.append(LabColor(L, a, b))

# Compute nearest neighbors for each color difference formula
nearest_neighbors = {'CIELAB': [], 'CIE94': [], 'CIEDE2000': []}
delta_e_values = {'CIELAB': [], 'CIE94': [], 'CIEDE2000': []}

for i, lab1 in enumerate(lab_values):
    print(i)
    min_dist_76 = min_dist_94 = min_dist_2000 = float('inf')
    nn_76 = nn_94 = nn_2000 = None

    for j, lab2 in enumerate(lab_values):
        if i == j:
            continue
        dist_76 = delta_e_cie1976(lab1, lab2)
        dist_94 = delta_e_cie1994(lab1, lab2)
        dist_2000 = delta_e_cie2000(lab1, lab2)
        
        if dist_76 < min_dist_76:
            min_dist_76 = dist_76
            nn_76 = j
        if dist_94 < min_dist_94:
            min_dist_94 = dist_94
            nn_94 = j
        if dist_2000 < min_dist_2000:
            min_dist_2000 = dist_2000
            nn_2000 = j

    nearest_neighbors['CIELAB'].append(nn_76)
    nearest_neighbors['CIE94'].append(nn_94)
    nearest_neighbors['CIEDE2000'].append(nn_2000)
    delta_e_values['CIELAB'].append(min_dist_76)
    delta_e_values['CIE94'].append(min_dist_94)
    delta_e_values['CIEDE2000'].append(min_dist_2000)

# Write
with open('1_4_1/nearest_neighbors.pkl', 'wb') as f:
    pickle.dump(nearest_neighbors, f)
with open('1_4_1/delta_e.pkl', 'wb') as f:
    pickle.dump(delta_e_values, f)

# Read   
with open('1_4_1/nearest_neighbors.pkl', 'rb') as f:
    nearest_neighbors = pickle.load(f)
with open('1_4_1/delta_e.pkl', 'rb') as f:
    delta_e_values = pickle.load(f)


# Analysis: Agreement with Visual Perception
# Compare average ΔE values to see which formula aligns better with perceived uniformity
avg_delta_e = {key: np.mean(values) for key, values in delta_e_values.items()}
print("Average ΔE values for nearest neighbors:")
for key, value in avg_delta_e.items():
    print(f"{key}: {value:.4f}")

# Analysis: Consistency in Nearest Neighbors
# Check if the nearest neighbor identified by each formula is the same for each chip
consistency_count = 0
total_chips = len(reflectances)

for i in range(total_chips):
    if nearest_neighbors['CIELAB'][i] == nearest_neighbors['CIE94'][i] == nearest_neighbors['CIEDE2000'][i]:
        consistency_count += 1

consistency_percentage = (consistency_count / total_chips) * 100
print(f"Percentage of chips with consistent nearest neighbors across all formulas: {consistency_percentage:.2f}%")

# Optional: Detailed comparison of nearest neighbors
for i in range(total_chips):
    print(f"Chip {i+1}:")
    print(f"  CIELAB nearest neighbor: {nearest_neighbors['CIELAB'][i]}")
    print(f"  CIE94 nearest neighbor: {nearest_neighbors['CIE94'][i]}")
    print(f"  CIEDE2000 nearest neighbor: {nearest_neighbors['CIEDE2000'][i]}")
    if nearest_neighbors['CIELAB'][i] != nearest_neighbors['CIE94'][i] or nearest_neighbors['CIELAB'][i] != nearest_neighbors['CIEDE2000'][i]:
        print("  Nearest neighbors are not consistent across formulas.")
    else:
        print("  Nearest neighbors are consistent across all formulas.")

plt.figure()
for key, values in delta_e_values.items():
    plt.plot(values, label=key)
plt.legend()
plt.xlabel('Munsell Color Index')
plt.ylabel('Delta E')
plt.show()
