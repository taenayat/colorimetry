import numpy as np
import pandas as pd
from scipy.integrate import simps

# Load spectral data
spd_d65_ref = pd.read_csv('1_4_3/D65_Il_2nm.txt', sep='\s+', header=None)[1]
spd_d65_test = pd.read_csv('1_4_3/D65_Lab.txt', sep='\s+', header=None)[0]

reflectance_samples = np.transpose(np.array(pd.read_csv('1_4_3/TestColorSamples_2nm.txt', sep='\s+', header=None)))
cmf_data = pd.read_csv('1_4_3/CMFs_2deg_2nm.txt', sep='\s+', header=None)
wavelengths = cmf_data[0]
cmf_x, cmf_y, cmf_z = cmf_data[1], cmf_data[2], cmf_data[3]



# Function to compute XYZ tristimulus values
def compute_xyz(reflectance, illuminant, cmf_x, cmf_y, cmf_z, wavelengths):
    k = 1 / simps(illuminant * cmf_y, wavelengths)
    X = k * simps(reflectance * illuminant * cmf_x, wavelengths)
    Y = k * simps(reflectance * illuminant * cmf_y, wavelengths)
    Z = k * simps(reflectance * illuminant * cmf_z, wavelengths)
    return X, Y, Z

# Function to compute CIE 1964 U*W*V* values
def xyz_to_uwv(X, Y, Z):
    U = 4 * X / (X + 15 * Y + 3 * Z)
    V = 6 * Y / (X + 15 * Y + 3 * Z)
    W = Y
    return U, W, V

# Function to compute color difference in U*W*V* space
def delta_e_uwv(U1, W1, V1, U2, W2, V2):
    return np.sqrt((U1 - U2)**2 + (W1 - W2)**2 + (V1 - V2)**2)

def von_kries_adaptation(ur, vr, uk, vk, uki, vki):
    cr = (4 - ur - 10*vr) / vr
    dr = (1.708*vr + 0.404 - 1.481*ur) / vr
    ck = (4 - uk - 10*vk) / vk
    dk = (1.708*vk + 0.404 - 1.481*uk) / vk
    cki = (4 - uki - 10*vki) / vki
    dki = (1.708*vki + 0.404 - 1.481*uki) / vki

    u_prime_k_i = (10.872 + 0.404*cr*cki/ck - 4*dr*dki/dk) / (16.518 + 1.481*cr*cki/ck - dr*dki/dk)
    v_prime_k_i = 5.520 / (16.518 + 1.481*cr*cki/ck - dr*dki/dk)
    
    return u_prime_k_i, v_prime_k_i


# Compute XYZ and U*W*V* values for each sample under both illuminants
xyz_ref = []
xyz_sim = []
uwv_ref = []
uwv_sim = []

X_n, Y_n, Z_n = compute_xyz(np.ones_like(wavelengths), spd_d65_ref, cmf_x, cmf_y, cmf_z, wavelengths)
ur, _, vr= xyz_to_uwv(X_n, Y_n, Z_n)
X_d, Y_d, Z_d = compute_xyz(np.ones_like(wavelengths), spd_d65_test, cmf_x, cmf_y, cmf_z, wavelengths)
uk, _, vk= xyz_to_uwv(X_d, Y_d, Z_d)


for reflectance in reflectance_samples:
    X_r, Y_r, Z_r = compute_xyz(reflectance, spd_d65_ref, cmf_x, cmf_y, cmf_z, wavelengths)
    X_s, Y_s, Z_s = compute_xyz(reflectance, spd_d65_test, cmf_x, cmf_y, cmf_z, wavelengths)
    
    ur_i, _, vr_i= xyz_to_uwv(X_r, Y_r, Z_r)
    uk_i, _, vk_i= xyz_to_uwv(X_s, Y_s, Z_s)
    
    u_prime_k_i, v_prime_k_i = von_kries_adaptation(ur, vr, uk, vk, uk_i, vk_i)

    U_r, W_r, V_r = xyz_to_uwv(X_r, Y_r, Z_r)
    U_s, W_s, V_s = u_prime_k_i, Y_s, v_prime_k_i
    
    xyz_ref.append((X_r, Y_r, Z_r))
    xyz_sim.append((X_s, Y_s, Z_s))
    uwv_ref.append((U_r, W_r, V_r))
    uwv_sim.append((U_s, W_s, V_s))


# Compute individual color rendering indices
cri_values = []
for (U_r, W_r, V_r), (U_s, W_s, V_s) in zip(uwv_ref, uwv_sim):
    delta_e = delta_e_uwv(U_r, W_r, V_r, U_s, W_s, V_s)
    cri = 100 - 4.6 * delta_e
    cri_values.append(cri)

# Compute general color rendering index Ra
Ra = np.mean(cri_values[:8])  
print(f"General Color Rendering Index (Ra): {Ra:.2f}")

## without chromatic adaptation CRI: 99.66
## with chromatic adaptation: 99.67