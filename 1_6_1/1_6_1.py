import numpy as np
import pandas as pd
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000

df = pd.read_excel('1_6_1/Colorimetry_HPC-ACSI_Homework-1_6-1_Dataset.xlsx')

df['DE_cielab'] = df.apply(lambda row: delta_e_cie1976(LabColor(row.L1, row.a1, row.b1), LabColor(row.L2, row.a2, row.b2)), axis=1)
df['DE_ciede2000'] = df.apply(lambda row: delta_e_cie2000(LabColor(row.L1, row.a1, row.b1), LabColor(row.L2, row.a2, row.b2)), axis=1)

def PF3(delta_E, delta_V):
    N = len(delta_E)
    F = np.sqrt(np.sum(delta_E / delta_V) / np.sum(delta_V / delta_E))
    V_AB = np.sqrt(np.sum(((delta_E - F * delta_V) ** 2) / (delta_E * F * delta_V)) / N)
    f = np.sum(delta_E * delta_V) / np.sum(delta_V ** 2)
    CV = 100 * np.sqrt(np.sum(((delta_E - f * delta_V) ** 2) / (delta_E ** 2)) / N)
    gamma = 10 ** np.sqrt(np.sum((np.log10(delta_E / delta_V) - np.log10(np.sum(delta_E / delta_V) / N)) ** 2) / N)
    return (100 * ((gamma - 1) + V_AB) + CV) / 3

def STRESS(delta_E, delta_V):
    F = np.sum(delta_E * delta_V) / np.sum(delta_E**2)
    numerator = np.sqrt(np.sum((delta_V - F*delta_E)**2))
    denominator = np.sum(delta_V**2)
    return 100 * (numerator / denominator)

stress_cielab = STRESS(df['DE_cielab'], df['DV'])
stress_ciede2000 = STRESS(df['DE_ciede2000'], df['DV'])
pf3_cielab = PF3(df['DE_cielab'], df['DV'])
pf3_ciede2000 = PF3(df['DE_ciede2000'], df['DV'])

print(f"100 * STRESS (CIELAB): {100 * stress_cielab:.2f}")
print(f"100 * STRESS (CIEDE2000): {100 * stress_ciede2000:.2f}")
print(f"PF/3 (CIELAB): {pf3_cielab:.2f}")
print(f"PF/3 (CIEDE2000): {pf3_ciede2000:.2f}")

