import numpy as np
import pandas as pd

hurvich_jameson = np.array([
    [0.85, 1.50, 0.01],
    [1.66, -2.23, 0.37],
    [0.34, 0.06, -0.71]
])

ingling_tsou = np.array([
    [0.6, 0.4, 0],
    [1.2, -1.6, 0.4],
    [0.24, 0.11, -0.7]
])

guth = np.array([
    [0.5967, 0.3634, 0],
    [0.9553, -1.2836, 0],
    [-0.0284, 0, 0.0483]
])

boynton = np.array([
    [1, 1, 0],
    [1, -2, 0],
    [1, 1, -1]
])

MHPE = np.array([
    [0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340, 0.04641],
    [0.00000, 0.00000, 1.00000]
])

XYZ = np.array([
    [95.041, 100.000, 108.869],
    [32.989, 29.788, 24.517],
    [27.485, 28.896, 14.915],
    [23.912, 30.427, 9.906],
    [20.436, 29.483, 21.261],
    [24.988, 30.845, 40.343],
    [28.214, 29.793, 57.802],
    [33.324, 29.375, 53.133],
    [37.633, 31.350, 45.355]
])

model_dict = {'hurvich_jameson':hurvich_jameson,
         'ingling_tsou':ingling_tsou,
         'guth':guth,
         'boynton':boynton}

final_stage = {'hurvich_jameson': lambda arr: np.sum(np.abs(arr)),
         'ingling_tsou':lambda arr: np.sqrt(np.sum(arr**2)),
         'guth':lambda arr: np.sqrt(np.sum(arr**2)),
         'boynton':lambda arr: np.sqrt(np.sum(arr**2))}

cone_values = np.zeros((len(XYZ), len(model_dict), 3))
final_stage_values = np.zeros((len(XYZ), len(model_dict)))


for i, color in enumerate(XYZ):
    for j, model in enumerate(model_dict.keys()):
        cone_values[i, j] = np.matmul(model_dict[model], np.matmul(MHPE,XYZ[i]))
        final_stage_values[i,j] = final_stage[model](cone_values[i,j])


# Define the color samples and models
samples = ['White', 'Color 2', 'Color 3', 'Color 4', 'Color 5', 'Color 6', 'Color 7', 'Color 8', 'Color 9']
models = ['Hurvich Jameson', 'Ingling Tsou', 'Guth', 'Boynton']

# Create dataframes for the opponent channel values and final stage values
df_cone_values = pd.DataFrame(cone_values.reshape(len(samples)*len(models), -1), 
                              index=pd.MultiIndex.from_product([samples, models], names=['Sample', 'Model']),
                              columns=['A', 'T', 'D'])

df_final_stage_values = pd.DataFrame(final_stage_values.flatten(), 
                                     index=pd.MultiIndex.from_product([samples, models], names=['Sample', 'Model']),
                                     columns=['Final Stage Value'])

# Display the dataframes
print("Opponent Channel Values:")
print(df_cone_values)
print("\nFinal Stage Values:")
print(df_final_stage_values)
