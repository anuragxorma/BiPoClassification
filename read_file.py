import numpy as np
import dask.dataframe as dd

df = dd.read_csv('/home/pipc06/Documents/OSIRIS/Toy Data/osiris_toydata_7.csv', sep=' ', header=None)
df = df.rename(columns={0: "time", 1: "energy", 2: "x", 3: "y", 4: "z", 9: "truth"})

df = df.drop([5,6,7,8], axis=1) 

# Convert all columns to float
df = df.astype(float)

df['energy'] = df['energy'] / 280

# Calculate additional features: time differences and distances
df['time_diff'] = df['time'].diff().fillna(0)
df['distance'] = np.sqrt(df['x'].diff().fillna(0)**2 + df['y'].diff().fillna(0)**2 + df['z'].diff().fillna(0)**2)
df['r'] = np.sqrt(df['x']**2 + df['y']**2)

# Persist the DataFrame to improve performance
df = df.persist()
