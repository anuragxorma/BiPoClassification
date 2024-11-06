import numpy as np
import dask.dataframe as dd

#reading the file using dask
df = dd.read_csv('osiris_toydata_7.csv', sep=' ', header=None)
df = df.rename(columns={0: "time", 1: "energy", 2: "x", 3: "y", 4: "z", 9: "truth"})

#dropping the empty clumns
df = df.drop([5,6,7,8], axis=1) 

# Convert all columns to float
df = df.astype(float)
