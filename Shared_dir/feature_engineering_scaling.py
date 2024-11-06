from read_file import df
import numpy as np
from dask import compute
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

#energy is saved as photoevents(p.e.) and 1MeV corresponds to about 280 events
df['energy'] = df['energy'] / 280

# Calculate additional features: time differences and distances between the consecutive events since the BiPo events appears very close to each other in time and distance  
df['time_diff'] = df['time'].diff().fillna(0)
df['distance'] = np.sqrt(df['x'].diff().fillna(0)**2 + df['y'].diff().fillna(0)**2 + df['z'].diff().fillna(0)**2)

#convert to cylindrical coordinates since the detector is cylindrical
df['r'] = np.sqrt(df['x']**2 + df['y']**2)

#convert Dask Dataframe to Pandas Dataframe
df = df.compute()

#FEATURE SCALING

# Initialize MinMaxScaler for energy and r
min_max_scaler = MinMaxScaler()

# Apply Min-Max Scaling to energy and r
df[['energy', 'r']] = min_max_scaler.fit_transform(df[['energy', 'r']])

# Initialize RobustScaler for z and time_diff
robust_scaler = RobustScaler()

# Apply Robust Scaling to z and time_diff
df[['z', 'time_diff']] = robust_scaler.fit_transform(df[['z', 'time_diff']])

# Ensure distance values are positive by adding a small constant if necessary
df['distance'] = np.log1p(df['distance'])  # log1p is used to apply log(1 + x) to avoid issues with 0 values

#SEPERATE THE TWO RADIOACTIVE CHAINS SINCE THE FEATURES OVERLAP SIGNIFICANTLY FOR THE TWO CHAINS

# SUBSET 1: Truths 0, 1, 2, 3
df_subset1 = df[df['truth'].isin([0, 1, 2, 3])]
X1 = df_subset1[['energy', 'r', 'z', 'time_diff', 'distance']]
y1 = df_subset1['truth']

# One-hot encode the target variable
y1_cat = to_categorical(y1)

total_events = len(X1)

# Further split using train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1_cat, test_size=0.2, random_state=0)
X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.2, random_state=0)

# SUBSET 2: Truths 0, 3, 4, 5
df_subset2 = df[df['truth'].isin([0, 3, 4, 5])]
X2 = df_subset2[['energy', 'r', 'z', 'time_diff', 'distance']]
y2 = df_subset2['truth']

# One-hot encode the target variable
y2_cat = to_categorical(y2)

# Further split using train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_cat, test_size=0.2, random_state=0)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2, random_state=0)

