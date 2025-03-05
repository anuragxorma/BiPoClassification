import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd

# Reading the file using dask
df = dd.read_csv('data_preprocessing/osiris_toydata_7.csv', sep=',', header=None)
df = df.rename(columns={0: "time", 1: "energy", 2: "x", 3: "y", 4: "z", 9: "truth"})

# Dropping the empty columns
df = df.drop([5,6,7,8], axis=1)

# Convert all columns to float
df = df.astype(float)

#energy is saved as photoevents(p.e.) and 1MeV corresponds to about 280 events
df['energy'] = df['energy'] / 280

# Compute dataframe to Pandas
df_pandas = df.compute()

# Function to calculate time differences for a given event pair
def calculate_time_differences(df, start_event, end_event):
    start_times = df[df.truth == start_event]['time'].values
    end_times = df[df.truth == end_event]['time'].values
    time_diffs = end_times - start_times[:len(end_times)]  # Ensure same length
    return time_diffs

# Calculate time differences for BiPo214 and BiPo212
time_diff_1 = calculate_time_differences(df_pandas, 1, 2)
time_diff_2 = calculate_time_differences(df_pandas, 4, 5)

# Threshold filter
time_threshold_1 = 0.001
time_diff_1_filtered = [x for x in time_diff_1 if x <= time_threshold_1 and x != 0]

time_threshold_2 = 5e-6
time_diff_2_filtered = [x for x in time_diff_2 if x <= time_threshold_2 and x != 0]

# Plot histogram for BiPo214
def plot_histogram(data, title, filename, color, time_threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=100, color=color, alpha=0.7)
    plt.xlabel('Time Differences (seconds)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, time_threshold)
    plt.savefig(filename)
    plt.show()

plot_histogram(time_diff_1_filtered, 'Histogram of Time Differences (BiPo214)', 'plots/data_viz/decay_time_bipo214.png', 'blue', time_threshold_1)
plot_histogram(time_diff_2_filtered, 'Histogram of Time Differences (BiPo212)', 'plots/data_viz/decay_time_bipo212.png', 'blue', time_threshold_2)

# Plot histograms of energies
plt.figure(figsize=(10, 5))
plt.hist(df_pandas[df_pandas.truth == 1].energy, bins=100, color='blue', alpha=0.7, label='Bi214')
plt.hist(df_pandas[df_pandas.truth == 4].energy, bins=100, color='salmon', alpha=0.7, label='Bi212')
plt.xlabel('Energy [MeV]')
plt.ylabel('Frequency')
plt.title('Energy Distribution of Bi214 and Bi212')
plt.legend()
plt.grid(True)
plt.savefig('plots/data_viz/energy_bi214_bi212.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(df_pandas[df_pandas.truth == 2].energy, bins=100, color='blue', alpha=0.7, label='Po214')
plt.hist(df_pandas[df_pandas.truth == 5].energy, bins=100, color='salmon', alpha=0.7, label='Po212')
plt.xlabel('Energy [MeV]')
plt.ylabel('Frequency')
plt.title('Energy Distribution of Po214 and Po212')
plt.legend()
plt.grid(True)
plt.savefig('plots/data_viz/energy_po214_po212.png')
plt.show()

# Plot histogram of all energies with truth distributions
plt.figure(figsize=(12, 6))
plt.hist(df_pandas.energy, bins=200, color='gray', alpha=0.5, label='All Events')

# Overlay histograms for individual truth labels
for label, color in zip([1, 2, 4, 5], ['blue', 'red', 'green', 'black']):
    plt.hist(df_pandas[df_pandas.truth == label].energy, bins=100, alpha=0.5, color=color, label=f'Truth {label}')

plt.xlabel('Energy [MeV]')
plt.ylabel('Frequency')
plt.title('Overall Energy Distribution with Individual Truth Contributions')
plt.legend()
plt.grid(True)
plt.savefig('plots/data_viz/all_energy_distribution.png')
plt.show()