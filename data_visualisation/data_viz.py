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
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Events')
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, time_threshold)
    plt.savefig(filename)
    plt.show()

plot_histogram(time_diff_1_filtered, 'Decay Curve(BiPo214)', 'plots/data_viz/decay_time_bipo214.png', 'blue', time_threshold_1)
plot_histogram(time_diff_2_filtered, 'Decay Curve (BiPo212)', 'plots/data_viz/decay_time_bipo212.png', 'blue', time_threshold_2)

# Plot histograms of energies
plt.figure(figsize=(10, 5))
plt.hist(df_pandas[df_pandas.truth == 1].energy, bins=100, color='blue', alpha=0.7, label='Bi214')
plt.hist(df_pandas[df_pandas.truth == 4].energy, bins=100, color='salmon', alpha=0.7, label='Bi212')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
plt.title('Energy Distribution of Bi214 and Bi212')
plt.legend()
plt.grid(True)
plt.savefig('plots/data_viz/energy_bi214_bi212.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(df_pandas[df_pandas.truth == 2].energy, bins=100, color='blue', alpha=0.7, label='Po214')
plt.hist(df_pandas[df_pandas.truth == 5].energy, bins=100, color='salmon', alpha=0.7, label='Po212')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
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
plt.ylabel('Number of Events')
plt.title('Overall Energy Distribution with Individual Truth Contributions')
plt.legend()
plt.grid(True)
plt.savefig('plots/data_viz/all_energy_distribution.png')
plt.show()


##### FITS FOR THE PLOTS ######

from scipy.optimize import curve_fit

# Define exponential decay function
def decay_function(t, N0, tau):
    return N0 * np.exp(-t / tau)

# Bin the histogram data for fitting
hist_1, bin_edges_1 = np.histogram(time_diff_1_filtered, bins=100)
bin_centers_1 = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2  # Compute bin centers

# Bin the histogram data
hist_2, bin_edges_2 = np.histogram(time_diff_2_filtered, bins=100)
bin_centers_2 = (bin_edges_2[:-1] + bin_edges_2[1:]) / 2

# Remove zero-count bins to avoid fitting issues
nonzero_indices = hist_2 > 0
bin_centers_2_filtered = bin_centers_2[nonzero_indices]
hist_2_filtered = hist_2[nonzero_indices]

# Fit the exponential decay function
popt_1, _ = curve_fit(decay_function, bin_centers_1, hist_1, p0=[max(hist_1), 0.0001])
# Fit the exponential decay function
popt_2, _ = curve_fit(decay_function, bin_centers_2_filtered, hist_2_filtered, p0=[max(hist_2_filtered), 0.00001])

# Extract fitted parameters
N0_1, tau_1 = popt_1
N0_2, tau_2 = popt_2

print(fr"Decay time ($\tau$) for BiPo214: {tau_1:.6f} s")
print(fr"Decay time (tau) for BiPo212: {tau_2:.9f} s")

# Plot fitted decay curves
plt.figure(figsize=(10, 5))
plt.scatter(bin_centers_1, hist_1, label='BiPo214', color='blue', alpha=0.6)
plt.plot(bin_centers_1, decay_function(bin_centers_1, *popt_1), label=fr'Fit: $\tau$={tau_1:.3e}s', color='black')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Events')
plt.title('Decay Curve Fitting (BiPo214)')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/decay_fit_bipo214.png')
plt.show()

# Plot fitted decay curve
plt.figure(figsize=(10, 5))
plt.scatter(bin_centers_2_filtered, hist_2_filtered, label='BiPo212', color='red', alpha=0.6)
plt.plot(
    bin_centers_2_filtered, 
    decay_function(bin_centers_2_filtered, *popt_2), 
    label=fr'Fit: $\tau = {tau_2:.3e}\,$s',  # Use scientific notation
    color='black'
)
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Events')
plt.title('Decay Curve Fitting (BiPo212)')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/decay_fit_bipo212.png')
plt.show()
plt.show()

from scipy.stats import norm

# Define Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Fit energy distributions for Bi214 and Bi212
bi_energy = df_pandas[df_pandas.truth == 1].energy.values
hist_bi, bin_edges_bi = np.histogram(bi_energy, bins=100)
bin_centers_bi = (bin_edges_bi[:-1] + bin_edges_bi[1:]) / 2

popt_bi, _ = curve_fit(gaussian, bin_centers_bi, hist_bi, p0=[max(hist_bi), np.mean(bi_energy), np.std(bi_energy)])
A_bi, mu_bi, sigma_bi = popt_bi

print(f"Bi214 Energy Peak: {mu_bi:.3f} MeV, Width: {sigma_bi:.3f} MeV")

# Fit energy distributions for Po214 and Po212
po_energy = df_pandas[df_pandas.truth == 2].energy.values
hist_po, bin_edges_po = np.histogram(po_energy, bins=100)
bin_centers_po = (bin_edges_po[:-1] + bin_edges_po[1:]) / 2

popt_po, _ = curve_fit(gaussian, bin_centers_po, hist_po, p0=[max(hist_po), np.mean(po_energy), np.std(po_energy)])
A_po, mu_po, sigma_po = popt_po

print(f"Po214 Energy Peak: {mu_po:.3f} MeV, Width: {sigma_po:.3f} MeV")

# Plot fitted energy distributions
plt.figure(figsize=(10, 5))
plt.hist(bi_energy, bins=100, color='blue', alpha=0.6, label='Bi214 Data')
plt.plot(bin_centers_bi, gaussian(bin_centers_bi, *popt_bi), color='black', 
         label=fr'Fit: $\mu$={mu_bi:.3f} MeV, $\sigma$={sigma_bi:.3f} MeV')
plt.axvline(mu_bi, color='black', linestyle='dashed', label='Mean (μ)')
plt.axvline(mu_bi - sigma_bi, color='gray', linestyle='dashed', alpha=0.7, label='μ - σ')
plt.axvline(mu_bi + sigma_bi, color='gray', linestyle='dashed', alpha=0.7, label='μ + σ')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
plt.title('Gaussian Fit for Bi214 Energy')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/gaussian_fit_bi214.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(po_energy, bins=100, color='red', alpha=0.6, label='Po214 Data')
plt.plot(bin_centers_po, gaussian(bin_centers_po, *popt_po), color='black', 
         label=fr'Fit: $\mu$={mu_po:.3f} MeV, $\sigma$={sigma_po:.3f} MeV')
plt.axvline(mu_po, color='black', linestyle='dashed', label='Mean (μ)')
plt.axvline(mu_po - sigma_po, color='gray', linestyle='dashed', alpha=0.7, label='μ - σ')
plt.axvline(mu_po + sigma_po, color='gray', linestyle='dashed', alpha=0.7, label='μ + σ')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
plt.title('Gaussian Fit for Po214 Energy')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/gaussian_fit_po214.png')
plt.show()

###BiPo212###

# Fit energy distributions for Bi212
bi_energy = df_pandas[df_pandas.truth == 4].energy.values
hist_bi, bin_edges_bi = np.histogram(bi_energy, bins=100)
bin_centers_bi = (bin_edges_bi[:-1] + bin_edges_bi[1:]) / 2

popt_bi, _ = curve_fit(gaussian, bin_centers_bi, hist_bi, p0=[max(hist_bi), np.mean(bi_energy), np.std(bi_energy)])
A_bi, mu_bi, sigma_bi = popt_bi

print(f"Bi212 Energy Peak: {mu_bi:.3f} MeV, Width: {sigma_bi:.3f} MeV")

# Fit energy distributions for Po214 and Po212
po_energy = df_pandas[df_pandas.truth == 5].energy.values
hist_po, bin_edges_po = np.histogram(po_energy, bins=100)
bin_centers_po = (bin_edges_po[:-1] + bin_edges_po[1:]) / 2

popt_po, _ = curve_fit(gaussian, bin_centers_po, hist_po, p0=[max(hist_po), np.mean(po_energy), np.std(po_energy)])
A_po, mu_po, sigma_po = popt_po

print(f"Po212 Energy Peak: {mu_po:.3f} MeV, Width: {sigma_po:.3f} MeV")

# Plot fitted energy distributions
plt.figure(figsize=(10, 5))
plt.hist(bi_energy, bins=100, color='blue', alpha=0.6, label='Bi212 Data')
plt.plot(bin_centers_bi, gaussian(bin_centers_bi, *popt_bi), color='black', 
         label=fr'Fit: $\mu$={mu_bi:.3f} MeV, $\sigma$={sigma_bi:.3f} MeV')
plt.axvline(mu_bi, color='black', linestyle='dashed', label='Mean (μ)')
plt.axvline(mu_bi - sigma_bi, color='gray', linestyle='dashed', alpha=0.7, label='μ - σ')
plt.axvline(mu_bi + sigma_bi, color='gray', linestyle='dashed', alpha=0.7, label='μ + σ')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
plt.title('Gaussian Fit for Bi212 Energy')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/gaussian_fit_bi212.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(po_energy, bins=100, color='red', alpha=0.6, label='Po212 Data')
plt.plot(bin_centers_po, gaussian(bin_centers_po, *popt_po), color='black', 
         label=fr'Fit: $\mu$={mu_po:.3f} MeV, $\sigma$={sigma_po:.3f} MeV')
plt.axvline(mu_po, color='black', linestyle='dashed', label='Mean (μ)')
plt.axvline(mu_po - sigma_po, color='gray', linestyle='dashed', alpha=0.7, label='μ - σ')
plt.axvline(mu_po + sigma_po, color='gray', linestyle='dashed', alpha=0.7, label='μ + σ')
plt.xlabel('Energy [MeV]')
plt.ylabel('Number of Events')
plt.title('Gaussian Fit for Po212 Energy')
plt.legend()
plt.grid()
plt.savefig('plots/data_viz/gaussian_fit_po212.png')
plt.show()

###EXTERNAL BACKGROUND

# Filter for truth label 3 (external gamma events)
df_truth3 = df_pandas[df_pandas.truth == 3]

# Compute radial distance r = sqrt(x² + y²)
df_truth3['r'] = np.sqrt(df_truth3['x']**2 + df_truth3['y']**2)

# Plot histogram of z-coordinate
plt.figure(figsize=(10, 5))
plt.hist(df_truth3.z, bins=100, color='purple', alpha=0.7)
plt.xlabel('z-coordinate [mm]')
plt.ylabel('Number of Events')
plt.title('Z-coordinate Distribution of External Background (Truth 3)')
plt.grid(True)
plt.savefig('plots/data_viz/z_distribution_truth3.png')
plt.show()

# Plot histogram of radial distance r
plt.figure(figsize=(10, 5))
plt.hist(df_truth3['r'], bins=100, color='green', alpha=0.7)
plt.xlabel(r'Radial Distance r = $\sqrt{x^2 + y^2}$ [mm]')
plt.ylabel('Number of Events')
plt.title('Radial Distribution of External Background (Truth 3)')
plt.grid(True)
plt.savefig('plots/data_viz/r_distribution_truth3.png')
plt.show()
