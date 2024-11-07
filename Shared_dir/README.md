# README for Shared Directory

## Overview

The data processing for this project involves handling a Monte Carlo simulation dataset that contains events from both the BiPo-214 and BiPo-212 decay chains. Initially, the dataset is loaded and preprocessed using simple techniques due to the simplicity of the current dataset format (CSV). However, when the official Monte Carlo simulation becomes available, this will be replaced with `.ROOT` format files, and the process will be adapted accordingly to handle more complex data structures.

## Dataset Structure

The dataset consists of the following categories:

- **0 (Internal Background)**: 2,156,000 events
- **1 (Bi214)**: 2,165,960 events
- **2 (Po214)**: 2,165,960 events
- **3 (External Background)**: 2,156,000 events
- **4 (Bi212)**: 2,121,073 events
- **5 (Po212)**: 2,121,073 events

Each event is associated with multiple features like time, energy, position (x, y, z), and the truth label (`truth`). Due to the similarities in the feature characteristics of the BiPo-214 and BiPo-212 chains within certain windows of the features, the events are separated into two subsets for model training. The two chains have overlapping characteristics, which may cause events to be classified by both models, leading to the need for further statistical methods to distinguish them.

## Data Preprocessing

The preprocessing pipeline consists of several steps, which are executed in two separate files: `read_file.py` and `feature_engineering_scaling.py`.

### `read_file.py`

The `read_file.py` script reads the dataset from a CSV file and processes it for further feature engineering. The file:

- Loads the dataset using the Dask library for efficient handling of large datasets.
- Renames columns for clarity (time, energy, x, y, z, truth).
- Drops unnecessary columns.
- Converts all values to float type to ensure consistent data format.

For further details, refer to the code in the `read_file.py` file.

### `feature_engineering_scaling.py`

The `feature_engineering_scaling.py` file applies various transformations to prepare the data for model training:

- **Feature Engineering**: 
  - Calculates **time differences** (`time_diff`) and **distances** (`distance`) between consecutive events.
  - Converts `x` and `y` positions into **cylindrical coordinates** (`r`).
- **Scaling**: 
  - Applies Min-Max scaling to `energy` and `r` (radial distance).
  - Applies Robust scaling to `z` and `time_diff` to handle outliers.
- **Log Transformation**: Applies a log transformation to `distance` to avoid issues with zeros.
- **Training Features**: The features used for model training include:
  - **energy**: The energy associated with each event (scaled).
  - **r**: Radial distance (scaled).
  - **z**: The z-coordinate (Robustly scaled).
  - **time_diff**: The time difference between consecutive events (Robustly scaled).
  - **distance**: The Euclidean distance between consecutive events (log-transformed).
- **Data Splitting**: 
  - Splits the dataset into two subsets based on the decay chains: BiPo-214 and BiPo-212, to prevent feature overlap issues during model training.

For further details on feature engineering and data splitting, refer to the code in the `feature_engineering_scaling.py` file.

### Future Work

Due to the overlap in feature characteristics between the BiPo-214 and BiPo-212 chains, events from both chains may be classified by both models. Statistical methods will be developed in the future to identify and distinguish these overlapping events.


