import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import feature_engineering_scaling as fes
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def create_sequences(X, y, seq_length):
    # Convert to NumPy arrays if not already
    X_np = X.to_numpy(dtype=np.float64)
    y = np.array(y)
    
    # Create sequences using stride_tricks
    num_sequences = len(X_np) - seq_length + 1
    Xs = np.lib.stride_tricks.as_strided(
        X_np,
        shape=(num_sequences, seq_length, X_np.shape[1]),
        strides=(X_np.strides[0], X_np.strides[0], X_np.strides[1])
    )

    # Create sequences for labels using stride_tricks
    ys = np.lib.stride_tricks.as_strided(
        y,
        shape=(num_sequences, seq_length, y.shape[1]),
        strides=(y.strides[0], y.strides[0], y.strides[1])
    )
    
    # Initialize feature indices
    time_column_idx = 4  # time is the 5th feature
    r_column_idx = 1     # r is the 2nd feature
    z_column_idx = 2     # z is the 3rd feature
    phi_column_idx = 3   # phi is the 4th feature

    # Compute relative time differences
    relative_time_diffs = Xs[:, :, time_column_idx] - Xs[:, 0, time_column_idx][:, None]

    # Remove the original time column
    Xs = np.delete(Xs, time_column_idx, axis=2)

    # Append relative time differences as a new feature
    Xs = np.concatenate((Xs, relative_time_diffs[:, :, None]), axis=2)

    # Extract r, z, phi columns
    r = Xs[:, :, r_column_idx]
    z = Xs[:, :, z_column_idx]
    phi = Xs[:, :, phi_column_idx]
    
    # First event values for distance calculations
    r0 = r[:, 0][:, None]
    z0 = z[:, 0][:, None]
    phi0 = phi[:, 0][:, None]
    
    # Compute distances relative to the first event
    distances = np.sqrt(
        ((r * np.cos(phi)) - (r0 * np.cos(phi0)))**2 +
        ((r * np.sin(phi)) - (r0 * np.sin(phi0)))**2 +
        (z - z0)**2
    )
    
    # Append distances as a new feature
    Xs = np.concatenate((Xs, distances[:, :, None]), axis=2)
    
    # Replace phi with scaled sin(phi)
    Xs[:, :, phi_column_idx] = np.sin(phi)
    
    return Xs, ys

seq_length=10

y_cat = to_categorical(fes.y)

# Generate sequences
X_seq, y_seq = create_sequences(fes.X_scaled, y_cat, seq_length)

# Further split using train_test_split
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=0)

