import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import feature_engineering_scaling as fes
import numpy as np
from sklearn.model_selection import train_test_split

def create_sequences(X, y, seq_length):
    # Convert to NumPy arrays if not already
    X = np.array(X)
    y = np.array(y)

    # Create sequences using stride_tricks
    num_sequences = len(X) - seq_length + 1
    Xs = np.lib.stride_tricks.as_strided(
        X,
        shape=(num_sequences, seq_length, X.shape[1]),
        strides=(X.strides[0], X.strides[0], X.strides[1])
    )

    ys = y[seq_length - 1 :]

    return Xs, ys

seq_length=10

# Generate sequences
X1_seq, y1_seq = create_sequences(fes.X1_train, fes.y1_train, seq_length)

X1_seq_train, X1_seq_val, y1_seq_train, y1_seq_val = train_test_split(X1_seq, y1_seq, test_size=0.2, random_state=0)

# Generate sequences
X2_seq, y2_seq = create_sequences(fes.X2_train, fes.y2_train, seq_length)

X2_seq_train, X2_seq_val, y2_seq_train, y2_seq_val = train_test_split(X2_seq, y2_seq, random_state=0)