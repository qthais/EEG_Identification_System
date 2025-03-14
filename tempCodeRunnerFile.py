X_train = np.transpose(X_train, (0, 2, 3, 1))  # Shape: (samples, freq_bins, time_bins, num_channels)
# X_test = np.transpose(X_test, (0, 2, 3, 1))