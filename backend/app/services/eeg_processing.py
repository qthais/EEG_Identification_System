import numpy as np
from scipy.signal import ShortTimeFFT, get_window
def eeg_to_spectrogram(eeg_segment, fs=160, nperseg=64, noverlap=32):
    """
    Convert EEG segment (2D: time x channels) to a spectrogram.
    Uses ShortTimeFFT.spectrogram (recommended modern replacement).
    """
    spectrograms = []

    # Create STFT object once and reuse for all channels
    win = get_window("hann", nperseg)  # Hann window of length nperseg
    hop = nperseg - noverlap           # Step size

    # Initialize STFT object
    stft = ShortTimeFFT(win=win, hop=hop, fs=fs, fft_mode="onesided")

    for ch in range(eeg_segment.shape[1]):  # Loop through each EEG channel
        Sxx = stft.spectrogram(eeg_segment[:, ch])
        Sxx = np.log1p(Sxx)  # Apply log transform to normalize values
        spectrograms.append(Sxx)

    return np.array(spectrograms)