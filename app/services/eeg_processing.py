import numpy as np
import scipy.signal
def eeg_to_spectrogram(eeg_segment, fs=160, nperseg=64, noverlap=32):
    """
    Convert EEG segment (2D: time x channels) to a spectrogram.
    Uses Short-Time Fourier Transform (STFT).
    """
    spectrograms = []
    for ch in range(eeg_segment.shape[1]):  # Loop through each EEG channel
        f, t, Sxx = scipy.signal.spectrogram(eeg_segment[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap) #(480,3)=>(480,ch)
        Sxx = np.log1p(Sxx)  # Apply log transform to make values more normally distributed
        spectrograms.append(Sxx)

    return np.array(spectrograms) 