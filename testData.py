import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
# Path to a sample EDF file (adjust this path accordingly)
edf_file = "app/data/raw/files/S002/S002R01.edf"

# Load EDF using MNE
print("edf file:",edf_file)
raw = mne.io.read_raw_edf(edf_file, preload=True)
print("Channel Labels:", raw.info)

# Pick a channel (using the new API: inst.pick(...))
raw.pick(["P7.."])



# Get data from the first 10 seconds for visualization
data, times = raw[:, int(40 * raw.info["sfreq"]):int(50 * raw.info["sfreq"])]
#times= time marks for each sample from 0-10s

plt.figure(figsize=(10, 4))
plt.plot(times, data[0])
plt.title("Raw EEG Signal (Fp1.) - First 10 Seconds")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Compute spectrogram for the first 2-second segment
segment = data[:, :int(3*raw.info["sfreq"])]# take the first 2 in 10s

f, t, Sxx = spectrogram(segment[0], fs=raw.info["sfreq"], nperseg=64, noverlap=32)
#(33,9) in Sxx, 33 corresponding to f shape in spectrogram , 9 corresponding to times from 0 to 1.8, while in Sxx[i,j] display the data segment for the time t[j] and the freq f[i]
Sxx_log = np.log1p(Sxx)

plt.figure(figsize=(8, 4))
plt.pcolormesh(t, f, Sxx_log, shading='gouraud')# can change Sxx_log to Sxx
plt.title("Spectrogram of 3-second EEG Segment (Iz.)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Log-scaled Power')
plt.show()
