import mne
import numpy as np

# Paths to files
full_edf = "app/data/raw/files/S004/S004R01.edf"
edf_12s = "app/data/raw/files/S004/S004_12s.edf"

# Channels to compare
CHANNELS = ['Oz..', 'Iz..', 'Cz..']

# Load the full 60s file
raw_full = mne.io.read_raw_edf(full_edf, preload=True, verbose=False)
raw_full.pick(CHANNELS)

# Get the last 12 seconds from the full EDF
sfreq = int(raw_full.info['sfreq'])  # Should be 160
start = int((raw_full.n_times / sfreq) - 12) * sfreq
end = raw_full.n_times
segment_from_full = raw_full.get_data(start=start, stop=end)

# Load the 12s cropped EDF
raw_12 = mne.io.read_raw_edf(edf_12s, preload=True, verbose=False)
raw_12.pick(CHANNELS)
segment_from_12 = raw_12.get_data()

# Compare both segments
are_equal = np.allclose(segment_from_full, segment_from_12, atol=1e-6)
print("Are the 12s segments equal?", are_equal)

# Optional: print stats if they differ
if not are_equal:
    diff = segment_from_full - segment_from_12
    print("Max difference:", np.max(np.abs(diff)))
