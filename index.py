import pyedflib
import matplotlib.pyplot as plt
# Load EDF file
edf_file = "app/data/raw/files"  # Replace with your file path
f = pyedflib.EdfReader(edf_file)

# Print general file information
print("Number of Signals:", f.signals_in_file)
print("Signal Labels:", f.getSignalLabels())
print("Sample Frequencies:", f.getSampleFrequencies())
print("Start Date:", f.getStartdatetime())
print("File Duration (seconds):", f.file_duration)
onsets, durations, descriptions = f.readAnnotations()

# Print event details
print("EDF Event Annotations:")
for i in range(len(onsets)):
    print(f"Event {i+1}:")
    print(f"  Onset (start time): {onsets[i]} sec")
    print(f"  Duration: {durations[i]} sec")
    print(f"  Description: {descriptions[i]}")
# Print detailed information for each signal
# for i in range(f.signals_in_file):
#     print(f"\nSignal {i + 1}: {f.getLabel(i)}")
#     print(f"  Sample Rate: {f.getSampleFrequency(i)} Hz")
#     print(f"  Physical Min/Max: {f.getPhysicalMinimum(i)} / {f.getPhysicalMaximum(i)}")
#     print(f"  Digital Min/Max: {f.getDigitalMinimum(i)} / {f.getDigitalMaximum(i)}")

# Close file
signal_index = 1  # Change to select another channel
n_samples = f.getNSamples()[signal_index]
print(n_samples)
signal = f.readSignal(signal_index)
print(signal)
# Plot EEG signal
plt.figure(figsize=(10, 4))
plt.plot(signal[:1000])  # Plot first 1000 samples
plt.title(f"EEG Signal: {f.getLabel(signal_index)}")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (µV)")
plt.show()

# Close file
f.close()
