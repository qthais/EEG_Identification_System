import mne
import tensorflow as tf
import numpy as np
import joblib
from updateCnn2d import eeg_to_spectrogram
model = tf.keras.models.load_model('app/models/best_cnn2d_model.keras')
edf_file = "app/data/raw/files/S006/S006_12s.edf"
CHANNELS = ['Oz..', 'Iz..','Cz..'] 
SAMPLE_RATE = 160  # EEG Sampling Rate
TIME_WINDOW = 3    # 3 seconds per segment
STRIDE = 0.25    
raw=mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
raw.pick(CHANNELS)  
samples_per_segment=SAMPLE_RATE*TIME_WINDOW
start_sample = raw.n_times - samples_per_segment-160
end_sample = raw.n_times-160
random_segment=raw.get_data(start=start_sample, stop=end_sample).T
print("Random segment shape:", random_segment.shape)
spec = eeg_to_spectrogram(random_segment)
num_channels, freq_bins, time_bins= spec.shape
scaler = joblib.load('app/models/scaler.pkl')
spec_flattened = spec.reshape(-1, freq_bins * time_bins)  # Shape: (channels, freq_bins * time_bins)
spec_transform = scaler.transform(spec_flattened).reshape(num_channels, freq_bins, time_bins)
spec_transform_reshaped = spec_transform.transpose(1, 2, 0)  # Shape: (freq_bins, time_bins, channels)
spec_transform_reshaped = np.expand_dims(spec_transform_reshaped, axis=0)  # Add batch dimension

print(spec_transform_reshaped.shape)

prediction=model.predict(spec_transform_reshaped)
confidence = max(prediction[0])
print(confidence)
predicted_class = np.argmax(prediction, axis=-1)
print(f"Predicted class: {predicted_class}")