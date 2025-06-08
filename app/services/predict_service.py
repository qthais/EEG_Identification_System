# app/services/predict_service.py
import mne
import numpy as np
import joblib
import tensorflow as tf
from app.services.eeg_processing import eeg_to_spectrogram
from app.services.config import CHANNELS, SAMPLE_RATE, TIME_WINDOW

# Load model once
model = tf.keras.models.load_model('models/best_cnn2d_model.keras')

# Load scaler once
scaler = joblib.load('models/scaler.pkl')

def preprocess_segment(segment):
    num_channels, freq_bins, time_bins = segment.shape
    spec_flattened = segment.reshape(-1, freq_bins * time_bins)
    spec_transform = scaler.transform(spec_flattened).reshape(num_channels, freq_bins, time_bins)
    spec_transform_reshaped = spec_transform.transpose(1, 2, 0)
    spec_transform_reshaped = np.expand_dims(spec_transform_reshaped, axis=0)
    return spec_transform_reshaped

def predict_random_segment(edf_file_path):
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    raw.pick(CHANNELS)
    raw.filter(0.5, 40, fir_design='firwin', verbose=False)

    samples_per_segment = SAMPLE_RATE * TIME_WINDOW
    max_start = raw.n_times - samples_per_segment
    start_sample = np.random.randint(0, max_start)
    end_sample = start_sample + samples_per_segment

    random_segment = raw.get_data(start=start_sample, stop=end_sample).T
    spec = eeg_to_spectrogram(random_segment)
    input_tensor = preprocess_segment(spec)

    prediction = model.predict(input_tensor)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    return {
        "predicted_class": int(predicted_class),
        "raw_prediction": prediction.tolist(),
        "segment_shape": random_segment.shape
    }
