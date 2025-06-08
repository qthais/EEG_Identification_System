import os
import numpy as np
import mne
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.signal import spectrogram
import cv2
import collections

# Parameters
DATA_DIR = ""
SAMPLE_RATE = 160
TIME_WINDOW = 2
STRIDE = 1
CHANNELS = ['Fp1.', 'Fpz.', 'Fp2.', 'Fz..', 'Cz..']  # Using 5 channels

def load_eeg_data(subjects, channels):
    X, y = [], []
    for subject in subjects:
        try:
            subject_id = int(subject[1:]) - 1
        except ValueError:
            continue
        subject_folder = os.path.join(DATA_DIR, subject)
        edf_files = [f for f in os.listdir(subject_folder) if f.endswith(".edf") and "R01" in f]
        for edf_file in edf_files:
            edf_path = os.path.join(subject_folder, edf_file)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue
            raw.pick(channels)  # Use new pick API
            raw.filter(0.5, 40, fir_design='firwin', verbose=False)
            annotations = raw.annotations
            t0_events = [ann for ann in annotations if ann['description'] == "T0"]
            if not t0_events:
                print(f"⚠️ No T0 events for {subject}")
                continue
            for ann in t0_events:
                event_start = int(ann['onset'] * SAMPLE_RATE)
                event_duration = int(ann['duration'] * SAMPLE_RATE) if ann['duration'] > 0 else raw.n_times - event_start
                seg_length = int(TIME_WINDOW * SAMPLE_RATE)
                stride_samples = int(STRIDE * SAMPLE_RATE)
                for offset in range(0, event_duration - seg_length + 1, stride_samples):
                    start_sample = event_start + offset
                    end_sample = start_sample + seg_length
                    if end_sample > raw.n_times:
                        break
                    segment = raw.get_data(start=start_sample, stop=end_sample)  # shape: (channels, samples)
                    spec_img = create_multichannel_spectrogram(segment)
                    if spec_img is not None:
                        X.append(spec_img)
                        y.append(subject_id)
    return np.array(X), np.array(y)

def create_multichannel_spectrogram(eeg_segment, target_shape=(64,64)):
    specs = []
    # Use nperseg=64, noverlap=32 to get sufficient time bins from each segment
    for ch in range(eeg_segment.shape[0]):
        f, t, Sxx = spectrogram(eeg_segment[ch, :], fs=SAMPLE_RATE, nperseg=64, noverlap=32)
        Sxx = np.log1p(Sxx)
        Sxx_resized = cv2.resize(Sxx, target_shape, interpolation=cv2.INTER_CUBIC)
        specs.append(Sxx_resized)
    return np.stack(specs, axis=-1)  # shape: (64, 64, num_channels)

subjects = sorted([s for s in os.listdir(DATA_DIR) if s.startswith("S") and len(s)==4])
X, y = load_eeg_data(subjects, CHANNELS)
print("Before expanding dims, X shape:", X.shape)
# (Already, each sample is (64,64,5)) – no need to expand dims further
print("Final spectrogram shape:", X.shape)
print("Class distribution:", collections.Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = 109
model = build_cnn_model((64,64,len(CHANNELS)), num_classes)
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_eeg_model.keras", monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=25, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, checkpoint])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)
# model.save("eeg_authentication_cnn.keras")
