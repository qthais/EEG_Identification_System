import os
import numpy as np
import joblib
import mne
import tensorflow as tf
import scipy.signal
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten ,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Global Parameters
DATA_DIR = "files/"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160  # EEG Sampling Rate
TIME_WINDOW = 3  # 3 seconds per segment
STRIDE = 0.3          # 1-second stride
CHANNELS = ['Oz..', 'Iz..','P7..']  # 5 EEG Channels
N_CLASSES = 109

# Function to Convert EEG Segment to Spectrogram
def eeg_to_spectrogram(eeg_segment, fs=160, nperseg=64, noverlap=32):
    """
    Convert EEG segment (2D: time x channels) to a spectrogram.
    Uses Short-Time Fourier Transform (STFT).
    """
    spectrograms = []
    for ch in range(eeg_segment.shape[1]):  # Loop through each EEG channel
        f, t, Sxx = scipy.signal.spectrogram(eeg_segment[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap) #(320,5)=>(320,ch)
        Sxx = np.log1p(Sxx)  # Apply log transform to make values more normally distributed
        spectrograms.append(Sxx)

    return np.array(spectrograms)  # Shape: (num_channels, freq_bins, time_bins)

# Function to Load and Process EEG Data
def load_eeg_split_by_time(data_dir, subject_prefix, edf_keyword, channels,
                           sample_rate=SAMPLE_RATE, time_window=TIME_WINDOW, stride=STRIDE):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    subject_folders = sorted([s for s in os.listdir(data_dir) if s.startswith(subject_prefix) and len(s) == 4])

    for folder_name in subject_folders:
        try:
            subject_id = int(folder_name[1:]) - 1
        except ValueError:
            continue

        folder_path = os.path.join(data_dir, folder_name)
        edf_files = [f for f in os.listdir(folder_path) if f.endswith(".edf") and edf_keyword in f]

        for edf_file in edf_files:
            edf_path = os.path.join(folder_path, edf_file)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue

            raw.pick(channels)
            raw.filter(0.5, 40, fir_design='firwin', verbose=False)

            t0_events = [ann for ann in raw.annotations if ann['description'] == "T0"]
            if not t0_events:
                continue

            seg_length = int(time_window * sample_rate)
            stride_samples = int(stride * sample_rate)

            for ann in t0_events:
                start_sample = int(ann['onset'] * sample_rate)
                event_duration = int(ann['duration'] * sample_rate) if ann['duration'] > 0 else raw.n_times - start_sample

                segments = []
                for offset in range(0, event_duration - seg_length + 1, stride_samples):
                    seg_start = start_sample + offset
                    seg_end = seg_start + seg_length
                    if seg_end > raw.n_times:
                        break
                    segment = raw.get_data(start=seg_start, stop=seg_end).T
                    spec = eeg_to_spectrogram(segment)
                    segments.append(spec)

                # Chia theo tỉ lệ thời gian
                total = len(segments)
                train_end = int(0.6 * total)
                val_end = int(0.8 * total)

                for i, spec in enumerate(segments):
                    if i < train_end:
                        X_train.append(spec)
                        y_train.append(subject_id)
                    elif i < val_end:
                        X_val.append(spec)
                        y_val.append(subject_id)
                    else:
                        X_test.append(spec)
                        y_test.append(subject_id)

    return (np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test))


# Build CNN 2D Model
def build_cnn2d_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN 2D layers
    x = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.02))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)  # Thêm L2 regularization
    x= Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
if __name__ == "__main__":
    # Load EEG Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_eeg_split_by_time(DATA_DIR, SUBJECT_PREFIX, EDF_KEYWORD, CHANNELS)


    # Standardization: Fit on train, transform on both train & test
    scaler = StandardScaler()

    # Reshape data for standardization (Flatten the frequency bins & time bins)
    num_train_samples, num_channels, freq_bins, time_bins = X_train.shape #(5096,5,33,9)
    print('f and t',freq_bins,time_bins)
    X_train_reshaped = X_train.reshape(-1, freq_bins * time_bins)  # Flatten spectrograms (25480,297)
    scaler.fit(X_train_reshaped)  # Fit only on train data
    joblib.dump(scaler, 'scaler.pkl') 
    # Apply Standardization to Train & Test Sets
    X_train = scaler.transform(X_train_reshaped).reshape(num_train_samples, num_channels, freq_bins, time_bins)
    X_val   = scaler.transform(X_val.reshape(-1, freq_bins * time_bins)).reshape(X_val.shape[0], num_channels, freq_bins, time_bins)
    X_test = scaler.transform(X_test.reshape(-1, freq_bins * time_bins)).reshape(X_test.shape[0], num_channels, freq_bins, time_bins)

    # Add channel dimension for Conv2D (Convert to shape: (samples, height, width, channels))
    X_train = np.transpose(X_train, (0, 2, 3, 1))  # Shape: (samples, freq_bins, time_bins, num_channels)
    X_val   = np.transpose(X_val, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))
    #(5096,33,9,5)
    # Define input shape (num_channels, freq_bins, time_bins, 1)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (freq_bins, time_bins, num_channels)
    model = build_cnn2d_model(input_shape, N_CLASSES)
    model.summary()

    # Training Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_cnn2d_model.keras", monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Train Model
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop, checkpoint, lr_scheduler])

    # Evaluate Model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test Accuracy:", test_acc)
