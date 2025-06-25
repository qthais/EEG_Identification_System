import os
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import collections
from sklearn.preprocessing import StandardScaler

# Global Parameters
DATA_DIR = "files/"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160
TIME_WINDOW = 2
STRIDE = 1
CHANNELS = ['Fp1.', 'Fp2.', 'O1..', 'O2..', 'Cz..']  # Using 5 channels
N_CLASSES = 109

scaler = StandardScaler()

def normalize_segment(segment):
    return scaler.fit_transform(segment)
# def normalize_segment(segment):
#     return (segment - np.mean(segment, axis=0)) / (np.std(segment, axis=0) + 1e-6)

def load_raw_eeg_segments(data_dir, subject_prefix, edf_keyword, channels,
                          sample_rate=SAMPLE_RATE, time_window=TIME_WINDOW, stride=STRIDE):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    subject_folders = sorted([s for s in os.listdir(data_dir)
                              if s.startswith(subject_prefix) and len(s)==4])
    for folder_name in subject_folders:
        try:
            #Convert to index (001 to 1-1=0 eg..)
            subject_id = int(folder_name[1:]) - 1 
        except ValueError:
            continue
        folder_path = os.path.join(data_dir, folder_name)        #folder_path= files/S001,files/S002,..
        edf_files = [f for f in os.listdir(folder_path) if f.endswith(".edf") and edf_keyword in f]        #find edf files open eye- R01, skip event files
        for edf_file in edf_files: #open eye so the edf files array is 1, the loop is just decoration
            edf_path = os.path.join(folder_path, edf_file)
            #eg files/S001/S001R01.edf
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue
            raw.pick(channels)  # Use new API
            raw.filter(0.5, 40, fir_design='firwin', verbose=False)
            # Áp dụng ICA để loại bỏ nhiễu
            ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter=800)
            ica.fit(raw)

            t0_events = [ann for ann in raw.annotations if ann['description'] == "T0"]
            if not t0_events:
                print(f"⚠️ No T0 events for {folder_name}")
                continue
            seg_length = int(time_window * sample_rate)             # 2*160=320
            stride_samples = int(stride * sample_rate)            # 1* 160
            for ann in t0_events:
                start_sample = int(ann['onset'] * sample_rate) #160*0
                event_duration = int(ann['duration'] * sample_rate) if ann['duration'] > 0 else raw.n_times - start_sample #160*60=9600s
                segments = []
                for offset in range(0, event_duration - seg_length + 1, stride_samples): #[0-320,160-480,320-640,...]
                    seg_start = start_sample + offset
                    seg_end = seg_start + seg_length
                    if seg_end > raw.n_times: # raw.n_times= 9600s-9760s
                        break
                    segment = raw.get_data(start=seg_start, stop=seg_end)  # (channels, time samples) (5,320)
                    segment = segment.T  # (time samples, channels) (320,5)
                    segments.append(segment)
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

# Load data

# Chia tập train và test
X_train, y_train, X_val, y_val, X_test, y_test = load_raw_eeg_segments(DATA_DIR, SUBJECT_PREFIX, EDF_KEYWORD, CHANNELS)

# Reshape lại thành dạng 2D để fit vào StandardScaler
num_train_samples, num_timesteps, num_channels = X_train.shape  # (num_samples, 320, 5)
X_train_reshaped = X_train.reshape(-1, num_channels)  # (num_samples * 320, 5)

# Fit trên tập train
scaler.fit(X_train_reshaped)

# Chuẩn hóa cả tập train và test
X_train = scaler.transform(X_train.reshape(-1, num_channels)).reshape(num_train_samples, num_timesteps, num_channels)
X_test = scaler.transform(X_test.reshape(-1, num_channels)).reshape(X_test.shape[0], num_timesteps, num_channels)

def build_cnn_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = (int(TIME_WINDOW * SAMPLE_RATE), len(CHANNELS))  # (320, 5)
model = build_cnn_lstm_model(input_shape, N_CLASSES)
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_cnn_lstm_model.keras", monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=40,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, checkpoint, lr_scheduler])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)

