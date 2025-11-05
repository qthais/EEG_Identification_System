import os
import numpy as np
import joblib
import mne
import tensorflow as tf
from scipy.signal import ShortTimeFFT, get_window
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Global Parameters
DATA_DIR = "backend/app/data/raw/files/"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160
TIME_WINDOW = 3
STRIDE = 0.3
CHANNELS = ['Iz..','O2..','Oz..']
N_CLASSES = 109
TARGET_SUBJECT = 101   # chọn 1 user để làm authentication
def compute_far_frr(y_true, y_scores, threshold=0.5):
    """
    Tính FAR và FRR cho bài toán authentication.
    y_true: 1 = genuine, 0 = impostor
    y_scores: output sigmoid score
    """
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    FAR = fp / (fp + tn + 1e-10)
    FRR = fn / (fn + tp + 1e-10)
    return FAR, FRR

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    eer_threshold = thresholds[idx]
    return eer, eer_threshold

# ----------------- Feature Extraction -----------------
def eeg_to_spectrogram(eeg_segment, fs=160, nperseg=64, noverlap=32):
    spectrograms = []
    win = get_window("hann", nperseg)
    hop = nperseg - noverlap
    stft = ShortTimeFFT(win=win, hop=hop, fs=fs, fft_mode="onesided")
    for ch in range(eeg_segment.shape[1]):
        Sxx = stft.spectrogram(eeg_segment[:, ch])
        Sxx = np.log1p(Sxx)
        spectrograms.append(Sxx)
    return np.array(spectrograms)

# ----------------- Dataset Loader -----------------
def load_eeg_authentication(data_dir, subject_prefix, edf_keyword, channels,
                            target_subject, sample_rate=SAMPLE_RATE,
                            time_window=TIME_WINDOW, stride=STRIDE):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    subject_folders = sorted([s for s in os.listdir(data_dir) if s.startswith(subject_prefix)])

    for folder_name in subject_folders:
        try:
            subject_id = int(folder_name[1:]) - 1
        except ValueError:
            continue

        folder_path = os.path.join(data_dir, folder_name)
        edf_files = [f for f in os.listdir(folder_path) if f.endswith(".edf") and edf_keyword in f]
        all_segments = []
        for edf_file in edf_files:
            edf_path = os.path.join(folder_path, edf_file)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue

            raw.pick(channels)
            t0_events = [ann for ann in raw.annotations if ann['description'] == "T0"]
            if not t0_events:
                continue

            seg_length = int(time_window * sample_rate)
            stride_samples = int(stride * sample_rate)

            for ann in t0_events:
                start_sample = int(ann['onset'] * sample_rate)
                event_duration = int(ann['duration'] * sample_rate) if ann['duration'] > 0 else raw.n_times - start_sample

                for offset in range(0, event_duration - seg_length + 1, stride_samples):
                    seg_start = start_sample + offset
                    seg_end = seg_start + seg_length
                    if seg_end > raw.n_times:
                        break
                    segment = raw.get_data(start=seg_start, stop=seg_end).T
                    spec = eeg_to_spectrogram(segment)
                    all_segments.append(spec)

        # nếu subject này không có dữ liệu thì skip
        if len(all_segments) == 0:
            continue

        # gắn nhãn binary: 1 nếu là target_subject, 0 nếu impostor
        labels = [1 if subject_id == target_subject else 0] * len(all_segments)

        # chia train/val/test cho riêng subject này
        total = len(all_segments)
        train_end = int(0.6 * total)
        val_end = int(0.8 * total)

        X_train.extend(all_segments[:train_end])
        y_train.extend(labels[:train_end])

        X_val.extend(all_segments[train_end:val_end])
        y_val.extend(labels[train_end:val_end])

        X_test.extend(all_segments[val_end:])
        y_test.extend(labels[val_end:])

    return (np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test))


# ----------------- Model Definition -----------------
def build_cnn2d_auth_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.03))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)   # binary output

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------- Main -----------------
if __name__ == "__main__":
    # Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_eeg_authentication(DATA_DIR, SUBJECT_PREFIX, EDF_KEYWORD, CHANNELS, TARGET_SUBJECT)
    # Standardize
    scaler = StandardScaler()
    num_train, num_channels, f_bins, t_bins = X_train.shape
    X_train_flat = X_train.reshape(-1, f_bins * t_bins)
    scaler.fit(X_train_flat)
    joblib.dump(scaler, "backend/app/models/scaler_auth.pkl")

    X_train = scaler.transform(X_train_flat).reshape(num_train, num_channels, f_bins, t_bins)
    X_val   = scaler.transform(X_val.reshape(-1, f_bins*t_bins)).reshape(X_val.shape[0], num_channels, f_bins, t_bins)
    X_test  = scaler.transform(X_test.reshape(-1, f_bins*t_bins)).reshape(X_test.shape[0], num_channels, f_bins, t_bins)

    # For Conv2D → (samples, f_bins, t_bins, channels)
    X_train = np.transpose(X_train, (0,2,3,1))
    X_val   = np.transpose(X_val, (0,2,3,1))
    X_test  = np.transpose(X_test, (0,2,3,1))

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_cnn2d_auth_model(input_shape)
    model.summary()

    # Training
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("backend/app/models/best_auth_model.keras", monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop, checkpoint, lr_scheduler])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Authentication Accuracy:", test_acc)

    # Predict scores
    y_scores = model.predict(X_test).ravel()

    # Compute FAR & FRR at threshold=0.5
    FAR, FRR = compute_far_frr(y_test, y_scores, threshold=0.05)
    print(f"FAR @0.5: {FAR}, FRR @0.5: {FRR}")

    # Compute EER
    eer, thr = compute_eer(y_test, y_scores)
    print(f"EER: {eer} at threshold={thr}")


    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc}")

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (1 - FRR)")
    plt.title("Authentication ROC Curve")
    plt.legend()
    plt.savefig("auth_roc.png", dpi=200)
    plt.show()
